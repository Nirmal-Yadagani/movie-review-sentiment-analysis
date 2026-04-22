import mlflow.pyfunc
import pandas as pd
import numpy as np
import torch
import joblib
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

class SentimentPipelineModel(mlflow.pyfunc.PythonModel):
    """
    A custom MLflow PyFunc model that bundles the Tokenizer, ONNX Embedding model, 
    and the Scikit-Learn MLP Classifier into a single, deployable API endpoint.
    """
    
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Executed once during model initialization (e.g., when a Docker container starts).
        Loads the necessary models into memory for inference.
        """
        onnx_path = context.artifacts["onnx_model_dir"]
        self.tokenizer = AutoTokenizer.from_pretrained(onnx_path)
        self.onnx_model = ORTModelForFeatureExtraction.from_pretrained(onnx_path)
        
        mlp_path = context.artifacts["mlp_model"]
        self.classifier = joblib.load(mlp_path)

    def _encode_texts(self, texts: list) -> np.ndarray:
        """Internal helper method to replicate mean-pooling text encoding logic."""
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.onnx_model(**inputs)
        
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        
        return (sum_embeddings / sum_mask).detach().numpy()

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input) -> pd.DataFrame:
        """
        Executed for every API request. Orchestrates the full pipeline from 
        raw text input to human-readable sentiment output.
        """
        if isinstance(model_input, pd.DataFrame):
            texts = model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, list):
            texts = model_input
        else:
            texts = [str(model_input)]

        embeddings = self._encode_texts(texts)
        predictions = self.classifier.predict(embeddings)
        
        sentiment_map = {1: "positive", 0: "negative"}
        results = [sentiment_map[p] for p in predictions]
        
        return pd.DataFrame({"sentiment": results})