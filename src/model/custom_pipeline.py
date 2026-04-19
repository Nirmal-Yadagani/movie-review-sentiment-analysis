import mlflow.pyfunc
import pandas as pd
import numpy as np
import torch
import joblib
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

class SentimentPipelineModel(mlflow.pyfunc.PythonModel):
    """
    A custom MLflow model that bundles the Tokenizer, ONNX Embedding model, 
    and the Scikit-Learn MLP Classifier into a single API endpoint.
    """
    
    def load_context(self, context):
        """
        This method is called ONLY ONCE when the Docker container starts.
        It loads the models into memory so they are ready for inference.
        """
        # 1. Load the ONNX model and Tokenizer from the artifact directory
        onnx_path = context.artifacts["onnx_model_dir"]
        self.tokenizer = AutoTokenizer.from_pretrained(onnx_path)
        self.onnx_model = ORTModelForFeatureExtraction.from_pretrained(onnx_path)
        
        # 2. Load the trained MLP classifier
        mlp_path = context.artifacts["mlp_model"]
        self.classifier = joblib.load(mlp_path)

    def _encode_texts(self, texts):
        """Internal helper method to replicate your mean-pooling logic."""
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.onnx_model(**inputs)
        
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        
        embeddings = (sum_embeddings / sum_mask).detach().numpy()
        return embeddings

    def predict(self, context, model_input):
        """
        This method is called EVERY TIME the API receives a request.
        `model_input` will be the JSON payload converted into a Pandas DataFrame.
        """
        # Extract the raw text from the input
        if isinstance(model_input, pd.DataFrame):
            # Assuming the API sends {"text": ["movie was great"]}
            texts = model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, list):
            texts = model_input
        else:
            texts = [str(model_input)]

        # 1. Convert raw text to 768-dimensional embeddings
        embeddings = self._encode_texts(texts)
        
        # 2. Pass embeddings to the MLP classifier
        predictions = self.classifier.predict(embeddings)
        
        # 3. Map numerical predictions back to human-readable labels
        sentiment_map = {1: "positive", 0: "negative"}
        results = [sentiment_map[p] for p in predictions]
        
        # Return a DataFrame as expected by MLflow's serving schema
        return pd.DataFrame({"sentiment": results})