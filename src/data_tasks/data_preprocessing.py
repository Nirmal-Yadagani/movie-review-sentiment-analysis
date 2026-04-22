import os
import sys
import numpy as np
import pandas as pd
import torch
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from tqdm import tqdm

from src.logger import logger
from src.exception import CustomException
from src.utils.helper_functions import load_params

# Set Pandas options globally after imports
pd.set_option('future.no_silent_downcasting', True)


def encode_series_in_batches(text_series: pd.Series, tokenizer: AutoTokenizer, onnx_model: ORTModelForFeatureExtraction, batch_size: int = 32) -> np.ndarray:
    """
    Converts a Pandas Series of text into an array of embeddings using batch processing
    to optimize memory usage and processing speed.
    
    Args:
        text_series (pd.Series): The column of text data to be encoded.
        tokenizer: The HuggingFace tokenizer.
        onnx_model: The exported ONNX model for feature extraction.
        batch_size (int): Number of texts to process simultaneously. Defaults to 32.
        
    Returns:
        np.ndarray: A stacked numpy array containing the mean-pooled embeddings.
    """
    texts = text_series.tolist()
    all_embeddings = []
    
    logger.info("Encoding text in batches", total_texts=len(texts), batch_size=batch_size)
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Text"):
        batch_texts = texts[i : i + batch_size]
        
        # 1. Tokenize the current batch
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        
        # 2. Run the ONNX model
        outputs = onnx_model(**inputs)
        
        # 3. Perform Mean Pooling (incorporating attention mask)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        
        batch_embeddings = (sum_embeddings / sum_mask).detach().numpy()
        all_embeddings.append(batch_embeddings)
        
    return np.vstack(all_embeddings)


def main() -> None:
    """
    Executes the preprocessing pipeline: loads train/test data, generates 
    ONNX-accelerated text embeddings, and saves the processed numpy arrays.
    """
    log = logger.bind(stage="data_preprocessing")
    
    try:
        log.info("Starting data preprocessing")
        
        params = load_params("params.yaml")
        text_column = params['data_preprocessing']['text_column']
        batch_size = params['data_preprocessing']['batch_size']
        embedder_folder = params['data_preprocessing']['embedder_folder']

        train_data = pd.read_csv(os.path.join('data', 'raw', 'train.csv'))
        test_data = pd.read_csv(os.path.join('data', 'raw', 'test.csv'))
        
        log.info("Raw data loaded successfully", 
                 train_data_shape=train_data.shape, 
                 test_data_shape=test_data.shape)
 
        log.info("Loading local ONNX model and tokenizer", model_path=embedder_folder)
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(embedder_folder)
        tokenizer = AutoTokenizer.from_pretrained(embedder_folder)

        # Process Train Data
        log.info("Processing training data")
        train_processed = encode_series_in_batches(train_data[text_column], tokenizer, onnx_model, batch_size)
        train_labels = train_data['sentiment']
        
        # Process Test Data
        log.info("Processing testing data")
        test_processed = encode_series_in_batches(test_data[text_column], tokenizer, onnx_model, batch_size)
        test_labels = test_data['sentiment']
        
        # Save Outputs
        data_save_dir = os.path.join('data', 'processed')
        os.makedirs(data_save_dir, exist_ok=True)
        
        train_path = os.path.join(data_save_dir, 'train_embeddings.npy')
        test_path = os.path.join(data_save_dir, 'test_embeddings.npy')
        
        np.save(train_path, train_processed)
        np.save(os.path.join(data_save_dir, 'train_labels.npy'), train_labels)
        np.save(test_path, test_processed)
        np.save(os.path.join(data_save_dir, 'test_labels.npy'), test_labels)
        
        log.info("Preprocessed data saved successfully", 
                 train_embeddings_path=train_path, 
                 test_embeddings_path=test_path,
                 embedding_dimensions=train_processed.shape)
    
    except Exception as e:
        log.error("Data preprocessing pipeline failed", error=str(e))
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()