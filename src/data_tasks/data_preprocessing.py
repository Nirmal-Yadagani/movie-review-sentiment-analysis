import os
import sys

import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from tqdm import tqdm
import torch

from src.logger import logger
from src.exception import MyException
from src.utils.helper_functions import load_params


def encode_series_in_batches(text_series, tokenizer, onnx_model, batch_size=32):
    """
    Safely converts a Pandas Series of text into an array of embeddings.
    """
    # Convert the Pandas Series to a standard Python list (much faster to slice)
    texts = text_series.tolist()
    all_embeddings = []
    
    # Loop through the list in chunks (batches)
    logger.info("Encoding text in batches...", total_texts=len(texts), batch_size=batch_size)
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Text"):
        batch_texts = texts[i : i + batch_size]
        
        # 1. Tokenize the current batch
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        
        # 2. Run the ONNX model
        outputs = onnx_model(**inputs)
        
        # 3. Perform Mean Pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        
        batch_embeddings = (sum_embeddings / sum_mask).detach().numpy()
        
        # Store the results of this batch
        all_embeddings.append(batch_embeddings)
        
    # Stack all the tiny batch arrays back into one massive matrix
    return np.vstack(all_embeddings)


def encode_text(text, tokenizer, onnx_model):
    """
    Safely converts a single string of text into an array of embeddings.
    """
    logger.info("Encoding single text input...")
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    outputs = onnx_model(**inputs)
    
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    
    sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
    
    return (sum_embeddings / sum_mask).detach().numpy()
    

def main():
    try:
        logger.info("Starting data preprocessing...")
        params = load_params("params.yaml")
        text_column = params['data_preprocessing']['text_column']
        embedder_folder = params['data_preprocessing']['embedder_folder']

        train_data = pd.read_csv(os.path.join('data', 'raw', 'train.csv'))
        test_data = pd.read_csv(os.path.join('data', 'raw', 'test.csv'))
        logger.info("Data loaded successfully.", train_data_shape=train_data.shape, test_data_shape=test_data.shape)
 
        # LOAD FROM LOCAL FOLDER (No export=True needed here because it's already an ONNX model)
        logger.info("Loading local ONNX model and tokenizer...", model_path=embedder_folder)
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(embedder_folder)
        tokenizer = AutoTokenizer.from_pretrained(embedder_folder)

        train_processed = encode_series_in_batches(train_data[text_column], tokenizer, onnx_model)
        train_labels = train_data['sentiment']
        test_processed = encode_series_in_batches(test_data[text_column], tokenizer, onnx_model)
        test_labels = test_data['sentiment']
        
        data_save_dir = os.path.join('data', 'processed')
        os.makedirs(data_save_dir, exist_ok=True)
        np.save(os.path.join(data_save_dir, 'train_embeddings.npy'), train_processed)
        np.save(os.path.join(data_save_dir, 'train_labels.npy'), train_labels)
        np.save(os.path.join(data_save_dir, 'test_embeddings.npy'), test_processed)
        np.save(os.path.join(data_save_dir, 'test_labels.npy'), test_labels)
        logger.info("Preprocessed data saved successfully.", 
                    train_embeddings_path=os.path.join(data_save_dir, 'train_embeddings.npy'), 
                    test_embeddings_path=os.path.join(data_save_dir, 'test_embeddings.npy'))
    
    except Exception as e:
        if isinstance(e, MyException):
            raise e
        raise MyException(e, sys)

if __name__ == "__main__":
    main()