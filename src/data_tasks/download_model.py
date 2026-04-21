import os
import sys
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

from src.logger import logger
from src.exception import MyException
from src.utils.helper_functions import load_params

def main():
    try:
        logger.info("Starting model download stage...")
        params = load_params("params.yaml")
        embedder_model_name = params['download_model']['embedder_model_name']
        save_folder = params['download_model']['embedder_save_folder']

        os.makedirs(save_folder, exist_ok=True)
 
        # Download and export to ONNX
        logger.info("Downloading ONNX model and tokenizer...", model_name=embedder_model_name)
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(embedder_model_name, export=True)
        tokenizer = AutoTokenizer.from_pretrained(embedder_model_name)

        # Save locally
        onnx_model.save_pretrained(save_folder)
        tokenizer.save_pretrained(save_folder)
        logger.info("ONNX model and tokenizer saved successfully.", save_folder=save_folder)
    
    except Exception as e:
        if isinstance(e, MyException):
            raise e
        raise MyException(e, sys)

if __name__ == "__main__":
    main()