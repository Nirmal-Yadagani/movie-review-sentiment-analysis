import os
import sys
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

from src.logger import logger
from src.exception import CustomException
from src.utils.helper_functions import load_params

def main() -> None:
    """
    Downloads a specified HuggingFace model and tokenizer, exports the model 
    to ONNX format for optimized inference, and saves it locally.
    """
    log = logger.bind(stage="model_download")
    
    try:
        log.info("Starting model download and ONNX export stage")
        
        params = load_params("params.yaml")
        embedder_model_name = params['download_model']['embedder_model_name']
        save_folder = params['download_model']['embedder_save_folder']

        os.makedirs(save_folder, exist_ok=True)
 
        log.info("Downloading HuggingFace model and exporting to ONNX", 
                 source_model=embedder_model_name)
                 
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(embedder_model_name, export=False)
        tokenizer = AutoTokenizer.from_pretrained(embedder_model_name)

        onnx_model.save_pretrained(save_folder)
        tokenizer.save_pretrained(save_folder)
        
        log.info("ONNX model and tokenizer saved successfully", save_directory=save_folder)
    
    except Exception as e:
        log.error("Model download and export failed", error=str(e))
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()