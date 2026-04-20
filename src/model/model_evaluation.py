import os
import sys
import joblib
import json

import dagshub
import mlflow
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from src.logger import logger
from src.exception import MyException
from src.utils.helper_functions import load_params
from src.model.custom_pipeline import SentimentPipelineModel


dagshub.init(repo_owner='Nirmal-Yadagani', repo_name='movie-review-sentiment-analysis', mlflow=True)

def load_model(model_file_path):
    """
    Load a trained model from the specified file path.

    Args:
        model_file_path (str): The file path to the saved model.
    Returns:
        The loaded model object.
    Raises:
        FileNotFoundError: If the specified model file does not exist.
        Exception: If there is an error during model loading.
    """
    try:
        if not os.path.exists(model_file_path):
            raise MyException(f"Model file not found at path: {model_file_path}")
        model = joblib.load(model_file_path)
        logger.info(f"Model loaded successfully", model_file_path=model_file_path)
        return model
    
    except Exception as e:
        if isinstance(e, MyException):
            raise e
        raise MyException(e, sys)
    
  
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of a trained model on the test dataset.

    Args:
        model: The trained model to be evaluated.
        X_test: The test features.
        y_test: The true labels for the test dataset.
    Returns:
        A dictionary containing evaluation metrics such as accuracy, precision, recall, and F1-score.
    Raises:
        Exception: If there is an error during model evaluation.
    """
    try:
        predictions = model.predict(X_test)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        evaluation_results = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1_score': f1_score(y_test, predictions, average='weighted')
        }
        logger.info(f"Model evaluation completed successfully", evaluation_results=evaluation_results)
        return evaluation_results
    
    except Exception as e:
        raise MyException(e, sys)
    
def save_metrics(metrics, metrics_file_path):
    """
    Save the evaluation metrics as Json to a specified file path.

    Args:
        metrics (dict): A dictionary containing evaluation metrics.
        metrics_file_path (str): The file path where the metrics should be saved.
    Raises:
        Exception: If there is an error during saving the metrics.
    """
    try:
        os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
        with open(metrics_file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Evaluation metrics saved successfully", metrics_file_path=metrics_file_path)
    
    except Exception as e:
        logger.error(f"Error saving evaluation metrics", error=str(e), metrics_file_path=metrics_file_path)
        raise MyException(e, sys)
    

def save_model_info(run_id, model_name, model_info_file_path):
    """
    Save the model information as Json to a specified file path.

    Args:
        run_id (str): mlflow run id for the current model training run.
        model_path (str): The file path where the trained model is saved.
        model_info_file_path (str): The file path where the model information should be saved.
    raises:
        Exception: If there is an error during saving the model information.
    """
    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_name
        }
        os.makedirs(os.path.dirname(model_info_file_path), exist_ok=True)
        with open(model_info_file_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        logger.info(f"Model information saved successfully", model_info_file_path=model_info_file_path)
    
    except Exception as e:
        logger.error(f"Error saving model information", error=str(e), model_info_file_path=model_info_file_path)
        raise MyException(e, sys)

    
def main():
    try:
        mlflow.set_experiment('Movie Review Classifier')
        with mlflow.start_run():
            logger.info('Loading testing data..', train_data_path='data/processed/')
            X_test = np.load('data/processed/test_embeddings.npy')
            y_test = np.load('data/processed/test_labels.npy')
            logger.info('Testing data loaded successfully.', X_test_shape=X_test.shape, y_test_shape=y_test.shape)

            logger.info("Loading model evaluation parameters...")
            params = load_params('params.yaml')['model_evaluation']
            embedder_folder_path = params['embedder_folder_path']
            model_file_path = params['model_file_path']
            metrics_file_path = params['metrics_file_path']
            model_info_file_path = params['model_info_file_path']
            register_model_name = params['register_model_name']
            logger.info("Model evaluation parameters loaded successfully.", model_file_path=model_file_path, metrics_file_path=metrics_file_path, model_info_file_path=model_info_file_path)
            
            logger.info("Starting model evaluation...")
            model = load_model(model_file_path)
            evaluation_results = evaluate_model(model, X_test, y_test)
            
            mlflow.log_metrics(evaluation_results)
            if hasattr(model, 'get_params'):
                params = model.get_params()
                mlflow.log_params(params)

            # 1. Define the physical files the wrapper class needs to operate
            artifacts = {
                # Assuming you saved the ONNX model to this folder in an earlier step
                "onnx_model_dir": embedder_folder_path, 
                "mlp_model": model_file_path
            }

            # 2. Define the exact Python environment the Docker container must build
            conda_env = {
                "channels": ["conda-forge"],
                "dependencies": [
                    "python=3.10",
                    "pip",
                    {
                        "pip": [
                            # 1. Point pip to the CPU-only repository
                            "--extra-index-url https://download.pytorch.org/whl/cpu",
                            # 2. Explicitly demand the CPU version of torch
                            "torch==2.1.2+cpu", 
                            "mlflow",
                            "optimum[onnxruntime]",
                            "transformers",
                            "scikit-learn",
                            "pandas"
                        ]
                    }
                ],
                "name": "sentiment_env"
            }

            # 3. Log the custom pipeline model
            mlflow.pyfunc.log_model(
                name=register_model_name,
                python_model=SentimentPipelineModel(),
                artifacts=artifacts,
                registered_model_name=register_model_name,
                conda_env=conda_env,
                code_paths=["src"]
            )

            save_metrics(evaluation_results, metrics_file_path)
            save_model_info(mlflow.active_run().info.run_id, os.path.basename(model_file_path).split('.')[0], model_info_file_path)
            logger.info("Model evaluation process completed successfully.")

    except Exception as e:
        if isinstance(e, MyException):
            raise e
        raise MyException(e, sys)
    

if __name__=='__main__':
    main()
    