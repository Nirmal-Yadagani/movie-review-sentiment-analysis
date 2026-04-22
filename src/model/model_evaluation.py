import os
import sys
import joblib
import json
import dagshub
import mlflow
import numpy as np
from dotenv import load_dotenv

from src.logger import logger
from src.exception import CustomException
from src.utils.helper_functions import load_params
from src.model.custom_pipeline import SentimentPipelineModel

load_dotenv()
dagshub.init(repo_owner='Nirmal-Yadagani', repo_name='movie-review-sentiment-analysis', mlflow=True)

def load_model(model_file_path: str):
    """Load a trained model from the specified file path."""
    try:
        if not os.path.exists(model_file_path):
            raise CustomException(f"Model file not found at path: {model_file_path}", sys)
        model = joblib.load(model_file_path)
        logger.info("Model loaded successfully", model_file_path=model_file_path)
        return model
    except Exception as e:
        raise CustomException(e, sys)
    
  
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model performance returning core classification metrics."""
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        predictions = model.predict(X_test)
        
        evaluation_results = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1_score': f1_score(y_test, predictions, average='weighted')
        }
        logger.info("Model evaluation completed", evaluation_results=evaluation_results)
        return evaluation_results
    except Exception as e:
        raise CustomException(e, sys)
    

def save_json(data: dict, file_path: str, context_message: str) -> None:
    """Generic helper to save dictionaries as JSON files."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(context_message, file_path=file_path)
    except Exception as e:
        raise CustomException(e, sys)

    
def main() -> None:
    log = logger.bind(stage="model_evaluation")
    try:
        mlflow.set_experiment('Movie Review Classifier')
        with mlflow.start_run() as run:
            log.info('Loading testing data')
            X_test = np.load('data/processed/test_embeddings.npy')
            y_test = np.load('data/processed/test_labels.npy')

            params = load_params('params.yaml')['model_evaluation']
            embedder_folder_path = params['embedder_folder_path']
            model_file_path = params['model_file_path']
            metrics_file_path = params['metrics_file_path']
            model_info_file_path = params['model_info_file_path']
            register_model_name = params['register_model_name']
            
            model = load_model(model_file_path)
            evaluation_results = evaluate_model(model, X_test, y_test)
            
            mlflow.log_metrics(evaluation_results)
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())

            artifacts = {
                "onnx_model_dir": embedder_folder_path, 
                "mlp_model": model_file_path
            }

            conda_env = {
                "channels": ["conda-forge"],
                "dependencies": [
                    "python=3.10",
                    "pip",
                    {
                        "pip": [
                            "--extra-index-url https://download.pytorch.org/whl/cpu",
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

            mlflow.pyfunc.log_model(
                name=register_model_name,
                python_model=SentimentPipelineModel(),
                artifacts=artifacts,
                registered_model_name=register_model_name,
                conda_env=conda_env,
                code_paths=["src"]
            )

            save_json(evaluation_results, metrics_file_path, "Evaluation metrics saved successfully")
            
            model_info = {'run_id': run.info.run_id, 'model_path': os.path.basename(model_file_path).split('.')[0]}
            save_json(model_info, model_info_file_path, "Model information saved successfully")
            
            log.info("Model evaluation process completed successfully")

    except Exception as e:
        raise CustomException(e, sys)

if __name__=='__main__':
    main()