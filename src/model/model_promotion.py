import os
import sys
import json
import dagshub
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

from src.logger import logger
from src.exception import MyException
from src.utils.helper_functions import load_params

load_dotenv()
dagshub.init(repo_owner='Nirmal-Yadagani', repo_name='movie-review-sentiment-analysis', mlflow=True)

def get_current_Production_metric(client, model_name, metric_name):
    """Fetches the specified metric for the current PRODUCTION model."""
    try:
        prod_model = client.get_model_version_by_alias(name=model_name, alias="Production")
        prod_run = client.get_run(prod_model.run_id)
        prod_metric = prod_run.data.metrics.get(metric_name)
        return prod_metric, prod_model.version
    except Exception as e:
        logger.info(f"No existing Production model found for {model_name}. Proceeding as first deployment.")
        return None, None

def promote_model():
    try:
        logger.info("Loading promotion parameters and artifacts...")
        params = load_params('params.yaml')
        
        eval_params = params['model_evaluation']
        promo_params = params.get('model_promotion', {})
        
        target_metric = promo_params.get('target_metric', 'f1_score')
        improvement_threshold = promo_params.get('improvement_threshold', 0.01)
        register_model_name = eval_params['register_model_name']
        
        with open(eval_params['metrics_file_path'], 'r') as f:
            new_metrics = json.load(f)
        
        with open(eval_params['model_info_file_path'], 'r') as f:
            model_info = json.load(f)

        new_metric_value = new_metrics.get(target_metric)
        new_run_id = model_info.get('run_id')
        
        if new_metric_value is None:
            raise MyException(f"Metric {target_metric} not found in evaluated metrics.", sys)

        client = MlflowClient()
        
        # 1. Get the newly registered model version
        versions = client.search_model_versions(f"name='{register_model_name}'")
        new_model_version = next((v.version for v in versions if v.run_id == new_run_id), None)
        
        if not new_model_version:
             raise MyException("Could not find the registered model version for the current run.", sys)

        # 2. Compare against PRODUCTION
        logger.info(f"Comparing new model (Run: {new_run_id}) against Production.")
        prod_metric_value, prod_version = get_current_Production_metric(client, register_model_name, target_metric)

        promote = False

        if prod_metric_value is None:
            logger.info("No Production model detected. Fast-tracking new model to Staging.")
            promote = True
        else:
            improvement = new_metric_value - prod_metric_value
            logger.info(f"Production {target_metric}: {prod_metric_value:.4f}")
            logger.info(f"New Model {target_metric}: {new_metric_value:.4f}")
            logger.info(f"Improvement: {improvement:.4f} (Threshold: {improvement_threshold})")

            if improvement >= improvement_threshold:
                logger.info("Model beat Production! Flagging for CI/CD.")
                promote = True
            else:
                logger.info("Model did not beat Production. Discarding.")

        # 3. Apply the STAGING alias (Waiting room for GitHub Actions)
        if promote:
            client.set_registered_model_alias(
                name=register_model_name, 
                alias="Staging",
                version=new_model_version
            )
            logger.info(f"Successfully flagged version {new_model_version} of {register_model_name} as 'Staging'.")

    except Exception as e:
        raise MyException(e, sys)

if __name__ == "__main__":
    promote_model()