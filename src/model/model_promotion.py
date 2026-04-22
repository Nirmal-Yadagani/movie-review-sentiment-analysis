import os
import sys
import json
import dagshub
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

from src.logger import logger
from src.exception import CustomException
from src.utils.helper_functions import load_params

load_dotenv()
dagshub.init(repo_owner='Nirmal-Yadagani', repo_name='movie-review-sentiment-analysis', mlflow=True)

def get_metric_by_alias(client: MlflowClient, model_name: str, alias: str, metric_name: str) -> float | None:
    """
    Generic helper to fetch the specified metric for a model holding a specific alias.
    Returns None if the alias does not exist.
    """
    try:
        model_ver = client.get_model_version_by_alias(name=model_name, alias=alias)
        run = client.get_run(model_ver.run_id)
        return run.data.metrics.get(metric_name)
    except Exception:
        logger.debug("Alias not found", model_name=model_name, alias=alias)
        return None

def promote_model() -> None:
    """
    Compares the newly trained model against both the current Production 
    and Staging models. Promotes to Staging only if it is the absolute best model.
    """
    log = logger.bind(stage="model_promotion")
    try:
        log.info("Loading promotion parameters and artifacts")
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
            raise CustomException(f"Metric {target_metric} not found in evaluated metrics.", sys)

        client = MlflowClient()
        
        # 1. Get the newly registered model version
        versions = client.search_model_versions(f"name='{register_model_name}'")
        new_model_version = next((v.version for v in versions if v.run_id == new_run_id), None)
        
        if not new_model_version:
             raise CustomException("Could not find the registered model version for the current run.", sys)

        # 2. Fetch existing metrics for comparison
        log.info("Fetching existing benchmark scores")
        prod_metric = get_metric_by_alias(client, register_model_name, "Production", target_metric)
        staging_metric = get_metric_by_alias(client, register_model_name, "staging", target_metric)

        # 3. Determine the absolute highest score we need to beat
        existing_scores = [score for score in [prod_metric, staging_metric] if score is not None]

        promote = False

        if not existing_scores:
            log.info("No Production or Staging models detected. Fast-tracking to Staging.")
            promote = True
        else:
            # Find the highest existing score (the "Champion")
            score_to_beat = max(existing_scores)
            improvement = new_metric_value - score_to_beat
            
            log.info("Metric comparison", 
                     highest_existing_score=round(score_to_beat, 4), 
                     new_score=round(new_metric_value, 4), 
                     improvement=round(improvement, 4), 
                     threshold=improvement_threshold)

            if improvement >= improvement_threshold:
                log.info("New model is the absolute best! Flagging for CI/CD Staging.")
                promote = True
            else:
                log.info("Model did not beat the highest existing benchmark. Discarding promotion.")

        # 4. Apply the STAGING alias
        if promote:
            client.set_registered_model_alias(
                name=register_model_name, 
                alias="Staging",
                version=new_model_version
            )
            log.info("Model successfully promoted", alias="Staging", version=new_model_version)

    except Exception as e:
        log.error("Model promotion pipeline failed", error=str(e))
        raise CustomException(e, sys)

if __name__ == "__main__":
    promote_model()