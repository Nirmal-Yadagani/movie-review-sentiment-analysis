import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

from src.logger import logger
from src.exception import CustomException

def transition_staging_to_production(model_name: str) -> None:
    """
    Safeguards the current Production model as a fallback, then promotes 
    the Staging model to Production.
    """
    log = logger.bind(stage="ci_cd_promotion", model_name=model_name)
    client = MlflowClient()
    
    try:
        log.info("Searching for Staging model to promote")
        staging_model = client.get_model_version_by_alias(name=model_name, alias="staging")
        version_to_promote = staging_model.version
        log.info("Found Staging model", version=version_to_promote)
        
        # --- NEW: SAFEGUARD CURRENT PRODUCTION ---
        try:
            current_prod = client.get_model_version_by_alias(name=model_name, alias="production")
            # If we found an active production model, give it the fallback alias
            client.set_registered_model_alias(
                name=model_name, 
                alias="fallback", 
                version=current_prod.version
            )
            log.info("Safeguarded old model as fallback", previous_production_version=current_prod.version)
        except mlflow.exceptions.RestException:
            # If this is the very first deployment, there is no Production model to safeguard
            log.info("No existing Production model found to safeguard. Proceeding.")

        # --- PROMOTE THE NEW MODEL ---
        client.set_registered_model_alias(
            name=model_name, 
            alias="production",
            version=version_to_promote
        )
        
        # Remove the Staging alias 
        client.delete_registered_model_alias(name=model_name, alias="staging")
        
        log.info("Successfully transitioned model to production", version=version_to_promote)

    except mlflow.exceptions.RestException as e:
        log.error("MLflow Registry error. Ensure a 'staging' model exists.", error=str(e))
        raise CustomException(e, sys)
    except Exception as e:
        log.error("Failed to transition model", error=str(e))
        raise CustomException(e, sys)

if __name__ == "__main__":
    target_model = os.environ.get("MODEL_NAME", "MLPClassifier")
    transition_staging_to_production(target_model)