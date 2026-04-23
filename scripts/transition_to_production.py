import os
import mlflow
from mlflow.tracking import MlflowClient
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def transition_staging_to_production(model_name: str) -> None:
    """
    Safeguards the current Production model as a fallback, then promotes 
    the Staging model to Production.
    """
    logger.info(f"Initiating transition of '{model_name}' from Staging to Production.")
    client = MlflowClient() 
    
    try:
        logger.info("Searching for Staging model to promote")
        staging_model = client.get_model_version_by_alias(name=model_name, alias="staging")
        version_to_promote = staging_model.version
        logger.info(f"Found Staging model, version={version_to_promote}")

        # --- NEW: SAFEGUARD CURRENT PRODUCTION ---
        try:
            current_prod = client.get_model_version_by_alias(name=model_name, alias="production")
            # If we found an active production model, give it the fallback alias
            client.set_registered_model_alias(
                name=model_name, 
                alias="fallback", 
                version=current_prod.version
            )
            logger.info(f"Safeguarded old model as fallback, previous_production_version={current_prod.version}")
        except mlflow.exceptions.RestException:
            # If this is the very first deployment, there is no Production model to safeguard
            logger.info("No existing Production model found to safeguard. Proceeding.")

        # --- PROMOTE THE NEW MODEL ---
        client.set_registered_model_alias(
            name=model_name, 
            alias="production",
            version=version_to_promote
        )
        
        # Remove the Staging alias 
        client.delete_registered_model_alias(name=model_name, alias="staging")
        
        logger.info(f"Successfully transitioned model to production, version={version_to_promote}")

    except mlflow.exceptions.RestException as e:
        logger.error(f"MLflow Registry error. Ensure a 'staging' model exists, error={str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Failed to transition model, error={str(e)}")
        raise e
    
if __name__ == "__main__":
    target_model = os.environ.get("MODEL_NAME", "MLPClassifier")
    transition_staging_to_production(target_model)
