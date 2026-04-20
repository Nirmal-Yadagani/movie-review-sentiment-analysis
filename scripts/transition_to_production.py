import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

# Notice we removed dagshub.init() and dotenv! 
# GitHub Actions injects the MLflow environment variables securely for us.

def transition_staging_to_production(model_name):
    client = MlflowClient()
    try:
        # Find which version currently holds the "Staging" alias
        staging_model = client.get_model_version_by_alias(name=model_name, alias="Staging")
        version_to_promote = staging_model.version
        
        print(f"Found Staging model (Version {version_to_promote}). Promoting to Production...")
        
        # Apply the Production alias to this version
        client.set_registered_model_alias(
            name=model_name, 
            alias="Production",
            version=version_to_promote
        )
        
        # Optionally, remove the Staging alias so it's clean for the next run
        client.delete_registered_model_alias(name=model_name, alias="Staging")
        
        print(f"Successfully transitioned Version {version_to_promote} to Production!")

    except Exception as e:
        print(f"Error transitioning model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure this matches your registered model name
    model_name = "MLPClassifier" 
    transition_staging_to_production(model_name)