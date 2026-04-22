import os
import sys
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

from src.logger import logger
from src.exception import CustomException
from src.utils.helper_functions import load_params


def train_model(X_train: np.ndarray, y_train: np.ndarray, model: MLPClassifier) -> MLPClassifier:
    """Train the multi-layer perceptron model on the provided data."""
    try:
        logger.info("Training MLPClassifier...", model_type=type(model).__name__)
        model.fit(X_train, y_train)
        logger.info("Model training completed")
        return model
    except Exception as e:
        raise CustomException(e, sys)
    

def save_model(model: MLPClassifier, model_path: str) -> None:
    """Save the trained model to disk securely."""
    try:
        logger.info("Saving model to disk")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logger.info("Model saved successfully", model_path=model_path)
    except Exception as e:
        raise CustomException(e, sys)


def main() -> None:
    """Executes the training pipeline."""
    log = logger.bind(stage="model_training")
    try:
        log.info('Loading training data')
        X_train = np.load('data/processed/train_embeddings.npy')
        y_train = np.load('data/processed/train_labels.npy')
        log.info('Training data loaded successfully', X_train_shape=X_train.shape, y_train_shape=y_train.shape)

        log.info("Loading model parameters")
        params = load_params('params.yaml')
        model_params = params['model_training']
        model_path = model_params['model_path']

        model = MLPClassifier(
            hidden_layer_sizes=tuple(model_params['hidden_layer_sizes']), 
            activation=model_params['activation'], 
            solver=model_params['solver'], 
            alpha=model_params['alpha'],
            learning_rate_init=model_params['learning_rate_init']
        )
        log.info("Model instantiated", model_params=model_params)

        trained_model = train_model(X_train, y_train, model)
        save_model(trained_model, model_path)
    
    except Exception as e:
        log.error("Model training pipeline failed", error=str(e))
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()