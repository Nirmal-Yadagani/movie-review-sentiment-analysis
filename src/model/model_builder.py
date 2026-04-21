import os
import sys
import joblib

from sklearn.neural_network import MLPClassifier
import numpy as np

from src.logger import logger
from src.exception import MyException
from src.utils.helper_functions import load_params


def train_model(X_train, y_train, model):
    """Train a model on the training data."""
    try:
        logger.info("Training MLPClassifier...", model=str(model))
        model.fit(X_train, y_train)
        logger.info("Model training completed.")
        return model
      
    except Exception as e:
        if isinstance(e, MyException):
            raise e
        raise MyException(e, sys)
    

def save_model(model, model_path):
    """Save the trained model to disk."""
    try:
        logger.info(f"Saving model..")
        joblib.dump(model, model_path)
        logger.info("Model saved successfully.", model_path=model_path)
      
    except Exception as e:
        if isinstance(e, MyException):
            raise e
        raise MyException(e, sys)


def main():
    try:
        logger.info('Loading training data..', train_data_path='data/processed/')
        X_train = np.load('data/processed/train_embeddings.npy')
        y_train = np.load('data/processed/train_labels.npy')
        logger.info('Training data loaded successfully.', X_train_shape=X_train.shape, y_train_shape=y_train.shape)

        logger.info("Loading model parameters...")
        params = load_params('params.yaml')
        model_params = params['model_training']

        model_path = params['model_training']['model_path']
        model = MLPClassifier(hidden_layer_sizes=tuple(model_params['hidden_layer_sizes']), 
                              activation=model_params['activation'], 
                              solver=model_params['solver'], 
                              alpha=model_params['alpha'],
                              learning_rate_init=model_params['learning_rate_init'])
        logger.info("Model parameters loaded successfully.", model_params=model_params)

        trained_model = train_model(X_train, y_train, model)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_model(trained_model, model_path)
    
    except Exception as e:
        if isinstance(e, MyException):
            raise e
        raise MyException(e, sys)
    

if __name__ == "__main__":
    main()