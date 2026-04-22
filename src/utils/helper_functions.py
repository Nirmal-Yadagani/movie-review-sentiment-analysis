import os
import sys
from typing import Tuple
import yaml
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException
from src.connections.s3_connection import S3Operations


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
            logger.debug("Parameters loaded successfully", params_file_path=params_path)
        return params
    except Exception as e:
        raise CustomException(e, sys)
    

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a specified URL or S3 bucket."""
    try:
        if data_url.startswith("s3://"):
            load_dotenv()
            bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
            access_key = os.getenv("AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            region = os.getenv("AWS_REGION", "us-east-1")
            
            s3 = S3Operations(bucket_name, access_key, secret_key, region)
            # Assuming you pass the exact S3 key as part of the data_url or params in the future
            df = s3.fetch_file_from_s3("IMDB.csv") 
            logger.debug("Data loaded successfully from S3", data_url=data_url)
        else:
            df = pd.read_csv(data_url)
            logger.debug("Data loaded successfully from local path", data_url=data_url)
        return df
    except Exception as e:
        raise CustomException(e, sys)
    

def clean_and_split_data(df: pd.DataFrame, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess the input DataFrame by dropping duplicates, handling missing values, and splitting."""
    try:
        df = df.drop_duplicates()
        df = df.dropna().replace({'positive': 1, 'negative': 0})
        logger.info("Data preprocessed successfully", original_shape=df.shape)

        train_data, test_data = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state
        )
        return train_data, test_data
    except Exception as e:
        raise CustomException(e, sys)


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save train and test data to specified local paths."""
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        
        train_file_path = os.path.join(raw_data_path, "train.csv")
        test_file_path = os.path.join(raw_data_path, "test.csv")
        
        train_data.to_csv(train_file_path, index=False)
        test_data.to_csv(test_file_path, index=False)
        
        logger.debug("Train and test data saved successfully", 
                     train_file_path=train_file_path, 
                     test_file_path=test_file_path)
    except Exception as e:
        raise CustomException(e, sys)