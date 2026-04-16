import os
import sys
import yaml
from dotenv import load_dotenv

import pandas as pd

from src.logger import logger
from src.exception import MyException
from src.connections.s3_connection import s3_operations


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file.

    Args:
        params_path (str): Path to the YAML file containing parameters.

    Returns:
        dict: A dictionary containing the loaded parameters.
    """
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
            logger.debug("Parameters loaded successfully.",params_file_path=params_path)
        return params
    
    except Exception as e:
        if isinstance(e, MyException):
            raise e
        raise MyException(e, sys)
    

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a specified URL.

    Args:
        data_url (str): The URL or local path to the data file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    try:
        if data_url.startswith("s3://"):
            load_dotenv()
            bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
            access_key = os.getenv("AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            region = os.getenv("AWS_REGION")
            s3 = s3_operations(bucket_name, access_key, secret_key, region)
            df = s3.fetch_file_from_s3("IMDB.csv")
            logger.debug("Data loaded successfully from S3.",data_url=data_url)
        else:
            df = pd.read_csv(data_url)
            logger.debug("Data loaded successfully from local path.",data_url=data_url)
        return df
    
    except Exception as e:
        if isinstance(e, MyException):
            raise e
        raise MyException(e, sys)
    

def clean_and_split_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the input DataFrame by dropping duplicates and handling missing values.

    Args:
        df (pd.DataFrame): The input DataFrame to be preprocessed.
    
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    try:
        df = df.drop_duplicates()
        df = df.dropna().replace({'positive': 1, 'negative': 0})
        logger.info("Data preprocessed successfully.",original_shape=df.shape)
        return df
    
    except Exception as e:
        if isinstance(e, MyException):
            raise e
        raise MyException(e, sys)


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save train and test data to specified paths.

    Args:
        train_data (pd.DataFrame): The training data to be saved.
        test_data (pd.DataFrame): The testing data to be saved.
        data_path (str): The directory path where the data files will be saved.

    Returns:
        None
    """
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        train_file_path = os.path.join(raw_data_path, "train.csv")
        test_file_path = os.path.join(raw_data_path, "test.csv")
        train_data.to_csv(train_file_path, index=False)
        test_data.to_csv(test_file_path, index=False)
        logger.debug("Train and test data saved successfully.",train_file_path=train_file_path,test_file_path=test_file_path)
    
    except Exception as e:
        if isinstance(e, MyException):
            raise e
        raise MyException(e, sys)