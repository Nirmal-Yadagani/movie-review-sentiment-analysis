import sys

import boto3
import pandas as pd

from src.logger import logger
from src.exception import MyException
from io import StringIO


class s3_operations:
    """
    A utility class to interact with AWS S3, providing method to fetch file from bucket.
    """
    def __init__(self, bucket_name, aws_access_key, aws_secret_key, region_name='east-us-1'):
        """
        Initializes the S3 client and resource.
        
        Raises:
            MyException: If connection to AWS fails.
        """
        self.bucket_name = bucket_name
        try:
          self.s3_client = boto3.client(
              's3',
              aws_access_key_id=aws_access_key,
              aws_secret_access_key=aws_secret_key,
              region_name=region_name)
          self.log = logger.bind(service='S3Storage')

        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            logger.error('s3 connection failed',error=str(custom_error))
            raise custom_error
        

    def fetch_file_from_s3(self, s3_key):
        """
        Fetches file object(s) matching a specific prefix from S3.

        Args:
            s3_key (str): The filename or prefix to search.

        Returns:
            pd.DataFrame: A DataFrame containing the contents of the file(s) if found, otherwise None.
        """
        try:
            self.log.info('fetching file from s3', bucket=self.bucket_name, key=s3_key)
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            self.log.info('file fetched successfully', bucket=self.bucket_name, key=s3_key)
            return df
        
        except Exception as e:
            if isinstance(e, MyException):
                raise e
            raise MyException(e, sys)