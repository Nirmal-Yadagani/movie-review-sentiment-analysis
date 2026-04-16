import sys

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import MyException
from src.utils.helper_functions import load_params, load_data, clean_and_split_data, save_data
    

def main():
    try:
        logger.info("Starting data ingestion process...")
        params = load_params("params.yaml")
        data_url = params['data_ingestion']['data_url']
        test_size = params['data_ingestion']['test_size']
        random_state = params['data_ingestion']['random_state']
        data_path = params['data_ingestion']['data_path']

        df = load_data(data_url)
        final_df = clean_and_split_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=random_state)
        save_data(train_data, test_data, data_path)
        logger.info("Data ingestion process completed successfully.", data_path)
    except Exception as e:
        if isinstance(e, MyException):
            raise e
        raise MyException(e, sys)
    

if __name__ == "__main__":
    main()
