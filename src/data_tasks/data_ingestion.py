import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException
from src.utils.helper_functions import load_params, load_data, clean_and_split_data, save_data

# Set Pandas options globally after imports
pd.set_option('future.no_silent_downcasting', True)

def main() -> None:
    """
    Executes the data ingestion pipeline: loads raw data, cleans it, splits it 
    into train/test sets, and saves the outputs locally.
    """
    log = logger.bind(stage="data_ingestion")
    
    try:
        log.info("Starting data ingestion process")
        
        params = load_params("params.yaml")
        data_url = params['data_ingestion']['data_url']
        test_size = params['data_ingestion']['test_size']
        random_state = params['data_ingestion']['random_state']
        data_path = params['data_ingestion']['data_path']

        # Log parameters being used
        log.info("Ingestion parameters loaded", test_size=test_size, random_state=random_state)

        df = load_data(data_url)
        
        train_data, test_data = clean_and_split_data(df, test_size, random_state)
        
        save_data(train_data, test_data, data_path)
        
        log.info("Data ingestion process completed successfully", 
                 train_size=len(train_data), 
                 test_size=len(test_data), 
                 output_path=data_path)
                 
    except Exception as e:
        log.error("Data ingestion pipeline failed", error=str(e))
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()