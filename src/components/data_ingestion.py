import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    fallback_csv_path: str = os.path.join("data", "raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple[str, str]:
        logging.info("Initiating Data Ingestion component.")
        try:
            mongo_url = os.getenv("MONGO_DB_URL", None)
            df = None

            if mongo_url:
                try:
                    import pymongo
                    logging.info("Connecting to MongoDB Atlas / Local MongoDB instance...")
                    client = pymongo.MongoClient(mongo_url)
                    db = client[os.getenv("DB_NAME", "customer_db")]
                    collection = db[os.getenv("COLLECTION_NAME", "transactions")]
                    data = list(collection.find())
                    if len(data) > 0:
                        df = pd.DataFrame(data)
                        if "_id" in df.columns:
                            df.drop(columns=["_id"], inplace=True)
                        logging.info(f"Fetched {len(df)} records from MongoDB successfully.")
                except Exception as mongo_err:
                    logging.warning(f"MongoDB connection failed: {mongo_err}. Falling back to local CSV dataset.")

            if df is None or len(df) == 0:
                logging.info(f"Loading raw data from local CSV source: {self.ingestion_config.fallback_csv_path}")
                df = pd.read_csv(self.ingestion_config.fallback_csv_path)

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Saved raw dataset artifact to {self.ingestion_config.raw_data_path}")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion completed successfully.")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print(f"Train Path: {train_path}, Test Path: {test_path}")
