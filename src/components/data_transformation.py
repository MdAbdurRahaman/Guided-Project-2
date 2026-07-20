import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    rfm_data_path: str = os.path.join("artifacts", "transformed_rfm.csv")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def calculate_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans transaction data and aggregates per-customer RFM metrics.
        """
        try:
            logging.info("Cleaning raw e-commerce transaction data...")
            # Drop records missing CustomerID
            df = df.dropna(subset=["CustomerID"]).copy()
            df["CustomerID"] = df["CustomerID"].astype(int)

            # Keep positive quantity & price
            df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

            # Calculate total spend per item
            df["TotalSpend"] = df["Quantity"] * df["UnitPrice"]

            # Parse InvoiceDate
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

            # Reference snapshot date for Recency calculation
            max_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

            logging.info(f"Aggregating RFM features for {df['CustomerID'].nunique()} unique customers...")
            rfm = df.groupby("CustomerID").agg({
                "InvoiceDate": lambda x: (max_date - x.max()).days,
                "InvoiceNo": "nunique",
                "TotalSpend": "sum"
            }).reset_index()

            rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
            logging.info("RFM feature aggregation completed.")
            return rfm
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str) -> tuple[pd.DataFrame, str]:
        try:
            logging.info("Initiating Data Transformation pipeline.")
            raw_df = pd.read_csv(train_path)

            rfm_df = self.calculate_rfm(raw_df)
            
            # Save unscaled RFM dataset for EDA and dashboard visualization
            os.makedirs(os.path.dirname(self.transformation_config.rfm_data_path), exist_ok=True)
            rfm_df.to_csv(self.transformation_config.rfm_data_path, index=False)

            # Feature Log transformation to resolve positive skewness
            rfm_features = rfm_df[["Recency", "Frequency", "Monetary"]].copy()
            rfm_log = np.log1p(rfm_features)

            logging.info("Fitting StandardScaler on log-transformed RFM features...")
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(rfm_log)

            scaled_rfm_df = pd.DataFrame(
                scaled_features, 
                columns=["Recency", "Frequency", "Monetary"]
            )
            scaled_rfm_df["CustomerID"] = rfm_df["CustomerID"].values

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=scaler
            )
            logging.info(f"Preprocessor object saved at {self.transformation_config.preprocessor_obj_file_path}")

            return scaled_rfm_df, self.transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)
