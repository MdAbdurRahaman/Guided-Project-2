import os
import sys
import yaml
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataValidationConfig:
    schema_file_path: str = os.path.join("config", "schema.yaml")
    validation_status_file: str = os.path.join("artifacts", "validation_status.txt")

class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()

    def validate_all_columns(self, file_path: str) -> bool:
        try:
            logging.info("Initiating Data Validation component.")
            df = pd.read_csv(file_path)

            with open(self.validation_config.schema_file_path, "r") as f:
                schema = yaml.safe_load(f)

            required_columns = schema.get("required_columns", [])
            df_cols = list(df.columns)
            
            validation_status = True
            missing_cols = []

            for col in required_columns:
                if col not in df_cols:
                    validation_status = False
                    missing_cols.append(col)

            os.makedirs(os.path.dirname(self.validation_config.validation_status_file), exist_ok=True)
            with open(self.validation_config.validation_status_file, "w") as f:
                if validation_status:
                    f.write(f"Validation status: TRUE. All required columns present.")
                    logging.info("Data Validation passed successfully.")
                else:
                    f.write(f"Validation status: FALSE. Missing required columns: {missing_cols}")
                    logging.error(f"Data Validation failed. Missing columns: {missing_cols}")

            return validation_status
        except Exception as e:
            raise CustomException(e, sys)
