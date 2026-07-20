import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Starting End-to-End Customer Segmentation Training Pipeline...")

            # Step 1: Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()
            logging.info(f"Ingestion complete. Train path: {train_path}")

            # Step 2: Validation
            validation = DataValidation()
            status = validation.validate_all_columns(train_path)
            if not status:
                raise Exception("Data Validation failed! Aborting pipeline.")
            logging.info("Validation complete.")

            # Step 3: Transformation
            transformation = DataTransformation()
            scaled_rfm_df, preprocessor_path = transformation.initiate_data_transformation(train_path)
            logging.info("Transformation & RFM feature engineering complete.")

            # Step 4: Model Training
            trainer = ModelTrainer()
            model_path, silhouette = trainer.initiate_model_trainer(scaled_rfm_df)
            logging.info(f"Model Training complete. Model path: {model_path}, Silhouette Score: {silhouette:.4f}")

            print("="*60)
            print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"Best Model Silhouette Score: {silhouette:.4f}")
            print(f"Preprocessor Saved: {preprocessor_path}")
            print(f"Model Saved: {model_path}")
            print("="*60)

            return silhouette
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
