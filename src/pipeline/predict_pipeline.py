import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, assign_customer_persona

class PredictPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.model_path = os.path.join("artifacts", "model.pkl")

    def predict(self, recency: float, frequency: float, monetary: float) -> dict:
        try:
            logging.info("Initiating Customer Segment Prediction...")
            
            # Load saved artifacts
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            # Input dataframe
            input_df = pd.DataFrame({
                "Recency": [recency],
                "Frequency": [frequency],
                "Monetary": [monetary]
            })

            # Apply log1p transformation identical to training pipeline
            log_input = np.log1p(input_df)
            scaled_input = preprocessor.transform(log_input)

            # Predict cluster
            cluster_id = int(model.predict(scaled_input)[0])

            # Get business persona metadata
            persona_info = assign_customer_persona(recency, frequency, monetary)

            result = {
                "cluster_id": cluster_id,
                "persona": persona_info["persona"],
                "description": persona_info["description"],
                "recommended_action": persona_info["recommended_action"],
                "input_rfm": {
                    "Recency": recency,
                    "Frequency": frequency,
                    "Monetary": monetary
                }
            }

            logging.info(f"Prediction successful: Cluster {cluster_id} - Persona: {persona_info['persona']}")
            return result
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, recency: float, frequency: float, monetary: float):
        self.recency = recency
        self.frequency = frequency
        self.monetary = monetary

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            return pd.DataFrame({
                "Recency": [self.recency],
                "Frequency": [self.frequency],
                "Monetary": [self.monetary]
            })
        except Exception as e:
            raise CustomException(e, sys)
