import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging

def save_object(file_path: str, obj: object) -> None:
    """
    Saves a Python object to disk using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Successfully saved object to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str) -> object:
    """
    Loads a Python object from disk using pickle.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at specified path: {file_path}")
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logging.info(f"Successfully loaded object from {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)

def assign_customer_persona(recency: float, frequency: float, monetary: float) -> dict:
    """
    Utility rule-based/segmentation mapper providing business interpretability
    alongside unsupervised clustering predictions.
    """
    if monetary > 1500 and frequency >= 5 and recency <= 30:
        persona = "Champions"
        description = "High spenders, frequent buyers, recently active. Strategy: VIP rewards & early product access."
        action = "Exclusive VIP Offers"
    elif frequency >= 3 and recency <= 60:
        persona = "Loyal Customers"
        description = "Consistent purchase history and good responsiveness. Strategy: Cross-sell & loyalty points."
        action = "Cross-sell Recommendations"
    elif recency > 90 and monetary > 500:
        persona = "At Risk / High Value"
        description = "Spent high amounts previously but haven't purchased in a long time. Strategy: Re-engagement campaigns."
        action = "Win-back Discount Email"
    elif recency > 90 and frequency <= 2:
        persona = "Hibernating / Lost"
        description = "Low monetary contribution and inactive for over 3 months. Strategy: Automated low-cost retargeting."
        action = "Automated Standard Promotion"
    else:
        persona = "Promising / Potential"
        description = "Recent buyers with moderate spending. Strategy: Onboarding nurture workflows."
        action = "Welcome Series & Discounts"

    return {
        "persona": persona,
        "description": description,
        "recommended_action": action
    }
