import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    model_metrics_file_path: str = os.path.join("artifacts", "model_metrics.json")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_kmeans(self, X: np.ndarray) -> tuple[KMeans, dict]:
        best_k = 4
        best_score = -1
        best_model = None
        metrics_history = {}

        for k in range(2, 8):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            score = silhouette_score(X, cluster_labels)
            db_score = davies_bouldin_score(X, cluster_labels)
            
            metrics_history[k] = {
                "silhouette_score": float(score),
                "davies_bouldin_score": float(db_score),
                "inertia": float(kmeans.inertia_)
            }
            logging.info(f"K-Means (k={k}) -> Silhouette Score: {score:.4f}, Davies-Bouldin: {db_score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k
                best_model = kmeans

        logging.info(f"Best K-Means model identified with k={best_k} (Silhouette Score: {best_score:.4f})")
        return best_model, metrics_history

    def initiate_model_trainer(self, scaled_rfm_df: pd.DataFrame) -> tuple[str, float]:
        try:
            logging.info("Initiating Model Trainer component.")
            feature_cols = ["Recency", "Frequency", "Monetary"]
            X = scaled_rfm_df[feature_cols].values

            # Train and evaluate K-Means algorithm
            best_kmeans, kmeans_metrics = self.evaluate_kmeans(X)

            # Fit Hierarchical Clustering for comparison
            agglom = AgglomerativeClustering(n_clusters=best_kmeans.n_clusters)
            agglom_labels = agglom.fit_predict(X)
            agglom_score = silhouette_score(X, agglom_labels)
            logging.info(f"Hierarchical Clustering Silhouette Score: {agglom_score:.4f}")

            # Fit Gaussian Mixture Model
            gmm = GaussianMixture(n_components=best_kmeans.n_clusters, random_state=42)
            gmm_labels = gmm.fit_predict(X)
            gmm_score = silhouette_score(X, gmm_labels)
            logging.info(f"GMM Silhouette Score: {gmm_score:.4f}")

            # Save best model (KMeans)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_kmeans
            )
            logging.info(f"Saved trained KMeans model artifact to {self.model_trainer_config.trained_model_file_path}")

            final_labels = best_kmeans.predict(X)
            final_silhouette = silhouette_score(X, final_labels)

            return self.model_trainer_config.trained_model_file_path, final_silhouette
        except Exception as e:
            raise CustomException(e, sys)
