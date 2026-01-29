import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
import os

class MarineIsolationForest:
    def __init__(self, config=None):
        # Use .get() to provide safety defaults
        self.config = config if config is not None else {}
        
        # Set defaults if keys are missing from the passed config
        n_estimators = self.config.get('n_estimators', 100)
        max_samples = self.config.get('max_samples', 'auto')
        contamination = self.config.get('contamination', 0.01)
        random_state = self.config.get('random_state', 42)

        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        print(f"Isolation Forest initialized with max_samples={max_samples}")

    def train_model(self, X_train):
        """
        Fits the model to normal operating data.
        X_train shape: (Samples, Features)
        """
        print(f"Training Isolation Forest on {X_train.shape[0]} samples...")
        self.model.fit(X_train)
        print("Training complete.")

    def predict_score(self, X):
        """
        Returns the anomaly score. 
        Lower values = more anomalous.
        """
        # score_samples returns the opposite of the anomaly score (lower is more abnormal)
        return self.model.score_samples(X)

    def predict_label(self, X):
        """
        Returns -1 for outliers and 1 for inliers.
        """
        return self.model.predict(X)

    def save_model(self, filename="iso_forest_model.pkl"):
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        if os.path.exists(filename):
            self.model = joblib.load(filename)
            print(f"Model loaded from {filename}")
        else:
            print(f"Error: {filename} not found.")