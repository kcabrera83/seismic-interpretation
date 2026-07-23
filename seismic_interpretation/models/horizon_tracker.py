"""Horizon tracker using GradientBoosting for porosity regression."""

import numpy as np
import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor


class HorizonTracker:
    def __init__(self, in_features=6, lr=0.001, epochs=100, batch_size=64, device=None):
        self.name = "GradientBoosting (HorizonTracker)"
        self.model = GradientBoostingRegressor(
            n_estimators=min(epochs, 300),
            learning_rate=lr,
            max_depth=4,
            random_state=42,
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        y_test = np.array(y_test, dtype=np.float64)
        mse = float(np.mean((y_test - preds) ** 2))
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(y_test - preds))),
            "r2": float(1 - np.sum((y_test - preds) ** 2) / np.sum((y_test - y_test.mean()) ** 2)),
            "n_test": int(len(y_test)),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        return self
