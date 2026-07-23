"""Horizon tracker using GradientBoosting for porosity regression."""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import os
import pickle


class HorizonTracker:
    def __init__(self, n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.name = "GradientBoostingRegressor (HorizonTracker)"

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        return {
            "mse": float(mean_squared_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "mae": float(mean_absolute_error(y_test, preds)),
            "r2": float(r2_score(y_test, preds)),
            "n_test": int(len(y_test)),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        return self
