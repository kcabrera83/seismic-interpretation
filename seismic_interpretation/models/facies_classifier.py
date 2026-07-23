"""Facies classifier using GradientBoosting for lithology classification."""

import numpy as np
import os
import joblib
from sklearn.ensemble import GradientBoostingClassifier


class FaciesClassifier:
    def __init__(self, in_features=6, num_classes=5, lr=0.001, epochs=100, batch_size=64, device=None):
        self.name = "GradientBoosting (FaciesClassifier)"
        self.num_classes = num_classes
        self.model = GradientBoostingClassifier(
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

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        y_test = np.array(y_test)
        accuracy = float(np.mean(preds == y_test))
        f1_scores = []
        for c in range(self.num_classes):
            tp = np.sum((preds == c) & (y_test == c))
            fp = np.sum((preds == c) & (y_test != c))
            fn = np.sum((preds != c) & (y_test == c))
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            f1_scores.append(f1)
        f1_macro = float(np.mean(f1_scores))
        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "n_test": int(len(y_test)),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        return self
