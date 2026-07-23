"""Facies classifier using RandomForest for lithology classification."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import numpy as np
import os
import pickle


class FaciesClassifier:
    def __init__(self, n_estimators=200, max_depth=12, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.name = "RandomForestClassifier (FaciesClassifier)"

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_macro": float(f1_score(y_test, preds, average="macro")),
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
