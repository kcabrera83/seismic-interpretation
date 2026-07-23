import pytest
import os
import pickle
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


def test_tracker_model_loads():
    path = os.path.join(MODELS_DIR, "horizon_tracker.pkl")
    assert os.path.exists(path)
    with open(path, "rb") as f:
        model = pickle.load(f)
    assert model is not None


def test_classifier_model_loads():
    path = os.path.join(MODELS_DIR, "facies_classifier.pkl")
    assert os.path.exists(path)
    with open(path, "rb") as f:
        model = pickle.load(f)
    assert model is not None


def test_scaler_loads():
    path = os.path.join(MODELS_DIR, "scaler.pkl")
    assert os.path.exists(path)
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    assert scaler is not None


def test_tracker_prediction():
    with open(os.path.join(MODELS_DIR, "horizon_tracker.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    X = np.array([[0.5, 30.0, 1.5, 8000.0, 3000.0, 2.5]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    assert pred is not None
    assert len(pred) == 1


def test_classifier_prediction():
    with open(os.path.join(MODELS_DIR, "facies_classifier.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    X = np.array([[0.5, 30.0, 1.5, 8000.0, 3000.0, 2.5]])
    X_scaled = scaler.transform(X)
    labels = model.predict(X_scaled)
    assert labels is not None
    assert len(labels) == 1


def test_classifier_predict_proba():
    with open(os.path.join(MODELS_DIR, "facies_classifier.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    X = np.array([[0.5, 30.0, 1.5, 8000.0, 3000.0, 2.5]])
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)
    assert proba is not None
    assert proba.shape[1] == len(model.classes_)
