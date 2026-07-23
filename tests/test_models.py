import pytest
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


def test_outputs_directory_exists():
    assert os.path.exists(MODELS_DIR)


def test_model_files_exist():
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith((".pkl", ".joblib", ".h5", ".pt"))]
    assert len(model_files) > 0


def test_tracker_model_loads():
    import pickle
    path = os.path.join(MODELS_DIR, "horizon_tracker.pkl")
    assert os.path.exists(path)
    with open(path, "rb") as f:
        model = pickle.load(f)
    assert model is not None


def test_classifier_model_loads():
    import pickle
    path = os.path.join(MODELS_DIR, "facies_classifier.pkl")
    assert os.path.exists(path)
    with open(path, "rb") as f:
        model = pickle.load(f)
    assert model is not None


def test_scaler_loads():
    import pickle
    path = os.path.join(MODELS_DIR, "scaler.pkl")
    assert os.path.exists(path)
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    assert scaler is not None


def test_tracker_prediction():
    import pickle
    import numpy as np
    with open(os.path.join(MODELS_DIR, "horizon_tracker.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    X = np.array([[0.5, 30.0, 1.5, 8000.0, 3000.0, 2.5]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    assert pred is not None
    assert len(pred) == 1
