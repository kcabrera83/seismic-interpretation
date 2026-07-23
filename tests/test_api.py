import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app, load_models

load_models()
client = app.test_client()


def _make_features(n=5):
    return [
        {
            "amplitude": 0.5,
            "frequency": 30.0,
            "phase": 1.5,
            "acoustic_impedance": 8000.0,
            "velocity": 3000.0,
            "density": 2.5,
        }
    ] * n


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True


def test_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.get_json()
    assert "horizon_tracker" in data
    assert "facies_classifier" in data


def test_api_docs():
    response = client.get("/api/docs")
    assert response.status_code == 200
    data = response.get_json()
    assert data["openapi"] == "3.0.0"


def test_track_valid():
    response = client.post("/api/track", json={
        "features": _make_features(5),
    })
    assert response.status_code == 200
    data = response.get_json()
    assert "predictions" in data
    assert "n" in data
    assert data["n"] == 5
    assert len(data["predictions"]) == 5


def test_track_missing_features():
    response = client.post("/api/track", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_classify_valid():
    response = client.post("/api/classify", json={
        "features": _make_features(5),
    })
    assert response.status_code == 200
    data = response.get_json()
    assert "labels" in data
    assert "probabilities" in data
    assert "n" in data
    assert data["n"] == 5
    assert len(data["labels"]) == 5
    assert len(data["probabilities"]) == 5


def test_classify_missing_features():
    response = client.post("/api/classify", json={})
    assert response.status_code == 400


def test_track_single_sample():
    response = client.post("/api/track", json={
        "features": [_make_features(1)[0]],
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data["n"] == 1
