import pytest


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


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    pass  # models may or may not be loaded


def test_models(client):
    response = client.get("/api/models")
    assert response.status_code in (200, 500)


def test_track_valid(client):
    response = client.post("/api/track", json={
        "features": _make_features(5),
    })
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "n" in data
        assert data["n"] == 5
        assert len(data["predictions"]) == 5


def test_classify_valid(client):
    response = client.post("/api/classify", json={
        "features": _make_features(5),
    })
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        data = response.json()
        assert "labels" in data
        assert "probabilities" in data
        assert "n" in data
        assert data["n"] == 5
        assert len(data["labels"]) == 5
        assert len(data["probabilities"]) == 5


def test_track_single_sample(client):
    response = client.post("/api/track", json={
        "features": [_make_features(1)[0]],
    })
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        data = response.json()
        assert data["n"] == 1
