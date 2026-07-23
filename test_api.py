"""API integration tests."""

import json
import sys
import time
import requests

BASE = "http://127.0.0.1:5013"
passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:
        failed += 1
        print(f"  FAIL  {name}: {e}")


def test_health():
    r = requests.get(f"{BASE}/api/health", timeout=5)
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "healthy"
    assert d["models_loaded"] is True


def test_models():
    r = requests.get(f"{BASE}/api/models", timeout=5)
    assert r.status_code == 200
    d = r.json()
    assert "horizon_tracker" in d
    assert "facies_classifier" in d


def test_index():
    r = requests.get(f"{BASE}/", timeout=5)
    assert r.status_code == 200
    assert "Seismic Interpretation" in r.text


def test_track():
    feat = [{
        "amplitude": 0.5, "frequency": 30.0, "phase": 0.0,
        "acoustic_impedance": 2000.0, "velocity": 2500.0, "density": 2.2,
    }]
    r = requests.post(f"{BASE}/api/track", json={"features": feat}, timeout=5)
    assert r.status_code == 200
    d = r.json()
    assert "predictions" in d
    assert len(d["predictions"]) == 1


def test_classify():
    feat = [{
        "amplitude": -0.3, "frequency": 45.0, "phase": 1.5,
        "acoustic_impedance": 1800.0, "velocity": 2200.0, "density": 2.0,
    }]
    r = requests.post(f"{BASE}/api/classify", json={"features": feat}, timeout=5)
    assert r.status_code == 200
    d = r.json()
    assert "labels" in d
    assert len(d["labels"]) == 1
    assert "probabilities" in d


def test_track_batch():
    feats = [
        {"amplitude": 0.1, "frequency": 20.0, "phase": 0.5, "acoustic_impedance": 1900.0, "velocity": 2300.0, "density": 2.1},
        {"amplitude": -0.7, "frequency": 60.0, "phase": -1.2, "acoustic_impedance": 2500.0, "velocity": 3000.0, "density": 2.5},
    ]
    r = requests.post(f"{BASE}/api/track", json={"features": feats}, timeout=5)
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_classify_batch():
    feats = [
        {"amplitude": 0.0, "frequency": 25.0, "phase": 0.0, "acoustic_impedance": 2100.0, "velocity": 2400.0, "density": 2.15},
        {"amplitude": 0.8, "frequency": 70.0, "phase": 2.0, "acoustic_impedance": 2800.0, "velocity": 3200.0, "density": 2.6},
    ]
    r = requests.post(f"{BASE}/api/classify", json={"features": feats}, timeout=5)
    assert r.status_code == 200
    assert len(r.json()["labels"]) == 2


def test_track_bad_input():
    r = requests.post(f"{BASE}/api/track", json={"bad": "data"}, timeout=5)
    assert r.status_code == 400


def test_classify_bad_input():
    r = requests.post(f"{BASE}/api/classify", json={}, timeout=5)
    assert r.status_code == 400


if __name__ == "__main__":
    print("=" * 50)
    print("  API Tests")
    print("=" * 50)

    print("\nWaiting for server...")
    for i in range(10):
        try:
            requests.get(f"{BASE}/api/health", timeout=2)
            print("  Server is up.\n")
            break
        except Exception:
            time.sleep(1)
    else:
        print("  Server did not start in time.")
        sys.exit(1)

    test("GET  /api/health", test_health)
    test("GET  /api/models", test_models)
    test("GET  /", test_index)
    test("POST /api/track (single)", test_track)
    test("POST /api/classify (single)", test_classify)
    test("POST /api/track (batch)", test_track_batch)
    test("POST /api/classify (batch)", test_classify_batch)
    test("POST /api/track (bad input)", test_track_bad_input)
    test("POST /api/classify (bad input)", test_classify_bad_input)

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")

    sys.exit(0 if failed == 0 else 1)
