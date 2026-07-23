"""Flask API for seismic interpretation models."""

import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

from seismic_interpretation.utils.preprocessor import FEATURE_COLS

app = Flask(__name__)

MODELS_DIR = os.path.join("outputs", "models")
tracker_model = None
classifier_model = None
scaler = None
metadata = None


def load_models():
    global tracker_model, classifier_model, scaler, metadata
    with open(os.path.join(MODELS_DIR, "horizon_tracker.pkl"), "rb") as f:
        tracker_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "facies_classifier.pkl"), "rb") as f:
        classifier_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": tracker_model is not None and classifier_model is not None,
    })


@app.route("/api/models", methods=["GET"])
def models_info():
    return jsonify(metadata)


@app.route("/api/track", methods=["POST"])
def track():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' key with array of feature objects"}), 400

    rows = data["features"]
    X = np.array([[r[c] for c in FEATURE_COLS] for r in rows])
    X_scaled = scaler.transform(X)
    predictions = tracker_model.predict(X_scaled).tolist()
    return jsonify({"predictions": predictions, "n": len(predictions)})


@app.route("/api/classify", methods=["POST"])
def classify():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' key with array of feature objects"}), 400

    rows = data["features"]
    X = np.array([[r[c] for c in FEATURE_COLS] for r in rows])
    X_scaled = scaler.transform(X)
    labels = classifier_model.predict(X_scaled).tolist()
    proba = classifier_model.predict_proba(X_scaled).tolist()
    return jsonify({"labels": labels, "probabilities": proba, "n": len(labels)})


@app.route("/api/docs", methods=["GET"])
def api_docs():
    return jsonify({
        "openapi": "3.0.0",
        "info": {"title": "Seismic Interpretation", "version": "1.0.0"},
        "paths": {
            "/api/health": {"get": {"summary": "Health check"}},
            "/api/models": {"get": {"summary": "Model info"}},
            "/api/track": {"post": {"summary": "Track seismic horizons"}},
            "/api/classify": {"post": {"summary": "Classify seismic facies"}},
        }
    })


if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5013, debug=False)
