"""FastAPI for seismic interpretation models."""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from seismic_interpretation.utils.preprocessor import FEATURE_COLS

app = FastAPI(
    title="Seismic Interpretation",
    description="Seismic horizon tracking and facies classification",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join("outputs", "models")
tracker_model = None
classifier_model = None
scaler = None
metadata = None


@app.on_event("startup")
async def load_models():
    global tracker_model, classifier_model, scaler, metadata
    with open(os.path.join(MODELS_DIR, "horizon_tracker.pkl"), "rb") as f:
        tracker_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "facies_classifier.pkl"), "rb") as f:
        classifier_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)


class SeismicFeature(BaseModel):
    amplitude: float
    frequency: float
    phase: float
    acoustic_impedance: float
    velocity: float
    density: float


class TrackRequest(BaseModel):
    features: List[SeismicFeature]


class TrackResponse(BaseModel):
    predictions: list
    n: int


class ClassifyRequest(BaseModel):
    features: List[SeismicFeature]


class ClassifyResponse(BaseModel):
    labels: list
    probabilities: List[list]
    n: int


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": tracker_model is not None and classifier_model is not None,
    }


@app.get("/api/models")
async def models_info():
    return metadata


@app.post("/api/track", response_model=TrackResponse)
async def track(request: TrackRequest):
    rows = [r.model_dump() for r in request.features]
    X = np.array([[r[c] for c in FEATURE_COLS] for r in rows])
    X_scaled = scaler.transform(X)
    predictions = tracker_model.predict(X_scaled).tolist()
    return TrackResponse(predictions=predictions, n=len(predictions))


@app.post("/api/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    rows = [r.model_dump() for r in request.features]
    X = np.array([[r[c] for c in FEATURE_COLS] for r in rows])
    X_scaled = scaler.transform(X)
    labels = classifier_model.predict(X_scaled).tolist()
    proba = classifier_model.predict_proba(X_scaled).tolist()
    return ClassifyResponse(labels=labels, probabilities=proba, n=len(labels))
