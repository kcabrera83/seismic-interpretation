# API Documentation - Seismic Interpretation

## Base URL

```
http://localhost:5013
```

## Endpoints

### GET /

Serve the main web dashboard UI.

**Response:** HTML page

---

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

---

### GET /api/models

Return model metadata and training metrics.

**Response:**
```json
{
  "horizon_tracker": {
    "model": "GradientBoostingRegressor",
    "metrics": {
      "mse": 0.000123,
      "rmse": 0.01109,
      "mae": 0.00875,
      "r2": 0.9234
    }
  },
  "facies_classifier": {
    "model": "RandomForestClassifier",
    "metrics": {
      "accuracy": 0.8856,
      "f1_macro": 0.8745
    }
  },
  "training_samples": 1600,
  "test_samples": 400,
  "features": ["amplitude", "frequency", "phase", "acoustic_impedance", "velocity", "density"]
}
```

---

### POST /api/track

Track seismic horizons by predicting porosity from seismic attributes.

**Request:**
```json
{
  "features": [
    {
      "amplitude": 0.5,
      "frequency": 30,
      "phase": 0.0,
      "acoustic_impedance": 2000,
      "velocity": 2500,
      "density": 2.2
    },
    {
      "amplitude": -0.3,
      "frequency": 45,
      "phase": 1.5,
      "acoustic_impedance": 1800,
      "velocity": 2200,
      "density": 2.0
    }
  ]
}
```

**Required Fields:**

| Field | Type | Description |
|-------|------|-------------|
| features | array[object] | Array of feature objects |

Each feature object must contain:

| Key | Type | Description |
|-----|------|-------------|
| amplitude | float | Seismic amplitude |
| frequency | float | Dominant frequency (Hz) |
| phase | float | Phase angle (radians) |
| acoustic_impedance | float | Acoustic impedance (kg/m2s) |
| velocity | float | Seismic velocity (m/s) |
| density | float | Rock density (g/cm3) |

**Response:**
```json
{
  "predictions": [0.1523, 0.1842],
  "n": 2
}
```

**Error Responses:**
| Status | Condition | Body |
|--------|-----------|------|
| 400 | Missing features | `{"error": "Missing 'features' key with array of feature objects"}` |

---

### POST /api/classify

Classify seismic facies (lithology) from seismic attributes.

**Request:**
```json
{
  "features": [
    {
      "amplitude": 0.5,
      "frequency": 30,
      "phase": 0.0,
      "acoustic_impedance": 2000,
      "velocity": 2500,
      "density": 2.2
    }
  ]
}
```

**Response:**
```json
{
  "labels": [1],
  "probabilities": [[0.05, 0.85, 0.05, 0.03, 0.02]],
  "n": 1
}
```

**Labels:** Integer class indices corresponding to lithology types.

**Error Responses:**
| Status | Condition | Body |
|--------|-----------|------|
| 400 | Missing features | `{"error": "Missing 'features' key with array of feature objects"}` |

---

### GET /api/docs

Return OpenAPI 3.0 specification.

---

## Feature Reference

| Feature | Unit | Description |
|---------|------|-------------|
| amplitude | - | Seismic reflection amplitude |
| frequency | Hz | Dominant frequency of seismic wavelet |
| phase | radians | Phase angle of seismic signal |
| acoustic_impedance | kg/m2s | Product of density and velocity |
| velocity | m/s | Seismic wave propagation velocity |
| density | g/cm3 | Rock bulk density |

## Error Codes

- **200**: Success
- **400**: Bad request (missing features)
- **500**: Internal server error

---

*Elaborado por Ing. Kelvin Cabrera*
