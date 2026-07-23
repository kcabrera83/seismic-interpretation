# User Guide - Seismic Interpretation

## Overview

The Seismic Interpretation system uses machine learning for horizon tracking (porosity prediction) and facies classification from seismic attributes. It provides a REST API and web dashboard for real-time seismic data analysis.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
cd seismic-interpretation
pip install -r requirements.txt
```

### Train Models

```bash
python train.py
```

This generates 2,000 synthetic seismic samples and trains:
- HorizonTracker (GradientBoostingRegressor) - Predicts porosity from seismic attributes
- FaciesClassifier (RandomForestClassifier) - Classifies lithology from seismic attributes

### Run the Server

```bash
python app.py
```

Open `http://localhost:5013` in your browser.

## Dashboard Features

- **Horizon Tracking Panel** - Input seismic attributes to predict porosity
- **Facies Classification Panel** - Input seismic attributes to classify rock type
- **Model Metrics** - View training accuracy, R2 scores, and feature importances
- **Chart.js Visualizations** - Visual analysis of prediction results
- **Dark Theme UI** - Modern dark-themed dashboard

## API Usage

### Horizon Tracking (Python)

```python
import requests

response = requests.post("http://localhost:5013/api/track", json={
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
})
result = response.json()
print(f"Porosity prediction: {result['predictions'][0]:.4f}")
```

### Horizon Tracking (curl)

```bash
curl -X POST http://localhost:5013/api/track \
  -H "Content-Type: application/json" \
  -d '{"features": [{"amplitude": 0.5, "frequency": 30, "phase": 0.0, "acoustic_impedance": 2000, "velocity": 2500, "density": 2.2}]}'
```

### Facies Classification (Python)

```python
import requests

response = requests.post("http://localhost:5013/api/classify", json={
    "features": [
        {
            "amplitude": -0.3,
            "frequency": 45,
            "phase": 1.5,
            "acoustic_impedance": 1800,
            "velocity": 2200,
            "density": 2.0
        }
    ]
})
result = response.json()
print(f"Facies label: {result['labels'][0]}")
print(f"Probabilities: {result['probabilities'][0]}")
```

### Check Health

```bash
curl http://localhost:5013/api/health
```

### Get Model Info

```bash
curl http://localhost:5013/api/models
```

## Batch Processing

Both `/api/track` and `/api/classify` support batch processing by passing multiple feature objects in the `features` array:

```python
import requests

response = requests.post("http://localhost:5013/api/track", json={
    "features": [
        {"amplitude": 0.5, "frequency": 30, "phase": 0.0, "acoustic_impedance": 2000, "velocity": 2500, "density": 2.2},
        {"amplitude": -0.3, "frequency": 45, "phase": 1.5, "acoustic_impedance": 1800, "velocity": 2200, "density": 2.0},
        {"amplitude": 0.8, "frequency": 35, "phase": 0.5, "acoustic_impedance": 2200, "velocity": 2800, "density": 2.4},
    ]
})
results = response.json()
for pred in results['predictions']:
    print(f"Porosity: {pred:.4f}")
```

## Running Tests

```bash
python test_api.py
```

## Troubleshooting

- **Models not loaded**: Run `python train.py` first
- **Missing features error**: All 6 seismic attributes are required per feature object
- **Port in use**: Change port in `app.py`

---

*Elaborado por Ing. Kelvin Cabrera*
