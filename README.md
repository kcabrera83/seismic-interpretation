# Seismic Interpretation

ML-based seismic attribute interpretation system for horizon tracking and facies classification.

## Overview

This project provides two core ML models:

- **HorizonTracker** (GradientBoostingRegressor) - Predicts porosity from seismic attributes for horizon identification and tracking.
- **FaciesClassifier** (RandomForestClassifier) - Classifies lithology facies from seismic attributes.

## Features

- Synthetic seismic data generation (amplitude, frequency, phase, acoustic impedance, velocity, density)
- Data preprocessing with StandardScaler
- Model training, evaluation, and persistence
- REST API with Flask (port 5013)
- Dark-themed web dashboard with Chart.js visualizations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train models

```bash
python train.py
```

### Run API server

```bash
python app.py
```

### Run tests

```bash
python test_api.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web dashboard |
| GET | `/api/health` | System health check |
| GET | `/api/models` | Model metadata and metrics |
| POST | `/api/track` | Horizon tracking (porosity prediction) |
| POST | `/api/classify` | Facies classification (lithology) |

## Project Structure

```
seismic-interpretation/
  seismic_interpretation/
    __init__.py
    data_generator.py
    models/
      __init__.py
      horizon_tracker.py
      facies_classifier.py
    utils/
      __init__.py
      preprocessor.py
  outputs/models/
  templates/
    index.html
  train.py
  app.py
  test_api.py
  requirements.txt
  setup.py
  .github/workflows/ci.yml
```

## Example API Calls

**Horizon Tracking:**
```bash
curl -X POST http://localhost:5013/api/track \
  -H "Content-Type: application/json" \
  -d '{"features": [{"amplitude": 0.5, "frequency": 30, "phase": 0.0, "acoustic_impedance": 2000, "velocity": 2500, "density": 2.2}]}'
```

**Facies Classification:**
```bash
curl -X POST http://localhost:5013/api/classify \
  -H "Content-Type: application/json" \
  -d '{"features": [{"amplitude": -0.3, "frequency": 45, "phase": 1.5, "acoustic_impedance": 1800, "velocity": 2200, "density": 2.0}]}'
```

## Elaborado por

Ing. Kelvin Cabrera
