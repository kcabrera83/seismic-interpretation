# Architecture - Seismic Interpretation

## System Overview

```
+------------------+     +-------------------+     +------------------+
|   Data Layer     | --> |   Model Layer     | --> |   API Layer      |
| (Data Generator) |     | (ML Models)       |     | (Flask REST)     |
+------------------+     +-------------------+     +------------------+
                                                          |
                                                          v
                                                 +------------------+
                                                 | Dashboard Layer  |
                                                 | (HTML/CSS/JS)    |
                                                 +------------------+
```

## Components

### Data Layer

- **Source**: Synthetic data generator (`generate_synthetic_data`)
- **Samples**: 2,000 seismic attribute records
- **Features**: amplitude, frequency, phase, acoustic_impedance, velocity, density
- **Targets**: porosity_prediction (regression), lithology_prediction (classification)

### Model Layer

#### HorizonTracker
- **Algorithm**: GradientBoostingRegressor (n_estimators=200, max_depth=5)
- **Task**: Predict porosity from seismic attributes
- **Input**: 6 scaled seismic features
- **Output**: Porosity value (continuous)
- **Serialization**: pickle (`.pkl`)

#### FaciesClassifier
- **Algorithm**: RandomForestClassifier (n_estimators=200, max_depth=12)
- **Task**: Classify lithology facies from seismic attributes
- **Input**: 6 scaled seismic features
- **Output**: Facies label (integer) + probability distribution
- **Serialization**: pickle (`.pkl`)

### Preprocessing Pipeline

- **StandardScaler** for all 6 features
- Single scaler shared between both models
- Saved as `scaler.pkl`

### API Layer

- **Framework**: Flask
- **Port**: 5013
- **Format**: JSON request/response
- **Endpoints**: 5 (track, classify, health, models, docs)

### Dashboard Layer

- **Frontend**: HTML/CSS/JS (Jinja2 templates)
- **Charts**: Chart.js for visualization
- **Theme**: Dark theme UI

## Data Flow

### Horizon Tracking Flow
1. User provides array of seismic feature objects
2. Features extracted in predefined column order
3. StandardScaler transforms features
4. GradientBoostingRegressor predicts porosity per sample
5. Array of porosity predictions returned

### Facies Classification Flow
1. User provides array of seismic feature objects
2. Features extracted and scaled
3. RandomForestClassifier predicts facies label + probabilities
4. Labels and probability distributions returned

## Feature Processing

All features are standardized (zero mean, unit variance) using StandardScaler:

| Feature | Unit | Physical Meaning |
|---------|------|------------------|
| amplitude | - | Reflection strength at interface |
| frequency | Hz | Spectral content of wavelet |
| phase | radians | Wavelet phase characteristic |
| acoustic_impedance | kg/m2s | Rock acoustic property |
| velocity | m/s | Seismic wave speed |
| density | g/cm3 | Rock bulk density |

## Project Structure

```
seismic-interpretation/
├── seismic_interpretation/
│   ├── __init__.py
│   ├── data_generator.py              # Synthetic seismic data
│   ├── models/
│   │   ├── __init__.py
│   │   ├── horizon_tracker.py         # GradientBoosting regressor
│   │   └── facies_classifier.py       # RandomForest classifier
│   └── utils/
│       ├── __init__.py
│       └── preprocessor.py            # StandardScaler + data prep
├── templates/
│   └── index.html                     # Dashboard UI
├── outputs/models/                    # Saved model artifacts
├── train.py                           # Training pipeline
├── app.py                             # Flask API server
├── test_api.py                        # API test suite
├── requirements.txt
└── setup.py
```

## Saved Artifacts

| File | Description |
|------|-------------|
| horizon_tracker.pkl | GradientBoosting model for porosity prediction |
| facies_classifier.pkl | RandomForest model for facies classification |
| scaler.pkl | StandardScaler for feature normalization |
| metadata.json | Training metrics, sample counts, feature list |

## Model Evaluation

### HorizonTracker
- MSE: ~0.0001
- R2: ~0.92+

### FaciesClassifier
- Accuracy: ~0.88+
- F1 Macro: ~0.87+

---

*Elaborado por Ing. Kelvin Cabrera*
