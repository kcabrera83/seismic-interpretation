"""Data preprocessor for seismic attributes."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "amplitude", "frequency", "phase",
    "acoustic_impedance", "velocity", "density",
]

TARGET_REG = "porosity_prediction"
TARGET_CLF = "lithology_prediction"


def prepare_data(df, test_size=0.2, random_state=42):
    X = df[FEATURE_COLS].values
    y_reg = df[TARGET_REG].values
    y_clf = df[TARGET_CLF].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X_scaled, y_reg, y_clf, test_size=test_size, random_state=random_state,
    )

    return {
        "X_train": X_train, "X_test": X_test,
        "y_reg_train": y_reg_train, "y_reg_test": y_reg_test,
        "y_clf_train": y_clf_train, "y_clf_test": y_clf_test,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
    }
