"""Synthetic seismic attribute data generator."""

import numpy as np
import pandas as pd


def generate_synthetic_data(n_samples=1000, random_state=42):
    rng = np.random.RandomState(random_state)

    amplitude = rng.uniform(-1.0, 1.0, n_samples)
    frequency = rng.uniform(5.0, 80.0, n_samples)
    phase = rng.uniform(-np.pi, np.pi, n_samples)

    acoustic_impedance = (
        1500 + 0.5 * amplitude * 1000 + 0.3 * frequency * 10 + rng.normal(0, 150, n_samples)
    )
    velocity = (
        2000 + 0.4 * acoustic_impedance * 0.5 + 100 * np.sin(phase) + rng.normal(0, 80, n_samples)
    )
    density = (
        2.0 + 0.0003 * acoustic_impedance / (velocity / 1000) + rng.normal(0, 0.05, n_samples)
    )
    porosity_prediction = np.clip(
        0.25 - 0.00005 * acoustic_impedance + 0.002 * frequency + rng.normal(0, 0.03, n_samples),
        0.0, 0.4,
    )
    lithology_raw = (
        0.3 * amplitude
        + 0.01 * frequency
        + 0.2 * np.cos(phase)
        + 0.0001 * acoustic_impedance
        + rng.normal(0, 0.5, n_samples)
    )
    lithology_prediction = np.clip(np.round((lithology_raw - lithology_raw.min()) / (lithology_raw.max() - lithology_raw.min()) * 4).astype(int), 0, 4)

    df = pd.DataFrame({
        "amplitude": amplitude,
        "frequency": frequency,
        "phase": phase,
        "acoustic_impedance": acoustic_impedance,
        "velocity": velocity,
        "density": density,
        "porosity_prediction": porosity_prediction,
        "lithology_prediction": lithology_prediction,
    })
    return df
