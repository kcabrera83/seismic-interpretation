import streamlit as st
import pickle
import joblib
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Seismic Interpretation", layout="wide")
st.title("Seismic Interpretation")
st.markdown("Interpret seismic data for horizon tracking and facies classification.")

@st.cache_resource
def load_models():
    d = Path(__file__).parent / "outputs" / "models"
    return {k: joblib.load(d / v) for k, v in [("porosity", "porosity_predictor.pkl"), ("facies", "facies_classifier.pkl")]}

models = load_models()

st.sidebar.header("Input Parameters")
amplitude = st.sidebar.slider("Amplitude", -100, 100, 0)
frequency_hz = st.sidebar.slider("Frequency Hz", 5, 100, 52)
phase_deg = st.sidebar.slider("Phase Deg", 0, 360, 180)
acoustic_impedance = st.sidebar.slider("Acoustic Impedance", 2000, 10000, 6000)
velocity_ms = st.sidebar.slider("Velocity Ms", 1500, 5000, 3250)
density_g_cc = st.sidebar.slider("Density G Cc", 1, 3, 2)

if st.sidebar.button("Run Prediction"):
    try:
        features = np.array([[amplitude, frequency_hz, phase_deg, acoustic_impedance, velocity_ms, density_g_cc]])
        m = models["porosity"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Porosity", result if isinstance(result, str) else f"{result:.4f}")
        m = models["facies"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Facies", result if isinstance(result, str) else f"{result:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")
