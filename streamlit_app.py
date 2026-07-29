import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Seismic Interpretation", layout="wide")
st.title("Seismic Interpretation")
st.markdown("Interpret seismic data for facies classification.")

import joblib, numpy as np
d = Path(__file__).parent / 'outputs' / 'models'
models = {'porosity': joblib.load(d / 'porosity_predictor.pkl'), 'facies': joblib.load(d / 'facies_classifier.pkl')}

st.sidebar.header("Input Parameters")
amplitude = st.sidebar.slider('Amplitude', -100, 100, 0)
frequency = st.sidebar.slider('Frequency', 5, 100, 52)
phase = st.sidebar.slider('Phase', 0, 360, 180)
impedance = st.sidebar.slider('Impedance', 2000, 10000, 6000)
velocity = st.sidebar.slider('Velocity', 1500, 5000, 3250)
density = st.sidebar.slider('Density', 1, 3, 2)

if st.sidebar.button("Run"):
    try:
        x = np.array([[amplitude, frequency, phase, impedance, velocity, density]])
        cols = st.columns(2)
        for i, (k, m) in enumerate(models.items()):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            if 'label_encoder' in m:
                val = m['label_encoder'].inverse_transform(p)[0]
            else:
                val = f'{p[0]:.2f}'
            cols[i].metric(k.title(), val)
    except Exception as e:
        st.error(str(e))