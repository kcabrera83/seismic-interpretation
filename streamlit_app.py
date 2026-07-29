import streamlit as st, joblib, numpy as np, matplotlib.pyplot as plt
from pathlib import Path; import sys; sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Seismic Interpretation", layout="wide")
st.title("Seismic Interpretation")

class Engine:
    def __init__(self):
        p = Path(__file__).parent / 'outputs' / 'models'
        self.facies = joblib.load(p / 'facies_classifier.pkl')
        self.porosity = joblib.load(p / 'porosity_predictor.pkl')
    def run(self, name, X):
        m = getattr(self, name)
        if isinstance(m, dict):
            x = m['scaler'].transform(X)
            r = m['model'].predict(x)
            if 'label_encoder' in m:
                return m['label_encoder'].inverse_transform(r)[0]
            return float(r[0])
        return float(m.predict(X)[0])

eng = Engine()

with st.sidebar:
    st.header('Inputs')
    amp = st.slider('Amp', -100, 100, 0)
    freq = st.slider('Freq', 5, 100, 52)
    phase = st.slider('Phase', 0, 360, 180)
    imp = st.slider('Imp', 2000, 10000, 6000)
    vel = st.slider('Vel', 1500, 5000, 3250)
    density = st.slider('Density', 1, 3, 2)
    go = st.button('Predict', type='primary', use_container_width=True)

if go:
    x = np.array([[amp, freq, phase, imp, vel, density]])
    out = {}
    out['facies'] = eng.run('facies', x)
    out['porosity'] = eng.run('porosity', x)
    cols = st.columns(len(out))
    for i, (k, v) in enumerate(out.items()):
        cols[i].metric(k.title(), str(v) if isinstance(v, str) else f'{v:.2f}')
    nums = [v for v in out.values() if isinstance(v, (int, float))]
    if nums:
        fig, ax = plt.subplots(figsize=(6,2))
        names = [k.title() for k, v in out.items() if isinstance(v, (int, float))]
        colors = ['#2E86AB','#A23B72','#F18F01']
        bars = ax.bar(names, nums, color=colors[:len(names)])
        ax.axhline(y=sum(nums)/len(nums), color='gray', ls='--', alpha=0.5)
        for bar, val in zip(bars, nums):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*0.9, f'{val:.1f}', ha='center', va='top', color='white', fontweight='bold')
        st.pyplot(fig)