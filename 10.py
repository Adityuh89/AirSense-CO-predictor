import streamlit as st
import joblib
import pandas as pd
import base64
import os

# ─── 1. Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirSense | CO Predictor",
    page_icon="🌫️",
    layout="wide"
)

# ─── 2. Background Image Injection (Base64) ────────────────────────────────────
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Checking if the file exists before trying to open it
if os.path.exists("bg.jpg"):
    try:
        bin_str = get_base64('bg.jpg')
        st.markdown(
            f"""
            <style>
            /* The main container background */
            .stApp {{
                background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                                  url("data:image/jpg;base64,{bin_str}") !important;
                background-size: cover !important;
                background-attachment: fixed !important;
                background-position: center !important;
            }}

            /* Removing Streamlit's white/dark default overlays */
            .stAppViewContainer, .stMainBlockContainer, .stHeader {{
                background-color: transparent !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Image processing error: {e}")
else:
    # If the image isn't found, we use a nice dark gradient so the app doesn't break
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0d0d1a 0%, #07070a 100%) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.info("💡 Tip: Place 'bg.jpg' in your projects folder to see the industrial background.")

# ─── 3. Global CSS Styling ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    color: #e8e8e8 !important;
    font-family: 'Syne', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1200px !important; }

/* Glassmorphism Effect */
.hero, .metric-box, .result-box, .waiting-box {
    background: rgba(25, 25, 35, 0.65) !important;
    backdrop-filter: blur(12px) saturate(160%);
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px;
}

.hero { padding: 2.5rem; margin-bottom: 2rem; }
.hero-tag { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #00ff80; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 0.8rem; }
.hero-title { font-size: 3.2rem; font-weight: 800; line-height: 1.1; margin: 0 0 0.8rem 0; color: #ffffff; }
.hero-sub { font-size: 1.05rem; color: #b0b0c0; max-width: 600px; line-height: 1.6; }
.hero-badge { display: inline-block; background: rgba(0,255,128,0.1); border: 1px solid rgba(0,255,128,0.3); border-radius: 8px; padding: 0.4rem 1rem; font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #00ff80; margin-top: 1.2rem; }

.metrics-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
.metric-box { flex: 1; padding: 1.2rem; text-align: center; }
.metric-val { font-family: 'DM Mono', monospace; font-size: 1.6rem; color: #ffffff; font-weight: 600; }
.metric-lbl { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #707085; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.3rem; }

.section-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #707085; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 1.2rem; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.1); }

/* Buttons */
.stButton > button {
    width: 100% !important; background: #00ff80 !important; color: #000000 !important; 
    border-radius: 12px !important; padding: 0.8rem !important; font-weight: 800 !important; 
    border: none !important; transition: 0.3s ease;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(0,255,128,0.3); }

/* Result Styles */
.result-number-safe { font-family: 'DM Mono', monospace; font-size: 4rem; color: #00ff80; font-weight: 500; }
.result-number-danger { font-family: 'DM Mono', monospace; font-size: 4rem; color: #ff5050; font-weight: 500; }

.footer { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #505065; text-align: center; margin-top: 4rem; text-transform: uppercase; letter-spacing: 0.1em; }
</style>
""", unsafe_allow_html=True)

# ─── 4. Header Section ────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">Predictive Analytics · Air Quality</div>
    <div class="hero-title">AirSense CO Predictor</div>
    <div class="hero-sub">Analyzing Carbon Monoxide concentration through deep sensor telemetry and high-precision XGBoost forecasting.</div>
    <div class="hero-badge">Model Status: Active · 90.84% R-Squared Precision</div>
</div>
""", unsafe_allow_html=True)

# ─── 5. Model Loading ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("airquality.pkl")

try:
    model = load_model()
except:
    st.error("❌ 'airquality.pkl' not found. Please ensure the model file is in the projects folder.")
    st.stop()

# ─── 6. Application Logic ─────────────────────────────────────────────────────
st.markdown("""<div class="metrics-row">
    <div class="metric-box"><div class="metric-val">90.84%</div><div class="metric-lbl">Inference R²</div></div>
    <div class="metric-box"><div class="metric-val">XGBoost</div><div class="metric-lbl">Architecture</div></div>
    <div class="metric-box"><div class="metric-val">UCI Dataset</div><div class="metric-lbl">Training Source</div></div>
</div>""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-label">— Sensor Diagnostics</div>', unsafe_allow_html=True)
    sensor = st.number_input("Sensor PT08.S1 Reading", 500, 2500, 1200)
    temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    rh = st.slider("Relative Humidity (%)", 0.0, 100.0, 50.0)
    ah = st.slider("Absolute Humidity", 0.1, 2.5, 0.8)
    
    c1, c2, c3 = st.columns(3)
    with c1: hour = st.selectbox("Hour", list(range(24)), 12)
    with c2: day = st.selectbox("Day", list(range(7)), format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
    with c3: month = st.selectbox("Month", list(range(1, 13)), 3)

    predict_btn = st.button("EXECUTE PREDICTION")

with col2:
    st.markdown('<div class="section-label">— Result Output</div>', unsafe_allow_html=True)
    
    if predict_btn:
        input_df = pd.DataFrame({
            'PT08.S1(CO)': [sensor], 'RH': [rh], 'T': [temp], 'AH': [ah],
            'dayofweek': [day], 'Month': [month], 'Hour': [hour]
        })
        
        with st.spinner("Processing Telemetry..."):
            res = model.predict(input_df)[0]
        
        is_safe = res <= 3.0
        color_class = "result-number-safe" if is_safe else "result-number-danger"
        status_text = "✓ ATMOSPHERE STABLE" if is_safe else "⚠ ELEVATED CONCENTRATION"
        
        st.markdown(f"""
        <div class="result-box" style="text-align: center; padding: 3rem;">
            <div style="text-transform: uppercase; font-size: 0.65rem; color: #707085; letter-spacing: 0.2em;">Predicted CO Level</div>
            <div class="{color_class}">{res:.2f}</div>
            <div style="font-family: 'DM Mono', monospace; color: #707085;">mg / m³</div>
            <div style="margin-top: 2rem; font-weight: 700; font-size: 0.85rem; color: {'#00ff80' if is_safe else '#ff5050'};">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""<div class="waiting-box" style="height: 335px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
            <div style="font-size: 3rem; opacity: 0.1;">🌫</div>
            <div style="font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #303040; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 1rem;">Awaiting Data Injection</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="footer">AirSense Systems · built by aditya · uci air quality v1.0</div>', unsafe_allow_html=True)