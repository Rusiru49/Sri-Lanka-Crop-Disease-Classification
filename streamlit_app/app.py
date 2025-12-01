"""
app.py

Main Streamlit application for Sri Lanka Crop Disease Classification.
Updated to use a separate sidebar component located in components/sidebar.py
"""

import streamlit as st
from pathlib import Path
import sys
import os
import io
import pickle
from PIL import Image, ImageOps
import numpy as np
from typing import Dict, Any

# --------------------------
# Project root setup
# --------------------------
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Import sidebar component
try:
    from components.sidebar import render_sidebar
except Exception as e:
    st.error(f"Sidebar import failed: {e}")
    def render_sidebar():
        return "Home"

# Try load utils (non-fatal)
try:
    from src.utils.config import get_config
    from src.utils.logger import get_logger
    config = get_config()
    logger = get_logger()
except Exception:
    config = None
    logger = None

# Streamlit UI configuration
st.set_page_config(
    page_title="Sri Lanka Crop Disease Classification",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Custom Style
# --------------------------
st.markdown(
    """
    <style>
    :root {
        --card-bg: #ffffff;
        --muted: #6b7280;
        --border: #e6e6e6;
    }
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1f7a1f;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: var(--muted);
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
    }
    .muted { color: var(--muted); }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Utility functions
# --------------------------
def check_system_status() -> Dict[str, bool]:
    return {
        "model_exists": (project_root / "models" / "saved_models" / "lightgbm_model.pkl").exists(),
        "data_dir_exists": (project_root / "data" / "raw" / "plantvillage").exists(),
        "config_loaded": config is not None
    }

def load_model(path: Path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except:
        return None

DEFAULT_CLASSES = [
    "Tomato__Early_blight", "Tomato__Late_blight", "Tomato__Leaf_Mold",
    "Tomato__Septoria_leaf_spot", "Tomato__Spider_mites", "Tomato__Target_Spot",
    "Tomato__Yellow_Leaf_Curl_Virus", "Tomato__Mosaic_Virus",
    "Tomato__Bacterial_spot", "Tomato__Healthy",
    "Potato__Early_blight", "Potato__Late_blight", "Potato__Healthy",
    "Pepper__Bacterial_spot", "Pepper__Healthy",
    "Corn__Common_rust", "Corn__Gray_leaf_spot", "Corn__Northern_leaf_blight", "Corn__Healthy"
]

def preprocess_image(img: Image.Image, size=(224, 224)) -> np.ndarray:
    img = img.convert("RGB")
    img = ImageOps.fit(img, size, Image.ANTIALIAS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_image(image_bytes: bytes, model, classes=DEFAULT_CLASSES):
    result = {"top": [], "raw": None, "used_model": False}
    img = Image.open(io.BytesIO(image_bytes))
    x = preprocess_image(img)

    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x.reshape(x.shape[0], -1))[0]
            idx = np.argsort(probs)[::-1][:3]
            top = [(classes[i], float(probs[i])) for i in idx]
            return {"top": top, "raw": probs.tolist(), "used_model": True}
        elif hasattr(model, "predict"):
            pred = model.predict(x.reshape(x.shape[0], -1))[0]
            return {"top": [(str(pred), 1.0)], "raw": None, "used_model": True}
    except:
        pass

    return {
        "top": [(classes[i], 1.0/3) for i in range(3)],
        "raw": None,
        "used_model": False
    }

# --------------------------
# Pages
# --------------------------
def page_home():
    st.markdown('<h1 class="main-header">🌾 Sri Lanka Crop Disease Classification</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered, farmer-friendly crop disease detection.</p>', unsafe_allow_html=True)

    status = check_system_status()
    with st.expander("🔧 System Status", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.success("✅ Config loaded" if status["config_loaded"] else "⚠️ Config missing")
        c2.success("✅ Model found" if status["model_exists"] else "ℹ️ No model — fallback mode")
        c3.success("✅ Dataset found" if status["data_dir_exists"] else "ℹ️ Dataset not present")

    st.markdown("## What this app does")
    st.markdown(
        """
        - Upload plant leaf images to classify diseases  
        - View top predictions with confidence  
        - Model insights and training utilities  
        """
    )

def page_disease_prediction():
    st.header("🔍 Disease Prediction")

    uploaded = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"])
    model_path = project_root / "models" / "saved_models" / "lightgbm_model.pkl"
    model = load_model(model_path) if model_path.exists() else None

    if uploaded:
        img_bytes = uploaded.read()
        st.image(Image.open(io.BytesIO(img_bytes)), use_column_width=True)

        with st.spinner("Analyzing..."):
            result = predict_image(img_bytes, model)

        st.subheader("Prediction Result")
        for label, score in result["top"]:
            st.write(f"**{label} — {score*100:.1f}%**")

def page_model_insights():
    st.header("📊 Model Insights")
    model_path = project_root / "models" / "saved_models" / "lightgbm_model.pkl"

    if not model_path.exists():
        st.info("No model file found.")
        return

    model = load_model(model_path)
    st.success(f"Model loaded: {model_path.name}")

    if hasattr(model, "get_params"):
        st.write("### Model Parameters")
        st.json(model.get_params())

def page_train_model():
    st.header("⚙️ Train Model")
    st.info("Training from UI is disabled. Run:")
    st.code("python scripts/train_model.py")

def page_about():
    st.header("ℹ️ About this App")
    st.markdown(
        """
        Built using Streamlit and classical ML models  
        Dataset: PlantVillage  
        Created for Sri Lankan agricultural support  
        """
    )

# --------------------------
# Main Router
# --------------------------
def main():
    # FIXED: pass project_root and check_system_status
    page = render_sidebar(project_root, check_system_status)

    if page == "Home":
        page_home()
    elif page == "Disease Prediction":
        page_disease_prediction()
    elif page == "Model Insights":
        page_model_insights()
    elif page == "Train Model":
        page_train_model()
    elif page == "About":
        page_about()
    else:
        page_home()

if __name__ == "__main__":
    main()
