import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Home", page_icon="🏡", layout="wide")

st.title("🌿 Sri Lanka Crop Disease Classification")
st.write("Welcome to the system! Use the sidebar to navigate across prediction, insights, and dataset overview.")

# Optional: Add logo
project_root = Path(__file__).parent.parent
logo_path = project_root / "assets" / "logo.png"

if logo_path.exists():
    st.image(str(logo_path), width=250)

st.header("📌 Project Features")
st.markdown("""
- 🔍 Diagnose plant diseases using deep learning  
- 📊 View model insights and performance  
- 🌾 Understand class distribution and dataset structure  
- 📁 Process images, extract features, evaluate predictions  
""")

st.info("Use the left sidebar to navigate to other pages.")
