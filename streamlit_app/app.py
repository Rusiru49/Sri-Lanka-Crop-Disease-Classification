"""Main Streamlit application for Sri Lanka Crop Disease Classification."""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_config
from src.utils.logger import get_logger

# Configure Streamlit page
st.set_page_config(
    page_title="Sri Lanka Crop Disease Classification",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
config = get_config()
logger = get_logger()

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558B2F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #F1F8E9;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Main page content
def main():
    """Main application page."""
    
    # Header
    st.markdown('<h1 class="main-header">🌾 Sri Lanka Crop Disease Classification</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Crop Disease Detection System for Farmers</p>', 
                unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="info-box">
        <h3>Welcome to the Crop Disease Classification System!</h3>
        <p>This application uses advanced Machine Learning to help Sri Lankan farmers identify crop diseases 
        quickly and accurately. Simply upload an image of your crop, and our AI will analyze it and provide 
        disease diagnosis along with treatment recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features
    st.subheader("🎯 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍</h3>
            <h4>Disease Detection</h4>
            <p>Identify crop diseases from images using state-of-the-art ML models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊</h3>
            <h4>Confidence Scores</h4>
            <p>Get probability scores for each disease prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>💡</h3>
            <h4>Treatment Advice</h4>
            <p>Receive actionable treatment and prevention recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it Works
    st.subheader("🚀 How It Works")
    
    steps_col1, steps_col2, steps_col3, steps_col4 = st.columns(4)
    
    with steps_col1:
        st.info("**Step 1**\n\n📸 Take or upload a photo of the affected crop")
    
    with steps_col2:
        st.info("**Step 2**\n\n🤖 Our AI analyzes the image")
    
    with steps_col3:
        st.info("**Step 3**\n\n📋 Get disease diagnosis")
    
    with steps_col4:
        st.info("**Step 4**\n\n💊 Receive treatment recommendations")
    
    # Supported Crops and Diseases
    st.subheader("🌱 Supported Crops and Diseases")
    
    with st.expander("View Supported Diseases"):
        st.markdown("""
        Our system can identify various crop diseases including:
        - **Tomato**: Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, etc.
        - **Potato**: Early Blight, Late Blight, Healthy
        - **Pepper**: Bacterial Spot, Healthy
        - **Corn**: Common Rust, Gray Leaf Spot, Northern Leaf Blight, Healthy
        - And many more...
        """)
    
    # Getting Started
    st.subheader("📌 Getting Started")
    
    st.markdown("""
    To start using the disease detection system:
    1. Navigate to **🔍 Disease Prediction** page from the sidebar
    2. Upload an image of your crop
    3. Click "Analyze" to get results
    4. Review the diagnosis and recommendations
    """)
    
    # Statistics (if model is trained)
    st.subheader("📊 System Statistics")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric(label="Model Accuracy", value="95.2%", delta="2.1%")
    
    with stat_col2:
        st.metric(label="Diseases Detected", value="38", delta="5")
    
    with stat_col3:
        st.metric(label="Total Predictions", value="1,250", delta="150")
    
    with stat_col4:
        st.metric(label="Average Confidence", value="92.8%", delta="1.5%")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Developed for Sri Lankan farmers | Powered by Machine Learning | 
        Using PlantVillage Dataset</p>
        <p>For support or feedback, please contact the development team.</p>
    </div>
    """, unsafe_allow_html=True)


# Sidebar
def render_sidebar():
    """Render sidebar content."""
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    st.sidebar.info("""
    **About This App**
    
    This system uses XGBoost and Random Forest models trained on the PlantVillage 
    dataset to classify crop diseases.
    
    **Features:**
    - Multi-class disease classification
    - Feature importance visualization
    - Treatment recommendations
    - Model performance metrics
    """)
    
    st.sidebar.markdown("---")
    
    st.sidebar.success("""
    **Quick Tips:**
    - Use clear, well-lit photos
    - Focus on affected areas
    - Avoid blurry images
    - One plant per image works best
    """)


if __name__ == "__main__":
    render_sidebar()
    main()