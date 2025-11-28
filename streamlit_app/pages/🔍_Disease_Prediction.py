"""Disease prediction page for Streamlit app."""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.predict import DiseasePredictor
from src.features.feature_engineering import FeatureExtractor
from src.utils.helpers import get_class_names

st.set_page_config(page_title="Disease Prediction", page_icon="🔍", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FF9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #F44336;
        font-weight: bold;
    }
    .recommendation-box {
        background-color: #FFF3E0;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF9800;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None


def load_predictor():
    """Load the trained model and predictor."""
    try:
        model_path = "models/saved_models/xgboost_model.pkl"
        
        if not os.path.exists(model_path):
            st.warning("⚠️ Model not found. Please train the model first.")
            return None
        
        # Load class names
        data_dir = "data/raw/plantvillage/train"
        if os.path.exists(data_dir):
            class_names = get_class_names(data_dir)
        else:
            # Default class names (example)
            class_names = [
                "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
                "Tomato_Healthy", "Potato_Early_blight", "Potato_Late_blight",
                "Potato_Healthy", "Pepper_Bacterial_spot", "Pepper_Healthy"
            ]
        
        # Create feature extractor
        feature_extractor = FeatureExtractor()
        
        # Create predictor
        predictor = DiseasePredictor(
            model_path=model_path,
            class_names=class_names,
            feature_extractor=feature_extractor
        )
        
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def display_prediction_result(result, image):
    """Display prediction results."""
    st.success("✅ Analysis Complete!")
    
    # Display image and prediction side by side
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        # Main prediction
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        
        # Confidence styling
        if confidence >= 0.8:
            conf_class = "confidence-high"
            conf_emoji = "🟢"
        elif confidence >= 0.5:
            conf_class = "confidence-medium"
            conf_emoji = "🟡"
        else:
            conf_class = "confidence-low"
            conf_emoji = "🔴"
        
        st.markdown(f"""
        <div class="prediction-box">
            <h2>🌿 {predicted_class.replace('_', ' ').title()}</h2>
            <p style='font-size: 1.5rem;'>{conf_emoji} Confidence: 
            <span class='{conf_class}'>{confidence:.1%}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Top 3 predictions
        st.subheader("📊 Top 3 Predictions")
        for i, pred in enumerate(result['top_3_predictions'], 1):
            progress_value = pred['confidence']
            st.write(f"**{i}. {pred['class'].replace('_', ' ').title()}**")
            st.progress(progress_value)
            st.write(f"Confidence: {pred['confidence']:.1%}")
            st.markdown("---")
    
    # Recommendations
    if 'recommendations' in result:
        st.subheader("💡 Treatment Recommendations")
        
        recommendations = result['recommendations']
        
        # Severity indicator
        severity = recommendations.get('severity', 'Unknown')
        if severity == 'High':
            st.error(f"⚠️ **Severity Level:** {severity}")
        elif severity == 'Medium':
            st.warning(f"⚠️ **Severity Level:** {severity}")
        elif severity == 'None':
            st.success(f"✅ **Status:** Healthy Plant")
        else:
            st.info(f"ℹ️ **Severity Level:** {severity}")
        
        # Description
        st.write(f"**Description:** {recommendations.get('description', 'N/A')}")
        
        # Treatment
        if 'treatment' in recommendations:
            st.markdown("**🔬 Treatment Steps:**")
            for step in recommendations['treatment']:
                st.write(f"- {step}")
        
        # Prevention
        if 'prevention' in recommendations:
            st.markdown("**🛡️ Prevention Measures:**")
            for measure in recommendations['prevention']:
                st.write(f"- {measure}")


def main():
    """Main prediction page."""
    st.title("🔍 Crop Disease Prediction")
    st.markdown("Upload an image of your crop to detect diseases and get treatment recommendations.")
    
    # Load predictor
    if st.session_state.predictor is None:
        with st.spinner("Loading model..."):
            st.session_state.predictor = load_predictor()
    
    if st.session_state.predictor is None:
        st.error("Failed to load model. Please ensure the model is trained and saved.")
        st.info("Run the training script first: `python scripts/train_model.py`")
        return
    
    # File upload
    st.subheader("📤 Upload Crop Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the crop leaf or plant"
    )
    
    # Option to use sample images
    use_sample = st.checkbox("Or use a sample image")
    
    if use_sample:
        sample_dir = "data/raw/plantvillage/test"
        if os.path.exists(sample_dir):
            # Get sample images
            sample_classes = os.listdir(sample_dir)
            selected_class = st.selectbox("Select disease type:", sample_classes)
            
            class_path = os.path.join(sample_dir, selected_class)
            sample_images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            if sample_images:
                selected_image = st.selectbox("Select image:", sample_images)
                uploaded_file = os.path.join(class_path, selected_image)
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🔬 Analyze Image", type="primary", use_container_width=True)
    
    # Process image
    if analyze_button and uploaded_file is not None:
        try:
            # Load image
            if isinstance(uploaded_file, str):
                image = Image.open(uploaded_file)
                image_path = uploaded_file
            else:
                image = Image.open(uploaded_file)
                # Save temporarily
                temp_path = "temp_upload.jpg"
                image.save(temp_path)
                image_path = temp_path
            
            # Display loading
            with st.spinner("🔄 Analyzing image..."):
                # Make prediction
                result = st.session_state.predictor.predict_with_recommendations(image_path)
                st.session_state.prediction_result = result
            
            # Display results
            display_prediction_result(result, image)
            
            # Download report button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("📥 Download Report", use_container_width=True):
                    st.info("Report download feature coming soon!")
            
            # Clean up temp file
            if os.path.exists("temp_upload.jpg"):
                os.remove("temp_upload.jpg")
        
        except Exception as e:
            st.error(f"❌ Error analyzing image: {str(e)}")
            st.info("Please try uploading a different image or check if the model is properly trained.")
    
    elif analyze_button:
        st.warning("⚠️ Please upload an image first!")
    
    # Tips
    with st.expander("📌 Tips for Best Results"):
        st.markdown("""
        - Use **clear, well-lit photos**
        - **Focus on the affected area** of the plant
        - Avoid **blurry or dark images**
        - Take photos during **daytime** for better lighting
        - **One plant/leaf per image** works best
        - Ensure the **disease symptoms are visible**
        - Keep the background **simple and uncluttered**
        """)


if __name__ == "__main__":
    main()