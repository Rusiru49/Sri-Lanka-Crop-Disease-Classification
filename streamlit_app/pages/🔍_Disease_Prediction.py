"""Disease prediction page for Streamlit app."""

import streamlit as st

# CRITICAL: set_page_config MUST be the first Streamlit command
st.set_page_config(page_title="Disease Prediction", page_icon="🔍", layout="wide")

import sys
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
import cv2
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def get_config():
    config_file = Path(__file__).resolve().parents[2] / "config/config.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r") as f:
        return yaml.safe_load(f)

from src.models.predict import DiseasePredictor
from src.features.feature_engineering import FeatureExtractor
from src.utils.helpers import get_class_names

# ============================================================================
# EMERGENCY RUNTIME PATCHES - Remove after updating source files
# ============================================================================

def patched_get_class_names(data_dir):
    """Patched version that handles missing directories."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return patched_get_default_class_names()
    try:
        class_names = []
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                class_names.append(item)
        return sorted(class_names) if class_names else patched_get_default_class_names()
    except:
        return patched_get_default_class_names()

def patched_get_default_class_names():
    return [
        "Corn_Common_rust", "Corn_Gray_leaf_spot", "Corn_Healthy",
        "Corn_Northern_Leaf_Blight", "Pepper_Bacterial_spot", "Pepper_Healthy",
        "Potato_Early_blight", "Potato_Healthy", "Potato_Late_blight",
        "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Healthy", 
        "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", 
        "Tomato_Spider_mites", "Tomato_Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", 
        "Tomato_mosaic_virus"
    ]

# Apply patches
from src.utils import helpers
helpers.get_class_names = patched_get_class_names
helpers.get_default_class_names = patched_get_default_class_names

from src.data.data_loader import ImageDataLoader
original_init = ImageDataLoader.__init__
def patched_init(self, data_dir=None):
    from src.utils.config import get_config
    from src.utils.helpers import set_seed
    config = get_config()
    self.data_dir = data_dir or config.get('data.raw_dir', 'data/raw/plantvillage')
    self.image_size = config.get('image_size', (224, 224))
    self.random_seed = config.get('random_seed', 42)
    set_seed(self.random_seed)
    if not os.path.exists(self.data_dir):
        self.class_names = patched_get_default_class_names()
        self.is_pre_split = False
    else:
        self.is_pre_split = self._check_pre_split()
        train_dir = os.path.join(self.data_dir, 'train') if self.is_pre_split else self.data_dir
        try:
            self.class_names = patched_get_class_names(train_dir)
        except:
            self.class_names = patched_get_default_class_names()
    self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
    self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
ImageDataLoader.__init__ = patched_init

def patched_preprocess_image(self, image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_size = getattr(self, 'image_size', (224, 224))
    image = cv2.resize(image, image_size)
    image = image.astype(np.float32) / 255.0
    return image
DiseasePredictor.preprocess_image = patched_preprocess_image

# Show success message after patches are applied
st.success("✓ Runtime patches applied - Ready for predictions!")
# ============================================================================

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
        model_path = Path(__file__).resolve().parents[2] / "models/saved_models/random_forest_model.pkl"
        
        if not os.path.exists(model_path):
            st.warning("⚠️ Model not found. Please train the model first.")
            return None
        
        # Try to load class names from data directory
        data_dir = Path(__file__).resolve().parents[2] / "data/raw/plantvillage/train"
        
        if data_dir.exists():
            try:
                class_names = get_class_names(str(data_dir))
                st.success(f"✅ Loaded {len(class_names)} classes from data directory")
            except Exception as e:
                st.warning(f"⚠️ Could not load class names from directory: {str(e)}")
                class_names = get_default_class_names()
        else:
            # Use default class names if directory doesn't exist
            st.info("ℹ️ Using default class names (data directory not found)")
            class_names = get_default_class_names()
        
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
        import traceback
        st.error(traceback.format_exc())
        return None


def get_default_class_names():
    """Return default class names if data directory is not available."""
    return [
        "Tomato_Early_blight", 
        "Tomato_Late_blight", 
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites",
        "Tomato_Target_Spot",
        "Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato_mosaic_virus",
        "Tomato_Healthy", 
        "Potato_Early_blight", 
        "Potato_Late_blight",
        "Potato_Healthy", 
        "Pepper_Bacterial_spot", 
        "Pepper_Healthy",
        "Corn_Common_rust",
        "Corn_Gray_leaf_spot",
        "Corn_Northern_Leaf_Blight",
        "Corn_Healthy"
    ]


def display_prediction_result(result, image):
    """Display prediction results."""
    st.success("✅ Analysis Complete!")
    
    # Display image and prediction side by side
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_column_width=True)
    
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
    
    # Option to use sample images (only if directory exists)
    sample_dir = Path(__file__).resolve().parents[2] / "data/raw/plantvillage/test"
    
    if sample_dir.exists():
        use_sample = st.checkbox("Or use a sample image")
        
        if use_sample:
            try:
                # Get sample images
                sample_classes = [d for d in os.listdir(sample_dir) 
                                if os.path.isdir(os.path.join(sample_dir, d))]
                
                if sample_classes:
                    selected_class = st.selectbox("Select disease type:", sample_classes)
                    
                    class_path = os.path.join(sample_dir, selected_class)
                    sample_images = [f for f in os.listdir(class_path) 
                                   if f.endswith(('.jpg', '.png', '.jpeg'))]
                    
                    if sample_images:
                        selected_image = st.selectbox("Select image:", sample_images)
                        uploaded_file = os.path.join(class_path, selected_image)
                    else:
                        st.warning("No sample images found in selected class")
                else:
                    st.info("No sample classes available")
            except Exception as e:
                st.warning(f"Could not load sample images: {str(e)}")
    else:
        st.info("💡 Upload your own image to get started (sample images not available)")
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🔬 Analyze Image", type="primary", use_container_width=True)
    
    # Process image
    if analyze_button and uploaded_file is not None:
        try:
            # Load image
            if isinstance(uploaded_file, str):
                # File path (from sample images)
                image = Image.open(uploaded_file)
                image_path = uploaded_file
            else:
                # Uploaded file
                image = Image.open(uploaded_file)
                # Save temporarily
                temp_dir = Path(__file__).resolve().parents[2] / "temp"
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / "temp_upload.jpg"
                image.save(temp_path)
                image_path = str(temp_path)
            
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
            temp_path = Path(__file__).resolve().parents[2] / "temp/temp_upload.jpg"
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                except:
                    pass  # Ignore cleanup errors
        
        except Exception as e:
            st.error(f"❌ Error analyzing image: {str(e)}")
            import traceback
            st.error("**Detailed Error:**")
            st.code(traceback.format_exc())
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
    
    # Debug info (collapsible)
    with st.expander("🔧 Debug Information"):
        st.write("**Model Path:**", Path(__file__).resolve().parents[2] / "models/saved_models/lightgbm_model.pkl")
        st.write("**Data Directory:**", Path(__file__).resolve().parents[2] / "data/raw/plantvillage")
        st.write("**Sample Directory Exists:**", sample_dir.exists())
        if st.session_state.predictor:
            st.write("**Number of Classes:**", len(st.session_state.predictor.class_names))
            st.write("**All Classes:**")
            for i, class_name in enumerate(st.session_state.predictor.class_names, 1):
                st.write(f"{i}. {class_name}")


if __name__ == "__main__":
    main()