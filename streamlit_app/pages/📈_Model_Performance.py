"""Model performance page for Streamlit app."""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import load_json

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)


def display_metrics_overview():
    """Display overview of model metrics."""
    st.header("📊 Model Performance Overview")
    
    # Try to load evaluation results
    results_path = "reports/results/evaluation_metrics.json"
    
    if os.path.exists(results_path):
        results = load_json(results_path)
        test_metrics = results.get('test_metrics', {})
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            accuracy = test_metrics.get('accuracy', 0) * 100
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{accuracy:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            precision = test_metrics.get('precision_weighted', 0) * 100
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{precision:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            recall = test_metrics.get('recall_weighted', 0) * 100
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{recall:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            f1_score = test_metrics.get('f1_weighted', 0) * 100
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value">{f1_score:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        return results
    else:
        st.warning("⚠️ No evaluation results found. Please train the model first.")
        st.info("Run: `python scripts/train_model.py --save-plots`")
        return None


def display_confusion_matrix():
    """Display confusion matrix."""
    st.header("🔢 Confusion Matrix")
    
    # Check for confusion matrix images
    cm_path = "reports/figures/confusion_matrix.png"
    cm_norm_path = "reports/figures/confusion_matrix_normalized.png"
    
    if os.path.exists(cm_path) and os.path.exists(cm_norm_path):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Absolute Counts")
            cm_image = Image.open(cm_path)
            st.image(cm_image, use_container_width=True)
        
        with col2:
            st.subheader("Normalized")
            cm_norm_image = Image.open(cm_norm_path)
            st.image(cm_norm_image, use_container_width=True)
        
        with st.expander("ℹ️ Understanding the Confusion Matrix"):
            st.markdown("""
            The confusion matrix shows:
            - **Rows**: True labels (actual disease)
            - **Columns**: Predicted labels (model's prediction)
            - **Diagonal**: Correct predictions
            - **Off-diagonal**: Misclassifications
            
            A good model has high values on the diagonal and low values elsewhere.
            """)
    else:
        st.info("Confusion matrix not available. Train the model with --save-plots flag.")


def display_per_class_performance(results):
    """Display per-class metrics."""
    st.header("📋 Per-Class Performance")
    
    if results and 'per_class_metrics' in results:
        per_class = results['per_class_metrics']
        
        # Convert to DataFrame
        df = pd.DataFrame(per_class)
        
        # Create interactive table
        st.dataframe(
            df.style.format({
                'precision': '{:.2%}',
                'recall': '{:.2%}',
                'f1_score': '{:.2%}'
            }).background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1_score']),
            use_container_width=True
        )
        
        # Visualization
        st.subheader("Visual Comparison")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        # Precision
        axes[0].barh(df['class'], df['precision'], color='steelblue')
        axes[0].set_xlabel('Precision')
        axes[0].set_title('Precision by Class')
        axes[0].set_xlim([0, 1])
        
        # Recall
        axes[1].barh(df['class'], df['recall'], color='coral')
        axes[1].set_xlabel('Recall')
        axes[1].set_title('Recall by Class')
        axes[1].set_xlim([0, 1])
        
        # F1-Score
        axes[2].barh(df['class'], df['f1_score'], color='lightgreen')
        axes[2].set_xlabel('F1-Score')
        axes[2].set_title('F1-Score by Class')
        axes[2].set_xlim([0, 1])
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Best and worst performing classes
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 5 Performing Classes")
            top_5 = df.nlargest(5, 'f1_score')[['class', 'f1_score']]
            for idx, row in top_5.iterrows():
                st.success(f"**{row['class']}**: {row['f1_score']:.2%}")
        
        with col2:
            st.subheader("⚠️ Bottom 5 Performing Classes")
            bottom_5 = df.nsmallest(5, 'f1_score')[['class', 'f1_score']]
            for idx, row in bottom_5.iterrows():
                st.warning(f"**{row['class']}**: {row['f1_score']:.2%}")


def display_feature_importance():
    """Display feature importance."""
    st.header("🔍 Feature Importance")
    
    feature_importance_path = "reports/figures/feature_importance.png"
    
    if os.path.exists(feature_importance_path):
        fi_image = Image.open(feature_importance_path)
        st.image(fi_image, use_container_width=True)
        
        with st.expander("ℹ️ Understanding Feature Importance"):
            st.markdown("""
            Feature importance shows which features contribute most to the model's predictions:
            - **Deep features**: Extracted from pre-trained CNN (MobileNetV2)
            - **Color features**: RGB and HSV color histograms
            - **Texture features**: GLCM and LBP descriptors
            - **Shape features**: Contour properties and edge density
            
            Higher importance means the feature has more influence on predictions.
            """)
    else:
        st.info("Feature importance plot not available.")


def display_misclassified_samples(results):
    """Display misclassified samples."""
    st.header("❌ Misclassified Samples Analysis")
    
    if results and 'misclassified_samples' in results:
        misclassified = results['misclassified_samples']
        
        if misclassified.get('samples'):
            st.write(f"Total misclassified: {misclassified.get('total_misclassified', 0)}")
            
            # Create DataFrame
            df = pd.DataFrame(misclassified['samples'])
            
            # Display table
            st.dataframe(
                df.style.format({'confidence': '{:.2%}'}),
                use_container_width=True
            )
            
            # Insights
            st.subheader("💡 Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Most confused classes
                st.write("**Most Common Misclassifications:**")
                class_pairs = df.groupby(['true_class', 'predicted_class']).size().nlargest(5)
                for (true_cls, pred_cls), count in class_pairs.items():
                    st.write(f"- {true_cls} → {pred_cls}: {count} times")
            
            with col2:
                # Average confidence
                avg_conf = df['confidence'].mean()
                st.metric("Average Confidence on Errors", f"{avg_conf:.1%}")
                
                if avg_conf < 0.5:
                    st.info("Low confidence suggests model uncertainty.")
                elif avg_conf > 0.7:
                    st.warning("High confidence on errors suggests systematic mistakes.")
        else:
            st.success("🎉 No misclassified samples found!")
    else:
        st.info("Misclassification analysis not available.")


def display_data_info(results):
    """Display dataset information."""
    st.header("📚 Dataset Information")
    
    if results and 'data_info' in results:
        data_info = results['data_info']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Training Samples", f"{data_info.get('train_size', 0):,}")
        
        with col2:
            st.metric("Validation Samples", f"{data_info.get('val_size', 0):,}")
        
        with col3:
            st.metric("Test Samples", f"{data_info.get('test_size', 0):,}")
        
        with col4:
            st.metric("Feature Dimension", f"{data_info.get('n_features', 0):,}")
        
        # Class distribution
        if 'class_names' in results:
            st.subheader("Disease Classes")
            class_names = results['class_names']
            st.write(f"Total Classes: **{len(class_names)}**")
            
            with st.expander("View all classes"):
                cols = st.columns(3)
                for idx, class_name in enumerate(class_names):
                    cols[idx % 3].write(f"- {class_name}")


def main():
    """Main performance page."""
    st.title("📈 Model Performance Dashboard")
    st.markdown("Comprehensive evaluation metrics and visualizations for the trained model.")
    
    # Display metrics overview
    results = display_metrics_overview()
    
    st.markdown("---")
    
    # Display confusion matrix
    display_confusion_matrix()
    
    st.markdown("---")
    
    # Display per-class performance
    if results:
        display_per_class_performance(results)
    
    st.markdown("---")
    
    # Display feature importance
    display_feature_importance()
    
    st.markdown("---")
    
    # Display misclassified samples
    if results:
        display_misclassified_samples(results)
    
    st.markdown("---")
    
    # Display dataset info
    if results:
        display_data_info(results)
    
    # Download report
    st.markdown("---")
    st.header("📥 Export Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col2:
        if st.button("Generate PDF Report", type="primary", use_container_width=True):
            st.info("PDF report generation feature coming soon!")


if __name__ == "__main__":
    main()