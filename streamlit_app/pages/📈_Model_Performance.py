"""Model performance page for Streamlit app."""

import streamlit as st

# CRITICAL: set_page_config MUST be the first Streamlit command
st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

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
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def display_metrics_overview():
    """Display overview of model metrics."""
    st.header("📊 Model Performance Overview")
    
    # Try to load evaluation results
    results_path = project_root / "reports/results/evaluation_metrics.json"
    
    if results_path.exists():
        try:
            results = load_json(str(results_path))
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
        except Exception as e:
            st.error(f"Error loading evaluation metrics: {str(e)}")
            return None
    else:
        st.warning("⚠️ No evaluation results found. Please train the model first.")
        st.info("**To train the model:**")
        st.code("python scripts/train_model.py --save-plots", language="bash")
        
        # Show sample metrics as placeholder
        st.markdown("### 📊 Sample Metrics (Placeholder)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", "95.2%", "High")
        with col2:
            st.metric("Precision", "94.8%", "Good")
        with col3:
            st.metric("Recall", "95.1%", "Good")
        with col4:
            st.metric("F1-Score", "94.9%", "Balanced")
        
        return None


def display_confusion_matrix():
    """Display confusion matrix."""
    st.header("🔢 Confusion Matrix")
    
    # Check for confusion matrix images
    cm_path = project_root / "reports/figures/confusion_matrix.png"
    cm_norm_path = project_root / "reports/figures/confusion_matrix_normalized.png"
    
    if cm_path.exists() and cm_norm_path.exists():
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Absolute Counts")
            cm_image = Image.open(cm_path)
            st.image(cm_image, use_column_width=True)
        
        with col2:
            st.subheader("Normalized")
            cm_norm_image = Image.open(cm_norm_path)
            st.image(cm_norm_image, use_column_width=True)
        
        with st.expander("ℹ️ Understanding the Confusion Matrix"):
            st.markdown("""
            The confusion matrix shows:
            - **Rows**: True labels (actual disease)
            - **Columns**: Predicted labels (model's prediction)
            - **Diagonal**: Correct predictions (darker = better)
            - **Off-diagonal**: Misclassifications (lighter = fewer errors)
            
            **Interpretation:**
            - A good model has high values on the diagonal and low values elsewhere
            - Look for patterns in misclassifications to identify similar diseases
            - Normalized matrix shows proportions (useful for imbalanced datasets)
            """)
    else:
        st.info("📊 Confusion matrix not available.")
        st.markdown("""
        <div class="info-box">
            <strong>How to generate:</strong><br>
            Train the model with the <code>--save-plots</code> flag:<br>
            <code>python scripts/train_model.py --save-plots</code>
        </div>
        """, unsafe_allow_html=True)


def display_per_class_performance(results):
    """Display per-class metrics."""
    st.header("📋 Per-Class Performance")
    
    if results and 'per_class_metrics' in results:
        per_class = results['per_class_metrics']
        
        # Convert to DataFrame
        df = pd.DataFrame(per_class)
        
        # Ensure numeric types
        df['precision'] = pd.to_numeric(df['precision'], errors='coerce')
        df['recall'] = pd.to_numeric(df['recall'], errors='coerce')
        df['f1_score'] = pd.to_numeric(df['f1_score'], errors='coerce')
        
        # Create interactive table
        st.dataframe(
            df.style.format({
                'precision': '{:.2%}',
                'recall': '{:.2%}',
                'f1_score': '{:.2%}',
                'support': '{:.0f}'
            }).background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1_score']),
            use_column_width=True
        )
        
        # Visualization
        st.subheader("📊 Visual Comparison")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, max(6, len(df) * 0.3)))
        
        # Sort by F1-score for better visualization
        df_sorted = df.sort_values('f1_score', ascending=True)
        
        # Precision
        axes[0].barh(df_sorted['class'], df_sorted['precision'], color='steelblue')
        axes[0].set_xlabel('Precision')
        axes[0].set_title('Precision by Class')
        axes[0].set_xlim([0, 1])
        axes[0].grid(axis='x', alpha=0.3)
        
        # Recall
        axes[1].barh(df_sorted['class'], df_sorted['recall'], color='coral')
        axes[1].set_xlabel('Recall')
        axes[1].set_title('Recall by Class')
        axes[1].set_xlim([0, 1])
        axes[1].grid(axis='x', alpha=0.3)
        
        # F1-Score
        axes[2].barh(df_sorted['class'], df_sorted['f1_score'], color='lightgreen')
        axes[2].set_xlabel('F1-Score')
        axes[2].set_title('F1-Score by Class')
        axes[2].set_xlim([0, 1])
        axes[2].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Best and worst performing classes
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 5 Performing Classes")
            top_5 = df.nlargest(5, 'f1_score')[['class', 'f1_score', 'support']]
            for idx, row in top_5.iterrows():
                st.markdown(f"""
                <div class="success-box">
                    <strong>{row['class']}</strong><br>
                    F1-Score: {row['f1_score']:.2%} | Support: {row['support']:.0f}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("⚠️ Bottom 5 Performing Classes")
            bottom_5 = df.nsmallest(5, 'f1_score')[['class', 'f1_score', 'support']]
            for idx, row in bottom_5.iterrows():
                st.markdown(f"""
                <div class="warning-box">
                    <strong>{row['class']}</strong><br>
                    F1-Score: {row['f1_score']:.2%} | Support: {row['support']:.0f}
                </div>
                """, unsafe_allow_html=True)
        
        # Summary statistics
        with st.expander("📊 Summary Statistics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Precision", f"{df['precision'].mean():.2%}")
            with col2:
                st.metric("Average Recall", f"{df['recall'].mean():.2%}")
            with col3:
                st.metric("Average F1-Score", f"{df['f1_score'].mean():.2%}")
    else:
        st.info("Per-class performance metrics not available. Train the model to see detailed metrics.")


def display_feature_importance():
    """Display feature importance."""
    st.header("🔍 Feature Importance")
    
    feature_importance_path = project_root / "reports/figures/feature_importance.png"
    
    if feature_importance_path.exists():
        fi_image = Image.open(feature_importance_path)
        st.image(fi_image, use_column_width=True)
        
        with st.expander("ℹ️ Understanding Feature Importance"):
            st.markdown("""
            Feature importance shows which features contribute most to the model's predictions:
            
            **Feature Types:**
            - **Deep features**: Extracted from pre-trained CNN (MobileNetV2)
            - **Color features**: RGB and HSV color histograms
            - **Texture features**: GLCM (Gray-Level Co-occurrence Matrix) and LBP (Local Binary Patterns)
            - **Shape features**: Contour properties and edge density
            
            **Interpretation:**
            - Higher importance = more influence on predictions
            - Top features are most critical for disease classification
            - Can help understand what the model "sees" in images
            """)
    else:
        st.info("📊 Feature importance plot not available.")
        st.markdown("""
        <div class="info-box">
            <strong>Note:</strong> Feature importance is automatically generated during model training.
            Train a tree-based model (LightGBM, RandomForest, XGBoost) to see feature importance.
        </div>
        """, unsafe_allow_html=True)


def display_misclassified_samples(results):
    """Display misclassified samples."""
    st.header("❌ Misclassified Samples Analysis")
    
    if results and 'misclassified_samples' in results:
        misclassified = results['misclassified_samples']
        
        if misclassified.get('samples'):
            total_misc = misclassified.get('total_misclassified', 0)
            st.metric("Total Misclassified", total_misc)
            
            # Create DataFrame
            df = pd.DataFrame(misclassified['samples'])
            
            # Display table with filtering
            st.subheader("🔍 Misclassification Details")
            
            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                if 'true_class' in df.columns:
                    true_classes = ['All'] + sorted(df['true_class'].unique().tolist())
                    selected_true = st.selectbox("Filter by True Class", true_classes)
            with col2:
                if 'predicted_class' in df.columns:
                    pred_classes = ['All'] + sorted(df['predicted_class'].unique().tolist())
                    selected_pred = st.selectbox("Filter by Predicted Class", pred_classes)
            
            # Apply filters
            filtered_df = df.copy()
            if selected_true != 'All':
                filtered_df = filtered_df[filtered_df['true_class'] == selected_true]
            if selected_pred != 'All':
                filtered_df = filtered_df[filtered_df['predicted_class'] == selected_pred]
            
            # Display filtered table
            st.dataframe(
                filtered_df.style.format({'confidence': '{:.2%}'} if 'confidence' in filtered_df.columns else {}),
                use_column_width=True
            )
            
            # Insights
            st.subheader("💡 Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Most Common Misclassifications:**")
                if 'true_class' in df.columns and 'predicted_class' in df.columns:
                    class_pairs = df.groupby(['true_class', 'predicted_class']).size().nlargest(5)
                    for (true_cls, pred_cls), count in class_pairs.items():
                        st.write(f"- **{true_cls}** → **{pred_cls}**: {count} times")
                else:
                    st.info("Class information not available")
            
            with col2:
                # Average confidence
                if 'confidence' in df.columns:
                    avg_conf = df['confidence'].mean()
                    st.metric("Average Confidence on Errors", f"{avg_conf:.1%}")
                    
                    if avg_conf < 0.5:
                        st.info("💡 Low confidence suggests model uncertainty. This is actually good - the model knows it's unsure!")
                    elif avg_conf > 0.7:
                        st.warning("⚠️ High confidence on errors suggests systematic mistakes. Model may need retraining.")
                    else:
                        st.success("✓ Moderate confidence on errors is typical.")
        else:
            st.markdown("""
            <div class="success-box">
                <h3>🎉 Perfect Predictions!</h3>
                <p>No misclassified samples found in the evaluation set.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Misclassification analysis not available. This data is generated during model evaluation.")


def display_data_info(results):
    """Display dataset information."""
    st.header("📚 Dataset Information")
    
    if results and 'data_info' in results:
        data_info = results['data_info']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            train_size = data_info.get('train_size', 0)
            st.metric("Training Samples", f"{train_size:,}")
        
        with col2:
            val_size = data_info.get('val_size', 0)
            st.metric("Validation Samples", f"{val_size:,}")
        
        with col3:
            test_size = data_info.get('test_size', 0)
            st.metric("Test Samples", f"{test_size:,}")
        
        with col4:
            n_features = data_info.get('n_features', 0)
            st.metric("Feature Dimension", f"{n_features:,}")
        
        # Total samples
        total = train_size + val_size + test_size
        if total > 0:
            st.metric("Total Samples", f"{total:,}")
            
            # Split ratios
            with st.expander("📊 Dataset Split Ratios"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Train %", f"{(train_size/total)*100:.1f}%")
                with col2:
                    st.metric("Validation %", f"{(val_size/total)*100:.1f}%")
                with col3:
                    st.metric("Test %", f"{(test_size/total)*100:.1f}%")
        
        # Class distribution
        if 'class_names' in results:
            st.subheader("🌱 Disease Classes")
            class_names = results['class_names']
            st.write(f"Total Classes: **{len(class_names)}**")
            
            with st.expander("📋 View all classes"):
                # Display in columns
                n_cols = 3
                cols = st.columns(n_cols)
                for idx, class_name in enumerate(sorted(class_names)):
                    cols[idx % n_cols].write(f"{idx + 1}. {class_name}")
    else:
        st.info("Dataset information not available. This data is saved during model training.")


def display_training_info(results):
    """Display training information."""
    st.header("⚙️ Training Configuration")
    
    if results and 'training_info' in results:
        training_info = results['training_info']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Settings")
            st.write(f"**Model Type:** {training_info.get('model_type', 'N/A')}")
            st.write(f"**Training Time:** {training_info.get('training_time', 'N/A')}")
            st.write(f"**Image Size:** {training_info.get('image_size', 'N/A')}")
        
        with col2:
            st.subheader("Hyperparameters")
            if 'hyperparameters' in training_info:
                for key, value in training_info['hyperparameters'].items():
                    st.write(f"**{key}:** {value}")


def main():
    """Main performance page."""
    st.title("📈 Model Performance Dashboard")
    st.markdown("Comprehensive evaluation metrics and visualizations for the trained model.")
    
    # Check if reports directory exists
    reports_dir = project_root / "reports"
    if not reports_dir.exists():
        st.warning("⚠️ Reports directory not found. Creating it now...")
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "results").mkdir(exist_ok=True)
        (reports_dir / "figures").mkdir(exist_ok=True)
    
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
    
    # Display training info
    if results:
        st.markdown("---")
        display_training_info(results)
    
    # Download/Export section
    st.markdown("---")
    st.header("📥 Export & Documentation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Generate PDF Report"):
            st.info("PDF report generation feature coming soon!")
    
    with col2:
        if st.button("📊 Export Metrics CSV"):
            if results:
                st.info("CSV export feature coming soon!")
            else:
                st.warning("No metrics available to export")
    
    with col3:
        if st.button("📸 Download Plots"):
            st.info("Plots download feature coming soon!")
    
    # Footer with tips
    st.markdown("---")  
    with st.expander("💡 Tips for Improving Model Performance"):
        st.markdown("""
        Here are some best practices to improve your model:

        ### 🔄 Data Improvements
        - Add more images, especially for underrepresented classes
        - Ensure images are clean and not duplicates
        - Use data augmentation (rotation, zoom, flips) to improve robustness

        ### ⚙️ Model Improvements
        - Train deeper or more efficient CNNs (EfficientNet, ResNet50, MobileNetV3)
        - Fine-tune the last few layers instead of full transfer learning
        - Experiment with different optimizers (AdamW, SGD with momentum)

        ### 🧪 Training Process
        - Increase epochs gradually (watch validation loss to avoid overfitting)
        - Use callbacks: EarlyStopping, ReduceLROnPlateau
        - Try different batch sizes (16, 32, 64)

        ### 🌳 Ensemble Models
        - Combine CNN features + tree-based models (RandomForest, XGBoost, LightGBM)
        - Average predictions from multiple models

        ### 🏗️ Evaluation
        - Check misclassified samples to see confusing classes
        - Improve class labels if two diseases look visually similar
        """)

if __name__ == "__main__":
    main()