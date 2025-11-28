"""Visualization utilities for plotting results."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger()

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str = None,
    normalize: bool = False
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_feature_importance(
    feature_importance: Dict[str, float],
    top_n: int = 20,
    save_path: str = None
):
    """
    Plot feature importance.
    
    Args:
        feature_importance: Dictionary of feature importances
        top_n: Number of top features to plot
        save_path: Path to save figure
    """
    # Get top N features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    features, importances = zip(*top_features)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(features)), importances, color='steelblue')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def plot_class_distribution(
    labels: np.ndarray,
    class_names: List[str],
    title: str = "Class Distribution",
    save_path: str = None
):
    """
    Plot class distribution.
    
    Args:
        labels: Label array
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(unique)), counts, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
    
    plt.show()


def plot_roc_curves(
    roc_data: Dict[int, tuple],
    class_names: List[str],
    save_path: str = None
):
    """
    Plot ROC curves for all classes.
    
    Args:
        roc_data: Dictionary with fpr, tpr, auc for each class
        class_names: List of class names
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 8))
    
    for class_idx, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(
            fpr, tpr,
            label=f'{class_names[class_idx]} (AUC = {roc_auc:.2f})',
            linewidth=2
        )
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for All Classes', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves saved to {save_path}")
    
    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = None
):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    
    # Plot accuracy
    if 'train_accuracy' in history:
        axes[1].plot(history['train_accuracy'], label='Train Accuracy', linewidth=2)
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_per_class_metrics(
    metrics_df: pd.DataFrame,
    save_path: str = None
):
    """
    Plot per-class metrics.
    
    Args:
        metrics_df: DataFrame with per-class metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Precision
    axes[0].barh(metrics_df['class'], metrics_df['precision'], color='skyblue')
    axes[0].set_xlabel('Precision', fontsize=12)
    axes[0].set_title('Precision per Class', fontsize=14, fontweight='bold')
    axes[0].set_xlim([0, 1])
    
    # Recall
    axes[1].barh(metrics_df['class'], metrics_df['recall'], color='lightcoral')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_title('Recall per Class', fontsize=14, fontweight='bold')
    axes[1].set_xlim([0, 1])
    
    # F1 Score
    axes[2].barh(metrics_df['class'], metrics_df['f1_score'], color='lightgreen')
    axes[2].set_xlabel('F1 Score', fontsize=12)
    axes[2].set_title('F1 Score per Class', fontsize=14, fontweight='bold')
    axes[2].set_xlim([0, 1])
    
    for ax in axes:
        ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Per-class metrics plot saved to {save_path}")
    
    plt.show()


def create_interactive_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> go.Figure:
    """
    Create interactive confusion matrix using Plotly.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Plotly figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title='Interactive Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=800,
        height=700
    )
    
    return fig