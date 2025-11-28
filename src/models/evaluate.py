"""Model evaluation utilities."""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import pandas as pd

from ..utils.logger import get_logger
from ..utils.helpers import save_json

logger = get_logger()


class ModelEvaluator:
    """Evaluate trained models."""
    
    def __init__(self, model, class_names: list = None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names or []
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test set")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Log metrics
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision (macro): {metrics['precision_macro']:.4f}")
        logger.info(f"Recall (macro): {metrics['recall_macro']:.4f}")
        logger.info(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
        
        return metrics
    
    def get_confusion_matrix(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> np.ndarray:
        """
        Get confusion matrix.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Confusion matrix
        """
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        return cm
    
    def get_classification_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> str:
        """
        Get detailed classification report.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Classification report string
        """
        y_pred = self.model.predict(X_test)
        
        target_names = self.class_names if self.class_names else None
        
        report = classification_report(
            y_test, y_pred,
            target_names=target_names,
            zero_division=0
        )
        
        return report
    
    def get_per_class_metrics(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Get metrics for each class.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with per-class metrics
        """
        y_pred = self.model.predict(X_test)
        
        # Calculate per-class metrics
        precision = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        # Get support (number of samples per class)
        unique, counts = np.unique(y_test, return_counts=True)
        support = dict(zip(unique, counts))
        
        # Create DataFrame
        data = []
        for idx, class_name in enumerate(self.class_names):
            data.append({
                'class': class_name,
                'precision': precision[idx],
                'recall': recall[idx],
                'f1_score': f1[idx],
                'support': support.get(idx, 0)
            })
        
        df = pd.DataFrame(data)
        
        return df
    
    def get_roc_curves(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, float]]:
        """
        Get ROC curves for each class.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with fpr, tpr, and auc for each class
        """
        y_pred_proba = self.model.predict_proba(X_test)
        
        roc_data = {}
        n_classes = len(self.class_names)
        
        for i in range(n_classes):
            # Binarize the output
            y_true_binary = (y_test == i).astype(int)
            y_score = y_pred_proba[:, i]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = roc_auc_score(y_true_binary, y_score)
            
            roc_data[i] = (fpr, tpr, roc_auc)
        
        return roc_data
    
    def get_top_k_accuracy(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        k: int = 5
    ) -> float:
        """
        Calculate top-k accuracy.
        
        Args:
            X_test: Test features
            y_test: Test labels
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy
        """
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Get top k predictions
        top_k_pred = np.argsort(y_pred_proba, axis=1)[:, -k:]
        
        # Check if true label is in top k
        correct = np.array([y_test[i] in top_k_pred[i] for i in range(len(y_test))])
        
        top_k_acc = np.mean(correct)
        
        logger.info(f"Top-{k} accuracy: {top_k_acc:.4f}")
        
        return top_k_acc
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """
        Save evaluation results to file.
        
        Args:
            results: Dictionary of results
            filepath: Path to save results
        """
        save_json(results, filepath)
        logger.info(f"Results saved to {filepath}")
    
    def get_misclassified_samples(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get misclassified samples for analysis.
        
        Args:
            X_test: Test features
            y_test: Test labels
            limit: Maximum number of samples to return
            
        Returns:
            Dictionary with misclassified sample information
        """
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Find misclassified samples
        misclassified_idx = np.where(y_test != y_pred)[0]
        
        if len(misclassified_idx) == 0:
            logger.info("No misclassified samples found!")
            return {}
        
        # Limit number of samples
        if len(misclassified_idx) > limit:
            misclassified_idx = misclassified_idx[:limit]
        
        misclassified_data = []
        
        for idx in misclassified_idx:
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            confidence = y_pred_proba[idx][pred_label]
            
            true_class = self.class_names[true_label] if self.class_names else str(true_label)
            pred_class = self.class_names[pred_label] if self.class_names else str(pred_label)
            
            misclassified_data.append({
                'index': int(idx),
                'true_class': true_class,
                'predicted_class': pred_class,
                'confidence': float(confidence)
            })
        
        logger.info(f"Found {len(misclassified_idx)} misclassified samples")
        
        return {
            'total_misclassified': len(misclassified_idx),
            'samples': misclassified_data
        }