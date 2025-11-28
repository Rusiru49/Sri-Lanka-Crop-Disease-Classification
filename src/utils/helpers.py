"""Helper utilities for the project."""

import os
import random
import numpy as np
import joblib
from pathlib import Path
from typing import Any, List, Dict
import json


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def create_directories(directories: List[str]):
    """
    Create multiple directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def save_model(model: Any, filepath: str):
    """
    Save model to disk.
    
    Args:
        model: Model object to save
        filepath: Path to save the model
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """
    Load model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    return joblib.load(filepath)


def save_json(data: Dict, filepath: str):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save the JSON file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> Dict:
    """
    Load JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def get_class_names(data_dir: str) -> List[str]:
    """
    Get list of class names from directory structure.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        List of class names
    """
    class_names = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            class_names.append(item)
    
    return sorted(class_names)


def calculate_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y_train: Training labels
        
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    
    return dict(zip(classes, weights))


def format_bytes(size: int) -> str:
    """
    Format bytes to human readable format.
    
    Args:
        size: Size in bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"