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
    Falls back to default class names if directory doesn't exist.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        List of class names (sorted)
    """
    # Convert to Path object
    data_path = Path(data_dir)
    
    # Check if directory exists
    if not data_path.exists():
        print(f"Warning: Data directory not found: {data_dir}")
        print("Using default class names...")
        return get_default_class_names()
    
    try:
        # Get all subdirectories (each subdirectory is a class)
        class_names = []
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                class_names.append(item)
        
        # If no classes found, use defaults
        if not class_names:
            print(f"Warning: No class directories found in {data_dir}")
            print("Using default class names...")
            return get_default_class_names()
        
        return sorted(class_names)
    
    except Exception as e:
        print(f"Error loading class names from {data_dir}: {str(e)}")
        print("Using default class names...")
        return get_default_class_names()


def get_default_class_names() -> List[str]:
    """
    Return default class names for crop disease classification.
    Used when data directory is not available.
    
    Returns:
        List of default class names
    """
    return [
        "Corn_Common_rust",
        "Corn_Gray_leaf_spot",
        "Corn_Healthy",
        "Corn_Northern_Leaf_Blight",
        "Pepper_Bacterial_spot",
        "Pepper_Healthy",
        "Potato_Early_blight",
        "Potato_Healthy",
        "Potato_Late_blight",
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_Healthy",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites",
        "Tomato_Target_Spot",
        "Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato_mosaic_virus"
    ]


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


def validate_image_path(image_path: str) -> bool:
    """
    Validate if the image path exists and is a valid image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(image_path):
        return False
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    _, ext = os.path.splitext(image_path)
    
    return ext.lower() in valid_extensions


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path object pointing to project root
    """
    current_file = Path(__file__).resolve()
    # Assuming helpers.py is in src/utils/
    return current_file.parent.parent.parent


def ensure_data_directories():
    """
    Ensure all required data directories exist.
    Creates them if they don't exist.
    """
    project_root = get_project_root()
    
    required_dirs = [
        project_root / "data" / "raw" / "plantvillage" / "train",
        project_root / "data" / "raw" / "plantvillage" / "test",
        project_root / "data" / "processed",
        project_root / "models" / "saved_models",
        project_root / "logs",
        project_root / "temp"
    ]
    
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created/verified {len(required_dirs)} directories")


def print_class_distribution(class_names: List[str], counts: List[int]):
    """
    Print class distribution in a formatted way.
    
    Args:
        class_names: List of class names
        counts: List of counts for each class
    """
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION")
    print("="*60)
    
    total = sum(counts)
    
    for name, count in zip(class_names, counts):
        percentage = (count / total) * 100 if total > 0 else 0
        bar_length = int(percentage / 2)  # Scale to 50 chars max
        bar = "█" * bar_length
        print(f"{name:35s} | {count:6d} ({percentage:5.2f}%) {bar}")
    
    print("="*60)
    print(f"Total samples: {total}")
    print("="*60 + "\n")