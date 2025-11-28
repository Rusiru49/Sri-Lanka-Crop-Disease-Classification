"""Data preprocessing utilities."""

import numpy as np
import cv2
from typing import Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder

from ..utils.config import get_config
from ..utils.logger import get_logger

logger = get_logger()
config = get_config()


class ImagePreprocessor:
    """Preprocess images for model training."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.image_size = config.image_size
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def normalize_images(self, images: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values.
        
        Args:
            images: Image array
            
        Returns:
            Normalized images
        """
        # Already normalized in data loader, but can apply additional normalization
        return images
    
    def flatten_images(self, images: np.ndarray) -> np.ndarray:
        """
        Flatten images for traditional ML models.
        
        Args:
            images: Image array with shape (n_samples, height, width, channels)
            
        Returns:
            Flattened images with shape (n_samples, height*width*channels)
        """
        n_samples = images.shape[0]
        return images.reshape(n_samples, -1)
    
    def standardize_features(
        self, 
        X_train: np.ndarray, 
        X_val: np.ndarray = None,
        X_test: np.ndarray = None
    ) -> Tuple[np.ndarray, ...]:
        """
        Standardize features using StandardScaler.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            
        Returns:
            Tuple of standardized features
        """
        # Fit on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        results = [X_train_scaled]
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            results.append(X_val_scaled)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            results.append(X_test_scaled)
        
        return tuple(results) if len(results) > 1 else results[0]
    
    def encode_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Encode string labels to integers.
        
        Args:
            labels: Label array
            
        Returns:
            Encoded labels
        """
        if labels.dtype == np.int64 or labels.dtype == np.int32:
            return labels
        
        return self.label_encoder.fit_transform(labels)
    
    def augment_brightness(self, image: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """
        Adjust image brightness.
        
        Args:
            image: Input image
            factor: Brightness factor
            
        Returns:
            Brightness adjusted image
        """
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb.astype(np.float32) / 255.0
    
    def remove_outliers(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        contamination: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers using Isolation Forest.
        
        Args:
            X: Feature array
            y: Label array
            contamination: Expected proportion of outliers
            
        Returns:
            Filtered X and y
        """
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=config.random_seed
        )
        
        # Fit on features
        predictions = iso_forest.fit_predict(X)
        
        # Keep inliers (prediction == 1)
        mask = predictions == 1
        
        logger.info(f"Removed {np.sum(~mask)} outliers from {len(X)} samples")
        
        return X[mask], y[mask]
    
    def balance_classes(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        method: str = 'oversample'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance classes using oversampling or undersampling.
        
        Args:
            X: Feature array
            y: Label array
            method: 'oversample' or 'undersample'
            
        Returns:
            Balanced X and y
        """
        from imblearn.over_sampling import RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler
        
        if method == 'oversample':
            sampler = RandomOverSampler(random_state=config.random_seed)
        else:
            sampler = RandomUnderSampler(random_state=config.random_seed)
        
        # Reshape for resampling
        original_shape = X.shape
        X_flat = X.reshape(X.shape[0], -1)
        
        X_resampled, y_resampled = sampler.fit_resample(X_flat, y)
        
        # Reshape back
        X_resampled = X_resampled.reshape(-1, *original_shape[1:])
        
        logger.info(f"Balanced dataset from {len(X)} to {len(X_resampled)} samples")
        
        return X_resampled, y_resampled