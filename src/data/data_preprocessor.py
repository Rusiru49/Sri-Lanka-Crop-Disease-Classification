"""Data preprocessing utilities."""

import numpy as np
import cv2
from typing import Tuple, Optional, Union
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
        X_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Standardize features using StandardScaler.
        
        Args:
            X_train: Training features
            X_val: Validation features (optional)
            X_test: Test features (optional)
            
        Returns:
            Tuple of standardized features or single array if only train provided
        """
        # Fit on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Handle different combinations of val/test
        if X_val is None and X_test is None:
            return X_train_scaled
        
        if X_val is not None and X_test is None:
            X_val_scaled = self.scaler.transform(X_val)
            return X_train_scaled, X_val_scaled
        
        if X_val is None and X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        # Both val and test provided
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def encode_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Encode string labels to integers.
        
        Args:
            labels: Label array
            
        Returns:
            Encoded labels
        """
        if labels.dtype in [np.int64, np.int32, np.int16, np.int8]:
            return labels
        
        return self.label_encoder.fit_transform(labels)
    
    def augment_brightness(self, image: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """
        Adjust image brightness.
        
        Args:
            image: Input image (float32, [0, 1])
            factor: Brightness factor
            
        Returns:
            Brightness adjusted image
        """
        # Convert to uint8 for OpenCV
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Convert to HSV
        hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        
        # Adjust brightness (V channel)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        
        # Convert back to RGB
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
            random_state=config.random_seed,
            n_jobs=-1
        )
        
        # Fit on features
        predictions = iso_forest.fit_predict(X)
        
        # Keep inliers (prediction == 1)
        mask = predictions == 1
        
        n_outliers = np.sum(~mask)
        logger.info(f"Removed {n_outliers} outliers from {len(X)} samples ({n_outliers/len(X)*100:.2f}%)")
        
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
        try:
            from imblearn.over_sampling import RandomOverSampler
            from imblearn.under_sampling import RandomUnderSampler
        except ImportError:
            logger.error("imbalanced-learn not installed. Install with: pip install imbalanced-learn")
            return X, y
        
        if method == 'oversample':
            sampler = RandomOverSampler(random_state=config.random_seed)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=config.random_seed)
        else:
            logger.warning(f"Unknown method '{method}'. Using 'oversample'")
            sampler = RandomOverSampler(random_state=config.random_seed)
        
        # Store original shape
        original_shape = X.shape
        
        # Reshape for resampling if needed
        if len(X.shape) > 2:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
        
        # Resample
        X_resampled, y_resampled = sampler.fit_resample(X_flat, y)
        
        # Reshape back to original dimensions
        if len(original_shape) > 2:
            X_resampled = X_resampled.reshape(-1, *original_shape[1:])
        
        logger.info(f"Balanced dataset: {len(X)} -> {len(X_resampled)} samples using {method}")
        
        return X_resampled, y_resampled