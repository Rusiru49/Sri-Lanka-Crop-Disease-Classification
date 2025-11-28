"""Data loading utilities for image datasets with pre-split structure."""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from tqdm import tqdm

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.helpers import get_class_names, set_seed

logger = get_logger()
config = get_config()


class ImageDataLoader:
    """Load and prepare image data for training."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize data loader.
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = data_dir or config.get('data.raw_dir')
        self.image_size = config.image_size
        self.random_seed = config.random_seed
        
        set_seed(self.random_seed)
        
        # Check if data is pre-split (train/val/test folders exist)
        self.is_pre_split = self._check_pre_split()
        
        if self.is_pre_split:
            logger.info("Detected pre-split dataset structure (train/val/test)")
            # Get class names from train folder
            train_dir = os.path.join(self.data_dir, 'train')
            self.class_names = get_class_names(train_dir)
        else:
            logger.info("Using single directory structure")
            self.class_names = get_class_names(self.data_dir)
        
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        logger.info(f"Found {len(self.class_names)} classes: {self.class_names[:5]}...")
    
    def _check_pre_split(self) -> bool:
        """
        Check if dataset is already split into train/val/test.
        
        Returns:
            True if pre-split, False otherwise
        """
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        test_dir = os.path.join(self.data_dir, 'test')
        
        return (os.path.exists(train_dir) and 
                os.path.exists(val_dir) and 
                os.path.exists(test_dir))
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, self.image_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def load_subset(self, subset: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load a specific subset (train/val/test).
        
        Args:
            subset: Subset name ('train', 'val', or 'test')
            
        Returns:
            Tuple of (images, labels, image_paths)
        """
        images = []
        labels = []
        image_paths = []
        
        if self.is_pre_split:
            subset_dir = os.path.join(self.data_dir, subset)
        else:
            subset_dir = self.data_dir
        
        if not os.path.exists(subset_dir):
            raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
        
        logger.info(f"Loading {subset} images from {subset_dir}")
        
        for class_name in self.class_names:
            class_dir = os.path.join(subset_dir, class_name)
            
            if not os.path.exists(class_dir):
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            logger.info(f"Loading {len(image_files)} images for class '{class_name}'")
            
            for img_file in tqdm(image_files, desc=f"Loading {class_name}"):
                img_path = os.path.join(class_dir, img_file)
                
                try:
                    image = self.load_image(img_path)
                    images.append(image)
                    labels.append(class_idx)
                    image_paths.append(img_path)
                except Exception as e:
                    logger.warning(f"Error loading {img_path}: {str(e)}")
        
        images = np.array(images)
        labels = np.array(labels)
        
        logger.info(f"Loaded {len(images)} images for {subset} set with shape {images.shape}")
        
        return images, labels, image_paths
    
    def load_all_splits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load all pre-split datasets (train/val/test).
        
        Returns:
            Dictionary with train, val, test splits
        """
        if not self.is_pre_split:
            raise ValueError("Dataset is not pre-split. Use load_dataset() and split_dataset() instead.")
        
        logger.info("\n" + "="*80)
        logger.info("Loading Pre-Split Dataset")
        logger.info("="*80)
        
        # Load train set
        logger.info("\n[1/3] Loading Training Set...")
        X_train, y_train, train_paths = self.load_subset('train')
        
        # Load validation set
        logger.info("\n[2/3] Loading Validation Set...")
        X_val, y_val, val_paths = self.load_subset('val')
        
        # Load test set
        logger.info("\n[3/3] Loading Test Set...")
        X_test, y_test, test_paths = self.load_subset('test')
        
        logger.info("\n" + "="*80)
        logger.info("Dataset Loading Summary")
        logger.info("="*80)
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"Total: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]} samples")
        logger.info("="*80 + "\n")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def load_dataset(self, subset: str = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load entire dataset from directory (legacy method for non-split datasets).
        
        Args:
            subset: Specific subset to load (train/val/test) or None for all
            
        Returns:
            Tuple of (images, labels, image_paths)
        """
        if self.is_pre_split and subset:
            return self.load_subset(subset)
        
        images = []
        labels = []
        image_paths = []
        
        data_path = self.data_dir
        if subset and not self.is_pre_split:
            data_path = os.path.join(data_path, subset)
        
        logger.info(f"Loading images from {data_path}")
        
        for class_name in self.class_names:
            class_dir = os.path.join(data_path, class_name)
            
            if not os.path.exists(class_dir):
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            logger.info(f"Loading {len(image_files)} images for class '{class_name}'")
            
            for img_file in tqdm(image_files, desc=f"Loading {class_name}"):
                img_path = os.path.join(class_dir, img_file)
                
                try:
                    image = self.load_image(img_path)
                    images.append(image)
                    labels.append(class_idx)
                    image_paths.append(img_path)
                except Exception as e:
                    logger.warning(f"Error loading {img_path}: {str(e)}")
        
        images = np.array(images)
        labels = np.array(labels)
        
        logger.info(f"Loaded {len(images)} images with shape {images.shape}")
        
        return images, labels, image_paths
    
    def get_class_distribution(self, labels: np.ndarray) -> Dict[str, int]:
        """
        Get class distribution from labels.
        
        Args:
            labels: Label array
            
        Returns:
            Dictionary with class counts
        """
        unique, counts = np.unique(labels, return_counts=True)
        distribution = {
            self.idx_to_class[idx]: count 
            for idx, count in zip(unique, counts)
        }
        return distribution
    
    def get_dataset_info(self) -> Dict[str, any]:
        """
        Get comprehensive dataset information.
        
        Returns:
            Dictionary with dataset statistics
        """
        info = {
            'data_dir': self.data_dir,
            'is_pre_split': self.is_pre_split,
            'n_classes': len(self.class_names),
            'class_names': self.class_names,
            'image_size': self.image_size
        }
        
        if self.is_pre_split:
            # Count images in each split
            for subset in ['train', 'val', 'test']:
                subset_dir = os.path.join(self.data_dir, subset)
                count = 0
                for class_name in self.class_names:
                    class_dir = os.path.join(subset_dir, class_name)
                    if os.path.exists(class_dir):
                        count += len([f for f in os.listdir(class_dir) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                info[f'{subset}_size'] = count
        
        return info