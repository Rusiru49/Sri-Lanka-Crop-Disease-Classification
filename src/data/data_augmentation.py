"""Data augmentation utilities for images."""

import numpy as np
import cv2
from typing import Tuple
import albumentations as A

from ..utils.config import get_config
from ..utils.logger import get_logger

logger = get_logger()
config = get_config()


class ImageAugmentor:
    """Augment images for training."""
    
    def __init__(self):
        """Initialize augmentor with configuration."""
        aug_config = config.get_augmentation_config()
        self.enabled = aug_config.get('enabled', True)
        
        if self.enabled:
            self.transform = A.Compose([
                A.Rotate(
                    limit=aug_config.get('rotation_range', 20),
                    p=0.5
                ),
                A.HorizontalFlip(p=0.5 if aug_config.get('horizontal_flip') else 0),
                A.VerticalFlip(p=0.5 if aug_config.get('vertical_flip') else 0),
                A.ShiftScaleRotate(
                    shift_limit=aug_config.get('width_shift_range', 0.2),
                    scale_limit=aug_config.get('zoom_range', 0.15),
                    rotate_limit=0,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.1)
            ])
        else:
            self.transform = None
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Augment a single image.
        
        Args:
            image: Input image (float32, [0, 1])
            
        Returns:
            Augmented image
        """
        if not self.enabled or self.transform is None:
            return image
        
        # Convert to uint8 for albumentations
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Apply augmentation
        augmented = self.transform(image=image_uint8)
        augmented_image = augmented['image']
        
        # Convert back to float32
        return augmented_image.astype(np.float32) / 255.0
    
    def augment_batch(
        self, 
        images: np.ndarray, 
        labels: np.ndarray,
        augmentation_factor: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment a batch of images.
        
        Args:
            images: Image array
            labels: Label array
            augmentation_factor: Number of augmented versions per image
            
        Returns:
            Tuple of augmented images and labels
        """
        if not self.enabled:
            return images, labels
        
        augmented_images = [images]
        augmented_labels = [labels]
        
        for _ in range(augmentation_factor - 1):
            aug_imgs = np.array([self.augment_image(img) for img in images])
            augmented_images.append(aug_imgs)
            augmented_labels.append(labels.copy())
        
        # Concatenate all augmented data
        final_images = np.concatenate(augmented_images, axis=0)
        final_labels = np.concatenate(augmented_labels, axis=0)
        
        logger.info(f"Augmented dataset from {len(images)} to {len(final_images)} samples")
        
        return final_images, final_labels
    
    def augment_minority_classes(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        target_samples_per_class: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment minority classes to balance dataset.
        
        Args:
            images: Image array
            labels: Label array
            target_samples_per_class: Target number of samples per class
            
        Returns:
            Balanced dataset with augmented images
        """
        if not self.enabled:
            return images, labels
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        if target_samples_per_class is None:
            target_samples_per_class = max(counts)
        
        augmented_images = list(images)
        augmented_labels = list(labels)
        
        for label, count in zip(unique_labels, counts):
            if count < target_samples_per_class:
                # Get images for this class
                class_indices = np.where(labels == label)[0]
                class_images = images[class_indices]
                
                # Calculate how many augmented samples needed
                needed_samples = target_samples_per_class - count
                
                # Randomly sample and augment
                for _ in range(needed_samples):
                    idx = np.random.choice(len(class_images))
                    aug_img = self.augment_image(class_images[idx])
                    augmented_images.append(aug_img)
                    augmented_labels.append(label)
                
                logger.info(f"Augmented class {label}: {count} -> {target_samples_per_class}")
        
        return np.array(augmented_images), np.array(augmented_labels)
    
    def create_synthetic_samples(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        n_synthetic: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic samples using mixup technique.
        
        Args:
            images: Image array
            labels: Label array
            n_synthetic: Number of synthetic samples to create
            
        Returns:
            Original and synthetic samples
        """
        synthetic_images = []
        synthetic_labels = []
        
        for _ in range(n_synthetic):
            # Randomly select two samples
            idx1, idx2 = np.random.choice(len(images), 2, replace=False)
            
            # Random mixing ratio
            alpha = np.random.beta(0.2, 0.2)
            
            # Mix images
            mixed_image = alpha * images[idx1] + (1 - alpha) * images[idx2]
            
            # Use label from dominant image
            mixed_label = labels[idx1] if alpha > 0.5 else labels[idx2]
            
            synthetic_images.append(mixed_image)
            synthetic_labels.append(mixed_label)
        
        # Combine with original data
        all_images = np.concatenate([images, np.array(synthetic_images)], axis=0)
        all_labels = np.concatenate([labels, np.array(synthetic_labels)], axis=0)
        
        logger.info(f"Created {n_synthetic} synthetic samples")
        
        return all_images, all_labels