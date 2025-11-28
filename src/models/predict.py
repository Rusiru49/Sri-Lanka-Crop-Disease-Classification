"""Prediction utilities for making inference with trained models."""

import numpy as np
from typing import Dict, List, Tuple, Any
import cv2

from ..utils.logger import get_logger
from ..utils.helpers import load_model
from ..data.data_loader import ImageDataLoader
from ..features.feature_engineering import FeatureExtractor

logger = get_logger()


class DiseasePredictor:
    """Make predictions on crop disease images."""
    
    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        feature_extractor: FeatureExtractor = None
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model
            class_names: List of class names
            feature_extractor: Feature extractor instance
        """
        self.model = load_model(model_path)
        self.class_names = class_names
        self.feature_extractor = feature_extractor or FeatureExtractor()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Number of classes: {len(class_names)}")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for prediction.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image
        """
        # Load image
        loader = ImageDataLoader()
        image = loader.load_image(image_path)
        
        return image
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from images.
        
        Args:
            images: Image array
            
        Returns:
            Feature array
        """
        # Ensure images is 4D
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        
        features = self.feature_extractor.extract_all_features(images)
        
        return features
    
    def predict_single(self, image_path: str) -> Dict[str, Any]:
        """
        Predict disease for a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        image = self.preprocess_image(image_path)
        image = np.expand_dims(image, axis=0)
        
        # Extract features
        features = self.extract_features(image)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        prediction_proba = self.model.predict_proba(features)[0]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(prediction_proba)[-3:][::-1]
        top_3_predictions = [
            {
                'class': self.class_names[idx],
                'confidence': float(prediction_proba[idx])
            }
            for idx in top_3_idx
        ]
        
        result = {
            'predicted_class': self.class_names[prediction],
            'confidence': float(prediction_proba[prediction]),
            'top_3_predictions': top_3_predictions,
            'all_probabilities': {
                self.class_names[i]: float(prob)
                for i, prob in enumerate(prediction_proba)
            }
        }
        
        logger.info(f"Prediction: {result['predicted_class']} "
                   f"(confidence: {result['confidence']:.2%})")
        
        return result
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Predict disease for multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_with_explanation(self, image_path: str) -> Dict[str, Any]:
        """
        Predict with model explanation (feature importance).
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with prediction and explanation
        """
        # Get basic prediction
        result = self.predict_single(image_path)
        
        # Add feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            top_features = np.argsort(self.model.feature_importances_)[-10:][::-1]
            result['important_features'] = [
                {
                    'feature_index': int(idx),
                    'importance': float(self.model.feature_importances_[idx])
                }
                for idx in top_features
            ]
        
        return result
    
    def get_disease_recommendations(self, disease_name: str) -> Dict[str, Any]:
        """
        Get treatment recommendations for a disease.
        
        Args:
            disease_name: Name of the disease
            
        Returns:
            Dictionary with recommendations
        """
        # Disease recommendations database (can be expanded)
        recommendations_db = {
            'early_blight': {
                'severity': 'Medium',
                'description': 'Fungal disease causing dark spots on leaves',
                'treatment': [
                    'Remove and destroy infected leaves',
                    'Apply fungicide (e.g., chlorothalonil)',
                    'Improve air circulation',
                    'Avoid overhead watering'
                ],
                'prevention': [
                    'Crop rotation',
                    'Use disease-resistant varieties',
                    'Mulch around plants',
                    'Maintain proper spacing'
                ]
            },
            'late_blight': {
                'severity': 'High',
                'description': 'Destructive fungal disease affecting leaves and fruits',
                'treatment': [
                    'Apply copper-based fungicides',
                    'Remove infected plants immediately',
                    'Improve drainage',
                    'Reduce humidity'
                ],
                'prevention': [
                    'Use certified disease-free seeds',
                    'Avoid overhead irrigation',
                    'Plant in well-drained soil',
                    'Monitor weather conditions'
                ]
            },
            'healthy': {
                'severity': 'None',
                'description': 'Plant appears healthy with no visible disease',
                'treatment': ['No treatment needed'],
                'prevention': [
                    'Continue regular monitoring',
                    'Maintain good cultural practices',
                    'Ensure proper nutrition',
                    'Water appropriately'
                ]
            }
        }
        
        # Normalize disease name
        disease_key = disease_name.lower().replace(' ', '_')
        
        if disease_key in recommendations_db:
            return recommendations_db[disease_key]
        else:
            return {
                'severity': 'Unknown',
                'description': 'Disease information not available',
                'treatment': ['Consult with agricultural extension officer'],
                'prevention': ['Follow general crop management practices']
            }
    
    def predict_with_recommendations(self, image_path: str) -> Dict[str, Any]:
        """
        Predict disease and provide recommendations.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Complete prediction with recommendations
        """
        # Get prediction
        prediction = self.predict_single(image_path)
        
        # Get recommendations
        disease_name = prediction['predicted_class']
        recommendations = self.get_disease_recommendations(disease_name)
        
        # Combine results
        result = {
            **prediction,
            'recommendations': recommendations
        }
        
        return result