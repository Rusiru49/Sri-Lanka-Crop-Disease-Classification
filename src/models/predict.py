"""Prediction utilities for making inference with trained models."""

import numpy as np
from typing import Dict, List, Tuple, Any
import cv2
import os
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.helpers import load_model, get_default_class_names
from ..data.data_loader import ImageDataLoader
from ..features.feature_engineering import FeatureExtractor

logger = get_logger()


class DiseasePredictor:
    """Make predictions on crop disease images."""
    
    def __init__(
        self,
        model_path: str,
        class_names: List[str] = None,
        feature_extractor: FeatureExtractor = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model
            class_names: List of class names (optional, uses defaults if None)
            feature_extractor: Feature extractor instance
            image_size: Target image size (width, height)
        """
        # Load model
        try:
            self.model = load_model(model_path)
            logger.info(f"✓ Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
        
        # Set class names
        if class_names is None:
            logger.info("No class names provided, using defaults")
            self.class_names = get_default_class_names()
        else:
            self.class_names = class_names
        
        logger.info(f"✓ Number of classes: {len(self.class_names)}")
        
        # Initialize feature extractor
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.image_size = image_size
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for prediction.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image
        """
        # Validate image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Read image directly instead of using ImageDataLoader to avoid circular dependency
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
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from images.
        
        Args:
            images: Image array
            
        Returns:
            Feature array
        """
        try:
            # Ensure images is 4D
            if len(images.shape) == 3:
                images = np.expand_dims(images, axis=0)
            
            features = self.feature_extractor.extract_all_features(images)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise
    
    def predict_single(self, image_path: str) -> Dict[str, Any]:
        """
        Predict disease for a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess image
            image = self.preprocess_image(image_path)
            image = np.expand_dims(image, axis=0)
            
            # Extract features
            features = self.extract_features(image)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            prediction_proba = self.model.predict_proba(features)[0]
            
            # Validate prediction index
            if prediction >= len(self.class_names):
                logger.error(f"Prediction index {prediction} out of range for {len(self.class_names)} classes")
                prediction = 0  # Default to first class
            
            # Get top 3 predictions
            top_3_idx = np.argsort(prediction_proba)[-3:][::-1]
            top_3_predictions = [
                {
                    'class': self.class_names[idx] if idx < len(self.class_names) else f"Class_{idx}",
                    'confidence': float(prediction_proba[idx])
                }
                for idx in top_3_idx
            ]
            
            result = {
                'predicted_class': self.class_names[prediction],
                'confidence': float(prediction_proba[prediction]),
                'top_3_predictions': top_3_predictions,
                'all_probabilities': {
                    self.class_names[i] if i < len(self.class_names) else f"Class_{i}": float(prob)
                    for i, prob in enumerate(prediction_proba)
                }
            }
            
            logger.info(f"Prediction: {result['predicted_class']} "
                       f"(confidence: {result['confidence']:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting {image_path}: {str(e)}")
            raise
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Predict disease for multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        logger.info(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                logger.info(f"Processing image {i}/{len(image_paths)}: {Path(image_path).name}")
                result = self.predict_single(image_path)
                result['image_path'] = image_path
                result['status'] = 'success'
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'status': 'error',
                    'error': str(e)
                })
        
        success_count = sum(1 for r in results if r.get('status') == 'success')
        logger.info(f"Completed: {success_count}/{len(image_paths)} successful predictions")
        
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
            try:
                top_features = np.argsort(self.model.feature_importances_)[-10:][::-1]
                result['important_features'] = [
                    {
                        'feature_index': int(idx),
                        'importance': float(self.model.feature_importances_[idx])
                    }
                    for idx in top_features
                ]
                logger.info("Added feature importance to prediction")
            except Exception as e:
                logger.warning(f"Could not extract feature importance: {str(e)}")
        
        return result
    
    def get_disease_recommendations(self, disease_name: str) -> Dict[str, Any]:
        """
        Get treatment recommendations for a disease.
        
        Args:
            disease_name: Name of the disease
            
        Returns:
            Dictionary with recommendations
        """
        # Expanded disease recommendations database
        recommendations_db = {
            # Tomato diseases
            'tomato_early_blight': {
                'severity': 'Medium',
                'description': 'Fungal disease causing dark spots with concentric rings on lower leaves',
                'treatment': [
                    'Remove and destroy infected leaves',
                    'Apply fungicide (chlorothalonil or mancozeb)',
                    'Improve air circulation around plants',
                    'Avoid overhead watering',
                    'Apply copper-based sprays'
                ],
                'prevention': [
                    'Practice crop rotation (3-4 year cycle)',
                    'Use disease-resistant varieties',
                    'Mulch around plants to prevent soil splash',
                    'Maintain proper plant spacing',
                    'Water at the base of plants'
                ]
            },
            'tomato_late_blight': {
                'severity': 'High',
                'description': 'Destructive fungal disease causing water-soaked lesions on leaves and fruits',
                'treatment': [
                    'Apply copper-based fungicides immediately',
                    'Remove and destroy infected plants',
                    'Improve drainage and reduce humidity',
                    'Apply mancozeb or chlorothalonil preventively'
                ],
                'prevention': [
                    'Use certified disease-free seeds and transplants',
                    'Avoid overhead irrigation',
                    'Plant in well-drained soil',
                    'Monitor weather for favorable disease conditions',
                    'Remove volunteer tomato and potato plants'
                ]
            },
            'tomato_leaf_mold': {
                'severity': 'Medium',
                'description': 'Fungal disease causing yellow spots on upper leaf surface',
                'treatment': [
                    'Improve greenhouse ventilation',
                    'Reduce humidity below 85%',
                    'Apply fungicides (chlorothalonil)',
                    'Remove infected leaves'
                ],
                'prevention': [
                    'Use resistant varieties',
                    'Maintain good air circulation',
                    'Avoid overhead watering',
                    'Space plants properly'
                ]
            },
            'tomato_septoria_leaf_spot': {
                'severity': 'Medium',
                'description': 'Fungal disease with small circular spots with gray centers',
                'treatment': [
                    'Remove infected leaves',
                    'Apply fungicides (copper-based or chlorothalonil)',
                    'Improve air circulation',
                    'Mulch to prevent soil splash'
                ],
                'prevention': [
                    'Crop rotation',
                    'Use disease-free transplants',
                    'Avoid working with wet plants',
                    'Remove plant debris at end of season'
                ]
            },
            'tomato_spider_mites': {
                'severity': 'Medium',
                'description': 'Tiny pests causing stippling and webbing on leaves',
                'treatment': [
                    'Spray with water to dislodge mites',
                    'Apply insecticidal soap or neem oil',
                    'Use miticides if infestation is severe',
                    'Increase humidity around plants'
                ],
                'prevention': [
                    'Monitor plants regularly',
                    'Avoid water stress',
                    'Encourage beneficial insects',
                    'Keep plants well-watered'
                ]
            },
            'tomato_target_spot': {
                'severity': 'Medium',
                'description': 'Fungal disease with concentric ring patterns on leaves',
                'treatment': [
                    'Apply fungicides (azoxystrobin or chlorothalonil)',
                    'Remove infected plant parts',
                    'Improve air circulation',
                    'Reduce leaf wetness'
                ],
                'prevention': [
                    'Use resistant varieties',
                    'Practice crop rotation',
                    'Avoid overhead irrigation',
                    'Maintain plant spacing'
                ]
            },
            'tomato_yellow_leaf_curl_virus': {
                'severity': 'High',
                'description': 'Viral disease causing leaf curling and yellowing',
                'treatment': [
                    'Remove and destroy infected plants',
                    'Control whitefly vectors',
                    'Use insecticides for whitefly control',
                    'No cure available - focus on prevention'
                ],
                'prevention': [
                    'Use virus-free transplants',
                    'Plant resistant varieties',
                    'Use reflective mulches',
                    'Control whitefly populations',
                    'Use physical barriers (row covers)'
                ]
            },
            'tomato_mosaic_virus': {
                'severity': 'High',
                'description': 'Viral disease causing mottled leaves and stunted growth',
                'treatment': [
                    'Remove and destroy infected plants',
                    'Disinfect tools between plants',
                    'No chemical treatment available',
                    'Focus on preventing spread'
                ],
                'prevention': [
                    'Use virus-free seeds',
                    'Avoid tobacco use around plants',
                    'Wash hands before handling plants',
                    'Control aphid vectors',
                    'Remove infected plants immediately'
                ]
            },
            'tomato_bacterial_spot': {
                'severity': 'Medium',
                'description': 'Bacterial disease causing dark spots on leaves and fruits',
                'treatment': [
                    'Apply copper-based bactericides',
                    'Remove infected plant material',
                    'Improve air circulation',
                    'Avoid overhead watering'
                ],
                'prevention': [
                    'Use disease-free seeds and transplants',
                    'Practice crop rotation',
                    'Avoid working with wet plants',
                    'Maintain proper spacing'
                ]
            },
            'tomato_healthy': {
                'severity': 'None',
                'description': 'Plant appears healthy with no visible disease symptoms',
                'treatment': ['No treatment needed - continue monitoring'],
                'prevention': [
                    'Continue regular monitoring',
                    'Maintain good cultural practices',
                    'Ensure proper nutrition and watering',
                    'Practice integrated pest management'
                ]
            },
            
            # Potato diseases
            'potato_early_blight': {
                'severity': 'Medium',
                'description': 'Fungal disease with concentric rings on leaves',
                'treatment': [
                    'Apply fungicides (chlorothalonil or mancozeb)',
                    'Remove infected foliage',
                    'Improve plant nutrition',
                    'Ensure adequate spacing'
                ],
                'prevention': [
                    'Use certified seed potatoes',
                    'Practice 3-4 year crop rotation',
                    'Maintain adequate potassium levels',
                    'Hill soil around plants',
                    'Remove volunteer plants'
                ]
            },
            'potato_late_blight': {
                'severity': 'High',
                'description': 'Devastating disease that caused the Irish Potato Famine',
                'treatment': [
                    'Apply fungicides immediately (mancozeb, chlorothalonil)',
                    'Destroy infected plants',
                    'Harvest immediately if tubers are unaffected',
                    'Improve drainage'
                ],
                'prevention': [
                    'Use certified disease-free seed potatoes',
                    'Monitor weather conditions',
                    'Apply preventive fungicides',
                    'Destroy cull piles',
                    'Hill plants properly'
                ]
            },
            'potato_healthy': {
                'severity': 'None',
                'description': 'Healthy potato plant with no disease symptoms',
                'treatment': ['No treatment needed'],
                'prevention': [
                    'Continue regular monitoring',
                    'Maintain proper fertility',
                    'Ensure adequate irrigation',
                    'Scout for pests regularly'
                ]
            },
            
            # Pepper diseases
            'pepper_bacterial_spot': {
                'severity': 'Medium',
                'description': 'Bacterial disease causing leaf spots and fruit lesions',
                'treatment': [
                    'Apply copper-based bactericides',
                    'Remove severely infected plants',
                    'Improve air circulation',
                    'Reduce leaf wetness'
                ],
                'prevention': [
                    'Use disease-free seeds and transplants',
                    'Practice crop rotation',
                    'Avoid overhead irrigation',
                    'Disinfect tools regularly',
                    'Remove plant debris'
                ]
            },
            'pepper_healthy': {
                'severity': 'None',
                'description': 'Healthy pepper plant with vigorous growth',
                'treatment': ['No treatment needed'],
                'prevention': [
                    'Continue monitoring',
                    'Maintain balanced nutrition',
                    'Water consistently',
                    'Ensure good drainage'
                ]
            },
            
            # Corn diseases
            'corn_common_rust': {
                'severity': 'Medium',
                'description': 'Fungal disease with rust-colored pustules on leaves',
                'treatment': [
                    'Apply fungicides if severe',
                    'Usually not economical to treat',
                    'Monitor disease progression',
                    'Ensure adequate plant nutrition'
                ],
                'prevention': [
                    'Plant resistant hybrids',
                    'Practice crop rotation',
                    'Destroy crop residue',
                    'Plant at recommended times'
                ]
            },
            'corn_gray_leaf_spot': {
                'severity': 'Medium',
                'description': 'Fungal disease with rectangular gray lesions',
                'treatment': [
                    'Apply foliar fungicides',
                    'Improve air circulation',
                    'Time applications with disease forecasts',
                    'Consider fungicide resistance'
                ],
                'prevention': [
                    'Use resistant hybrids',
                    'Practice crop rotation',
                    'Till under crop residue',
                    'Avoid continuous corn planting'
                ]
            },
            'corn_northern_leaf_blight': {
                'severity': 'Medium',
                'description': 'Fungal disease with long gray-green lesions',
                'treatment': [
                    'Apply fungicides at early symptoms',
                    'Monitor disease levels',
                    'Consider economic thresholds',
                    'Rotate fungicide classes'
                ],
                'prevention': [
                    'Plant resistant hybrids',
                    'Practice crop rotation',
                    'Manage crop residue',
                    'Scout fields regularly'
                ]
            },
            'corn_healthy': {
                'severity': 'None',
                'description': 'Healthy corn plant with normal development',
                'treatment': ['No treatment needed'],
                'prevention': [
                    'Continue monitoring',
                    'Maintain fertility program',
                    'Control weeds and pests',
                    'Ensure adequate moisture'
                ]
            },
            
            # Generic healthy
            'healthy': {
                'severity': 'None',
                'description': 'Plant appears healthy with no visible disease',
                'treatment': ['No treatment needed'],
                'prevention': [
                    'Continue regular monitoring',
                    'Maintain good cultural practices',
                    'Ensure proper nutrition',
                    'Water appropriately',
                    'Practice integrated pest management'
                ]
            }
        }
        
        # Normalize disease name
        disease_key = disease_name.lower().replace(' ', '_')
        
        # Try exact match first
        if disease_key in recommendations_db:
            return recommendations_db[disease_key]
        
        # Try partial match
        for key in recommendations_db.keys():
            if disease_key in key or key in disease_key:
                return recommendations_db[key]
        
        # Return default if no match
        logger.warning(f"No recommendations found for disease: {disease_name}")
        return {
            'severity': 'Unknown',
            'description': f'Detailed information for {disease_name} is not currently available',
            'treatment': [
                'Consult with local agricultural extension service',
                'Take clear photos of symptoms',
                'Consider laboratory diagnosis if needed',
                'Monitor disease progression'
            ],
            'prevention': [
                'Follow general crop management practices',
                'Practice crop rotation',
                'Maintain plant health through proper nutrition',
                'Scout fields regularly for early detection'
            ]
        }
    
    def predict_with_recommendations(self, image_path: str) -> Dict[str, Any]:
        """
        Predict disease and provide recommendations.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Complete prediction with recommendations
        """
        try:
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
            
            logger.info(f"Generated recommendations for {disease_name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in predict_with_recommendations: {str(e)}")
            raise