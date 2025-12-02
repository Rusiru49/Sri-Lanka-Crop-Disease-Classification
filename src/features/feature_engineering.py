import numpy as np
import cv2
from typing import List, Tuple
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.spatial import distance
from scipy.stats import skew, kurtosis
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

from ..utils.config import get_config
from ..utils.logger import get_logger

logger = get_logger()
config = get_config()


class FeatureExtractor:
    """Extract deep + traditional + plant-specific features from images."""

    def __init__(self):
        self.image_size = config.image_size
        self.feature_config = config.get('features', {})
        self.cnn_model = None
        self.preprocess_fn = None

        if self.feature_config.get('use_deep_features', True):
            self._initialize_cnn_model()

    # -------------------------------------------------------------
    # Initialize CNN
    # -------------------------------------------------------------
    def _initialize_cnn_model(self):
        model_name = config.get('model.feature_extraction.pretrained_model', 'MobileNetV2')
        logger.info(f"Initializing {model_name} for feature extraction")

        if model_name == 'MobileNetV2':
            base_model = MobileNetV2(weights='imagenet', include_top=False,
                                     input_shape=(*self.image_size, 3))
            self.preprocess_fn = mobilenet_preprocess

        elif model_name == 'ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False,
                                  input_shape=(*self.image_size, 3))
            self.preprocess_fn = resnet_preprocess

        elif model_name == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False,
                               input_shape=(*self.image_size, 3))
            self.preprocess_fn = vgg_preprocess

        else:
            raise ValueError(f"Unknown model: {model_name}")

        base_model.trainable = False

        x = GlobalAveragePooling2D()(base_model.output)
        self.cnn_model = Model(inputs=base_model.input, outputs=x)

        logger.info(f"CNN model initialized with output shape: {self.cnn_model.output_shape}")

    # -------------------------------------------------------------
    # Extract deep CNN features
    # -------------------------------------------------------------
    def extract_deep_features(self, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
        if self.cnn_model is None:
            raise ValueError("CNN model not initialized")

        if self.preprocess_fn is None:
            raise ValueError("Preprocessing function missing")

        n_samples = len(images)
        features_list = []

        logger.info(f"Extracting deep features (batch size={batch_size})")

        for i in range(0, n_samples, batch_size):
            batch = images[i:i + batch_size]
            batch_uint8 = (batch * 255).astype(np.uint8)
            batch_preprocessed = self.preprocess_fn(batch_uint8)
            batch_features = self.cnn_model.predict(batch_preprocessed, verbose=0)
            features_list.append(batch_features)

        features = np.vstack(features_list)
        logger.info(f"Deep features shape: {features.shape}")

        return features

    # -------------------------------------------------------------
    # Color features (Enhanced)
    # -------------------------------------------------------------
    def extract_color_features(self, image: np.ndarray) -> np.ndarray:
        features = []
        img_uint8 = (image * 255).astype(np.uint8)

        # RGB histograms
        for c in range(3):
            hist = cv2.calcHist([img_uint8], [c], None, [32], [0, 256])
            features.extend(hist.flatten())

        # RGB mean/std
        features.extend([image[:, :, i].mean() for i in range(3)])
        features.extend([image[:, :, i].std() for i in range(3)])

        # HSV mean/std
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        features.extend([hsv[:, :, i].mean() for i in range(3)])
        features.extend([hsv[:, :, i].std() for i in range(3)])

        # NEW: Color moments (skewness & kurtosis) - captures color distribution
        for c in range(3):
            channel = image[:, :, c].flatten()
            features.append(skew(channel))
            features.append(kurtosis(channel))

        # NEW: Green channel dominance (healthy leaf indicator)
        green_dominance = image[:, :, 1].mean() / (image.mean() + 1e-6)
        features.append(green_dominance)

        return np.array(features)

    # -------------------------------------------------------------
    # Texture features (Enhanced GLCM + LBP)
    # -------------------------------------------------------------
    def extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        features = []
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # GLCM with more angles
        glcm = graycomatrix(
            gray,
            distances=[1, 2],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )

        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
            features.extend(graycoprops(glcm, prop).flatten())

        # LBP
        radius = 3
        n_points = radius * 8
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2))
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
        features.extend(lbp_hist)

        # NEW: Multi-scale LBP (captures texture at different scales)
        for r in [1, 2, 5]:
            n_p = r * 8
            lbp_ms = local_binary_pattern(gray, n_p, r, method='uniform')
            lbp_ms_hist, _ = np.histogram(lbp_ms, bins=n_p + 2, range=(0, n_p + 2))
            lbp_ms_hist = lbp_ms_hist / (lbp_ms_hist.sum() + 1e-6)
            # Use only first 10 bins to save dimensions
            features.extend(lbp_ms_hist[:10])

        return np.array(features)

    # -------------------------------------------------------------
    # Shape features (Enhanced)
    # -------------------------------------------------------------
    def extract_shape_features(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Default values
        area = perimeter = aspect_ratio = extent = solidity = 0
        compactness = circularity = rectangularity = 0

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)

            x, y, w, h = cv2.boundingRect(largest)
            if h > 0 and w > 0:
                aspect_ratio = w / h
                extent = area / (w * h)
                rectangularity = area / (w * h)

            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area

            # NEW: Compactness (4π × area / perimeter²)
            if perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter ** 2)
            
            # NEW: Circularity
            if perimeter > 0:
                circularity = (perimeter ** 2) / (4 * np.pi * area + 1e-6)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)

        return np.array([
            area, perimeter, aspect_ratio, extent, solidity, 
            edge_density, compactness, circularity, rectangularity
        ])

    # -------------------------------------------------------------
    # NEW: Plant-Specific Features (KEY FOR TOMATO vs POTATO)
    # -------------------------------------------------------------
    def extract_plant_specific_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features that distinguish plant species
        Tomato leaves: compound, serrated edges, darker green
        Potato leaves: simpler, smoother edges, lighter green
        """
        features = []
        img_uint8 = (image * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        # 1. Leaf Edge Complexity (serration detection)
        edges = cv2.Canny(gray, 50, 150)
        edge_complexity = self._calculate_edge_complexity(edges)
        features.append(edge_complexity)
        
        # 2. Leaf Shape Descriptors
        contours, _ = cv2.findContours(
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            
            # Convexity defects (indicates compound leaves)
            hull = cv2.convexHull(largest, returnPoints=False)
            if len(largest) > 3 and len(hull) > 0:
                try:
                    defects = cv2.convexityDefects(largest, hull)
                    if defects is not None:
                        # Count significant defects
                        significant_defects = np.sum(defects[:, 0, 3] > 1000)
                        features.append(significant_defects)
                        
                        # Average defect depth
                        avg_defect_depth = np.mean(defects[:, 0, 3]) / 256.0
                        features.append(avg_defect_depth)
                    else:
                        features.extend([0, 0])
                except:
                    features.extend([0, 0])
            else:
                features.extend([0, 0])
            
            # Moments-based shape descriptors
            moments = cv2.moments(largest)
            if moments['m00'] != 0:
                hu_moments = cv2.HuMoments(moments).flatten()
                # Use log transform to normalize
                hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
                features.extend(hu_moments[:4])  # First 4 Hu moments
            else:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0])
        
        # 3. Color-based plant identification
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        
        # Green intensity and saturation (tomato = darker, more saturated)
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        green_ratio = np.sum(green_mask > 0) / green_mask.size
        features.append(green_ratio)
        
        # Mean hue and saturation of green regions
        if green_ratio > 0:
            green_pixels = hsv[green_mask > 0]
            mean_hue = np.mean(green_pixels[:, 0])
            mean_sat = np.mean(green_pixels[:, 1])
            mean_val = np.mean(green_pixels[:, 2])
        else:
            mean_hue = mean_sat = mean_val = 0
        
        features.extend([mean_hue, mean_sat, mean_val])
        
        # 4. Texture directionality (leaf vein patterns)
        # Tomatoes have more prominent parallel veins
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_direction = np.arctan2(sobel_y, sobel_x)
        
        # Histogram of gradient directions
        dir_hist, _ = np.histogram(gradient_direction, bins=8, range=(-np.pi, np.pi))
        dir_hist = dir_hist / (dir_hist.sum() + 1e-6)
        
        # Dominant direction strength (parallel veins have stronger peaks)
        direction_strength = np.max(dir_hist) - np.mean(dir_hist)
        features.append(direction_strength)
        
        # 5. Frequency domain features (leaf texture periodicity)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Energy in different frequency bands
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # High frequency energy (edges, serrations)
        high_freq_mask = np.ones_like(magnitude_spectrum)
        high_freq_mask[center_h-20:center_h+20, center_w-20:center_w+20] = 0
        high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask) / np.sum(magnitude_spectrum)
        features.append(high_freq_energy)
        
        return np.array(features)
    
    def _calculate_edge_complexity(self, edges: np.ndarray) -> float:
        """
        Calculate edge complexity score
        Serrated leaves (tomato) have higher complexity
        """
        # Find edge contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return 0.0
        
        total_complexity = 0
        for contour in contours:
            if len(contour) > 10:
                # Perimeter to area ratio of contour's bounding box
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                if area > 0:
                    complexity = perimeter / np.sqrt(area + 1e-6)
                    total_complexity += complexity
        
        return total_complexity / (len(contours) + 1e-6)

    # -------------------------------------------------------------
    # Combine all feature types
    # -------------------------------------------------------------
    def extract_all_features(self, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
        all_features = []

        # Deep features
        if self.feature_config.get('use_deep_features', True):
            deep = self.extract_deep_features(images, batch_size)
            all_features.append(deep)

        # Traditional + Plant-specific features
        traditional = []
        logger.info("Extracting traditional and plant-specific features...")

        for img in images:
            f = []

            if self.feature_config.get('extract_color_features', True):
                f.extend(self.extract_color_features(img))

            if self.feature_config.get('extract_texture_features', True):
                f.extend(self.extract_texture_features(img))

            if self.feature_config.get('extract_shape_features', True):
                f.extend(self.extract_shape_features(img))

            # NEW: Plant-specific features
            if self.feature_config.get('extract_plant_features', True):
                f.extend(self.extract_plant_specific_features(img))

            traditional.append(f)

        if traditional:
            all_features.append(np.array(traditional))

        # Final combined feature vector
        final = np.concatenate(all_features, axis=1)
        logger.info(f"Final feature shape: {final.shape}")

        return final