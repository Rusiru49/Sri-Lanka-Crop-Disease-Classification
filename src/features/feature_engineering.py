import numpy as np
import cv2
from typing import List
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
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
    """Extract deep + traditional features from images."""

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

        base_model.trainable = False  # << important for speed + stability

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

            # Ensure uint8
            batch_uint8 = (batch * 255).astype(np.uint8)

            batch_preprocessed = self.preprocess_fn(batch_uint8)
            batch_features = self.cnn_model.predict(batch_preprocessed, verbose=0)
            features_list.append(batch_features)

        features = np.vstack(features_list)
        logger.info(f"Deep features shape: {features.shape}")

        return features

    # -------------------------------------------------------------
    # Color features
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

        return np.array(features)

    # -------------------------------------------------------------
    # Texture features (GLCM + LBP)
    # -------------------------------------------------------------
    def extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        features = []

        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

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
        lbp_hist = lbp_hist / lbp_hist.sum()
        features.extend(lbp_hist)

        return np.array(features)

    # -------------------------------------------------------------
    # Shape features
    # -------------------------------------------------------------
    def extract_shape_features(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Default values
        area = perimeter = aspect_ratio = extent = solidity = 0

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)

            x, y, w, h = cv2.boundingRect(largest)
            if h > 0 and w > 0:
                aspect_ratio = w / h
                extent = area / (w * h)

            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)

        return np.array([area, perimeter, aspect_ratio, extent, solidity, edge_density])

    # -------------------------------------------------------------
    # Combine all feature types
    # -------------------------------------------------------------
    def extract_all_features(self, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
        all_features = []

        # Deep features
        if self.feature_config.get('use_deep_features', True):
            deep = self.extract_deep_features(images, batch_size)
            all_features.append(deep)

        # Traditional features
        traditional = []
        logger.info("Extracting traditional features...")

        for img in images:
            f = []

            if self.feature_config.get('extract_color_features', True):
                f.extend(self.extract_color_features(img))

            if self.feature_config.get('extract_texture_features', True):
                f.extend(self.extract_texture_features(img))

            if self.feature_config.get('extract_shape_features', True):
                f.extend(self.extract_shape_features(img))

            traditional.append(f)

        if traditional:
            all_features.append(np.array(traditional))

        # Final combined feature vector
        final = np.concatenate(all_features, axis=1)
        logger.info(f"Final feature shape: {final.shape}")

        return final
