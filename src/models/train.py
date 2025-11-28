"""Model training utilities."""

import numpy as np
from typing import Dict, Any
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.helpers import save_model, save_json

logger = get_logger()
config = get_config()


class ModelTrainer:
    """Train classification models."""

    def __init__(self, model_type: str = None):
        self.model_type = model_type or config.get('model.model_type', 'xgboost')
        self.model = None
        self.training_history = {}

    def _create_xgboost_model(self) -> xgb.XGBClassifier:
        xgb_config = config.get('model.xgboost', {})

        model = xgb.XGBClassifier(
            n_estimators=xgb_config.get('n_estimators', 100),
            max_depth=xgb_config.get('max_depth', 6),
            learning_rate=xgb_config.get('learning_rate', 0.1),
            subsample=xgb_config.get('subsample', 0.8),
            colsample_bytree=xgb_config.get('colsample_bytree', 0.8),
            random_state=xgb_config.get('random_state', 42),
            eval_metric="mlogloss",      # valid for xgboost 3.x
            tree_method="hist",
            n_jobs=-1
        )
        return model

    def _create_random_forest_model(self) -> RandomForestClassifier:
        rf_config = config.get('model.random_forest', {})

        model = RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 100),
            max_depth=rf_config.get('max_depth', 8),
            min_samples_split=rf_config.get('min_samples_split', 5),
            min_samples_leaf=rf_config.get('min_samples_leaf', 2),
            random_state=rf_config.get('random_state', 42),
            n_jobs=-1,
            verbose=1
        )
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> Dict[str, Any]:

        logger.info(f"Training {self.model_type} model")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Feature dimension: {X_train.shape[1]}")

        if self.model_type == 'xgboost':
            self.model = self._create_xgboost_model()

            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=20,
                    verbose=True
                )

                # XGBoost 3.x uses same evals_result() API
                results = self.model.evals_result()

                self.training_history = {
                    'val_loss': results['validation_0']['mlogloss']
                }
            else:
                self.model.fit(X_train, y_train, verbose=True)

        elif self.model_type == 'random_forest':
            self.model = self._create_random_forest_model()
            self.model.fit(X_train, y_train)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Evaluate metrics
        train_score = accuracy_score(y_train, self.model.predict(X_train))
        logger.info(f"Training accuracy: {train_score:.4f}")

        metrics = {'train_accuracy': train_score}

        if X_val is not None and y_val is not None:
            val_score = accuracy_score(y_val, self.model.predict(X_val))
            logger.info(f"Validation accuracy: {val_score:.4f}")
            metrics['val_accuracy'] = val_score

        return metrics

    def save(self, filepath: str = None):
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        if filepath is None:
            model_dir = config.get('training.model_save_path', 'models/saved_models')
            filepath = f"{model_dir}/{self.model_type}_model.pkl"

        save_model(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def get_feature_importance(self, feature_names: list = None) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("No trained model available.")

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]

            feature_importance = dict(zip(feature_names, importances))
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            return feature_importance

        logger.warning("Model does not support feature importance")
        return {}


class EnsembleTrainer:
    """Train ensemble of models."""

    def __init__(self):
        self.models = {}
        self.weights = {}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ):
        logger.info("Training ensemble models")

        # Train XGBoost
        xgb_trainer = ModelTrainer('xgboost')
        xgb_metrics = xgb_trainer.train(X_train, y_train, X_val, y_val)
        self.models['xgboost'] = xgb_trainer.model
        self.weights['xgboost'] = xgb_metrics.get("val_accuracy", 1.0)

        # Train Random Forest
        rf_trainer = ModelTrainer('random_forest')
        rf_metrics = rf_trainer.train(X_train, y_train, X_val, y_val)
        self.models['random_forest'] = rf_trainer.model
        self.weights['random_forest'] = rf_metrics.get("val_accuracy", 1.0)

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

        logger.info(f"Ensemble weights: {self.weights}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []

        for name, model in self.models.items():
            prob = model.predict_proba(X)
            weighted = prob * self.weights[name]
            predictions.append(weighted)

        ensemble_scores = np.sum(predictions, axis=0)
        return np.argmax(ensemble_scores, axis=1)

    def save(self, directory: str = "models/saved_models"):
        for name, model in self.models.items():
            filepath = f"{directory}/ensemble_{name}.pkl"
            save_model(model, filepath)

        save_json(self.weights, f"{directory}/ensemble_weights.json")
        logger.info(f"Ensemble models saved to {directory}")
