import numpy as np
from typing import Dict, Any
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.helpers import save_model, save_json

logger = get_logger()
config = get_config()


class ModelTrainer:
    """Train classification models with support for LightGBM, HistGradientBoosting, and Random Forest."""

    def __init__(self, model_type: str = None, n_classes: int = None):
        self.model_type = model_type or config.get('model.model_type', 'lightgbm')
        self.model = None
        self.training_history = {}
        self.n_classes = n_classes

    def _create_lightgbm_model(self) -> lgb.LGBMClassifier:
        """Create a LightGBM classifier - excellent XGBoost alternative."""
        lgbm_config = config.get('model.lightgbm', {})

        model = lgb.LGBMClassifier(
            n_estimators=lgbm_config.get('n_estimators', 100),
            max_depth=lgbm_config.get('max_depth', 6),
            learning_rate=lgbm_config.get('learning_rate', 0.1),
            subsample=lgbm_config.get('subsample', 0.8),
            colsample_bytree=lgbm_config.get('colsample_bytree', 0.8),
            random_state=lgbm_config.get('random_state', 42),
            n_jobs=-1,
            verbose=-1,
            objective='multiclass' if self.n_classes and self.n_classes > 2 else 'binary',
            num_class=self.n_classes if self.n_classes and self.n_classes > 2 else None,
        )
        return model

    def _create_histgradient_model(self) -> HistGradientBoostingClassifier:
        """Create sklearn's HistGradientBoosting - built-in, no extra dependencies."""
        hgb_config = config.get('model.histgradient', {})

        model = HistGradientBoostingClassifier(
            max_iter=hgb_config.get('max_iter', 100),
            max_depth=hgb_config.get('max_depth', 6),
            learning_rate=hgb_config.get('learning_rate', 0.1),
            random_state=hgb_config.get('random_state', 42),
            verbose=1,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        return model

    def _create_random_forest_model(self) -> RandomForestClassifier:
        """Create a Random Forest classifier."""
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
        """Train the model with optional validation and early stopping."""

        logger.info(f"Training {self.model_type} model")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Feature dimension: {X_train.shape[1]}")

        if self.n_classes is None:
            self.n_classes = len(np.unique(y_train))

        if self.model_type == 'lightgbm':
            self.model = self._create_lightgbm_model()

            if X_val is not None and y_val is not None:
                callbacks = [
                    lgb.early_stopping(stopping_rounds=10, verbose=True),
                    lgb.log_evaluation(period=10)
                ]
                
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=callbacks
                )

                if hasattr(self.model, 'evals_result_'):
                    results = self.model.evals_result_
                    self.training_history = {
                        'val_loss': results.get('valid_0', {}).get('multi_logloss', [])
                    }
            else:
                self.model.fit(X_train, y_train)

        elif self.model_type == 'histgradient':
            self.model = self._create_histgradient_model()
            self.model.fit(X_train, y_train)

        elif self.model_type == 'random_forest':
            self.model = self._create_random_forest_model()
            self.model.fit(X_train, y_train)

        else:
            raise ValueError(
                f"Unknown model type: {self.model_type}. "
                f"Supported types: 'lightgbm', 'histgradient', 'random_forest'"
            )

        train_score = accuracy_score(y_train, self.model.predict(X_train))
        logger.info(f"Training accuracy: {train_score:.4f}")

        metrics = {'train_accuracy': train_score}

        if X_val is not None and y_val is not None:
            val_score = accuracy_score(y_val, self.model.predict(X_val))
            logger.info(f"Validation accuracy: {val_score:.4f}")
            metrics['val_accuracy'] = val_score

        return metrics
    
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        """Wrapper for probability predictions (required for ModelEvaluator)."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError(f"{self.model_type} model does not support predict_proba()")

    def save(self, filepath: str = None):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        if filepath is None:
            model_dir = config.get('training.model_save_path', 'models/saved_models')
            filepath = f"{model_dir}/{self.model_type}_model.pkl"

        save_model(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def get_feature_importance(self, feature_names: list = None) -> Dict[str, float]:
        """Get feature importance from the trained model."""
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
        """Train multiple models and create an ensemble."""
        logger.info("Training ensemble models")
        n_classes = len(np.unique(y_train))

        # Train LightGBM
        lgbm_trainer = ModelTrainer('lightgbm', n_classes=n_classes)
        lgbm_metrics = lgbm_trainer.train(X_train, y_train, X_val, y_val)
        self.models['lightgbm'] = lgbm_trainer.model
        self.weights['lightgbm'] = lgbm_metrics.get("val_accuracy", 1.0)

        # Train HistGradientBoosting
        hgb_trainer = ModelTrainer('histgradient', n_classes=n_classes)
        hgb_metrics = hgb_trainer.train(X_train, y_train, X_val, y_val)
        self.models['histgradient'] = hgb_trainer.model
        self.weights['histgradient'] = hgb_metrics.get("val_accuracy", 1.0)

        # Train Random Forest
        rf_trainer = ModelTrainer('random_forest', n_classes=n_classes)
        rf_metrics = rf_trainer.train(X_train, y_train, X_val, y_val)
        self.models['random_forest'] = rf_trainer.model
        self.weights['random_forest'] = rf_metrics.get("val_accuracy", 1.0)

        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        else:
            logger.warning("All ensemble weights are 0, using equal weighting")
            self.weights = {k: 1.0 / len(self.weights) for k in self.weights.keys()}

        logger.info(f"Ensemble weights: {self.weights}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using weighted ensemble."""
        predictions = []

        for name, model in self.models.items():
            prob = model.predict_proba(X)
            weighted = prob * self.weights[name]
            predictions.append(weighted)

        ensemble_scores = np.sum(predictions, axis=0)
        return np.argmax(ensemble_scores, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions from ensemble."""
        predictions = []

        for name, model in self.models.items():
            prob = model.predict_proba(X)
            weighted = prob * self.weights[name]
            predictions.append(weighted)

        return np.sum(predictions, axis=0)

    def save(self, directory: str = "models/saved_models"):
        """Save all ensemble models and weights."""
        for name, model in self.models.items():
            filepath = f"{directory}/ensemble_{name}.pkl"
            save_model(model, filepath)

        save_json(self.weights, f"{directory}/ensemble_weights.json")
        logger.info(f"Ensemble models saved to {directory}")
