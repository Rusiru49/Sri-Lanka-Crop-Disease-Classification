"""Script to train the crop disease classification model with pre-split data."""

import sys
from pathlib import Path
import argparse
import traceback

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import ImageDataLoader
from src.data.data_preprocessor import ImagePreprocessor
from src.data.data_augmentation import ImageAugmentor
from src.features.feature_engineering import FeatureExtractor
from src.models.train import ModelTrainer, EnsembleTrainer
from src.models.evaluate import ModelEvaluator
from src.visualization.plot_utils import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_class_distribution
)
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.helpers import set_seed, save_json, create_directories

# Initialize
config = get_config()
logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description='Train crop disease classification model')
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='lightgbm',
        choices=['lightgbm', 'histgradient', 'random_forest', 'ensemble'],
        help='Type of model to train'
    )

    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--balance', action='store_true', help='Balance classes using augmentation')
    parser.add_argument('--save-plots', action='store_true', help='Save evaluation plots')

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Sri Lanka Crop Disease Classification - Training Pipeline")
    logger.info("=" * 80)

    set_seed(config.random_seed)
    create_directories([
        'models/saved_models', 'models/checkpoints',
        'reports/figures', 'reports/results', 'logs'
    ])

    # STEP 1 — LOAD DATA
    logger.info("\n[Step 1/7] Loading Pre-Split Dataset...")
    data_loader = ImageDataLoader()

    try:
        if getattr(data_loader, "is_pre_split", False):
            logger.info("✓ Detected pre-split dataset")
            splits = data_loader.load_all_splits()
            X_train, y_train = splits.get('train', (None, None))
            X_val, y_val = splits.get('val', (None, None))
            X_test, y_test = splits.get('test', (None, None))
        else:
            logger.warning("Dataset not pre-split — splitting automatically...")
            images, labels, _ = data_loader.load_dataset()
            splits = data_loader.split_dataset(images, labels)
            X_train, y_train = splits['train']
            X_val, y_val = splits['val']
            X_test, y_test = splits['test']

    except Exception:
        logger.error("Error loading dataset:\n" + traceback.format_exc())
        return

    # STEP 2 — DATA AUGMENTATION
    if args.augment:
        logger.info("[Step 2/7] Applying Data Augmentation...")
        augmentor = ImageAugmentor()

        try:
            if args.balance:
                X_train, y_train = augmentor.augment_minority_classes(X_train, y_train)
            else:
                X_train, y_train = augmentor.augment_batch(
                    X_train, y_train, augmentation_factor=2
                )

            logger.info(f"Training set size after augmentation: {len(X_train)}")

        except Exception:
            logger.warning("Augmentation failed — continuing without augmentation.")
    else:
        logger.info("Skipping data augmentation")

    # STEP 3 — FEATURE EXTRACTION
    logger.info("[Step 3/7] Extracting Features...")
    feature_extractor = FeatureExtractor()

    X_train_features = feature_extractor.extract_all_features(X_train)
    X_val_features = feature_extractor.extract_all_features(X_val) if X_val is not None else None
    X_test_features = feature_extractor.extract_all_features(X_test) if X_test is not None else None

    # STEP 4 — PREPROCESSING
    logger.info("[Step 4/7] Preprocessing...")
    preprocessor = ImagePreprocessor()

    X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.standardize_features(
        X_train_features, X_val_features, X_test_features
    )

    # STEP 5 — MODEL TRAINING
    logger.info(f"[Step 5/7] Training {args.model_type.upper()} Model...")

    if args.model_type == "ensemble":
        trainer = EnsembleTrainer()
        trainer.train(X_train_scaled, y_train, X_val_scaled, y_val)
        training_metrics = {}
    else:
        trainer = ModelTrainer(
            model_type=args.model_type,
            n_classes=len(set(y_train))
        )
        training_metrics = trainer.train(
            X_train_scaled, y_train, X_val_scaled, y_val
        )

    trainer.save()
    logger.info("✓ Model saved successfully")

    # STEP 6 — FEATURE IMPORTANCE
    if args.model_type != "ensemble" and args.save_plots:
        fi = trainer.get_feature_importance()

        if fi:
            plot_feature_importance(
                fi,
                top_n=20,
                save_path="reports/figures/feature_importance.png"
            )

    # STEP 7 — EVALUATION
    logger.info("[Step 7/7] Evaluating Model...")

    # Now safe because both trainer types have predict + predict_proba
    evaluator = ModelEvaluator(trainer, data_loader.class_names)

    if X_test_scaled is not None:
        test_metrics = evaluator.evaluate(X_test_scaled, y_test)

        if args.model_type == "ensemble":
            y_pred = trainer.predict(X_test_scaled)
        else:
            y_pred = trainer.model.predict(X_test_scaled)

        if args.save_plots:
            plot_confusion_matrix(
                y_test,
                y_pred,
                data_loader.class_names,
                save_path="reports/figures/confusion_matrix.png"
            )
    else:
        test_metrics = {}

    save_json(
        {
            'model_type': args.model_type,
            'training_metrics': training_metrics,
            'test_metrics': test_metrics,
            'class_names': data_loader.class_names
        },
        "reports/results/evaluation_metrics.json"
    )

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
