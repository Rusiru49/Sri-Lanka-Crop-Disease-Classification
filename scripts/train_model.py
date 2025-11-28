"""Script to train the crop disease classification model with pre-split data."""

import sys
from pathlib import Path
import numpy as np
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import ImageDataLoader
from src.data.data_preprocessor import ImagePreprocessor
from src.data.data_augmentation import ImageAugmentor
from src.features.feature_engineering import FeatureExtractor
from src.models.train import ModelTrainer
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train crop disease classification model')
    parser.add_argument(
        '--model-type',
        type=str,
        default='xgboost',
        choices=['xgboost', 'random_forest', 'ensemble'],
        help='Type of model to train'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Enable data augmentation'
    )
    parser.add_argument(
        '--balance',
        action='store_true',
        help='Balance classes using augmentation'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        default=True,
        help='Save evaluation plots'
    )
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("Sri Lanka Crop Disease Classification - Training Pipeline")
    logger.info("=" * 80)
    
    # Set random seed for reproducibility
    set_seed(config.random_seed)
    
    # Create necessary directories
    create_directories([
        'models/saved_models',
        'models/checkpoints',
        'reports/figures',
        'reports/results',
        'logs'
    ])
    
    # Step 1: Load Pre-Split Data
    logger.info("\n[Step 1/7] Loading Pre-Split Dataset...")
    data_loader = ImageDataLoader()
    
    try:
        # Check if dataset is pre-split
        if data_loader.is_pre_split:
            logger.info("✓ Detected pre-split dataset structure")
            
            # Get dataset info
            dataset_info = data_loader.get_dataset_info()
            logger.info(f"\nDataset Information:")
            logger.info(f"  - Classes: {dataset_info['n_classes']}")
            logger.info(f"  - Training samples: {dataset_info.get('train_size', 'N/A')}")
            logger.info(f"  - Validation samples: {dataset_info.get('val_size', 'N/A')}")
            logger.info(f"  - Test samples: {dataset_info.get('test_size', 'N/A')}")
            
            # Load all splits
            splits = data_loader.load_all_splits()
            X_train, y_train = splits['train']
            X_val, y_val = splits['val']
            X_test, y_test = splits['test']
            
            # Get class distribution for each split
            logger.info("\nClass Distribution:")
            for split_name, (X, y) in [('Train', (X_train, y_train)), 
                                        ('Val', (X_val, y_val)), 
                                        ('Test', (X_test, y_test))]:
                dist = data_loader.get_class_distribution(y)
                logger.info(f"\n{split_name} Set:")
                for class_name, count in sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5]:
                    logger.info(f"  - {class_name}: {count} samples")
            
            # Plot class distribution for training set
            if args.save_plots:
                plot_class_distribution(
                    y_train,
                    data_loader.class_names,
                    title="Training Set Class Distribution",
                    save_path="reports/figures/class_distribution_train.png"
                )
                
                plot_class_distribution(
                    y_val,
                    data_loader.class_names,
                    title="Validation Set Class Distribution",
                    save_path="reports/figures/class_distribution_val.png"
                )
                
                plot_class_distribution(
                    y_test,
                    data_loader.class_names,
                    title="Test Set Class Distribution",
                    save_path="reports/figures/class_distribution_test.png"
                )
        
        else:
            # For non-split datasets (fallback)
            logger.warning("Dataset is not pre-split. Using automatic splitting...")
            images, labels, image_paths = data_loader.load_dataset()
            logger.info(f"Loaded {len(images)} images")
            
            splits = data_loader.split_dataset(images, labels)
            X_train, y_train = splits['train']
            X_val, y_val = splits['val']
            X_test, y_test = splits['test']
    
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        logger.info("\nPlease ensure your dataset is structured as:")
        logger.info("data/raw/plantvillage/")
        logger.info("├── train/")
        logger.info("│   ├── class1/")
        logger.info("│   ├── class2/")
        logger.info("│   └── ...")
        logger.info("├── val/")
        logger.info("│   ├── class1/")
        logger.info("│   └── ...")
        logger.info("└── test/")
        logger.info("    ├── class1/")
        logger.info("    └── ...")
        return
    
    # Step 2: Data Augmentation (Optional)
    logger.info(f"\n[Step 2/7] Data Augmentation...")
    if args.augment:
        logger.info("Applying data augmentation to training set...")
        augmentor = ImageAugmentor()
        
        original_train_size = len(X_train)
        
        if args.balance:
            logger.info("Balancing classes using augmentation...")
            X_train, y_train = augmentor.augment_minority_classes(X_train, y_train)
        else:
            logger.info("Applying standard augmentation (2x)...")
            X_train, y_train = augmentor.augment_batch(X_train, y_train, augmentation_factor=2)
        
        logger.info(f"Training set size: {original_train_size} → {len(X_train)} samples")
        logger.info(f"Augmentation factor: {len(X_train) / original_train_size:.2f}x")
    else:
        logger.info("Skipping data augmentation (use --augment flag to enable)")
    
    # Step 3: Feature Extraction
    logger.info("\n[Step 3/7] Extracting Features...")
    logger.info("This may take several minutes depending on dataset size...")
    
    feature_extractor = FeatureExtractor()
    
    logger.info("Extracting training features...")
    X_train_features = feature_extractor.extract_all_features(X_train)
    
    logger.info("Extracting validation features...")
    X_val_features = feature_extractor.extract_all_features(X_val)
    
    logger.info("Extracting test features...")
    X_test_features = feature_extractor.extract_all_features(X_test)
    
    logger.info(f"✓ Feature extraction complete!")
    logger.info(f"  Feature vector size: {X_train_features.shape[1]}")
    logger.info(f"  Train features shape: {X_train_features.shape}")
    logger.info(f"  Val features shape: {X_val_features.shape}")
    logger.info(f"  Test features shape: {X_test_features.shape}")
    
    # Step 4: Preprocessing
    logger.info("\n[Step 4/7] Preprocessing Features...")
    preprocessor = ImagePreprocessor()
    
    X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.standardize_features(
        X_train_features,
        X_val_features,
        X_test_features
    )
    
    logger.info("✓ Feature standardization complete!")
    
    # Step 5: Model Training
    logger.info(f"\n[Step 5/7] Training {args.model_type.upper()} Model...")
    logger.info("-" * 80)
    
    trainer = ModelTrainer(model_type=args.model_type)
    training_metrics = trainer.train(
        X_train_scaled, y_train,
        X_val_scaled, y_val
    )
    
    logger.info("-" * 80)
    logger.info("✓ Training completed!")
    logger.info(f"  Training accuracy: {training_metrics.get('train_accuracy', 0):.4f}")
    logger.info(f"  Validation accuracy: {training_metrics.get('val_accuracy', 0):.4f}")
    
    # Save model
    trainer.save()
    logger.info(f"✓ Model saved to models/saved_models/{args.model_type}_model.pkl")
    
    # Get and save feature importance
    feature_importance = trainer.get_feature_importance()
    if feature_importance and args.save_plots:
        plot_feature_importance(
            feature_importance,
            top_n=20,
            save_path="reports/figures/feature_importance.png"
        )
        logger.info("✓ Feature importance plot saved")
    
    # Step 6: Model Evaluation on Test Set
    logger.info("\n[Step 6/7] Evaluating Model on Test Set...")
    logger.info("-" * 80)
    
    evaluator = ModelEvaluator(trainer.model, data_loader.class_names)
    
    # Calculate metrics
    test_metrics = evaluator.evaluate(X_test_scaled, y_test)
    
    logger.info("\nTest Set Performance:")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    logger.info(f"  Precision: {test_metrics['precision_weighted']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall_weighted']:.4f}")
    logger.info(f"  F1-Score:  {test_metrics['f1_weighted']:.4f}")
    
    # Get classification report
    classification_report = evaluator.get_classification_report(X_test_scaled, y_test)
    logger.info(f"\nDetailed Classification Report:\n{classification_report}")
    
    # Get per-class metrics
    per_class_metrics = evaluator.get_per_class_metrics(X_test_scaled, y_test)
    logger.info(f"\nPer-Class Metrics:\n{per_class_metrics.to_string()}")
    
    # Plot confusion matrix
    if args.save_plots:
        logger.info("\nGenerating visualizations...")
        
        plot_confusion_matrix(
            y_test,
            trainer.model.predict(X_test_scaled),
            data_loader.class_names,
            save_path="reports/figures/confusion_matrix.png",
            normalize=False
        )
        
        plot_confusion_matrix(
            y_test,
            trainer.model.predict(X_test_scaled),
            data_loader.class_names,
            save_path="reports/figures/confusion_matrix_normalized.png",
            normalize=True
        )
        
        logger.info("✓ Confusion matrices saved")
    
    # Get misclassified samples
    misclassified = evaluator.get_misclassified_samples(X_test_scaled, y_test, limit=20)
    
    if misclassified.get('total_misclassified', 0) > 0:
        logger.info(f"\nMisclassified Samples: {misclassified['total_misclassified']}")
        logger.info("Top 5 misclassifications:")
        for i, sample in enumerate(misclassified['samples'][:5], 1):
            logger.info(f"  {i}. True: {sample['true_class']} → Predicted: {sample['predicted_class']} "
                       f"(confidence: {sample['confidence']:.2%})")
    
    # Step 7: Save Results
    logger.info("\n[Step 7/7] Saving Results...")
    
    results = {
        'model_type': args.model_type,
        'training_config': {
            'augmentation': args.augment,
            'balance_classes': args.balance,
            'image_size': config.image_size,
            'batch_size': config.batch_size
        },
        'dataset_info': {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'n_classes': len(data_loader.class_names),
            'n_features': X_train_features.shape[1]
        },
        'training_metrics': training_metrics,
        'test_metrics': test_metrics,
        'class_names': data_loader.class_names,
        'per_class_metrics': per_class_metrics.to_dict(),
        'misclassified_samples': misclassified
    }
    
    save_json(results, 'reports/results/evaluation_metrics.json')
    logger.info("✓ Results saved to reports/results/evaluation_metrics.json")
    
    # Final Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"\n📊 Summary:")
    logger.info(f"  Model Type: {args.model_type.upper()}")
    logger.info(f"  Dataset: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    logger.info(f"  Classes: {len(data_loader.class_names)}")
    logger.info(f"  Features: {X_train_features.shape[1]}")
    logger.info(f"\n🎯 Performance:")
    logger.info(f"  Test Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    logger.info(f"  Test F1-Score:  {test_metrics['f1_weighted']:.4f}")
    logger.info(f"  Misclassified:  {misclassified.get('total_misclassified', 0)}/{len(X_test)}")
    logger.info(f"\n📁 Output Files:")
    logger.info(f"  Model: models/saved_models/{args.model_type}_model.pkl")
    logger.info(f"  Results: reports/results/evaluation_metrics.json")
    logger.info(f"  Plots: reports/figures/")
    logger.info(f"\n🚀 Next Steps:")
    logger.info(f"  1. Run: streamlit run streamlit_app/app.py")
    logger.info(f"  2. Upload images for disease prediction")
    logger.info(f"  3. View performance metrics in the app")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()