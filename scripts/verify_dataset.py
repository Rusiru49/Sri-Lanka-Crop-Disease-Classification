"""Script to verify the manually downloaded dataset structure."""

import os
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger()
config = get_config()


def verify_dataset_structure(data_dir='data/raw/plantvillage'):
    """
    Verify that the dataset is properly structured.
    
    Args:
        data_dir: Path to dataset root directory
    """
    logger.info("=" * 80)
    logger.info("Dataset Verification Script")
    logger.info("=" * 80)
    
    if not os.path.exists(data_dir):
        logger.error(f"❌ Dataset directory not found: {data_dir}")
        logger.info("\nExpected structure:")
        logger.info("data/raw/plantvillage/")
        logger.info("├── train/")
        logger.info("├── val/")
        logger.info("└── test/")
        return False
    
    logger.info(f"\n✓ Dataset directory found: {data_dir}")
    
    # Check for train/val/test folders
    required_folders = ['train', 'val', 'test']
    missing_folders = []
    
    for folder in required_folders:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            missing_folders.append(folder)
    
    if missing_folders:
        logger.error(f"❌ Missing folders: {', '.join(missing_folders)}")
        return False
    
    logger.info("✓ All required folders found (train, val, test)")
    
    # Analyze each split
    stats = {}
    class_counts = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    
    for split in required_folders:
        split_dir = os.path.join(data_dir, split)
        
        # Get all class folders
        classes = [d for d in os.listdir(split_dir) 
                  if os.path.isdir(os.path.join(split_dir, d))]
        
        total_images = 0
        class_details = []
        
        for class_name in sorted(classes):
            class_path = os.path.join(split_dir, class_name)
            
            # Count images
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
            
            count = len(image_files)
            total_images += count
            class_counts[class_name][split] = count
            class_details.append((class_name, count))
        
        stats[split] = {
            'total_images': total_images,
            'n_classes': len(classes),
            'class_details': class_details
        }
    
    # Display statistics
    logger.info("\n" + "=" * 80)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 80)
    
    total_all = 0
    for split in required_folders:
        total = stats[split]['total_images']
        n_classes = stats[split]['n_classes']
        total_all += total
        
        logger.info(f"\n{split.upper()} SET:")
        logger.info(f"  Total Images: {total:,}")
        logger.info(f"  Number of Classes: {n_classes}")
        logger.info(f"  Average per Class: {total/n_classes:.1f}")
    
    logger.info(f"\nTOTAL DATASET:")
    logger.info(f"  Total Images: {total_all:,}")
    
    # Verify expected counts
    expected = {
        'train': 3627,
        'val': 373,
        'test': 1404
    }
    
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION")
    logger.info("=" * 80)
    
    all_match = True
    for split, expected_count in expected.items():
        actual_count = stats[split]['total_images']
        match = "✓" if actual_count == expected_count else "✗"
        
        if actual_count == expected_count:
            logger.info(f"{match} {split.capitalize()}: {actual_count} images (expected {expected_count})")
        else:
            logger.warning(f"{match} {split.capitalize()}: {actual_count} images (expected {expected_count})")
            all_match = False
    
    # Display class distribution
    logger.info("\n" + "=" * 80)
    logger.info("CLASS DISTRIBUTION")
    logger.info("=" * 80)
    
    all_classes = sorted(class_counts.keys())
    logger.info(f"\nTotal Classes: {len(all_classes)}")
    
    logger.info("\nPer-Class Distribution (Train/Val/Test):")
    logger.info("-" * 80)
    logger.info(f"{'Class Name':<40} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    logger.info("-" * 80)
    
    for class_name in all_classes[:10]:  # Show first 10 classes
        train = class_counts[class_name]['train']
        val = class_counts[class_name]['val']
        test = class_counts[class_name]['test']
        total = train + val + test
        
        logger.info(f"{class_name:<40} {train:>8} {val:>8} {test:>8} {total:>8}")
    
    if len(all_classes) > 10:
        logger.info(f"... and {len(all_classes) - 10} more classes")
    
    logger.info("-" * 80)
    
    # Check for class consistency
    logger.info("\n" + "=" * 80)
    logger.info("CLASS CONSISTENCY CHECK")
    logger.info("=" * 80)
    
    # Get classes from each split
    train_classes = set(class_counts.keys())
    val_classes = {c for c, counts in class_counts.items() if counts['val'] > 0}
    test_classes = {c for c, counts in class_counts.items() if counts['test'] > 0}
    
    # Check if all splits have the same classes
    if train_classes == val_classes == test_classes:
        logger.info("✓ All splits have the same classes")
    else:
        logger.warning("✗ Classes are not consistent across splits")
        
        only_train = train_classes - val_classes - test_classes
        only_val = val_classes - train_classes - test_classes
        only_test = test_classes - train_classes - val_classes
        
        if only_train:
            logger.warning(f"  Classes only in train: {only_train}")
        if only_val:
            logger.warning(f"  Classes only in val: {only_val}")
        if only_test:
            logger.warning(f"  Classes only in test: {only_test}")
    
    # Check for empty classes
    empty_classes = []
    for class_name, counts in class_counts.items():
        if counts['train'] == 0 or counts['val'] == 0 or counts['test'] == 0:
            empty_classes.append(class_name)
    
    if empty_classes:
        logger.warning(f"\n⚠️  Classes with missing data in some splits: {len(empty_classes)}")
        for class_name in empty_classes[:5]:
            counts = class_counts[class_name]
            logger.warning(f"  - {class_name}: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    else:
        logger.info("✓ All classes have data in all splits")
    
    # Check for image file integrity
    logger.info("\n" + "=" * 80)
    logger.info("IMAGE FILE CHECK")
    logger.info("=" * 80)
    
    logger.info("Checking for valid image extensions...")
    
    valid_extensions = {'.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.JPEG'}
    invalid_files = []
    
    for split in required_folders:
        split_dir = os.path.join(data_dir, split)
        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext not in {'.png', '.jpg', '.jpeg'}:
                        invalid_files.append(os.path.join(class_path, file))
    
    if invalid_files:
        logger.warning(f"⚠️  Found {len(invalid_files)} files with non-image extensions")
        for file in invalid_files[:5]:
            logger.warning(f"  - {file}")
    else:
        logger.info("✓ All files have valid image extensions")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    if all_match and not empty_classes and not invalid_files:
        logger.info("✅ Dataset verification PASSED!")
        logger.info("Your dataset is properly structured and ready for training.")
        logger.info("\nNext steps:")
        logger.info("1. Run: python scripts/train_model.py --model-type xgboost --augment")
        logger.info("2. Wait for training to complete")
        logger.info("3. Run: streamlit run streamlit_app/app.py")
        return True
    else:
        logger.warning("⚠️  Dataset verification completed with warnings")
        if not all_match:
            logger.warning("- Image counts don't match expected values")
        if empty_classes:
            logger.warning("- Some classes are missing data in certain splits")
        if invalid_files:
            logger.warning("- Some files have invalid extensions")
        logger.info("\nDataset can still be used, but review warnings above.")
        return True
    
    logger.info("=" * 80 + "\n")


def main():
    """Main function."""
    data_dir = config.get('data.raw_dir', 'data/raw/plantvillage')
    verify_dataset_structure(data_dir)


if __name__ == "__main__":
    main()