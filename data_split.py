import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

def split_from_single_folder(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split images from a single folder containing class subfolders into train/val/test
    
    Structure expected:
    source_dir/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img1.jpg
    
    Args:
        source_dir: Path to folder containing class subfolders with images
        dest_dir: Path to processed folder where splits will be created
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1.0"
    
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist!")
        return
    
    # Create destination directories
    splits = ['train_processed', 'val_processed', 'test_processed']
    for split in splits:
        (dest_path / split).mkdir(parents=True, exist_ok=True)
    
    # Get all class folders (skip hidden folders and files)
    class_folders = [f for f in source_path.iterdir() 
                    if f.is_dir() and not f.name.startswith('.')]
    
    if not class_folders:
        print(f"Error: No class folders found in '{source_dir}'")
        print(f"Contents: {list(source_path.iterdir())}")
        return
    
    stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    
    print(f"Found {len(class_folders)} classes")
    print(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}\n")
    
    # Process each class
    for class_folder in class_folders:
        class_name = class_folder.name
        
        # Get all images in this class
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.BMP'}
        images = [f for f in class_folder.iterdir() 
                 if f.is_file() and f.suffix in image_extensions]
        
        if not images:
            print(f"Warning: No images found in '{class_name}'")
            continue
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # Split images
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        # Create class folders in each split
        for split in splits:
            (dest_path / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Copy images to respective folders
        for img in train_images:
            shutil.copy2(img, dest_path / 'train_processed' / class_name / img.name)
            stats[class_name]['train'] += 1
        
        for img in val_images:
            shutil.copy2(img, dest_path / 'val_processed' / class_name / img.name)
            stats[class_name]['val'] += 1
        
        for img in test_images:
            shutil.copy2(img, dest_path / 'test_processed' / class_name / img.name)
            stats[class_name]['test'] += 1
        
        print(f"{class_name}: Total={total}, Train={len(train_images)}, "
              f"Val={len(val_images)}, Test={len(test_images)}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    total_train = sum(v['train'] for v in stats.values())
    total_val = sum(v['val'] for v in stats.values())
    total_test = sum(v['test'] for v in stats.values())
    total_all = total_train + total_val + total_test
    
    if total_all == 0:
        print("ERROR: No images were processed!")
        print("Please check your folder structure.")
        return
    
    print(f"Total images: {total_all}")
    print(f"Train: {total_train} ({total_train/total_all*100:.1f}%)")
    print(f"Validation: {total_val} ({total_val/total_all*100:.1f}%)")
    print(f"Test: {total_test} ({total_test/total_all*100:.1f}%)")
    print("="*70)


def merge_and_split(train_dir, val_dir, test_dir, dest_dir, 
                    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Merge existing train/val/test folders and re-split them
    
    Use this if you already have train/val/test folders with class subfolders
    and want to create a new split
    
    Args:
        train_dir: Path to existing train folder
        val_dir: Path to existing val folder  
        test_dir: Path to existing test folder
        dest_dir: Path to processed folder where new splits will be created
        train_ratio, val_ratio, test_ratio: New split ratios
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1.0"
    
    dest_path = Path(dest_dir)
    
    # Create destination directories
    splits = ['train_processed', 'val_processed', 'test_processed']
    for split in splits:
        (dest_path / split).mkdir(parents=True, exist_ok=True)
    
    # Collect all images by class from all splits
    all_images_by_class = defaultdict(list)
    
    print("Collecting images from existing splits...")
    for split_dir in [train_dir, val_dir, test_dir]:
        split_path = Path(split_dir)
        if not split_path.exists():
            print(f"Warning: '{split_dir}' does not exist, skipping...")
            continue
            
        class_folders = [f for f in split_path.iterdir() 
                        if f.is_dir() and not f.name.startswith('.')]
        
        for class_folder in class_folders:
            class_name = class_folder.name
            image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.BMP'}
            images = [f for f in class_folder.iterdir() 
                     if f.is_file() and f.suffix in image_extensions]
            all_images_by_class[class_name].extend(images)
    
    if not all_images_by_class:
        print("ERROR: No images found in any of the source folders!")
        return
    
    stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    
    print(f"\nFound {len(all_images_by_class)} classes")
    print(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}\n")
    
    # Process each class
    for class_name, images in all_images_by_class.items():
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # Split images
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        # Create class folders in each split
        for split in splits:
            (dest_path / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Copy images to respective folders
        for img in train_images:
            shutil.copy2(img, dest_path / 'train_processed' / class_name / img.name)
            stats[class_name]['train'] += 1
        
        for img in val_images:
            shutil.copy2(img, dest_path / 'val_processed' / class_name / img.name)
            stats[class_name]['val'] += 1
        
        for img in test_images:
            shutil.copy2(img, dest_path / 'test_processed' / class_name / img.name)
            stats[class_name]['test'] += 1
        
        print(f"{class_name}: Total={total}, Train={len(train_images)}, "
              f"Val={len(val_images)}, Test={len(test_images)}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    total_train = sum(v['train'] for v in stats.values())
    total_val = sum(v['val'] for v in stats.values())
    total_test = sum(v['test'] for v in stats.values())
    total_all = total_train + total_val + total_test
    
    print(f"Total images: {total_all}")
    print(f"Train: {total_train} ({total_train/total_all*100:.1f}%)")
    print(f"Validation: {total_val} ({total_val/total_all*100:.1f}%)")
    print(f"Test: {total_test} ({total_test/total_all*100:.1f}%)")
    print("="*70)


# Example usage:
if __name__ == "__main__":
    
    # OPTION 1: If your images are in a single folder with class subfolders
    # Example: data/raw/plantvillage/class1/, data/raw/plantvillage/class2/, etc.
    # Uncomment these lines:
    """
    SOURCE_DIR = "data/raw/plantvillage"
    DEST_DIR = "data/processed"
    split_from_single_folder(SOURCE_DIR, DEST_DIR, 
                            train_ratio=0.7, 
                            val_ratio=0.15, 
                            test_ratio=0.15)
    """
    
    # OPTION 2: If you already have train/val/test folders with class subfolders
    # Example: data/raw/plantvillage/train/class1/, data/raw/plantvillage/val/class1/, etc.
    # Uncomment these lines:
    
    TRAIN_DIR = "data/raw/plantvillage/train"
    VAL_DIR = "data/raw/plantvillage/val"
    TEST_DIR = "data/raw/plantvillage/test"
    DEST_DIR = "data/processed"
    
    merge_and_split(TRAIN_DIR, VAL_DIR, TEST_DIR, DEST_DIR,
                   train_ratio=0.7,
                   val_ratio=0.15, 
                   test_ratio=0.15)