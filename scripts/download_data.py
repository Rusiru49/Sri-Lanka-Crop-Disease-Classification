"""Script to download PlantVillage dataset."""

import os
import sys
from pathlib import Path
import kaggle
import zipfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.helpers import create_directories

logger = get_logger()


def download_plantvillage_dataset(data_dir='data/raw'):
    """
    Download PlantVillage dataset from Kaggle.
    
    Args:
        data_dir: Directory to save the dataset
    """
    logger.info("Starting PlantVillage dataset download...")
    
    # Create directories
    create_directories([data_dir])
    
    try:
        # Check if kaggle.json exists
        kaggle_config = os.path.expanduser('~/.kaggle/kaggle.json')
        if not os.path.exists(kaggle_config):
            logger.error("Kaggle API credentials not found!")
            logger.info("Please follow these steps:")
            logger.info("1. Go to https://www.kaggle.com/account")
            logger.info("2. Create a new API token (this will download kaggle.json)")
            logger.info("3. Place kaggle.json in ~/.kaggle/ directory")
            logger.info("4. Run: chmod 600 ~/.kaggle/kaggle.json (on Linux/Mac)")
            return False
        
        # Download dataset
        logger.info("Downloading dataset from Kaggle...")
        dataset_name = "abdallahalidev/plantvillage-dataset"
        
        # Download using Kaggle API
        os.system(f"kaggle datasets download -d {dataset_name} -p {data_dir}")
        
        # Extract zip file
        zip_path = os.path.join(data_dir, "plantvillage-dataset.zip")
        
        if os.path.exists(zip_path):
            logger.info("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Remove zip file
            os.remove(zip_path)
            logger.info("Dataset extracted successfully!")
            
            # Organize dataset structure
            organize_dataset_structure(data_dir)
            
            logger.info("✅ Dataset download and setup completed!")
            return True
        else:
            logger.error("Download failed. Please try again or download manually.")
            return False
    
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        logger.info("\nAlternative: Download manually from:")
        logger.info("https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        logger.info(f"Extract to: {os.path.abspath(data_dir)}/plantvillage/")
        return False


def organize_dataset_structure(data_dir):
    """
    Organize dataset into train/val/test splits.
    
    Args:
        data_dir: Root data directory
    """
    logger.info("Organizing dataset structure...")
    
    # Find the extracted dataset folder
    plantvillage_dir = None
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and 'plant' in item.lower():
            plantvillage_dir = item_path
            break
    
    if plantvillage_dir:
        # Rename to standard name
        target_dir = os.path.join(data_dir, 'plantvillage')
        if plantvillage_dir != target_dir:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            os.rename(plantvillage_dir, target_dir)
        
        logger.info(f"Dataset organized at: {target_dir}")
    else:
        logger.warning("Could not find PlantVillage folder. Please organize manually.")


def verify_dataset(data_dir='data/raw/plantvillage'):
    """
    Verify dataset structure and count images.
    
    Args:
        data_dir: Dataset directory
    """
    if not os.path.exists(data_dir):
        logger.error(f"Dataset not found at {data_dir}")
        return False
    
    logger.info("Verifying dataset...")
    
    total_images = 0
    class_count = 0
    
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        
        if os.path.isdir(class_path):
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            num_images = len(image_files)
            total_images += num_images
            class_count += 1
            
            logger.info(f"  {class_name}: {num_images} images")
    
    logger.info(f"\nTotal: {class_count} classes, {total_images} images")
    
    if total_images > 0:
        logger.info("✅ Dataset verification successful!")
        return True
    else:
        logger.error("❌ No images found in dataset!")
        return False


def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("PlantVillage Dataset Download Script")
    logger.info("=" * 80)
    
    data_dir = 'data/raw'
    
    # Check if dataset already exists
    plantvillage_path = os.path.join(data_dir, 'plantvillage')
    if os.path.exists(plantvillage_path):
        logger.info("Dataset already exists.")
        choice = input("Do you want to re-download? (y/n): ")
        if choice.lower() != 'y':
            logger.info("Using existing dataset.")
            verify_dataset(plantvillage_path)
            return
        else:
            shutil.rmtree(plantvillage_path)
    
    # Download dataset
    success = download_plantvillage_dataset(data_dir)
    
    if success:
        # Verify dataset
        verify_dataset(plantvillage_path)
    else:
        logger.error("Dataset download failed.")
        logger.info("\nManual Download Instructions:")
        logger.info("1. Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        logger.info("2. Download the dataset")
        logger.info(f"3. Extract to: {os.path.abspath(plantvillage_path)}/")
        logger.info("4. Run this script again to verify")


if __name__ == "__main__":
    main()