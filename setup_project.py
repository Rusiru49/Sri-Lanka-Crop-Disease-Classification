"""
Setup script to create the complete project structure.
Run this script after cloning/creating the project to set up all directories and files.
"""

import os
from pathlib import Path


def create_directory_structure():
    """Create all necessary directories for the project."""
    
    directories = [
        # Data directories
        'data/raw/plantvillage/train',
        'data/raw/plantvillage/val',
        'data/raw/plantvillage/test',
        'data/processed/train_processed',
        'data/processed/val_processed',
        'data/processed/test_processed',
        'data/augmented/synthetic_samples',
        
        # Model directories
        'models/saved_models',
        'models/checkpoints',
        
        # Notebook directory
        'notebooks',
        
        # Source code directories
        'src/data',
        'src/features',
        'src/models',
        'src/visualization',
        'src/utils',
        
        # Streamlit app directories
        'streamlit_app/pages',
        'streamlit_app/components',
        'streamlit_app/assets/images',
        'streamlit_app/assets/icons',
        
        # Test directory
        'tests',
        
        # Reports directory
        'reports/figures',
        'reports/results',
        
        # Documentation directory
        'docs',
        
        # Config directory
        'config',
        
        # Scripts directory
        'scripts',
        
        # Logs directory
        'logs'
    ]
    
    print("Creating directory structure...")
    print("=" * 80)
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print("=" * 80)
    print("✓ All directories created successfully!\n")


def create_init_files():
    """Create __init__.py files in all Python packages."""
    
    packages = [
        'src',
        'src/data',
        'src/features',
        'src/models',
        'src/visualization',
        'src/utils',
        'streamlit_app',
        'streamlit_app/components',
        'streamlit_app/pages',
        'tests'
    ]
    
    print("Creating __init__.py files...")
    print("=" * 80)
    
    for package in packages:
        init_file = os.path.join(package, '__init__.py')
        
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                package_name = package.split('/')[-1].replace('_', ' ').title()
                f.write(f'"""{package_name} package."""\n')
            print(f"✓ Created: {init_file}")
        else:
            print(f"⊙ Already exists: {init_file}")
    
    print("=" * 80)
    print("✓ All __init__.py files created!\n")


def create_gitkeep_files():
    """Create .gitkeep files in empty directories."""
    
    empty_dirs = [
        'data/raw/plantvillage/train',
        'data/raw/plantvillage/val',
        'data/raw/plantvillage/test',
        'data/processed',
        'data/augmented',
        'models/saved_models',
        'models/checkpoints',
        'reports/figures',
        'reports/results',
        'logs'
    ]
    
    print("Creating .gitkeep files...")
    print("=" * 80)
    
    for directory in empty_dirs:
        gitkeep_file = os.path.join(directory, '.gitkeep')
        
        if not os.path.exists(gitkeep_file):
            Path(gitkeep_file).touch()
            print(f"✓ Created: {gitkeep_file}")
    
    print("=" * 80)
    print("✓ All .gitkeep files created!\n")


def verify_setup():
    """Verify that all essential files and directories exist."""
    
    print("Verifying project setup...")
    print("=" * 80)
    
    # Check essential directories
    essential_dirs = [
        'data/raw/plantvillage',
        'models/saved_models',
        'src',
        'streamlit_app',
        'scripts',
        'config'
    ]
    
    missing_dirs = []
    for directory in essential_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}")
        else:
            print(f"✗ {directory} - MISSING")
            missing_dirs.append(directory)
    
    print("=" * 80)
    
    if missing_dirs:
        print("⚠️  Some directories are missing!")
        return False
    else:
        print("✓ All essential directories verified!\n")
        return True


def print_next_steps():
    """Print instructions for next steps."""
    
    print("\n" + "=" * 80)
    print("PROJECT SETUP COMPLETE!")
    print("=" * 80)
    
    print("\n📋 Next Steps:\n")
    
    print("1. Place your dataset in the correct structure:")
    print("   data/raw/plantvillage/")
    print("   ├── train/   (place 3,627 training images here)")
    print("   ├── val/     (place 373 validation images here)")
    print("   └── test/    (place 1,404 test images here)")
    
    print("\n2. Verify your dataset:")
    print("   python scripts/verify_dataset.py")
    
    print("\n3. Install dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n4. Train your model:")
    print("   python scripts/train_model.py --model-type xgboost --augment")
    
    print("\n5. Launch the Streamlit app:")
    print("   streamlit run streamlit_app/app.py")
    
    print("\n" + "=" * 80)
    print("For detailed instructions, see:")
    print("  - README.md")
    print("  - MANUAL_DATASET_SETUP.md")
    print("  - PROJECT_SUMMARY.md")
    print("=" * 80 + "\n")


def main():
    """Main setup function."""
    
    print("\n" + "=" * 80)
    print("Sri Lanka Crop Disease Classification - Project Setup")
    print("=" * 80 + "\n")
    
    # Create directory structure
    create_directory_structure()
    
    # Create __init__.py files
    create_init_files()
    
    # Create .gitkeep files
    create_gitkeep_files()
    
    # Verify setup
    success = verify_setup()
    
    if success:
        # Print next steps
        print_next_steps()
        print("✅ Setup completed successfully!")
    else:
        print("❌ Setup completed with warnings. Please check missing directories.")
    
    print("\n🚀 You're ready to start!")


if __name__ == "__main__":
    main()