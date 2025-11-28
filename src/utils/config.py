"""Configuration utilities for loading and managing project settings."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the project."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports nested keys with dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config.get('data', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get('training', {})
    
    def get_augmentation_config(self) -> Dict[str, Any]:
        """Get augmentation configuration."""
        return self.config.get('augmentation', {})
    
    @property
    def image_size(self):
        """Get image size tuple."""
        return tuple(self.get('data.image_size', [224, 224]))
    
    @property
    def batch_size(self):
        """Get batch size."""
        return self.get('data.batch_size', 32)
    
    @property
    def random_seed(self):
        """Get random seed."""
        return self.get('data.random_seed', 42)


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance."""
    return config