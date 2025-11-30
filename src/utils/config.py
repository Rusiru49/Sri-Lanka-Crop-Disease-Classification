"""Configuration utilities for loading and managing project settings."""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the project."""

    def __init__(self):
        """
        Initialize configuration.
        Automatically detects the project root and loads config/config.yaml
        no matter where the script is executed (Streamlit, CLI, VS Code).
        """
        # Project root = 2 levels above this file
        self.project_root = Path(__file__).resolve().parents[2]

        # Absolute path to config.yaml
        self.config_path = self.project_root / "config/config.yaml"

        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key using dot notation.
        Example: get("data.batch_size")
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    # Shortcuts
    def get_data_config(self) -> Dict[str, Any]:
        return self.config.get("data", {})

    def get_model_config(self) -> Dict[str, Any]:
        return self.config.get("model", {})

    def get_training_config(self) -> Dict[str, Any]:
        return self.config.get("training", {})

    def get_augmentation_config(self) -> Dict[str, Any]:
        return self.config.get("augmentation", {})

    @property
    def image_size(self):
        return tuple(self.get("data.image_size", [224, 224]))

    @property
    def batch_size(self):
        return self.get("data.batch_size", 32)

    @property
    def random_seed(self):
        return self.get("data.random_seed", 42)


# Global instance — safe to import anywhere
config = Config()


def get_config() -> Config:
    """Return global configuration instance."""
    return config
