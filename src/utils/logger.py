"""Logging utilities for the project."""

import os
import sys
from pathlib import Path
from loguru import logger
from .config import get_config


def setup_logger():
    """Setup logger with file and console handlers."""
    config = get_config()
    log_level = config.get('logging.level', 'INFO')
    log_dir = config.get('logging.log_dir', 'logs')
    log_file = config.get('logging.log_file', 'crop_disease_classification.log')
    
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Remove default logger
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Add file handler
    logger.add(
        os.path.join(log_dir, log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    return logger


# Initialize logger
log = setup_logger()


def get_logger():
    """Get logger instance."""
    return log