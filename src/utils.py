"""Utility functions for Wikipedia RAG project."""

import yaml
import logging
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    """Setup logging configuration."""
    log_level = getattr(logging, config['logging']['level'].upper())
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler()
        ]
    )


def create_directories(config: dict):
    """Create necessary directories based on config."""
    dirs = [
        config['data']['processed_dir'],
        config['data']['embeddings_dir'],
        "logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)