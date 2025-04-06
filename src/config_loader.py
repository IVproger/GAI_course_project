# src/config_loader.py
import yaml
from types import SimpleNamespace
import os

def load_config(config_path):
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    # Optionally convert to SimpleNamespace for attribute access (config.model_name)
    # Or just return the dictionary
    return config_dict # Or SimpleNamespace(**config_dict)
