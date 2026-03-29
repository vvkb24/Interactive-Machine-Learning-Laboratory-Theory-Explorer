"""
Configuration management for SyllabusML.
Loads YAML configurations using Hydra and provides a unified interface.
"""
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
import os
from pathlib import Path

def get_config(config_name: str = "config") -> DictConfig:
    """
    Initializes Hydra and composes the configuration.
    
    Returns:
        DictConfig: The loaded configuration dictionary.
    """
    # Get the absolute path to the configs directory
    # Expected structure: src/syllabusml/config.py, configs is in root
    base_path = Path(__file__).resolve().parent.parent.parent
    config_path = base_path / "configs"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration directory not found at {config_path}")

    with initialize_config_dir(version_base=None, config_dir=str(config_path)):
        cfg = compose(config_name=config_name)
    return cfg

if __name__ == "__main__":
    # Test loading
    cfg = get_config()
    print(OmegaConf.to_yaml(cfg))
