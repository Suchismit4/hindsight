"""
Built-in data configurations for Hindsight.

This module provides standard, academically-relevant data loading configurations
that can be used as-is or extended for research purposes.
"""

from pathlib import Path
from typing import Dict, List
from src.data.managers.config_schema import ConfigLoader, DataConfig

# Path to built-in configurations
CONFIGS_DIR = Path(__file__).parent

def get_available_configs() -> List[str]:
    """
    Get list of available built-in configurations.
    
    Returns:
        List of configuration names (without .yaml extension)
    """
    yaml_files = CONFIGS_DIR.glob("*.yaml")
    return [f.stem for f in yaml_files]

def load_builtin_config(config_name: str) -> DataConfig:
    """
    Load a built-in configuration by name.
    
    Args:
        config_name: Name of the configuration (without .yaml extension)
        
    Returns:
        Parsed DataConfig object
        
    Raises:
        FileNotFoundError: If the configuration doesn't exist
        
    Example:
        >>> config = load_builtin_config("equity_standard")
        >>> # Customize the config if needed
        >>> config.start_date = "2015-01-01"
    """
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        available = get_available_configs()
        raise FileNotFoundError(
            f"Built-in configuration '{config_name}' not found. "
            f"Available configs: {available}"
        )
    
    return ConfigLoader.load_from_yaml(config_path)

# Standard configurations
EQUITY_STANDARD = "equity_standard"

__all__ = [
    'get_available_configs',
    'load_builtin_config', 
    'EQUITY_STANDARD'
] 