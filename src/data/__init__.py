"""
Data Module for Hindsight.

This module provides access to financial data from various sources, with support for:
- Loading data from different providers (WRDS, OpenBB, etc.)
- Caching data to avoid redundant loading
- Processing and transforming data with various operations
- Filtering and selecting data subsets

The module is designed around the DataManager class, which serves as the
primary interface for users to load and work with financial data.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any

# Import core components
from src.data.core import FrequencyType, TimeSeriesIndex
from src.data.core.provider import _PROVIDER_REGISTRY

# Import the main DataManager class
from src.data.managers.data_manager import DataManager

# Define public exports
__all__ = [
    'DataManager',
    'FrequencyType',
    'TimeSeriesIndex',
    'get_default_cache_root',
    'initialize_cache_directories'
]

def get_default_cache_root() -> str:
    """
    Get the default cache root directory path.
    
    Returns:
        Path to the default cache directory ('~/data/cache')
    """
    return os.path.expanduser(os.path.join('~', 'data', 'cache'))

def initialize_cache_directories(cache_root: Optional[str] = None) -> None:
    """
    Initialize cache directories for all registered data loaders.
    
    Creates the necessary directory structure for caching data from all
    registered data loaders.
    
    Args:
        cache_root: Optional custom cache directory. If None, uses default.
    """
    # Use provided cache_root or default
    cache_root = cache_root or get_default_cache_root()

    for provider in _PROVIDER_REGISTRY.values():
        for data_loader in provider.data_loaders.values():
            # Each data_loader is an instance of BaseDataSource
            # Build the cache directory path similar to BaseDataSource
            cache_path = os.path.join(
                cache_root,
                data_loader.data_path.strip('/')
            )
            cache_dir = Path(cache_path)
            # Create the cache directory if it doesn't exist
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True, exist_ok=True)

# Run the initialization function when the module is imported
initialize_cache_directories()
