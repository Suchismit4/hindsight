# src/data/__init__.py

import os
from pathlib import Path
from typing import Optional

from .provider import _PROVIDER_REGISTRY

def get_default_cache_root() -> str:
    """Returns the default cache root directory."""
    return os.path.expanduser(os.path.join('~', 'data', 'cache'))

def initialize_cache_directories(cache_root: Optional[str] = None):
    """
    Initialize cache directories for all registered data loaders.

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

from .manager import DataManager
