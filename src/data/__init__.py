# src/data/__init__.py
import os
from pathlib import Path
from .registry import data_loader_registry
from . import loaders

def initialize_cache_directories():
    """
    Initialize cache directories for all registered data loaders.
    
    This function iterates over all registered data loaders, constructs their
    corresponding cache paths based on their registry paths, and ensures that
    the necessary directories exist.
    """
    # Define the cache root path (consistent with BaseDataSource)
    cache_root = os.path.expanduser(os.path.join('~', 'data', 'cache'))

    for registry_path in data_loader_registry.keys():
        # Build the cache path similar to BaseDataSource.get_cache_path()
        cache_path = os.path.join(cache_root, registry_path.lstrip('/'))
        # Add '.parquet' extension as in get_cache_path()
        cache_file_path = f"{cache_path}.parquet"
        cache_file = Path(cache_file_path)
        cache_dir = cache_file.parent

        # Create the cache directory if it doesn't exist
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)

# Run the initialization function when the module is imported
initialize_cache_directories()

from .manager import DataManager