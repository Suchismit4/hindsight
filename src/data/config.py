import os
from typing import Optional

def get_cache_config() -> dict:
    """Get configuration from environment variables."""
    return {
        'cache_root': os.getenv('HINDSIGHT_CACHE_ROOT'),
        'max_cache_size': os.getenv('HINDSIGHT_MAX_CACHE_SIZE'),
        'cache_ttl': os.getenv('HINDSIGHT_CACHE_TTL')
    } 