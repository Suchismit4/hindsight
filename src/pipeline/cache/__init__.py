"""
Pipeline cache system for content-addressable hierarchical caching.

This module provides the infrastructure for caching pipeline artifacts at
different stages (L1-L4), with automatic dependency tracking and cache
invalidation.

The cache system uses content-addressable storage, where cache keys are
computed from the configuration and parent dependencies. This enables
automatic cache reuse when different pipeline specifications share common
stages.

Main Components:
- GlobalCacheManager: Main interface for cache operations
- CacheStage: Enumeration of cache levels (L1-L4)
- CacheMetadata: Metadata for cache entries
- MetadataManager: Metadata persistence and querying

Example:
    >>> from src.pipeline.cache import GlobalCacheManager, CacheStage
    >>> 
    >>> cache_mgr = GlobalCacheManager()
    >>> 
    >>> def load_data():
    ...     # Expensive data loading operation
    ...     return dataset
    >>> 
    >>> # Get or compute with automatic caching
    >>> data, key = cache_mgr.get_or_compute(
    ...     stage=CacheStage.L1_RAW,
    ...     config={"provider": "wrds", "dataset": "crsp"},
    ...     parent_keys=[],
    ...     compute_fn=load_data
    ... )
"""

from .manager import GlobalCacheManager
from .stages import CacheStage
from .metadata import CacheMetadata, MetadataManager

__all__ = [
    'GlobalCacheManager',
    'CacheStage',
    'CacheMetadata',
    'MetadataManager',
]

