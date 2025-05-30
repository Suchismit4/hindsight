"""
Core components for Hindsight data layer.

This module provides the foundational classes and utilities for the Hindsight
data layer, including:

1. Time series data manipulation through DateTimeAccessors
2. Frequency type definitions for data series
3. Time series indexing for efficient slicing and selection
4. Core operations for performing calculations on time series data
"""

from .struct import DatasetDateTimeAccessor
from .util import FrequencyType, TimeSeriesIndex, Loader, Rolling, prepare_for_jit, restore_from_jit
from .operations import TimeSeriesOps
from .cache import CacheManager
from .provider import Provider, register_provider, get_provider

# Define public exports
__all__ = [
    # Data structures and accessors
    'DatasetDateTimeAccessor',
    'FrequencyType',
    'TimeSeriesIndex',
    'Rolling',
    
    # Operations and utilities
    'TimeSeriesOps',
    'Loader',
    'prepare_for_jit',
    'restore_from_jit',

    # Infrastructure components
    'CacheManager',
    'Provider',
    'register_provider',
    'get_provider'
]
