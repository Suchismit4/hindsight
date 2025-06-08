"""
Data management system for coordinated data operations in Hindsight.

This module provides a centralized management system for data operations,
including:

1. Loading data from multiple sources
2. Applying transformations and processors
3. Filtering and selecting data subsets
4. Coordinating caching and persistence

The DataManager class serves as the primary interface for users,
providing a high-level API that abstracts away the complexities of
the underlying data infrastructure.
"""

from .data_manager import DataManager, CharacteristicsManager
from .config_schema import (
    DataConfig, 
    DataSourceConfig, 
    ConfigLoader, 
    DataConfigBuilder
)

__all__ = [
    'DataManager',
    'CharacteristicsManager', 
    'DataConfig',
    'DataSourceConfig',
    'ConfigLoader',
    'DataConfigBuilder'
]
