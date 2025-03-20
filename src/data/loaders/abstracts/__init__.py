"""
Abstract base classes for data loaders in Hindsight.

This module defines the abstract interfaces that all data loaders must implement,
providing a consistent API for data retrieval regardless of the data source.

Key components:
1. BaseDataSource: Abstract base class that defines the core interface for all data sources
2. DataLoader-specific interfaces: Specialized abstract classes for different types of data

These abstractions enable Hindsight to support multiple data providers while maintaining
a consistent interface for data consumption.
"""

from .base import BaseDataSource

__all__ = ['BaseDataSource']
