"""
Hindsight: Financial Data Analysis Library

Hindsight is a comprehensive library for processing, analyzing, and visualizing 
financial data from various sources. It provides tools for:

1. Data loading from multiple sources (WRDS, OpenBB, etc.)
2. Data transformation and filtering
3. Advanced time-series analysis
4. Financial calculations and metrics

The library uses xarray as its primary data structure, which allows for efficient
handling of multi-dimensional financial data with labeled dimensions.
"""

# Version information
__version__ = "0.0.21a"
__author__ = "Hindsight Team"

# Core components for public use
from src.data import DataManager
from src.data.core import FrequencyType
from src.data.core.util import TimeSeriesIndex

# Define what is available when using "from src import *"
__all__ = [
    'DataManager',
    'FrequencyType',
    'TimeSeriesIndex'
]