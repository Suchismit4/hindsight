"""
Data filters for DataFrames.

This module provides filters and filter utilities for pandas DataFrames.
"""

from typing import Dict, Any, List, Union

import pandas as pd
from src.data.processors.registry import Registry

# Create a registry for DataFrame filters
filter_registry = Registry[pd.DataFrame, pd.DataFrame]("filter_registry")

# Import all the filters and utility functions
from src.data.filters.filters import (
    # Filter registry
    filter_registry,
    
    # Core filter implementations
    equality_filter,
    comparison_filter,
    in_filter,
    not_in_filter,
    range_filter,
    date_range_filter,
    
    # Django-style filter utilities
    parse_django_style_filters,
    DJANGO_FILTER_SUFFIX_MAP,
    
    # Filter application function
    apply_filters
)

# Public API
__all__ = [
    # Registry
    'filter_registry',
    
    # Filter functions
    'equality_filter',
    'comparison_filter',
    'in_filter',
    'not_in_filter',
    'range_filter',
    'date_range_filter',
    
    # Django-style filter utilities
    'parse_django_style_filters',
    'DJANGO_FILTER_SUFFIX_MAP',
    
    # Filter application
    'apply_filters'
]

def create_filter(df: pd.DataFrame, filters: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Convenience function to apply filters to a DataFrame.
    
    This function handles both Django-style filter dictionaries and explicit filter configurations.
    
    Args:
        df: DataFrame to filter
        filters: Either a Django-style filter dictionary or a list of filter configurations
        
    Returns:
        Filtered DataFrame
    """
    if isinstance(filters, dict):
        filter_list = parse_django_style_filters(filters)
    else:
        filter_list = filters
        
    return apply_filters(df, filter_list) 