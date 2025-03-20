"""
Data filters for DataFrames.

This module provides a filtering system for pandas DataFrames, allowing users to:

1. Apply predefined filters (equality, comparison, range, etc.)
2. Use Django-style filter syntax for more intuitive filtering
3. Chain multiple filters together for complex conditions

The filtering system is designed to be extensible, with a registry for filter functions
and utilities for parsing different filter formats.

Example:
    # Using explicit filter configurations
    filters = [
        {"type": "equality", "column": "ticker", "value": "AAPL"},
        {"type": "comparison", "column": "price", "operator": ">=", "value": 150}
    ]
    filtered_df = apply_filters(df, filters)
    
    # Using Django-style filter syntax
    django_filters = {
        "ticker": "AAPL",
        "price__gte": 150
    }
    filtered_df = create_filter(df, django_filters)
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
    'apply_filters',
    'create_filter'
]

def create_filter(df: pd.DataFrame, filters: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Apply filters to a DataFrame using either Django-style or explicit configurations.
    
    This convenience function handles both filter formats, making it easier to apply
    filters without having to determine the format first.
    
    Args:
        df: DataFrame to filter
        filters: Either:
            - Django-style filter dictionary (e.g., {"column__gte": value})
            - List of explicit filter configurations
        
    Returns:
        Filtered DataFrame
        
    Examples:
        >>> # Django-style filtering
        >>> create_filter(df, {"ticker": "AAPL", "price__gte": 150})
        
        >>> # Explicit filter configuration
        >>> create_filter(df, [
        ...     {"type": "equality", "column": "ticker", "value": "AAPL"},
        ...     {"type": "comparison", "column": "price", "operator": ">=", "value": 150}
        ... ])
    """
    if isinstance(filters, dict):
        filter_list = parse_django_style_filters(filters)
    else:
        filter_list = filters
        
    return apply_filters(df, filter_list) 