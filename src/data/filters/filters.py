"""
DataFrame filters implementation.

This module provides a comprehensive set of filters for pandas DataFrames,
along with utilities for parsing different filter formats. The filters are
registered in a central registry for extensibility.

The module supports two primary filter formats:
1. Explicit filter configurations (lists of filter dictionaries)
2. Django-style filter syntax (more concise dictionary format)

Each filter function returns a filtered DataFrame, allowing them to be
chained together for complex filtering operations.
"""

import pandas as pd
from typing import Dict, Any, List, Tuple, Union, Optional
from src.data.processors.registry import Registry

# Create a filter registry
filter_registry = Registry[pd.DataFrame, pd.DataFrame]("filter_registry")

# Mapping of Django-style suffixes to filter types and operators
DJANGO_FILTER_SUFFIX_MAP = {
    "eq": ("equality_filter", None),
    "ne": ("comparison_filter", "!="),
    "gt": ("comparison_filter", ">"),
    "gte": ("comparison_filter", ">="),
    "lt": ("comparison_filter", "<"),
    "lte": ("comparison_filter", "<="),
    "in": ("in_filter", None),
    "nin": ("not_in_filter", None),
    "range": ("range_filter", None),
    "date_range": ("date_range_filter", None)
}

# ===============================================================================
# Core Filter Functions
# ===============================================================================

@filter_registry
def equality_filter(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply equality filter to a DataFrame.
    
    Selects rows where a column equals a specific value.
    
    Args:
        df: The DataFrame to filter
        params: Parameters with keys:
            - column: The column to filter on
            - value: The value to filter for
            
    Returns:
        Filtered DataFrame
        
    Raises:
        ValueError: If required parameters are missing
    """
    column = params.get("column")
    value = params.get("value")
    
    if column is None or value is None:
        raise ValueError("Both 'column' and 'value' must be provided for equality_filter.")
    
    pre_filter_len = len(df)
    df = df[df[column] == value]
    post_filter_len = len(df)
    
    print(f"Filter {column}={value}: {pre_filter_len} -> {post_filter_len} rows")
    return df

@filter_registry
def comparison_filter(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply comparison filter to a DataFrame.
    
    Selects rows where a column satisfies a comparison with a specific value.
    
    Args:
        df: The DataFrame to filter
        params: Parameters with keys:
            - column: The column to filter on
            - operator: One of '=', '==', '!=', '>', '>=', '<', '<='
            - value: The value to compare against
            
    Returns:
        Filtered DataFrame
        
    Raises:
        ValueError: If required parameters are missing or operator is unsupported
    """
    column = params.get("column")
    operator = params.get("operator")
    value = params.get("value")
    
    if column is None or operator is None or value is None:
        raise ValueError("'column', 'operator', and 'value' must be provided for comparison_filter.")
    
    pre_filter_len = len(df)
    
    if operator == '=' or operator == '==':
        df = df[df[column] == value]
    elif operator == '!=':
        df = df[df[column] != value]
    elif operator == '>':
        df = df[df[column] > value]
    elif operator == '>=':
        df = df[df[column] >= value]
    elif operator == '<':
        df = df[df[column] < value]
    elif operator == '<=':
        df = df[df[column] <= value]
    else:
        raise ValueError(f"Unsupported operator '{operator}' in filter for column '{column}'.")
    
    post_filter_len = len(df)
    print(f"Filter {column} {operator} {value}: {pre_filter_len} -> {post_filter_len} rows")
    return df

@filter_registry
def in_filter(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply an 'in' filter to a DataFrame.
    
    Selects rows where a column's value is in a list of values.
    
    Args:
        df: The DataFrame to filter
        params: Parameters with keys:
            - column: The column to filter on
            - values: List of values to check for
            
    Returns:
        Filtered DataFrame
        
    Raises:
        ValueError: If required parameters are missing
    """
    column = params.get("column")
    values = params.get("values")
    
    if column is None or values is None:
        raise ValueError("Both 'column' and 'values' must be provided for in_filter.")
    
    pre_filter_len = len(df)
    df = df[df[column].isin(values)]
    post_filter_len = len(df)
    
    print(f"Filter {column} in {values}: {pre_filter_len} -> {post_filter_len} rows")
    return df

@filter_registry
def not_in_filter(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply a 'not in' filter to a DataFrame.
    
    Selects rows where a column's value is not in a list of values.
    
    Args:
        df: The DataFrame to filter
        params: Parameters with keys:
            - column: The column to filter on
            - values: List of values to exclude
            
    Returns:
        Filtered DataFrame
        
    Raises:
        ValueError: If required parameters are missing
    """
    column = params.get("column")
    values = params.get("values")
    
    if column is None or values is None:
        raise ValueError("Both 'column' and 'values' must be provided for not_in_filter.")
    
    pre_filter_len = len(df)
    df = df[~df[column].isin(values)]
    post_filter_len = len(df)
    
    print(f"Filter {column} not in {values}: {pre_filter_len} -> {post_filter_len} rows")
    return df

@filter_registry
def range_filter(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply a range filter to a DataFrame.
    
    Selects rows where a column's value is within a specified range.
    Both min and max are inclusive boundaries.
    
    Args:
        df: The DataFrame to filter
        params: Parameters with keys:
            - column: The column to filter on
            - min_value: Minimum value (inclusive, optional)
            - max_value: Maximum value (inclusive, optional)
            
    Returns:
        Filtered DataFrame
        
    Raises:
        ValueError: If column is missing or both min and max are missing
    """
    column = params.get("column")
    min_value = params.get("min_value")
    max_value = params.get("max_value")
    
    if column is None:
        raise ValueError("'column' must be provided for range_filter.")
    
    if min_value is None and max_value is None:
        raise ValueError("At least one of 'min_value' or 'max_value' must be provided for range_filter.")
    
    pre_filter_len = len(df)
    
    if min_value is not None and max_value is not None:
        df = df[(df[column] >= min_value) & (df[column] <= max_value)]
    elif min_value is not None:
        df = df[df[column] >= min_value]
    elif max_value is not None:
        df = df[df[column] <= max_value]
    
    post_filter_len = len(df)
    range_str = f"[{min_value or ''}, {max_value or ''}]"
    print(f"Filter {column} in range {range_str}: {pre_filter_len} -> {post_filter_len} rows")
    return df

@filter_registry
def date_range_filter(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply date range filter to a DataFrame.
    
    Selects rows where a date column's value falls within a specified date range.
    Both start and end dates are inclusive boundaries.
    
    Args:
        df: The DataFrame to filter
        params: Parameters with keys:
            - date_column: The date column to filter on (defaults to 'date')
            - start_date: The start date (inclusive, optional)
            - end_date: The end date (inclusive, optional)
            
    Returns:
        Filtered DataFrame
        
    Note:
        If neither start_date nor end_date is provided, returns the original DataFrame
    """
    date_column = params.get("date_column", "date")
    start_date = params.get("start_date")
    end_date = params.get("end_date")
    
    if start_date is None and end_date is None:
        return df
    
    pre_filter_len = len(df)
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df = df[df[date_column] >= start_date]
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df = df[df[date_column] <= end_date]
    
    post_filter_len = len(df)
    print(f"Date range filter {date_column} [{start_date} to {end_date}]: {pre_filter_len} -> {post_filter_len} rows")
    return df

# ===============================================================================
# Filter Parsing and Application
# ===============================================================================

def parse_django_style_filters(filters_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse Django-style filters into filter configurations.
    
    Converts a user-friendly Django-style filter dictionary into the explicit
    filter configuration format required by the filter functions.
    
    Supported formats:
    - column: value                  -> equality_filter
    - column__eq: value              -> equality_filter
    - column__ne: value              -> comparison_filter (!=)
    - column__gt: value              -> comparison_filter (>)
    - column__gte: value             -> comparison_filter (>=)
    - column__lt: value              -> comparison_filter (<)
    - column__lte: value             -> comparison_filter (<=)
    - column__in: [values]           -> in_filter
    - column__nin: [values]          -> not_in_filter
    - column__range: [min, max]      -> range_filter
    - column__date_range: [start, end] -> date_range_filter
    
    Args:
        filters_dict: Dictionary of Django-style filters
        
    Returns:
        List of filter configurations ready for apply_filters
        
    Examples:
        >>> parse_django_style_filters({"ticker": "AAPL"})
        [{"type": "equality_filter", "column": "ticker", "value": "AAPL"}]
        
        >>> parse_django_style_filters({"price__gte": 150})
        [{"type": "comparison_filter", "column": "price", "operator": ">=", "value": 150}]
    """
    if not filters_dict:
        return []
    
    filter_configs = []
    
    for key, value in filters_dict.items():
        # Check if key contains a double underscore separator for Django-style filters
        parts = key.split('__')
        column = parts[0]
        
        # Simple equality check without double underscore (e.g., "ticker": "AAPL")
        if len(parts) == 1:
            filter_configs.append({
                "type": "equality_filter",
                "column": column,
                "value": value
            })
            continue
        
        # Extract suffix (e.g., "gte" from "price__gte")
        suffix = parts[1]
        
        # Check suffix against mapping
        if suffix not in DJANGO_FILTER_SUFFIX_MAP:
            raise ValueError(f"Unsupported filter suffix '{suffix}' in '{key}'.")
        
        filter_type, operator = DJANGO_FILTER_SUFFIX_MAP[suffix]
        
        # Handle specific filter types
        if filter_type == "equality_filter":
            filter_configs.append({
                "type": filter_type,
                "column": column,
                "value": value
            })
        elif filter_type == "comparison_filter":
            filter_configs.append({
                "type": filter_type,
                "column": column,
                "operator": operator,
                "value": value
            })
        elif filter_type == "in_filter" or filter_type == "not_in_filter":
            if not isinstance(value, (list, tuple)):
                value = [value]  # Convert single value to list
            filter_configs.append({
                "type": filter_type,
                "column": column,
                "values": value
            })
        elif filter_type == "range_filter":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"Range filter for '{key}' must provide a list or tuple with two values: [min, max].")
            filter_configs.append({
                "type": filter_type,
                "column": column,
                "min_value": value[0],
                "max_value": value[1]
            })
        elif filter_type == "date_range_filter":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"Date range filter for '{key}' must provide a list or tuple with two values: [start_date, end_date].")
            filter_configs.append({
                "type": filter_type,
                "date_column": column,
                "start_date": value[0],
                "end_date": value[1]
            })
    
    return filter_configs

def apply_filters(df: pd.DataFrame, filters_config: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Apply a series of filters to a DataFrame.
    
    Processes a DataFrame through multiple filter operations, applying each filter
    in sequence. The filters are applied in the order they appear in the list.
    
    Args:
        df: The DataFrame to filter
        filters_config: List of filter configurations, each containing:
            - type: The name of the filter function to use
            - Additional parameters specific to the filter type
    
    Returns:
        Filtered DataFrame
        
    Raises:
        ValueError: If a filter configuration is invalid or a filter is not found
        
    Example:
        >>> filters = [
        ...     {"type": "equality_filter", "column": "ticker", "value": "AAPL"},
        ...     {"type": "comparison_filter", "column": "price", "operator": ">=", "value": 150}
        ... ]
        >>> filtered_df = apply_filters(df, filters)
    """
    if not filters_config:
        print("WARNING: No filters provided, returning original DataFrame.")
        return df
    
    # Create a copy to avoid modifying the original DataFrame
    result = df.copy()
    
    for filter_config in filters_config:
        filter_type = filter_config.get("type")
        
        if not filter_type:
            raise ValueError("Filter configuration must include 'type' key")
        
        # Extract parameters for the filter (excluding 'type')
        params = {k: v for k, v in filter_config.items() if k != "type"}
        
        # Get the filter function from the registry
        filter_func = filter_registry.get(filter_type)
        if not filter_func:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # Apply the filter
        result = filter_func(result, params)
        print(f"Filter {filter_type} applied: {len(result)} rows")
    return result 