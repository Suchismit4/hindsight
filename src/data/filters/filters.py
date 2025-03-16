# src/data/filters/filters.py

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

@filter_registry
def equality_filter(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply equality filter to a DataFrame.
    
    Args:
        df: The DataFrame to filter
        params: Parameters with keys:
            - column: The column to filter on
            - value: The value to filter for
            
    Returns:
        Filtered DataFrame
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
    
    Args:
        df: The DataFrame to filter
        params: Parameters with keys:
            - column: The column to filter on
            - operator: One of '=', '==', '!=', '>', '>=', '<', '<='
            - value: The value to compare against
            
    Returns:
        Filtered DataFrame
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
    
    Args:
        df: The DataFrame to filter
        params: Parameters with keys:
            - column: The column to filter on
            - values: List of values to check for
            
    Returns:
        Filtered DataFrame
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
    
    Args:
        df: The DataFrame to filter
        params: Parameters with keys:
            - column: The column to filter on
            - values: List of values to exclude
            
    Returns:
        Filtered DataFrame
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
    
    Args:
        df: The DataFrame to filter
        params: Parameters with keys:
            - column: The column to filter on
            - min_value: Minimum value (inclusive, optional)
            - max_value: Maximum value (inclusive, optional)
            
    Returns:
        Filtered DataFrame
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
    
    Args:
        df: The DataFrame to filter
        params: Parameters with keys:
            - date_column: The date column to filter on
            - start_date: The start date (inclusive)
            - end_date: The end date (inclusive)
            
    Returns:
        Filtered DataFrame
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

def parse_django_style_filters(filters_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse Django-style filters into filter configurations.
    
    Supports:
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
        List of filter configurations
    """
    if not filters_dict:
        return []
    
    filter_configs = []
    
    for key, value in filters_dict.items():
        # Parse the key to extract column and any suffix
        parts = key.split("__")
        
        if len(parts) == 1:
            # Simple column: value format (equality filter)
            column = parts[0]
            
            # Auto-detect list and treat as 'in' filter if it's a list but not for range
            if isinstance(value, (list, tuple)) and len(value) > 0:
                filter_configs.append({
                    "type": "in_filter",
                    "column": column,
                    "values": value
                })
            else:
                # Regular equality filter
                filter_configs.append({
                    "type": "equality_filter",
                    "column": column,
                    "value": value
                })
        else:
            # Django-style column__suffix format
            column = parts[0]
            suffix = parts[1]
            
            if suffix not in DJANGO_FILTER_SUFFIX_MAP:
                raise ValueError(f"Unsupported filter suffix '{suffix}' in '{key}'")
            
            filter_type, operator = DJANGO_FILTER_SUFFIX_MAP[suffix]
            
            # Create the appropriate filter config based on type
            if filter_type == "equality_filter":
                filter_configs.append({
                    "type": "equality_filter",
                    "column": column,
                    "value": value
                })
            elif filter_type == "comparison_filter":
                filter_configs.append({
                    "type": "comparison_filter",
                    "column": column,
                    "operator": operator,
                    "value": value
                })
            elif filter_type in ("in_filter", "not_in_filter"):
                # Convert single value to list if needed
                values = value if isinstance(value, (list, tuple)) else [value]
                filter_configs.append({
                    "type": filter_type,
                    "column": column,
                    "values": values
                })
            elif filter_type == "range_filter":
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise ValueError(f"Range filter for '{key}' expects a list/tuple of [min, max]")
                filter_configs.append({
                    "type": "range_filter",
                    "column": column,
                    "min_value": value[0] if value[0] is not None else None,
                    "max_value": value[1] if value[1] is not None else None
                })
            elif filter_type == "date_range_filter":
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise ValueError(f"Date range filter for '{key}' expects a list/tuple of [start_date, end_date]")
                filter_configs.append({
                    "type": "date_range_filter",
                    "date_column": column,
                    "start_date": value[0] if value[0] is not None else None,
                    "end_date": value[1] if value[1] is not None else None
                })
    
    return filter_configs
                
def apply_filters(df: pd.DataFrame, filters_config: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Apply a series of filters to a DataFrame.
    
    Args:
        df: The DataFrame to filter
        filters_config: List of filter configurations
        
    Returns:
        Filtered DataFrame
    """
    if not filters_config:
        return df
    
    filtered_df = df.copy()
    
    for filter_config in filters_config:
        filter_type = filter_config.get("type")
        if filter_type is None:
            raise ValueError("Filter type must be specified in filter config.")
        
        filter_func = filter_registry.get(filter_type)
        if filter_func is None:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        filtered_df = filter_func(filtered_df, filter_config)
    
    return filtered_df 