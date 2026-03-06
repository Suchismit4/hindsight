"""
Data processors for xarray datasets.

This module provides post-processors for transforming xarray datasets. Each processor
implements a specific financial data transformation, including:

1. Coordinate handling (e.g., setting PERMNO, PERMCO as coordinates)
2. Financial calculations and corrections (e.g., preferred stock, market equity)

The module defines both traditional and Django-style processor configurations.
Django-style configuration uses a more user-friendly format with shortcuts
for common processor operations.

Example:
    # Traditional format
    processor_config = [
        {"proc": "set_permno", "options": {}},
        {"proc": "fix_mke", "options": {}}
    ]
    
    # Django-style format
    processor_config = {
        "set_permno_coord": True,
        "fix_market_equity": True
    }
"""

from typing import Dict, Any, List, Union, Optional, TypeVar, Callable, Sequence

import xarray as xr
import pandas as pd
from src.data.processors.registry import Registry, post_processor

# Import processors (excluding merge-related ones which are now handled at DataFrame level)
from src.data.processors.processors import (
    # Coordinate-related processors
    set_permno,
    set_permco,
    
    # Financial data processors
    ps,
    fix_mke
)

# Type definitions
ProcessorFunc = Callable[[xr.Dataset, Dict[str, Any]], xr.Dataset]
ProcessorConfig = Dict[str, Any]
ProcessorsList = List[ProcessorConfig]
ProcessorsDictConfig = Dict[str, Union[bool, Dict[str, Any], List[Dict[str, Any]]]]

# Define processor configuration shortcuts (Django-style to traditional mapping)
# NOTE: Merge-related processors have been removed. External table merging now happens
# at the DataFrame level before xarray conversion. See GenericWRDSDataLoader.
PROCESSOR_SHORTCUTS = {
    # Coordinate handling
    "set_permno_coord": {
        "proc": "set_permno",
        "options_mapping": {}
    },
    "set_permco_coord": {
        "proc": "set_permco",
        "options_mapping": {}
    },
    
    # Financial calculations
    "fix_market_equity": {
        "proc": "fix_mke",
        "options_mapping": {}
    },
    "preferred_stock": {
        "proc": "ps",
        "options_mapping": {}
    }
}

# Public API
__all__ = [
    # Registry
    'post_processor',
    'Registry',
    
    # Processors
    'set_permno',
    'set_permco',
    'ps',
    'fix_mke',
    
    # Helper functions
    'apply_processors',
    'parse_processors_config',
    
    # Constants and types
    'PROCESSOR_SHORTCUTS',
    'ProcessorFunc',
    'ProcessorConfig',
    'ProcessorsList',
    'ProcessorsDictConfig'
]


def _parse_transform_item(transform_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a single transform item from the nested 'transforms' list.
    
    Converts the user-friendly transform configuration to the internal processor format.
    
    Args:
        transform_config: Dictionary with transform configuration including 'type' field
        
    Returns:
        Standard processor configuration dictionary
        
    Raises:
        ValueError: If transform type is unknown
    """
    transform_type = transform_config.get("type")
    if not transform_type:
        raise ValueError("Transform item must have a 'type' field")
    
    # Map transform types to processor names
    if transform_type == "set_coordinates":
        coord_type = transform_config.get("coord_type", "permno")
        if coord_type == "permno":
            return {"proc": "set_permno", "options": {}}
        elif coord_type == "permco":
            return {"proc": "set_permco", "options": {}}
        else:
            raise ValueError(f"Unknown coord_type: {coord_type}")
    elif transform_type == "fix_market_equity":
        return {"proc": "fix_mke", "options": {}}
    elif transform_type == "preferred_stock":
        return {"proc": "ps", "options": {}}
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


def parse_processors_config(processors_dict: ProcessorsDictConfig) -> ProcessorsList:
    """
    Parse a Django-style processors dictionary into the standard format.
    
    Converts the user-friendly configuration to the internal processors format.
    Supports both flat shortcut keys and nested 'transforms' structure.
    
    NOTE: 'merges' are no longer handled here. External table merging now happens
    at the DataFrame level before xarray conversion. See GenericWRDSDataLoader.
    
    Examples:
        >>> # Simple processor with no options
        >>> parse_processors_config({"set_permno_coord": True})
        [{"proc": "set_permno", "options": {}}]
        
        >>> # Nested transforms structure
        >>> parse_processors_config({
        ...     "transforms": [
        ...         {"type": "set_coordinates", "coord_type": "permno"},
        ...         {"type": "fix_market_equity"}
        ...     ]
        ... })
        [
            {"proc": "set_permno", "options": {}},
            {"proc": "fix_mke", "options": {}}
        ]
    
    Args:
        processors_dict: Dictionary of Django-style processor configurations
        
    Returns:
        List of standard processor configurations
        
    Raises:
        ValueError: If an unknown processor shortcut is provided or a processor 
                    value is not a boolean, dictionary, or list
    """
    if not processors_dict:
        return []
    
    processors_list = []
    
    for key, value in processors_dict.items():
        # Handle nested 'transforms' list
        if key == "transforms":
            if not isinstance(value, list):
                raise ValueError("'transforms' must be a list of transform configurations")
            for transform_config in value:
                if not isinstance(transform_config, dict):
                    raise ValueError("Each transform item must be a dictionary")
                processors_list.append(_parse_transform_item(transform_config))
            continue
        
        # Handle standard shortcuts
        if key not in PROCESSOR_SHORTCUTS:
            raise ValueError(f"Unknown processor shortcut: {key}")
        
        shortcut = PROCESSOR_SHORTCUTS[key]
        proc_name = shortcut["proc"]
        
        # Handle simple flag case (e.g., "set_permno_coord": True)
        if value is True:
            processors_list.append({
                "proc": proc_name,
                "options": {}
            })
        # Handle dictionary of options (currently none of our processors need options)
        elif isinstance(value, dict):
            processors_list.append({
                "proc": proc_name,
                "options": value
            })
        else:
            raise ValueError(f"Invalid value for processor {key}: {value}. "
                           f"Expected True or a dictionary of options.")
    
    return processors_list


def apply_processors(ds: xr.Dataset, processors: Union[ProcessorsList, ProcessorsDictConfig]) -> Union[xr.Dataset, List]:
    """
    Apply a series of post-processors to an xarray Dataset.
    
    Processes an xarray Dataset through a sequence of processors, which transform
    the data according to their respective functions. Supports both traditional
    and Django-style processor configurations.
    
    Args:
        ds: Dataset to process
        processors: Either:
            - List of processor configurations (traditional format)
            - Dictionary of Django-style processor configurations
        
    Returns:
        Tuple of (processed Dataset, list of applied processors)
        
    Raises:
        ValueError: If a processor configuration is invalid or a processor is not found
        
    Example:
        >>> # Traditional format
        >>> processors_list = [
        ...     {"proc": "set_permno", "options": {}},
        ...     {"proc": "fix_mke", "options": {}}
        ... ]
        >>> processed_ds, applied = apply_processors(ds, processors_list)
        
        >>> # Django-style format
        >>> processors_dict = {
        ...     "set_permno_coord": True,
        ...     "fix_market_equity": True
        ... }
        >>> processed_ds, applied = apply_processors(ds, processors_dict)
    """
    if not processors:
        return ds, []
        
    # Make a defensive copy to avoid modifying the original dataset
    result = ds.copy()
    applied_postprocessors = []
    
    # Parse Django-style processors if needed
    if isinstance(processors, dict):
        processors_list = parse_processors_config(processors)
    else:
        processors_list = processors
    
    for processor_config in processors_list:
        proc_name = processor_config.get("proc")
        options = processor_config.get("options", {})
                
        if not proc_name:
            raise ValueError("Processor configuration must include 'proc' key")
            
        processor_func = post_processor.get(proc_name)
        if not processor_func:
            raise ValueError(f"Unknown processor: {proc_name}")
        
        result = processor_func(result, options)
        applied_postprocessors.append(processor_config)
        
    return result, applied_postprocessors
