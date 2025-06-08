"""
Data processors for xarray datasets.

This module provides post-processors for transforming xarray datasets. Each processor
implements a specific financial data transformation, including:

1. Data merging and replacement operations
2. Coordinate handling (e.g., setting PERMNO, PERMCO as coordinates)
3. Financial calculations and corrections (e.g., preferred stock, market equity)

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
from src.data.core.util import Loader
from src.data.processors.registry import Registry, post_processor

# Import all processors
from src.data.processors.processors import (
    # Dataset transformation processors
    merge_2d_table,
    merge_4d_table,
    replace,
    
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
PROCESSOR_SHORTCUTS = {
    # Data replacement and merging
    "replace_values": {
        "proc": "replace",
        "options_mapping": {
            "source": "src",
            "from_var": "from",
            "to_var": "to",
            "identifier": "identifier",
            "rename": "rename"
        }
    },
    "merge_table": {
        "proc": "merge_2d_table",
        "options_mapping": {
            "source": "src",
            "axis": "ax1",
            "column": "ax2",
            "identifier": "identifier"
        }
    },
    "merge_4d_table": {
        "proc": "merge_4d_table",
        "options_mapping": {
            "source": "src",
            "variables": "variables",
            "identifier": "identifier",
            "time_column": "time_column",
            "rename": "rename"
        }
    },
    
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
    'merge_2d_table',
    'merge_4d_table',
    'replace',
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

def _map_options(options: Dict[str, Any], options_mapping: Dict[str, str]) -> Dict[str, Any]:
    """
    Maps options from user-friendly keys to internal option keys.
    
    Converts keys from the Django-style format to the format expected by processor functions.
    
    Args:
        options: Dictionary of user-provided options
        options_mapping: Mapping from user-friendly keys to internal keys
        
    Returns:
        Dictionary with mapped option keys
    """
    mapped_options = {}
    
    for option_key, option_value in options.items():
        mapped_key = options_mapping.get(option_key, option_key)
        mapped_options[mapped_key] = option_value
                
    return mapped_options

def parse_processors_config(processors_dict: ProcessorsDictConfig) -> ProcessorsList:
    """
    Parse a Django-style processors dictionary into the standard format.
    
    Converts the user-friendly configuration to the internal processors format.
    
    Examples:
        >>> # Simple processor with no options
        >>> parse_processors_config({"set_permno_coord": True})
        [{"proc": "set_permno", "options": {}}]
        
        >>> # Processor with options
        >>> parse_processors_config({
        ...     "merge_table": {
        ...         "source": "msenames",
        ...         "axis": "asset",
        ...         "column": "comnam"
        ...     }
        ... })
        [{"proc": "merge_2d_table", "options": {"external_ds": "msenames", "ax1": "asset", "ax2": "comnam"}}]
        
        >>> # Multiple processors
        >>> parse_processors_config({
        ...     "set_permno_coord": True,
        ...     "fix_market_equity": True
        ... })
        [{"proc": "set_permno", "options": {}}, {"proc": "fix_mke", "options": {}}]
        
        >>> # List of configurations for the same processor
        >>> parse_processors_config({
        ...     "merge_table": [
        ...         {"source": "msenames", "axis": "asset", "column": "comnam"},
        ...         {"source": "msenames", "axis": "asset", "column": "exchcd"}
        ...     ]
        ... })
        [
            {"proc": "merge_2d_table", "options": {"external_ds": "msenames", "ax1": "asset", "ax2": "comnam"}},
            {"proc": "merge_2d_table", "options": {"external_ds": "msenames", "ax1": "asset", "ax2": "exchcd"}}
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
        if key not in PROCESSOR_SHORTCUTS:
            raise ValueError(f"Unknown processor shortcut: {key}")
        
        shortcut = PROCESSOR_SHORTCUTS[key]
        proc_name = shortcut["proc"]
        options_mapping = shortcut["options_mapping"]
        
        # Handle simple flag case (e.g., "set_permno_coord": True)
        if value is True:
            processors_list.append({
                "proc": proc_name,
                "options": {}
            })
        # Handle list of configurations for the same processor
        elif isinstance(value, list):
            for config in value:
                if not isinstance(config, dict):
                    raise ValueError(f"Each item in the processor list for {key} must be a dictionary")
                mapped_options = _map_options(config, options_mapping)
                processors_list.append({
                    "proc": proc_name,
                    "options": mapped_options
                })
        # Handle dictionary of options
        elif isinstance(value, dict):
            mapped_options = _map_options(value, options_mapping)
            processors_list.append({
                "proc": proc_name,
                "options": mapped_options
            })
        else:
            raise ValueError(f"Invalid value for processor {key}: {value}. "
                           f"Expected True, a dictionary of options, or a list of option dictionaries.")
    
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
    
    print(f"DEBUG: Applying processors: {processors_list}")
    # quit()
    for processor_config in processors_list:
        proc_name = processor_config.get("proc")
        options = processor_config.get("options", {})
                
        if not proc_name:
            raise ValueError("Processor configuration must include 'proc' key")
        
        # If an external source is specified, load the external dataset.
        if "src" in options and isinstance(options["src"], str):
            external_identifier = options.get("identifier")
            if not external_identifier:
                raise ValueError("Postprocessor requires an 'identifier' for external source.")
            external_rename = options.get("rename")
            options["external_ds"] = Loader.load_external_proc_file(
                    options["src"],
                    external_identifier,
                    external_rename
            )
            
        processor_func = post_processor.get(proc_name)
        if not processor_func:
            raise ValueError(f"Unknown processor: {proc_name}")
        
        result = processor_func(result, options)
        
        # Remove the external_ds from options to avoid caching issues.
        if "external_ds" in options:
            del options["external_ds"]
        
        applied_postprocessors.append(processor_config)
        
    return result, applied_postprocessors