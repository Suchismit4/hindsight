"""
Data Manager Module.

This module provides the central DataManager class which serves as the main entry point
for accessing financial data in the Hindsight system. It coordinates between various 
data providers, handles caching, and manages the configuration of data requests.
"""

import xarray as xr
import xarray_jax
import yaml
from typing import Union, List, Dict, Any, Optional
import os

from src.data.core.provider import _PROVIDER_REGISTRY
from src.data.loaders import *
from src.data.core.cache import CacheManager

class CharacteristicsManager:
    """
    Manager for computing and caching financial characteristics.
    
    This class computes financial characteristics similar to the ones in the
    GlobalFactors codebase. It serves as a higher-level cache (L3) on top of
    the raw data cache.
    
    Attributes:
        cache_manager: Reference to the cache manager for storing computed characteristics
    """
    
    def __init__(self, cache_manager: CacheManager):
        """
        Initialize the CharacteristicsManager.
        
        Args:
            cache_manager: The cache manager to use for caching computed characteristics
        """
        self.cache_manager = cache_manager
        
    def compute_characteristics(self, data: Dict[str, xr.Dataset], config: Dict[str, Any]) -> Dict[str, xr.Dataset]:
        """
        Compute financial characteristics based on the input data and configuration.
        
        Args:
            data: Dictionary of raw datasets from different sources
            config: Configuration specifying which characteristics to compute
            
        Returns:
            Dictionary of datasets with computed characteristics
            
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        # This will be implemented later to compute characteristics from the GlobalFactors codebase
        raise NotImplementedError("Characteristics computation is not yet implemented")

class DataManager:
    """
    Central manager for data access across all data sources.
    
    This class serves as the main interface for clients to interact with the
    data framework. It centralizes access to various data sources through a
    unified API, handles configuration parsing, and coordinates with the
    cache system for efficient data retrieval.
    
    Attributes:
        cache_manager: Manager for the two-level caching system
        _data_loaders: Dictionary mapping data paths to their respective loaders
        characteristics_manager: Manager for computing and caching financial characteristics
    """
    
    def __init__(self):
        """
        Initialize the DataManager.
        
        Sets up the cache manager and collects all registered data loaders
        from the provider registry.
        """
        self.cache_manager = CacheManager()
        self._data_loaders = {}
        
        # Collect data loaders from all registered providers
        for provider in _PROVIDER_REGISTRY.values():
            self._data_loaders.update(provider.data_loaders)
            
        # Initialize the characteristics manager
        self.characteristics_manager = CharacteristicsManager(self.cache_manager)

    def _get_raw_data(self, data_requests: Union[List[Dict[str, Any]], str]) -> Dict[str, xr.Dataset]:
        """
        Retrieve raw data based on specified configurations.
        
        This is an internal method that provides lower-level access to raw financial data.
        It supports two ways of specifying data requests:
        1. A list of request dictionaries with data_path and config keys
        2. A path to a YAML configuration file containing the request list
        
        Each data request must specify both start_date and end_date in its config.
        
        Args:
            data_requests: Either a list of request dictionaries or a path to a YAML config file
                
                Example request dictionary:
                {
                    "data_path": "wrds/equity/crsp",
                    "config": {
                        "start_date": "2000-01-01",
                        "end_date": "2024-01-01",
                        "freq": "M",
                        "filters": {"date__gte": "2000-01-01"},
                        "processors": {"set_permno_coord": True}
                    }
                }

        Returns:
            Dictionary mapping data paths to their corresponding xarray Datasets
            
        Raises:
            TypeError: If the YAML file doesn't contain a list of requests
            ValueError: If start_date or end_date is missing, or if no loader exists
            BrokenPipeError: If data fetching fails unexpectedly
        """
        # If data_requests is a string, load it as a YAML config file
        if isinstance(data_requests, str):
            with open(data_requests, 'r') as f:
                data_requests = yaml.safe_load(f)
                if not isinstance(data_requests, list):
                    raise TypeError("YAML config file must contain a list of data requests.")

        collected_data = {}

        for request in data_requests:
            data_path = request.get('data_path')
            config = request.get('config', {})
            
            # Enforce that both start_date and end_date are provided
            if not config.get("start_date") or not config.get("end_date"):
                raise ValueError(f"Request for '{data_path}' must specify both 'start_date' and 'end_date'.")

            # Check if we have a loader for this data_path
            if data_path not in self._data_loaders:
                raise ValueError(f"No DataLoader available for data path '{data_path}'.")

            loader = self._data_loaders[data_path]

            # Fetch data through the cache manager
            data = self.cache_manager.fetch(
                relative_path=data_path, 
                parameters=request, 
                data_loader=loader
            )
            
            if data is None:
                raise BrokenPipeError(f"Failed to fetch data for path '{data_path}'")

            collected_data[data_path] = data

        return collected_data
    
    def get_data(self, config: Dict[str, Any]) -> Dict[str, xr.Dataset]:
        """
        Retrieve data with computed financial characteristics.
        
        This is the main method for accessing financial data with computed characteristics.
        The config parameter specifies both the raw data to retrieve and the characteristics
        to compute from that data.
        
        Args:
            config: Configuration specifying data sources and characteristics
                
                Example config:
                {
                    "data_sources": [
                        {
                            "data_path": "wrds/equity/crsp",
                            "config": {
                                "start_date": "2000-01-01",
                                "end_date": "2024-01-01",
                                "freq": "M"
                            }
                        },
                        {
                            "data_path": "wrds/equity/compustat",
                            "config": {
                                "start_date": "2000-01-01",
                                "end_date": "2024-01-01",
                                "freq": "Y"
                            }
                        }
                    ],
                    "characteristics": {
                        "accounting": ["assets", "sales", "book_equity", "debt_gr1"],
                        "market": ["ret_1_0", "ret_12_1", "chcsho_12m", "eqnpo_12m"]
                    }
                }
                
        Returns:
            Dictionary mapping data paths to their corresponding xarray Datasets with 
            computed characteristics
            
        Raises:
            NotImplementedError: Characteristics computation is not yet implemented
        """
        # Get the raw data first
        data_sources = config.get("data_sources", [])
        raw_data = self._get_raw_data(data_sources)
        
        # If no characteristics are requested, just return the raw data
        if "characteristics" not in config:
            return raw_data
        
        # Compute and return characteristics
        try:
            return self.characteristics_manager.compute_characteristics(raw_data, config)
        except NotImplementedError:
            # For now, just return the raw data with a warning
            print("WARNING: Characteristics computation is not yet implemented. Returning raw data.")
            return raw_data
    
    def get_available_data_paths(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available data paths in the system.
        
        Returns a dictionary with details about each data path, including
        its provider and any sub-providers (for providers like OpenBB that
        offer multiple data sources).
        
        Returns:
            Dictionary mapping each data_path to its provider information:
            {
                "wrds/equity/crsp": {
                    "provider": "wrds",
                    "sub_providers": [None]
                },
                "openbb/equity/price/historical": {
                    "provider": "openbb",
                    "sub_providers": ["yfinance", "fmp", ...]
                },
                ...
            }
        """
        # Build a mapping from "openbb/equities/price/historical" -> ["yfinance", "fmp", ...]
        coverage_map = {}
        
        # Handle OpenBB provider specially since it has sub-providers
        if "openbb" in _PROVIDER_REGISTRY: 
            try:
                import openbb as obb
                coverage_dict = obb.coverage.providers
                
                for subp, coverage_paths in coverage_dict.items():
                    for dot_path in coverage_paths:
                        # convert "equities.price.historical" > "equities/price/historical"
                        slash_path = dot_path.replace(".", "/")
                        
                        # Ensure path starts with "openbb/"
                        full_data_path = f"openbb/{slash_path}"
                        
                        # Initialize list if not exists, then append
                        if full_data_path not in coverage_map:
                            coverage_map[full_data_path] = []
                        coverage_map[full_data_path].append(subp)
            except ImportError:
                # OpenBB might not be installed, continue without it
                pass

        # For each registered provider and each data_path it supports, collect info
        results = {}
        
        for provider in _PROVIDER_REGISTRY.values(): 
            # provider._data_loaders is a dict: { "wrds/equities/compustat": loader, ... }
            for dp in provider.data_loaders.keys():
                if provider.name == "openbb":
                    # Look up sub-providers from coverage_map
                    # Normalize the data path to ensure consistent lookup
                    normalized_dp = dp if dp.startswith("openbb/") else f"openbb/{dp}"
                    sub_providers = coverage_map.get(normalized_dp, [])
                else:
                    # For WRDS or others, no concept of sub-providers
                    sub_providers = [None]
                    
                results[dp] = {
                    "provider": provider.name,
                    "sub_providers": sub_providers
                }
        
        return results
    
    def get_available_characteristics(self) -> Dict[str, List[str]]:
        """
        Get information about all available financial characteristics.
        
        Returns:
            Dictionary mapping characteristic categories to lists of characteristics:
            {
                "accounting": ["assets", "sales", "book_equity", ...],
                "market": ["ret_1_0", "ret_12_1", "chcsho_12m", ...],
                ...
            }
        """
        # This will be implemented later when the characteristics computation is implemented
        return {
            "accounting": [],
            "market": []
        }
