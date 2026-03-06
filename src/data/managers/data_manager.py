"""
Data Manager Module.

This module provides the central DataManager class which serves as the main entry point
for accessing financial data in the Hindsight system. It coordinates between various 
data providers, handles caching, and manages the configuration of data requests.
"""

import xarray as xr
import yaml
from typing import Union, List, Dict, Any
from pathlib import Path

from src.data.core.provider import _PROVIDER_REGISTRY
from src.data.loaders import *
from src.data.core.cache import CacheManager
from src.data.managers.config_schema import ConfigLoader, DataConfig

class DataManager:
    """
    Central manager for data access across all data sources.
    
    This class serves as the main interface for clients to interact with the
    data framework. It centralizes access to various data sources through a
    unified API, handles configuration parsing, and coordinates with the
    cache system for efficient data retrieval.
    
    The DataManager supports both legacy and semantic configuration formats:
    
    1. Legacy format: Dictionary-based configuration (backward compatible)
    2. New format: YAML-based semantic configuration with cleaner structure
    
    Attributes:
        cache_manager: Manager for the two-level caching system
        _data_loaders: Dictionary mapping data paths to their respective loaders
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
            
    def _get_raw_data(self, data_requests: Union[List[Dict[str, Any]], str]) -> Dict[str, xr.Dataset]:
        """
        Retrieve raw data based on specified configurations.
        
        This is an internal method that provides lower-level access to raw financial data.
        It supports two ways of specifying data requests:
        1. A list of request dictionaries with data_path and config keys
        2. A path to a YAML configuration file containing the request list
        
        Each data request must specify both start_date and end_date in its config.
        These dates are used to filter data at the DataFrame level during processing.
        
        Args:
            data_requests: Either a list of request dictionaries or a path to a YAML config file
                
                Example request dictionary:
                {
                    "data_path": "wrds/equity/crsp",
                    "config": {
                        "start_date": "2000-01-01",
                        "end_date": "2024-01-01",
                        "freq": "M",
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
            start_date = config.get("start_date")
            end_date = config.get("end_date")
            
            if not start_date or not end_date:
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
    
    def load_from_config(self, config_path: Union[str, Path]) -> Dict[str, xr.Dataset]:
        """
        Load data using the semantic YAML configuration format.
        
        This method provides a cleaner, more intuitive way to define data loading
        pipelines using YAML configuration files. It automatically handles the
        conversion to the legacy format for backward compatibility.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary mapping source names to their corresponding xarray Datasets
            
        Example:
            ```python
            dm = DataManager()
            data = dm.load_from_config("configs/equity_analysis.yaml")
            equity_data = data['equity_prices']
            ```
            
        Example YAML configuration:
            ```yaml
            data:
              name: "equity-analysis"
              start_date: "2020-01-01"
              end_date: "2024-01-01"
              
              sources:
                equity_prices:
                  provider: "wrds"
                  dataset: "crsp"
                  frequency: "daily"
                  
                  processors:
                    filters:
                      share_classes: [10, 11]
                      exchanges: [1, 2, 3]
                      
                    merges:
                      - source: "company_names"
                        type: "2d_table"
                        on: "permno"
                        columns: ["comnam", "exchcd"]
                        
                    transforms:
                      - type: "set_coordinates" 
                        coord_type: "permno"
                      - type: "fix_market_equity"
                      
                company_names:
                  provider: "wrds"
                  dataset: "crsp_names"
            ```
        """
        # Load the semantic configuration
        config = ConfigLoader.load_from_yaml(config_path)
        
        # Convert to legacy format for compatibility with existing infrastructure
        legacy_config = ConfigLoader.convert_to_legacy_format(config)
        
        # Load data using the existing pipeline
        raw_data = self._get_raw_data(legacy_config["data_sources"])
        
        # Map back from legacy data_path keys to semantic source names
        result = {}
        for source_name, source_config in config.sources.items():
            data_path = ConfigLoader._map_to_data_path(
                source_config.provider, 
                source_config.dataset
            )
            if data_path in raw_data:
                result[source_name] = raw_data[data_path]
        
        return result
    
    def load_from_built_config(self, config: DataConfig) -> Dict[str, xr.Dataset]:
        """
        Load data using a programmatically built configuration.
        
        Args:
            config: DataConfig object built using DataConfigBuilder
            
        Returns:
            Dictionary mapping source names to their corresponding xarray Datasets
        """
        # Convert to legacy format
        legacy_config = ConfigLoader.convert_to_legacy_format(config)
        
        # Load data using existing pipeline
        raw_data = self._get_raw_data(legacy_config["data_sources"])
        
        # Map back to semantic source names
        result = {}
        for source_name, source_config in config.sources.items():
            data_path = ConfigLoader._map_to_data_path(
                source_config.provider, 
                source_config.dataset
            )
            if data_path in raw_data:
                result[source_name] = raw_data[data_path]
        
        return result

    def load_builtin(self, config_name: str, start_date: str = None, end_date: str = None) -> Dict[str, xr.Dataset]:
        """
        Load data using a built-in configuration.
        
        This method provides easy access to standard, academically-relevant
        data loading configurations. Date ranges can be overridden.
        
        Args:
            config_name: Name of built-in configuration (e.g., "equity_standard")
            start_date: Optional override for start date
            end_date: Optional override for end date
            
        Returns:
            Dictionary mapping source names to their corresponding xarray Datasets
            
        Example:
            ```python
            dm = DataManager()
            
            # Load standard equity data for a specific period
            data = dm.load_builtin("equity_standard", "2015-01-01", "2020-12-31")
            equity_data = data['equity_prices']
            ```
        """
        from src.data.configs import load_builtin_config
        
        # Load the built-in configuration
        config = load_builtin_config(config_name)
        
        # Override date range if provided
        if start_date:
            config.start_date = start_date
        if end_date:
            config.end_date = end_date
            
        return self.load_from_built_config(config)
