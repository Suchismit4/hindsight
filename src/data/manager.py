# src/data/manager.py

import xarray as xr
import xarray_jax
from xarray import DataTree

from typing import Union, List, Dict, Any
import os
import yaml  

from .provider import _PROVIDER_REGISTRY
from src.data.loaders import *

class DataManager:
    """
    Central manager class for handling data loading and processing operations.

    This class serves as the main interface for clients to interact with the
    data framework. It coordinates between data loaders to provide a unified data access layer.
    """

    def __init__(self):
        """
        Initialize the DataManager.

        The manager collects data loaders from all registered providers upon initialization.
        """
        self.data_loaders = {}
        for provider in _PROVIDER_REGISTRY.values():
            self.data_loaders.update(provider.data_loaders)

    def get_data(self, data_requests: Union[List[Dict[str, Any]], str]) -> xr.DataTree:
        """
        Retrieve data for the specified data paths with their configurations.

        Args:
            data_requests: Either a list of dictionaries as before, or a string path to a YAML config file.

        Returns:
            xr.DataTree: The requested data merged into a single DataTree.

        Raises:
            ValueError: If no suitable loader is available for a data path.
        """
        # If data_requests is a string, assume it's a path to a YAML config file
        if isinstance(data_requests, str):
            with open(data_requests, 'r') as f:
                data_requests = yaml.safe_load(f)
                if not isinstance(data_requests, list):
                    raise TypeError("YAML config file must contain a list of data requests.")

        data_dict = {}
        
        for request in data_requests:
          
            data_path = request.get('data_path')
            config = request.get('config', {})
            
            # Convert to tuples for imm.
            filters = config.get('filters', {})
            for k, v in filters.items():
                if isinstance(v, list) and len(v) == 2:
                    filters[k] = tuple(v)
            
            if data_path not in self.data_loaders:
                raise ValueError(f"No DataLoader available for data path '{data_path}'.")
            
            loader = self.data_loaders[data_path]
            data = loader.load_data(**config)
    
            if isinstance(data, xr.Dataset):
                data_dict[data_path] = data
            elif isinstance(data, DataTree):
                # Convert the DataTree to a dictionary and merge paths
                tree_dict = self.data_tree_to_dict(data, prefix=data_path)
                data_dict.update(tree_dict)
            else:
                raise TypeError("DataLoader returned an unsupported data type.")
            
        # Create the DataTree from the data_dict
        merged_tree = DataTree.from_dict(data_dict)
        return merged_tree

    def data_tree_to_dict(self, tree: xr.DataTree, prefix: str = "") -> Dict[str, xr.Dataset]:
        """
        Convert a DataTree into a dictionary mapping paths to datasets.

        Args:
            tree: The DataTree to convert.
            prefix: A string to prefix to each path (used for data paths).

        Returns:
            A dictionary mapping paths to datasets.
        """
        data_dict = {}
        for node in tree.subtree:
            path = node.path
            if node.ds is not None:
                # Build the full path by combining the prefix and node path
                if path in ['', '/', '.', './']:
                    full_path = prefix
                else:
                    full_path = os.path.join(prefix, path.strip('/')).replace('\\', '/')
                data_dict[full_path] = node.ds
        return data_dict

    def get_available_data_paths(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of all available data paths in the registry, along with
        the top-level provider and any sub-providers (if applicable).
        
        Returns:
            Dict[str, Dict[str, Any]]: For each data_path, a dictionary with:
            {
                "provider": "openbb" | "wrds" | etc.,
                "sub_providers": ["yfinance", "fmp", ...] or [None]
            }
        """
        # Build a mapping from "openbb/equities/price/historical" -> ["yfinance", "fmp", ...]
        coverage_map = {}
        
        if "openbb" in _PROVIDER_REGISTRY: 
            coverage_dict = obb.coverage.providers
            
            for subp, coverage_paths in coverage_dict.items():
                for dot_path in coverage_paths:
                    # convert "equities.price.historical" > "equities/price/historical"
                    slash_path = dot_path.replace(".", "/")
                    
                    # Ensure path starts with "openbb/"
                    full_data_path = (
                        f"openbb{slash_path}"
                    )
                    
                    # Initialize list if not exists, then append
                    if full_data_path not in coverage_map:
                        coverage_map[full_data_path] = []
                    coverage_map[full_data_path].append(subp)

        # For each registered provider and each data_path it supports, collect info
        results = {}
        
        for provider in _PROVIDER_REGISTRY.values(): 
            # provider.data_loaders is a dict: { "wrds/equities/compustat": loader, ... }
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
