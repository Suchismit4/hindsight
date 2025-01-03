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

        # # Create a root DataTree (initially empty)
        # root_tree = DataTree(name="root")
        
        collected_data = {}

        for request in data_requests:
            data_path = request.get('data_path')
            config = request.get('config', {})

            # Convert 2-element lists to tuples if needed (for filters, etc.)
            filters = config.get('filters', {})
            for k, v in filters.items():
                if isinstance(v, list) and len(v) == 2:
                    filters[k] = tuple(v)

            # Check if we have a loader for this data_path
            if data_path not in self.data_loaders:
                raise ValueError(f"No DataLoader available for data path '{data_path}'.")

            loader = self.data_loaders[data_path]

            # Load the data using the config. This should return an xr.Dataset or a DataTree.
            data = loader.load_data(**config)

            collected_data[data_path] = data
            # # Break the path into segments
            # path_parts = [p for p in data_path.split('/') if p.strip() != '']

            # # traverse or create intermediate nodes
            # node = root_tree
            # for part in path_parts[:-1]:  # all but the last part
            #     if part not in node.children:
            #         node[part] = DataTree(name=part)
            #     node = node[part]

            # # The final path segment is where we create the node with data
            # final_part = path_parts[-1]

            # if isinstance(data, xr.Dataset):
            #     # Create/overwrite this final node with the dataset directly
            #     node[final_part] = DataTree(dataset=data, name=final_part)
            # else:
            #     raise TypeError("DataLoader returned an unsupported data type.")

        return collected_data
    
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
