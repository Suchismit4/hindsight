"""
Cache management system for financial datasets.

This module implements a two-level caching system for xarray datasets:
 - Level 1 (L1): Raw datasets before post-processing
 - Level 2 (L2): Post-processed datasets ready for analysis

The cache system helps improve performance by avoiding repetitive loading and 
processing of large financial datasets, while also ensuring data consistency
across multiple sessions.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import xarray as xr
from datetime import datetime
import pandas as pd
from copy import deepcopy

from src.data.core.util import FrequencyType, TimeSeriesIndex
from src.data.processors.registry import post_processor
from src.data.core.util import Loader
from src.data.processors import apply_processors


class CacheManager:
    """
    Manager for two-level dataset caching system.
    
    Responsible for:
    1. Determining cache keys based on data parameters
    2. Finding and loading cached datasets
    3. Saving datasets to cache with appropriate metadata
    4. Managing the cache filesystem structure
    
    Attributes:
        cache_root (str): Root directory for all cached data
    """
    
    def __init__(self, cache_root: Optional[str] = None):
        """
        Initialize the CacheManager with a root directory for cache storage.
        
        Args:
            cache_root: Path to the cache directory. If None, defaults to '~/data/cache'
        """
        self.cache_root = cache_root or os.path.expanduser("~/data/cache")

    def _convert_to_serializable(self, obj: Any) -> Any:
        """
        Recursively convert an object into a JSON-serializable structure.
        
        For types that are not natively serializable, returns their string representation.
        This is used for cache metadata storage.
        
        Args:
            obj: The object to convert to a JSON-serializable form
            
        Returns:
            A JSON-serializable representation of the object
            
        Raises:
            ValueError: If attempting to serialize an xr.Dataset directly
        """
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(value) for value in obj]
        elif isinstance(obj, xr.Dataset):
            raise ValueError("Attempted to serialize an xr.Dataset, which is not supported.")
        else:
            return str(obj)

    def _normalize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize parameter keys for consistency in cache key generation.
        
        Handles common parameter naming variations, such as 'freq' vs 'frequency'.
        
        Args:
            parameters: Original parameters dictionary
            
        Returns:
            Dictionary with normalized parameter keys
        """
        normalized = {}
        for key, value in parameters.items():
            normalized_key = "frequency" if key == "freq" else key
            normalized[normalized_key] = value
        return normalized

    def _get_cache_directory(self, relative_path: str) -> Path:
        """
        Determine and create the cache directory for a specific data path.
        
        Args:
            relative_path: Data path identifier (e.g., 'wrds/equity/crsp')
            
        Returns:
            Path object for the cache directory
        """
        sub_directory = relative_path.strip("/")
        cache_directory = os.path.join(self.cache_root, sub_directory)
        os.makedirs(cache_directory, exist_ok=True)
        return Path(cache_directory)

    def _compute_cache_key(self, parameters: Dict[str, Any]) -> str:
        """
        Compute a unique cache key based on the request parameters.
        
        The key excludes time range parameters (start_date, end_date) since
        these are handled separately in the cache filename.
        
        Args:
            parameters: Request parameters
            
        Returns:
            MD5 hash string representing the cache key
        """
        # Exclude 'start_date' and 'end_date' from the key computation.
        key_parameters = {key: value for key, value in parameters.items() if key not in ("start_date", "end_date")}
        serializable_params = self._convert_to_serializable(key_parameters)
        params_string = json.dumps(serializable_params, sort_keys=True)
        return hashlib.md5(params_string.encode("utf-8")).hexdigest()

    def _find_existing_cache(self, cache_key: str, cache_directory: Path, parameters: Dict[str, Any]) -> Optional[Path]:
        """
        Search for an existing cache file matching the parameters.
        
        Validates that any candidate cache files match the requested frequency
        and time range requirements.
        
        Args:
            cache_key: The computed cache key
            cache_directory: Directory to search for cache files
            parameters: Request parameters with time range and frequency info
            
        Returns:
            Path to matching NetCDF file if found, None otherwise
        """
        found_cache_path = None
        candidate_metadata_files = list(cache_directory.glob(f"{cache_key}_*.json"))

        requested_start = parameters.get("start_date")
        requested_end = parameters.get("end_date")
        requested_frequency = parameters.get("frequency")

        for metadata_file in candidate_metadata_files:
            try:
                with open(metadata_file, "r") as file:
                    metadata = json.load(file)
                # Validate that the frequency matches.
                if metadata.get("frequency") != requested_frequency:
                    continue
                # If a time range is requested, ensure the cached metadata contains a valid time range.
                if requested_start and requested_end:
                    cache_start = metadata.get("start_date")
                    cache_end = metadata.get("end_date")
                    if not cache_start or not cache_end:
                        continue
                # Construct the corresponding NetCDF file path.
                base_file = metadata_file.with_suffix("")  # Remove .json extension.
                netcdf_file = base_file.with_suffix(".nc")
                if not netcdf_file.exists():
                    continue
                found_cache_path = netcdf_file
                break
            except Exception as error:
                print(f"Error loading cache from {metadata_file}: {error}")
                continue

        return found_cache_path

    def _load_from_cache(self, relative_path: str, normalized_params: Dict[str, Any],
                         cache_directory: Path, cache_key: str) -> Optional[xr.Dataset]:
        """
        Attempt to load a dataset from cache.
        
        Args:
            relative_path: Data path identifier for logging
            normalized_params: Normalized request parameters
            cache_directory: Directory to search for cache files
            cache_key: The computed cache key
            
        Returns:
            Loaded xarray Dataset if found and valid, None otherwise
        """
        loaded_dataset = None
        cache_file_path = self._find_existing_cache(cache_key, cache_directory, normalized_params)
        if cache_file_path is not None:
            print(f"{relative_path}: Attemping to load found cache({cache_file_path}).")
            loaded_dataset = xr.load_dataset(cache_file_path)
            # Restore custom time index if available.
            if "time" in loaded_dataset.coords:
                ts_index = TimeSeriesIndex(loaded_dataset.coords["time"])
                loaded_dataset.coords["time"].attrs["indexes"] = {"time": ts_index}
            # TODO: Subset the dataset if needed based on requested start_date and end_date.
            print(f"{relative_path}: Successfully loaded from {cache_file_path}")
        return loaded_dataset

    def fetch(self, relative_path: str, parameters: Dict[str, Any], data_loader: Optional[Any]) -> Optional[xr.Dataset]:
        """
        Retrieve a dataset using the cache system or load it if necessary.
        
        This is the main method for accessing datasets through the cache manager.
        The process follows these steps:
        1. Try to load a Level 2 (post-processed) cache
        2. If not found, try to load a Level 1 (raw) cache
        3. If no cache exists, load raw data using the provided data_loader
        4. Apply post-processing to create a Level 2 dataset
        5. Cache both levels as needed
        
        Args:
            relative_path: Data path identifier (e.g., 'wrds/equity/crsp')
            parameters: Request parameters including time range, filters, processors
            data_loader: Object with load_data method to fetch data if cache misses
            
        Returns:
            The requested dataset, either from cache or freshly loaded and processed
            
        Raises:
            ValueError: If no cache exists and no data_loader is provided
            RuntimeError: If data loading or post-processing fails
        """
        # Determine the cache directory based on the relative path.
        cache_directory = self._get_cache_directory(relative_path)
        
        # Create a deep copy of parameters to avoid side-effects.
        parameters = deepcopy(parameters)
        # quit()
        
        # Extract postprocessor configuration (if any) from the 'config' section.
        postprocessors_config = deepcopy(parameters.get("config", {}).get("postprocessors", []))
        
        # Also check for processors in config (Django-style) if postprocessors wasn't found
        if not postprocessors_config:
            postprocessors_config = deepcopy(parameters.get("config", {}).get("processors", []))
        
        # Build cache keys:
        # Level 2 parameters include the postprocessors; Level 1 (raw) parameters exclude them.
        normalized_l2_params = self._normalize_parameters(deepcopy(parameters))
        
        # Clone parameters and remove processors for L1 key
        l1_params = deepcopy(parameters)
        if "postprocessors" in l1_params.get("config", {}):
            del l1_params["config"]["postprocessors"]
        if "processors" in l1_params.get("config", {}):
            del l1_params["config"]["processors"]
            
        normalized_l1_params = self._normalize_parameters(l1_params)
        
        # Compute unique cache keys for both Level 2 and Level 1 caches.
        L2Key = self._compute_cache_key(normalized_l2_params)
        L1Key = self._compute_cache_key(normalized_l1_params)

        # Attempt to load the Level 2 (post-processed) cache.
        final_dataset = self._load_from_cache(relative_path, normalized_l2_params, cache_directory, L2Key)
        if final_dataset is not None:
            return final_dataset
        
        # If Level 2 cache is not found, try to load the Level 1 (raw) cache.
        dataset_to_process = self._load_from_cache(relative_path, normalized_l1_params, cache_directory, L1Key)
        
        if dataset_to_process is not None:
            print(f"{relative_path}: Processing loaded L1 cache.")
        else:
            # No Level 1 cache available; load raw data using the provided loader.
            if data_loader is None:
                raise ValueError(f"{relative_path}: No cache found and no data loader provided.")
            print(f"{relative_path}: Level 1 cache not found. Loading data using loader (id: {id(data_loader)}).")
            config = deepcopy(parameters.get("config", {}))
            if "postprocessors" in config:
                del config["postprocessors"]
            if "processors" in config:
                del config["processors"]
                
            try:
                dataset_to_process = data_loader.load_data(**config)
            except Exception as e:
                raise RuntimeError(f"{relative_path}: Data loader failed to load raw data. Error: {e}") from e
            
            # Save the newly loaded raw data as a L1 cache.
            try:
                self.cache(dataset_to_process, relative_path, normalized_l1_params)
            except Exception as error:
                # Log caching errors for L1 without stopping further processing.
                print(f"{relative_path}: Warning - Failed to cache L1 dataset. Error: {error}")

        
        # Apply post-processing to generate the L2 dataset.
        try:
            final_dataset, applied_postprocessors = apply_processors(dataset_to_process, postprocessors_config)
            # Update parameters with the applied postprocessors for accurate caching metadata.
            parameters['postprocessors'] = applied_postprocessors
        except Exception as error:
            raise RuntimeError(f"{relative_path}: Post-processing steps failed for dataset. Error: {error}") from error
        
        print(f"{relative_path}: Post-processing finished.")
        
        # Cache the L2 dataset.
        if L1Key != L2Key:
            try:
                self.cache(final_dataset, relative_path, normalized_l2_params)
            except Exception as error:
                # Log the caching error without preventing the return of the final dataset.
                print(f"{relative_path}: Warning - Failed to cache L2 dataset. Error: {error}")
        
        return final_dataset

    def cache(self, dataset: xr.Dataset, relative_path: str, parameters: Dict[str, Any]) -> None:
        """
        Save a dataset to the cache system with metadata.
        
        Creates both a NetCDF file for the dataset and a JSON file for its metadata.
        The metadata includes information about the data source, time range,
        frequency, and processing instructions.
        
        Args:
            dataset: The xarray Dataset to cache
            relative_path: Data path identifier (e.g., 'wrds/equity/crsp')
            parameters: Parameters used to generate the dataset
            
        Raises:
            Exception: If saving the dataset or metadata fails
        """
        normalized_params = self._normalize_parameters(parameters)
        cache_directory = self._get_cache_directory(relative_path)
        cache_key = self._compute_cache_key(normalized_params)

        # Determine the time range from the dataset if available.
        if "time" in dataset.coords:
            time_values = pd.to_datetime(dataset.coords["time"].values)
            start_date = str(time_values.min().date())
            end_date = str(time_values.max().date())
        else:
            start_date = normalized_params.get("start_date", "")
            end_date = normalized_params.get("end_date", "")

        base_filename = f"{cache_key}_{start_date}_{end_date}"
        base_filepath = cache_directory / base_filename

        netcdf_filepath = base_filepath.with_suffix(".nc")
        metadata_filepath = base_filepath.with_suffix(".json")
        netcdf_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Temporarily remove non-serializable attributes from the time coordinate.
        time_indexes = None
        if "time" in dataset.coords and "indexes" in dataset.coords["time"].attrs:
            time_indexes = dataset.coords["time"].attrs.pop("indexes")

        try:
            # Save the dataset in NetCDF format.
            dataset.to_netcdf(path=netcdf_filepath, mode="w", format="NETCDF4", engine="netcdf4")
            metadata = {
                "src": relative_path,
                "frequency": normalized_params.get("frequency"),
                "start_date": start_date,
                "end_date": end_date,
                "instructions": self._convert_to_serializable(
                    {key: value for key, value in normalized_params.items() if key not in ("start_date", "end_date")}
                ),
                "timestamp": datetime.now().isoformat(),
            }
            with open(metadata_filepath, "w") as meta_file:
                json.dump(metadata, meta_file, indent=2)
            print(f"{relative_path}: Successfully saved cache to {netcdf_filepath}")
        except Exception as error:
            raise Exception(f"Failed to save dataset to cache: {error}") from error
        finally:
            # Restore any removed time indexes.
            if time_indexes is not None:
                dataset.coords["time"].attrs["indexes"] = time_indexes
                
