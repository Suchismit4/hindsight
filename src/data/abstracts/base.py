# src/data/loaders/base.py

import os
import pandas as pd
import xarray as xr
import json
from typing import Dict, Any, Union, Optional
from pathlib import Path
import hashlib
from abc import ABC, abstractmethod

from ..core.struct import DatasetDateTimeAccessor
from ..core.struct import FrequencyType

class BaseDataSource(ABC):
    """
    Base class for handling data sources configuration and path management.

    This class provides the foundational logic for accessing different data sources
    defined in the paths.yaml configuration file. It also handles interactions with
    the cache system, including saving/loading xarray Datasets as NetCDF.
    """

    def __init__(self, data_path: str):
        """Initialize the data source with the given data path."""
        self.data_path = data_path  # This is used in cache path generation
        self.cache_root = os.path.expanduser('~/data/cache')

    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply filters to the DataFrame.

        Args:
            df: The DataFrame to filter.
            filters: A dictionary where keys are column names and values are filter conditions.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """

        for column, condition in filters.items():
            if isinstance(condition, tuple) or isinstance(condition, list) \
                    and len(condition) == 2:
                # Condition is a tuple like ('>=', '1959-01-01')
                operator, value = condition
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
            else:
                # Condition is a simple equality
                df = df[df[column] == condition]
        return df

    def get_cache_path(self, **params) -> Path:
        """
        Generate a base 'cache path' (WITHOUT extension) based on self.data_path + hashed params.

        We append '.nc' or '.json' (for NetCDF caching) or '.parquet' (for DataFrame caching).

        Example:
          - data_path='wrds/equity/compustat'
          - cache_root='~/data/cache'
          => ~/data/cache/wrds/equity/compustat/{md5hash_of_params}

        Args:
            **params: Arbitrary parameters to be hashed into the cache filename.

        Returns:
            Path: The fully-qualified path (no file extension).
        """
        
        # Serialize the parameters
        params_string = json.dumps(params, sort_keys=True)
        
        # Create an MD5 hash of the parameters
        params_hash = hashlib.md5(params_string.encode('utf-8')).hexdigest()
        
        # Build the subdirectory path, e.g. ~/data/cache/wrds/equity/compustat
        sub_dir = self.data_path.strip('/')  # remove leading/trailing slashes
        
        cache_dir = os.path.join(self.cache_root, sub_dir)
        os.makedirs(cache_dir, exist_ok=True)

        # Return something like ~/data/cache/wrds/equity/compustat/<hash>
        base_path = os.path.join(cache_dir, params_hash)
        return Path(base_path)

    def check_cache_netcdf(self, base_path: Path) -> bool:
        """
        Check if the NetCDF (.nc) file exists at this path.
        """
        netcdf_path = base_path.with_suffix('.nc')
        return netcdf_path.exists()
    
    def _metadata_matches(self, metadata_path: Path, request_params: Dict[str, Any]) -> bool:
        """
        Compare an existing JSON metadata file with the requested params.
        
        If they match exactly, return True; otherwise False.

        Args:
            metadata_path (Path): Path to the .json file containing cached metadata.
            request_params (Dict[str, Any]): The parameters requested for the current load.

        Returns:
            bool: True if the metadata file exists and matches request_params, False otherwise.
        """
        if not metadata_path.exists():
            return False
        try:
            with open(metadata_path, 'r') as f:
                cached_params = json.load(f)
            # Direct comparison
            # Convert any sequences to sorted lists before comparison
            def normalize(params):
                if isinstance(params, dict):
                    return {k: normalize(v) for k, v in params.items()}
                if isinstance(params, (list, tuple)):
                    return sorted(normalize(x) for x in params)
                return params
                
            is_equal = normalize(cached_params) == normalize(request_params)
            
            if not is_equal:
                print(f"Parameters differ:\nCached: {cached_params}\nRequest: {request_params}")
                
            return is_equal
        except Exception as e:
            print(f"Error reading metadata file {metadata_path}: {e}")
            return False

    def load_from_cache(
        self,
        base_path: Path,
        request_params: Dict[str, Any],
        frequency: FrequencyType = FrequencyType.DAILY
    ) -> Optional[xr.Dataset]:
        """
        Load an xarray.Dataset from a NetCDF (.nc) file, if it exists and matches params.

        We also check a .json file with the same base path to ensure the parameters match.

        Args:
            base_path (Path): The base path (no extension). We'll look for <base>.nc and <base>.json.
            request_params (Dict[str, Any]): The params used for generating the data. We'll compare
                                             these to what's stored in the .json sidecar.
            frequency (FrequencyType): Frequency type to assign to the dataset if needed.

        Returns:
            xr.Dataset or None: The loaded Dataset if everything matches; None otherwise.
        """
        print(f"{self.data_path} : Attempting to load from cache")
        netcdf_path = base_path.with_suffix('.nc')
        metadata_path = base_path.with_suffix('.json')

        if not netcdf_path.exists() or not self._metadata_matches(metadata_path, request_params):
            print(f"{self.data_path} : No cache found. Falling back to fetching data...")
            return None
        
        try:
            ds = xr.load_dataset(netcdf_path)  # or xr.open_dataset, either is OK

            from ..core.struct import TimeSeriesIndex
            
            time_coord = ds.coords['time']
            ts_index = TimeSeriesIndex(time_coord)
            ds.coords['time'].attrs['indexes'] = {'time': ts_index}
            
            # TODO: Enable force loads...
            print(f"{self.data_path} : Loaded data from cache. Use force_load=True in config to not use cache.")
            
            return ds
        except Exception as e:
            print(f"{self.data_path} : Failed to load from NetCDF: {e}")
            return None

    def save_to_cache(
        self,
        ds: xr.Dataset,
        base_path: Path,
        params: Dict[str, Any]
    ) -> None:
        """
        Save an xarray.Dataset to a NetCDF (.nc) file, plus store parameters in a .json sidecar.

        Args:
            ds (xr.Dataset): The dataset to save.
            base_path (Path): The base path (no extension).
            params (dict): Parameters used to generate this dataset, stored in the JSON.
        """
        netcdf_path = base_path.with_suffix('.nc')
        metadata_path = base_path.with_suffix('.json')
        netcdf_path.parent.mkdir(parents=True, exist_ok=True)

        # Pop the 'indexes' attribute if it exists. We'll restore it after saving.
        time_indexes = ds.coords['time'].attrs.pop('indexes', None)
        
        try:
            # TODO: Add params to ds.attrs.
            # ds.attrs.update(params)

            ds.to_netcdf(
                path=netcdf_path,
                mode='w',  
                format="NETCDF4",
                engine="netcdf4"
            )
        
            # Save the same params into a JSON sidecar
            with open(metadata_path, 'w') as f:
                json.dump(params, f, indent=2)
            
        except Exception as e:
            raise Exception(f"Failed to save Dataset to NetCDF cache: {e}")
        finally:
            # restore the attribute in memory
            # so that selection methods continue to work.
            if time_indexes is not None:
                ds.coords['time'].attrs['indexes'] = time_indexes
            
    def _convert_to_xarray(self, df: pd.DataFrame, columns, frequency: FrequencyType = FrequencyType.DAILY) -> xr.Dataset:
        """
        Convert pandas DataFrame to xarray Dataset.
        """
        # Ensure 'date' is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        return DatasetDateTimeAccessor.from_table(
            df,
            time_column='date',
            asset_column='identifier',
            feature_columns=columns,
            frequency=frequency
        )

    @abstractmethod
    def load_data(self, **kwargs) -> Union[xr.Dataset, xr.DataTree]:
        """Abstract method to load data."""
        pass
