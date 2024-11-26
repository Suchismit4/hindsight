# src/data/loaders/base.py

import os
import pandas as pd
import xarray as xr
import json
from typing import Dict, Any, Union
from pathlib import Path
import hashlib
from abc import ABC, abstractmethod

from ..core.struct import DatasetDateTimeAccessor
from ..core.struct import FrequencyType

class BaseDataSource(ABC):
    """
    Base class for handling data sources configuration and path management.

    This class provides the foundational logic for accessing different data sources
    defined in the paths.yaml configuration file. It handles both online and offline
    data sources and manages the interaction with the cache system.
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
        Generate a cache file path based on the data path and parameters.

        Args:
            **params: Parameters used in the data loading function.

        Returns:
            Path: The corresponding cache file path.
        """
        # Serialize the parameters to a JSON-formatted string
        params_string = json.dumps(params, sort_keys=True)
        # Generate a hash of the parameters string
        params_hash = hashlib.md5(params_string.encode('utf-8')).hexdigest()
        # Create the cache path using the data path and hash
        cache_path = os.path.join(
            self.cache_root,
            self.data_path.strip('/'),  # Remove leading/trailing slashes
            params_hash
        )
        return Path(f"{cache_path}.parquet")

    def check_cache(self, cache_path: Path) -> bool:
        """Check if valid cache exists for the given path."""
        return cache_path.exists()

    def load_from_cache(self, cache_path: Path, frequency: FrequencyType = FrequencyType.DAILY) -> Union[xr.Dataset, None]:
        """
        Load data from cache file.

        Args:
            cache_path: Path to the cache file.

        Returns:
            xr.Dataset or None: The loaded dataset, or None if loading failed.
        """
        if self.check_cache(cache_path):
            try:
                # Load the parquet file into a pandas DataFrame
                df = pd.read_parquet(cache_path)
                data = self._convert_to_xarray(df, 
                                               list(df.columns.drop(['date', 'identifier'])), 
                                               frequency=frequency)
                return data
            except Exception as e:
                print(f"Failed to load from cache: {e}")
                return None
        else:
            return None

    def save_to_cache(self, df: pd.DataFrame, cache_path: Path, params: dict):
        """
        Save data to cache file.

        Args:
            df: The DataFrame to save.
            cache_path: Path to the cache file.
            params: Parameters used in data loading, saved as metadata.
        """
        # Save the DataFrame to parquet
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)

        # Save metadata
        metadata_path = cache_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(params, f)

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
