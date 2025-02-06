# src/data/loaders/abstracts/base.py

import os
import pandas as pd
import xarray as xr
import json
from typing import Dict, Any, Union, Optional, List
from pathlib import Path
import hashlib
from abc import ABC, abstractmethod

from src.data.core.util import FrequencyType
from src.data.core.util import Loader as load
from src.data.processors.registry import post_processor
from src.data.core.cache import CacheManager

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
        self.cache_manager = CacheManager()  # use the centralized cache manager
            
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

    def _apply_postprocessors(self, ds: xr.Dataset, postprocessors: List[str]) -> xr.Dataset:
        """
        Apply registered postprocessors to an xarray.Dataset sequentially as given.

        Args:
            ds (xr.Dataset): The dataset to be processed.
            postprocessors (List[str]): A list of names identifying the postprocessors to apply.

        Returns:
            xr.Dataset: The postprocessed dataset.
        """
        for processor_name in postprocessors:
            processor_func = post_processor.get(processor_name)
            if processor_func is None:
                raise ValueError(f"Postprocessor '{processor_name}' is not registered.")
            ds = processor_func(ds)
        return ds
   
    def _convert_to_xarray(self, df: pd.DataFrame, columns, frequency: FrequencyType = FrequencyType.DAILY) -> xr.Dataset:
        """
        Convert pandas DataFrame to xarray Dataset.
        """
        # Ensure 'date' is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        return load.from_table(
            df,
            time_column='date',
            asset_column='identifier',
            feature_columns=columns,
            frequency=frequency
        )

    def subset_by_date(self, ds: xr.Dataset, config: Dict[str, Any]) -> xr.Dataset:
        """
        Validate that both start_date and end_date are provided, and subset the dataset's time dimension.
        """
        if not config.get("start_date") or not config.get("end_date"):
            raise ValueError("Both 'start_date' and 'end_date' must be provided in the config.")
        return ds.dt.sel(time=slice(config["start_date"], config["end_date"]))

    @abstractmethod
    def load_data(self, **kwargs) -> Union[xr.Dataset, xr.DataTree]:
        """Abstract method to load data."""
        pass
