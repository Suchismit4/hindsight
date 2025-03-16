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
from src.data.filters.filters import apply_filters, parse_django_style_filters
from src.data.processors import apply_processors, ProcessorsList, ProcessorsDictConfig

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
            
    def apply_filters(self, df: pd.DataFrame, filters_config: Union[List[Dict[str, Any]], Dict[str, Any], None]) -> pd.DataFrame:
        """
        Apply filters to the DataFrame.
        
        Supports both explicit filter configurations and Django-style filter dictionaries.
        
        Args:
            df: The DataFrame to filter.
            filters_config: Either:
                - List of filter configurations (explicit format)
                - Dictionary of Django-style filters (e.g., {"column__gte": value})
                - None (no filtering will be performed)
            
        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        if filters_config is None:
            return df
            
        if isinstance(filters_config, dict):
            # Convert Django-style filters to explicit format
            filters_list = parse_django_style_filters(filters_config)
        else:
            # Already in explicit format
            filters_list = filters_config
            
        return apply_filters(df, filters_list)

    def _apply_postprocessors(self, ds: xr.Dataset, postprocessors: Union[ProcessorsList, ProcessorsDictConfig]) -> xr.Dataset:
        """
        Apply registered postprocessors to an xarray.Dataset.
        
        Supports both explicit post-processor configurations and Django-style dictionary format.

        Args:
            ds: The dataset to be processed.
            postprocessors: Either:
                - List of post-processor configurations (traditional format)
                - Dictionary of Django-style post-processors (e.g., {"set_permno_coord": True})

        Returns:
            The postprocessed dataset.
            
        Raises:
            ValueError: If a processor configuration is invalid or a processor is not found
        """
        if not postprocessors:
            return ds
        
        # Use the shared apply_processors function from the processors module
        return apply_processors(ds, postprocessors)
   
    def _convert_to_xarray(self, df: pd.DataFrame, columns: List[str], frequency: FrequencyType = FrequencyType.DAILY) -> xr.Dataset:
        """
        Convert pandas DataFrame to xarray Dataset.
        
        Args:
            df: DataFrame to convert
            columns: Feature columns to include in the Dataset
            frequency: Time frequency for the Dataset
            
        Returns:
            xarray Dataset with proper dimensions and coordinates
            
        Raises:
            ValueError: If 'date' or 'identifier' columns are missing
        """
        # Validate required columns
        if 'date' not in df.columns:
            raise ValueError("DataFrame must have a 'date' column")
        if 'identifier' not in df.columns:
            raise ValueError("DataFrame must have an 'identifier' column")
            
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
        
        Args:
            ds: Dataset to subset
            config: Configuration dictionary with start_date and end_date
            
        Returns:
            Subset of the dataset with time dimension filtered
            
        Raises:
            ValueError: If start_date or end_date is missing
        """
        if not config.get("start_date") or not config.get("end_date"):
            raise ValueError("Both 'start_date' and 'end_date' must be provided in the config.")
        return ds.dt.sel(time=slice(config["start_date"], config["end_date"]))

    @abstractmethod
    def load_data(self, **kwargs) -> Union[xr.Dataset, xr.DataTree]:
        """
        Abstract method to load data. Must be implemented by subclasses.
        
        Args:
            **kwargs: Configuration parameters for loading data
            
        Returns:
            Either an xarray Dataset or DataTree containing the loaded data
        """
        pass
