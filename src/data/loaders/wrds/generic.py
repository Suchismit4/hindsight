# data/loaders/wrds/generic.py

import os
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import xarray as xr
import pyreadstat

from src.data.abstracts.base import BaseDataSource
from src.data.core import FrequencyType

class GenericWRDSDataLoader(BaseDataSource):
    """
    A generic WRDS data loader that:
      - Loads data from a SAS file via pyreadstat (with optional multiprocessing).
      - Provides caching support stubs (to be implemented).
      - Offers a shared `_preprocess_df` workflow, including:
          * SAS date conversion
          * Identifier renaming
          * Column ordering
          * Simple filters
      - Converts the resulting DataFrame to an xarray.Dataset.
    
    Child classes (e.g., CRSP, Compustat) can override `_preprocess_df` or call it
    with different parameters to handle dataset-specific columns and logic.
    """

    LOCAL_SRC: str = ""                             # Path to local SAS file (override in child classes)
    FREQUENCY: FrequencyType = FrequencyType.DAILY  # Default frequency (override in child classes)

    def load_data(self, **config) -> xr.Dataset:
        """
        Load WRDS data with optional caching and transformation.

        Args:
            **config: Configuration parameters that might include:
                - columns_to_read (List[str]): Subset of columns to read from the file.
                - filters (Dict[str, Any]): Simple equality filters to apply post-load.
                - num_processes (int): Number of processes for pyreadstat (default=16).
                - date_col (str): Column name in the SAS file that represents the date.
                - identifier_col (str): Column name in the SAS file for the entity identifier.
                - frequency (str): The desired frequency (e.g., "D", "W", "M", "Y").
                - etc.

        Returns:
            xr.Dataset: The final dataset.
        """
        # Extract common configurations
        num_processes = config.get('num_processes', 16)
        user_freq_str = config.get('frequency', None)  # e.g., "D", "W", "M", "Y"

        # Parse frequency string -> FrequencyType enum; fallback to self.FREQUENCY if invalid
        freq_enum = self._parse_frequency(user_freq_str) if user_freq_str else self.FREQUENCY

        # Prepare params for caching
        params = {
            'freq': freq_enum.value  # store the enum's underlying value (e.g. "D", "Y", etc.)
        }
        for key in ['columns_to_read', 'filters']:
            if key in config:
                params[key] = config[key]

        # Construct a cache path (not fully implemented)
        cache_path = self.get_cache_path(**params)

        # Attempt cache load
        cached_ds = self.load_from_cache(cache_path, request_params=params)
        if cached_ds is not None:
            print("Loaded from NetCDF cache")
            return cached_ds

        # Load from local source
        df = self._load_local(**config)

        # Child classes or the base `_preprocess_df` will handle cleaning
        df = self._preprocess_df(df, **config)

        # Convert DataFrame to xarray.Dataset
        non_data_cols = ['date', 'identifier']
        data_cols = [col for col in df.columns if col not in non_data_cols]
        ds = self._convert_to_xarray(df, data_cols, frequency=freq_enum)

        # Save to cache
        self.save_to_cache(ds, cache_path, params)

        return ds

    def _load_local(
        self, 
        num_processes: int = 16,
        columns_to_read: Optional[List[str]] = None,
        **config
    ) -> pd.DataFrame:
        """
        Loads data from LOCAL_SRC using pyreadstat with optional multiprocessing.

        Args:
            num_processes (int): Number of processes for pyreadstat.
            columns_to_read (List[str]): Subset of columns to read.
            **config: Additional keyword arguments.

        Returns:
            pd.DataFrame: The raw loaded data.
        """
        read_kwargs = {
            "num_processes": num_processes,
        }
        if columns_to_read:
            read_kwargs["usecols"] = columns_to_read

        # Load using pyreadstat's multiprocessing read for .sas7bdat
        df, meta = pyreadstat.read_file_multiprocessing(
            pyreadstat.read_sas7bdat,
            self.LOCAL_SRC,
            **read_kwargs
        )

        return df

    def _preprocess_df(
        self, 
        df: pd.DataFrame,
        date_col: Optional[str] = None,
        identifier_col: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        **config
    ) -> pd.DataFrame:
        """
        A 'generic' preprocessing step that child classes can call or override.

        - Lower all column names
        - Resets index
        - Converts `date_col` from SAS numeric to datetime and renames it to 'date'
        - Renames `identifier_col` to 'identifier'
        - Applies any simple filters
        - Sorts by ('date', 'identifier') at the end

        Args:
            df (pd.DataFrame): The initial, raw DataFrame.
            date_col (str): Name of the SAS date column (e.g., 'date', 'datadate').
            identifier_col (str): Name of the entity column (e.g., 'permno', 'gvkey').
            filters (Dict[str, Any]): Simple equality filters to apply after loading.
            **config: Additional arguments.

        Returns:
            pd.DataFrame: The cleaned/preprocessed DataFrame.
        """
        # 1. Normalize column names
        df.columns = df.columns.str.lower()

        # 2. Reset index
        df.reset_index(inplace=True, drop=True)

        # 3. Date handling
        if date_col and date_col.lower() in df.columns:
            df[date_col] = self.convert_sas_date(df[date_col])
            df.rename(columns={date_col: 'date'}, inplace=True)

        # 4. Rename the identifier column
        if identifier_col and identifier_col.lower() in df.columns:
            df.rename(columns={identifier_col: 'identifier'}, inplace=True)

        # 5. Apply user-defined filters (simple equality)
        if filters:
            df = self._apply_filters(df, filters)

        # 6. Sort by date and identifier if they exist
        possible_sort_cols = [c for c in ['date', 'identifier'] if c in df.columns]
        if possible_sort_cols:
            df.sort_values(possible_sort_cols, inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df

    @staticmethod
    def convert_sas_date(sas_date_col: pd.Series, epoch: str = '1960-01-01') -> pd.Series:
        """
        Convert a numeric SAS date column to a proper Pandas datetime.

        Args:
            sas_date_col (pd.Series): Column of SAS date ints.
            epoch (str): Base epoch for SAS (default '1960-01-01').

        Returns:
            pd.Series: Date column in datetime format.
        """
        sas_epoch = pd.to_datetime(epoch)
        return sas_epoch + pd.to_timedelta(sas_date_col.astype(int), unit='D')

    @staticmethod
    def _parse_frequency(freq_str: str) -> FrequencyType:
        """
        Convert a frequency string (e.g. 'D', 'W', 'M', 'Y') to a FrequencyType enum.
        Defaults to DAILY if freq_str is unrecognized.

        Args:
            freq_str (str): One of "D", "W", "M", "Y", etc.

        Returns:
            FrequencyType: The corresponding enum value.
        """
        # Map the known single-letter codes to FrequencyType
        freq_map = {
            'D': FrequencyType.DAILY,
            'W': FrequencyType.WEEKLY,
            'M': FrequencyType.MONTHLY,
            'Y': FrequencyType.YEARLY,
            'A': FrequencyType.ANNNUAL,
        }
        return freq_map.get(freq_str.upper(), FrequencyType.DAILY)
