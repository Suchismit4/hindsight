# data/loaders/wrds/generic.py

import os
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import xarray as xr
import pyreadstat

from src.data.abstracts.base import BaseDataSource
from src.data.core.struct import FrequencyType

class GenericWRDSDataLoader(BaseDataSource):
    """
    Generic data loader for WRDS data from a locally mounted path.

    This class encapsulates:
      - Checking file stats for caching logic
      - Optionally reading specific columns
      - Handling multi-processing with pyreadstat
      - A placeholder for specialized DataFrame preprocessing (overridden in child classes)
    """

    LOCAL_SRC: str = ""                             # Child classes should override
    FREQUENCY: FrequencyType = FrequencyType.DAILY  # Default, can be overridden

    def load_data(self, **config) -> xr.Dataset:
        """
        Load WRDS data with caching support (if desired). 

        Args:
            **config: Configuration parameters which might include:
                - columns_to_read
                - filters
                - num_processes
                - etc.
        Returns:
            xr.Dataset
        """

        # Extract configurations
        num_processes = config.get('num_processes', 16)

        # Get file stats
        file_stat = os.stat(self.LOCAL_SRC)
        file_size = file_stat.st_size
        file_mod_time = file_stat.st_mtime

        # Collect parameters for caching
        params = {
            'file_size': file_size,
            'file_mod_time': file_mod_time,
        }
        # Add any other config-based parameters to `params`
        # (e.g., columns_to_read, filters) so they’re included in the hash
        for key in ['columns_to_read', 'filters']:
            if key in config:
                params[key] = config[key]

        # Generate the cache path based on parameters
        cache_path = self.get_cache_path(**params)

        # TODO: Try to load from cache first
        # data = self.load_from_cache(cache_path, frequency=self.FREQUENCY)
        # if data is not None:
        #     return data

        # If no cache found or load failed, read from local source
        df = self._load_local(num_processes=num_processes, **config)
        df = self._preprocess_df(df, **config)

        # Convert to xarray
        # Identify which columns are data columns
        non_data_cols = ['date', 'identifier']
        data_cols = [col for col in df.columns if col not in non_data_cols]
        ds = self._convert_to_xarray(df, data_cols, frequency=self.FREQUENCY)

        # TODO: Cache the resulting xr.Dataset
        # self.save_to_cache(ds, cache_path, frequency=self.FREQUENCY)

        return ds

    def _load_local(
        self, 
        num_processes: int = 16,
        columns_to_read: Optional[List[str]] = None,
        **config
    ) -> pd.DataFrame:
        """
        Loads data from LOCAL_SRC using pyreadstat (with optional multi-processing).

        Args:
            num_processes: Number of processes to use.
            columns_to_read: Subset of columns to read from the file.
            **config: Additional keyword arguments.

        Returns:
            pd.DataFrame
        """
        
        read_kwargs = {
            "num_processes": num_processes,
        }
        
        if columns_to_read:
            read_kwargs["usecols"] = columns_to_read

        # Load using pyreadstat
        df, meta = pyreadstat.read_file_multiprocessing(
            pyreadstat.read_sas7bdat,
            self.LOCAL_SRC,
            **read_kwargs
        )
        
        return df

    def _preprocess_df(self, df: pd.DataFrame, **config) -> pd.DataFrame:
        """
        Placeholder for child classes to implement dataset-specific cleaning steps:
          - SAS date conversions
          - Column renaming
          - Filter application
          - Sorting, resetting index
          - etc.

        Returns:
            pd.DataFrame
        """
        return df
