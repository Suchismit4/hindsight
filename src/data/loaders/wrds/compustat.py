# data/loaders/compustat/fundv.py

from src.data.abstracts.base import BaseDataSource
import pandas as pd
import xarray as xr
from typing import *
from pathlib import Path
import xarray_jax

import pyreadstat
import os

from src.data.core.struct import FrequencyType

class CompustatDataFetcher(BaseDataSource):
    """
    Data loader for Compustat data.

    This loader provides access to Compustat data from a local mounted path.
    Filters work similar to SQL examples at /wrds/crsp/samples/sample_programs/ResearchApps/ff3_crspCIZ.ipynb
    or the official WRDS API.
    """

    LOCAL_SRC: str = "/wrds/comp/sasdata/d_na/funda.sas7bdat"

    def load_data(self, **config) -> xr.Dataset:
        """
        Load Compustat data with caching support.

        Args:
            columns_to_read: List of columns to read from the dataset.
            filters: Optional dictionary of filters to apply to the data.
            num_processes: Number of processes to use for reading the data.
            **kwargs: Additional arguments (not used).

        Returns:
            xr.Dataset: Dataset containing the requested Compustat data.
        """

        # Extract configurations
        columns_to_read = config.get('columns_to_read', [])
        filters = config.get('filters', {})
        num_processes = config.get('num_processes', 16)

        # Get file stats
        file_stat = os.stat(self.LOCAL_SRC)
        file_size = file_stat.st_size
        file_mod_time = file_stat.st_mtime

        # Collect all parameters into a dictionary
        params = {
            'columns_to_read': columns_to_read,
            'filters': filters,
            'funda_file_size': file_size,
            'funda_file_mod_time': file_mod_time,
        }

        # Generate the cache path based on parameters
        cache_path = self.get_cache_path(**params)

        # Try to load from cache first
        data = self.load_from_cache(cache_path, frequency=FrequencyType.YEARLY)
        if data is not None:
            return data

        # If no cache or cache failed, load from source
        loaded_data = self._load_local(columns_to_read, filters, num_processes)

        loaded_data = self._convert_to_xarray(loaded_data, 
                                              list(loaded_data.columns.drop(['date', 'identifier'])), 
                                              frequency=FrequencyType.YEARLY)

        return loaded_data

    def _load_local(self, columns_to_read: List[str], filters: Dict[str, Any], num_processes: int) -> xr.Dataset:
        """
        Load data from Compustat source file and cache it.

        Args:
            columns_to_read: List of columns to read.
            filters: Dictionary of filters to apply to the data.
            num_processes: Number of processes to use in reading the file.
        """

        # Load the data using pyreadstat
        df, meta = pyreadstat.read_file_multiprocessing(
            pyreadstat.read_sas7bdat,
            self.LOCAL_SRC,
            usecols=columns_to_read,
            num_processes=num_processes
        )
        
        # Convert 'datadate' from SAS date to datetime
        # SAS epoch is January 1, 1960
        sas_epoch = pd.to_datetime('1960-01-01')
        df['datadate'] = sas_epoch + pd.to_timedelta(df['datadate'], unit='D')
                
        # Apply filters to the DataFrame
        df = self._apply_filters(df, filters)

        # Ensure date is datetime and rename 'datadate' to 'date'
        df.rename(columns={'datadate': 'date'}, inplace=True)

        # Rename 'gvkey' to 'identifier'
        df.rename(columns={'gvkey': 'identifier'}, inplace=True)

        # Select and order columns
        required_columns = ['date', 'identifier'] + [col for col in df.columns if col not in ['date', 'identifier']]

        df = df[required_columns]

        # Sort by date and identifier
        df.sort_values(['date', 'identifier'], inplace=True)
        df.reset_index(drop=True, inplace=True)
                  
        return df