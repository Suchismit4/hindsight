# data/loaders/wrds/compustat.py

from src.data.abstracts.base import BaseDataSource
import pandas as pd
import xarray as xr
from typing import *
from pathlib import Path
import xarray_jax

import pyreadstat
import os

from src.data.core.struct import FrequencyType

class CRSPDataFetcher(BaseDataSource):
    """
    Data loader for CRSP data.

    This loader provides access to CRSP data from a local mounted path.
    Filters work similar to SQL examples at /wrds/crsp/samples/sample_programs/ResearchApps/ff3_crspCIZ.ipynb
    or the official WRDS API.
    """

    LOCAL_SRC: str = "/wrds/crsp/sasdata/a_stock/dsf.sas7bdat" 

    def load_data(self, **config) -> xr.Dataset:
        """
        Load CRSP data with caching support.

        Args:
            columns_to_read: List of columns to read from the dataset.
            num_processes: Number of processes to use for reading the data.
            **kwargs: Additional arguments (not used).

        Returns:
            xr.Dataset: Dataset containing the requested Compustat data.
        """

        # Extract configurations
        num_processes = config.get('num_processes', 16)

        # Get file stats
        file_stat = os.stat(self.LOCAL_SRC)
        file_size = file_stat.st_size
        file_mod_time = file_stat.st_mtime

        # Collect all parameters into a dictionary
        params = {
            'funda_file_size': file_size,
            'funda_file_mod_time': file_mod_time,
        }

        # Generate the cache path based on parameters
        cache_path = self.get_cache_path(**params)

        # Try to load from cache first
        data = self.load_from_cache(cache_path, frequency=FrequencyType.DAILY)
        if data is not None:
            return data

        # If no cache or cache failed, load from source
        loaded_data = self._load_local(num_processes)

        # Set frequency type
        freq = {
            'D': FrequencyType.DAILY,
            'W': FrequencyType.WEEKLY,
            'M': FrequencyType.MONTHLY
            'Y': FrequencyType.YEARLY
        }[config.get("frequency")]

        loaded_data = self._convert_to_xarray(loaded_data, 
                                              ['issuno', 'hexcd', 'hsiccd',
       'bidlo', 'askhi', 'prc', 'vol', 'ret', 'bid', 'ask', 'shrout', 'cfacpr',
       'cfacshr', 'openprc', 'numtrd', 'retx'], 
                                              frequency=freq)
        
        # TODO: Cache

        return loaded_data

    def _load_local(self, num_processes: int) -> xr.Dataset:
        """
        Load data from CRSP source file and cache it.
        Args:
            num_processes: Number of processes to use in reading the file.
        """
        
        # Load the data using pyreadstat
        df, meta = pyreadstat.read_file_multiprocessing(
            pyreadstat.read_sas7bdat,
            self.LOCAL_SRC,
            num_processes=num_processes
        )
        df.columns = df.columns.str.lower()
        df.reset_index(inplace=True)
        
        sas_epoch = pd.to_datetime('1960-01-01')
        df['date'] = df['date'].astype(int)
        df['date'] = sas_epoch + pd.to_timedelta(df['date'], unit='D')

        # Convert 'permco' and 'permno' to integers
        df[['permco', 'permno']] = df[['permco', 'permno']].astype(int)

        # Rename 'permno' to 'identifier'
        df.rename(columns={'permno': 'identifier'}, inplace=True)

        # Select and order columns
        required_columns = ['date', 'identifier'] + [col for col 
                                                     in df.columns if col 
                                                     not in ['date', 'identifier']]
        df = df[required_columns]

        return df
