# data/loaders/wrds/crsp.py

from src.data.core.struct import FrequencyType
from src.data.abstracts.base import BaseDataSource
import pandas as pd
import xarray as xr
from typing import *
from pathlib import Path
import pyreadstat
import os

class CRSPDataFetcher(BaseDataSource):
    """
    Data loader for CRSP data.

    This loader provides access to CRSP data from a local mounted path.
    """

    LOCAL_SRC: str = "/wrds/crsp/sasdata/a_stock/dsf.sas7bdat"

    def _load_data(self, **config) -> xr.Dataset:
        """
        Load CRSP data with caching support.

        Args:
            columns_to_read: List of columns to read from the dataset.
            filters: Optional dictionary of filters to apply to the data.
            num_processes: Number of processes to use for reading the data.
            **kwargs: Additional arguments (not used).

        Returns:
            xr.Dataset: Dataset containing the requested CRSP data.
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
            'dsf_file_size': file_size,
            'dsf_file_mod_time': file_mod_time,
        }

        # If no cache or cache failed, load from source
        loaded_data = self._load_local(columns_to_read, filters, num_processes)

        # Set frequency type based on config parameters
        freq_map = {
            'D': FrequencyType.DAILY,
            'W': FrequencyType.WEEKLY,
            'M': FrequencyType.MONTHLY,
            'Y': FrequencyType.YEARLY
        }
        freq = freq_map.get(config.get('frequency'), None)

        # convert to xarray given params
        loaded_data = self._convert_to_xarray(
            loaded_data,
            list(loaded_data.columns.drop(['date', 'identifier'])),
            frequency=freq
        )

        return loaded_data

    def _get_cache_params(self, **config) -> Dict[str, Any]:
        """Get parameters used for cache key generation."""
        file_stat = os.stat(self.LOCAL_SRC)
        return {
            'columns_to_read': config.get('columns_to_read', []),
            'filters': config.get('filters', {}),
            'dsf_file_size': file_stat.st_size,
            'dsf_file_mod_time': file_stat.st_mtime,
        }

    def _load_local(self, columns_to_read: List[str], filters: Dict[str, Any], num_processes: int) -> pd.DataFrame:
        """
        Load data from CRSP source file and cache it.

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

        # Convert 'date' from SAS date to datetime
        sas_epoch = pd.to_datetime('1960-01-01')
        df['DATE'] = sas_epoch + pd.to_timedelta(df['DATE'], unit='D')

        # Apply filters to the DataFrame
        df = self._apply_filters(df, filters)

        # Rename 'permno' to 'identifier'
        df.rename(columns={'PERMNO': 'identifier'}, inplace=True)
        df.rename(columns={'DATE': 'date'}, inplace=True)

        # Select and order columns
        required_columns = ['date', 'identifier'] + [col for col in df.columns if col not in ['date', 'identifier']]
        df = df[required_columns]

        # Sort by date and identifier
        df.sort_values(['date', 'identifier'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df
