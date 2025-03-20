# data/loaders/wrds/generic.py

"""
Generic WRDS data loader implementation.

This module provides a GenericWRDSDataLoader class that serves as the foundation
for loading financial data from WRDS (Wharton Research Data Services) databases.
The loader reads a primary SAS table, converts it to an xarray.Dataset, and applies
a sequence of post-processing operations defined in the configuration.

Key features:
- Multi-process SAS file reading for performance
- Configurable column selection and filtering
- Standardized date handling and identifier mapping
- Integration with the post-processing framework
"""

import os
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import xarray as xr
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from src.data.loaders.abstracts.base import BaseDataSource
from src.data.core.util import FrequencyType
from src.data.processors import ProcessorsList, ProcessorsDictConfig

def multiprocess_read(src: str, num_processes: int, columns_to_read: Optional[List[str]]) -> pd.DataFrame:
    """
    Read a SAS file using multiprocessing for better performance.
    
    Utilizes pyreadstat's multiprocessing capability to efficiently read large SAS files
    by distributing the work across multiple CPU cores.
    
    Args:
        src: Path to the SAS file
        num_processes: Number of processes for parallel reading
        columns_to_read: Optional list of columns to include
        
    Returns:
        DataFrame containing the data from the SAS file
        
    Raises:
        FileNotFoundError: If the source file doesn't exist
        RuntimeError: If there's an error reading the SAS file
    """
    import pyreadstat  # local import to ensure clean process state
    read_kwargs = {"num_processes": num_processes}
    if columns_to_read:
        read_kwargs["usecols"] = columns_to_read
    
    try:
        df, meta = pyreadstat.read_file_multiprocessing(
            pyreadstat.read_sas7bdat,
            src,
            **read_kwargs
        )
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"SAS file not found: {src}")
    except Exception as e:
        raise RuntimeError(f"Error reading SAS file: {str(e)}")

class GenericWRDSDataLoader(BaseDataSource):
    """
    A generic WRDS data loader that loads and assembles SAS data into xarray Datasets.
    
    This class provides the foundation for all WRDS data loaders, handling the common
    tasks of loading SAS files, preprocessing the data, and converting to xarray format.
    Specific WRDS data loaders (like CRSP, Compustat) inherit from this class and
    customize the behavior as needed.
    
    Attributes:
        FREQUENCY (FrequencyType): Default frequency for the data (e.g., DAILY)
        LOCAL_SRC (str): Path to the SAS file (must be set by subclasses)
    """

    # Default frequency; can be overridden in subclasses if needed
    FREQUENCY: FrequencyType = FrequencyType.DAILY
    
    # Subclasses must define the LOCAL_SRC attribute
    LOCAL_SRC: str = ""

    def load_data(self, **config) -> xr.Dataset:
        """
        Load the primary SAS table and apply post-processing operations.
        
        This method orchestrates the data loading process:
        1. Loads the raw data from the SAS file
        2. Preprocess the data (column name normalization, date conversion, etc.)
        3. Converts to xarray Dataset with proper dimensions
        
        Standard configuration keys include:
          - num_processes: number of processes to use (default 16)
          - frequency: "D", "W", "M", "Y", etc.
          - columns_to_read: list of columns for the primary table
          - date_col: name of the SAS date column (e.g. "datadate")
          - identifier_col: name of the entity column (e.g. "gvkey")
          - filters: Django-style filter dictionary (e.g., {"column__gte": value})
          - filters_config: explicit filter configuration list (for advanced cases)
          
        Post-processor options (one of the following):
          - postprocessors: traditional format post-processors (list of dicts)
          - processors: Django-style post-processors (dictionary) - recommended
            
        Example of Django-style processors:
        ```python
        processors = {
            "set_permno_coord": True,
            "set_permco_coord": True,
            "fix_market_equity": True,
            "merge_table": [
                {
                    "source": "msenames",
                    "axis": "asset",
                    "column": "comnam"
                },
                {
                    "source": "msenames",
                    "axis": "asset",
                    "column": "exchcd"
                }
            ]
        }
        ```

        Args:
            **config: Configuration parameters for data loading and processing
            
        Returns:
            xr.Dataset: The assembled dataset
            
        Raises:
            ValueError: If LOCAL_SRC is not defined in the subclass
        """
        if not self.LOCAL_SRC:
            raise ValueError("LOCAL_SRC must be defined in subclass")
            
        # Parse frequency
        user_freq_str = config.get("frequency", None)
        freq_enum = (
            self._parse_frequency(user_freq_str) if user_freq_str else self.FREQUENCY
        )

        # Load and process the primary table
        df = self._load_local(**config)
        df = self._preprocess_df(df, **config)
    
        # Identify data columns vs. metadata columns
        non_data_cols = ["date", "identifier"]
        data_cols = [col for col in df.columns if col not in non_data_cols]
        
        # Convert to xarray Dataset
        ds = self._convert_to_xarray(df, data_cols, frequency=freq_enum)

        return ds

    def _load_local(
        self, 
        num_processes: int = 16,
        columns_to_read: Optional[List[str]] = None,
        **config
    ) -> pd.DataFrame:
        """
        Load data from LOCAL_SRC using pyreadstat with multiprocessing.
        
        Creates a clean multiprocessing context to ensure thread safety when
        loading SAS files, which is especially important when working with
        JAX-enabled environments.

        Args:
            num_processes: Number of processes for pyreadstat
            columns_to_read: Subset of columns to read
            **config: Additional keyword arguments

        Returns:
            The raw loaded data as a DataFrame
            
        Raises:
            ValueError: If LOCAL_SRC is not defined
        """
        if not self.LOCAL_SRC:
            raise ValueError("LOCAL_SRC must be defined in subclass")

        # Create a spawn-context to ensure a clean process without JAX/xarray state
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
            future = executor.submit(multiprocess_read, self.LOCAL_SRC, num_processes, columns_to_read)
            df = future.result()
        
        return df

    def _preprocess_df(
        self, 
        df: pd.DataFrame,
        date_col: Optional[str] = None,
        identifier_col: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        filters_config: Optional[List[Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """
        Preprocess the raw DataFrame for standardization and cleanup.
        
        This method performs standard data preprocessing steps:
        1. Normalizes column names to lowercase
        2. Converts date columns to proper datetime format
        3. Renames identifier columns for consistency
        4. Applies filters to subset the data
        5. Sorts and resets the index
        
        Subclasses can override or extend this method to implement
        data source-specific preprocessing.

        Args:
            df: The initial, raw DataFrame
            date_col: Name of the SAS date column (e.g., 'date', 'datadate')
            identifier_col: Name of the entity column (e.g., 'permno', 'gvkey')
            filters: Django-style filters (e.g., {"column__gte": value})
            filters_config: Explicit filter configurations (for advanced cases)
            **config: Additional keyword arguments

        Returns:
            The cleaned/preprocessed DataFrame
        """            
        if df.empty:
            return df
            
        # Normalize column names
        df.columns = df.columns.str.lower()
        df.reset_index(inplace=True, drop=True)

        # Date handling
        if date_col and (date_col_lower := date_col.lower()) in df.columns:
            df['date'] = self.convert_sas_date(df[date_col_lower])
            
        # Rename the identifier column
        if identifier_col and (id_col_lower := identifier_col.lower()) in df.columns:
            df = df.rename(columns={id_col_lower: 'identifier'})

        # Apply filters - give precedence to filters_config if both are provided
        if filters_config:
            df = self.apply_filters(df, filters_config)
        elif filters:
            df = self.apply_filters(df, filters)

        # Sort by date and identifier if they exist
        possible_sort_cols = [c for c in ['date', 'identifier'] if c in df.columns]
        if possible_sort_cols:
            df = df.sort_values(possible_sort_cols)  # Returns new DataFrame but optimized
        
        return df.reset_index(drop=True)

    @staticmethod
    def convert_sas_date(sas_date_col: pd.Series, epoch: str = '1960-01-01') -> pd.Series:
        """
        Convert a numeric SAS date column to a proper Pandas datetime.
        
        SAS dates are stored as number of days since the SAS epoch (January 1, 1960).
        This method converts these numeric values to proper datetime objects.

        Args:
            sas_date_col: Column of SAS date ints
            epoch: Base epoch for SAS (default '1960-01-01')

        Returns:
            Date column in datetime format
            
        Raises:
            ValueError: If the date conversion fails
        """
        sas_epoch = pd.to_datetime(epoch)
        try:
            return sas_epoch + pd.to_timedelta(sas_date_col.astype(int), unit='D')
        except (ValueError, TypeError):
            raise ValueError(f"Failed to convert SAS dates to datetime. Ensure the column contains valid numeric values.")
    
    @staticmethod
    def _parse_frequency(freq_str: Optional[str]) -> FrequencyType:
        """
        Convert a frequency string to a FrequencyType enum.
        
        Maps common frequency codes (D, W, M, Y) to the corresponding FrequencyType enum value.
        Defaults to DAILY if the frequency string is unrecognized.

        Args:
            freq_str: One of "D", "W", "M", "Y", etc.

        Returns:
            The corresponding FrequencyType enum value
        """
        # Map the known single-letter codes to FrequencyType
        freq_map = {
            'D': FrequencyType.DAILY,
            'W': FrequencyType.WEEKLY,
            'M': FrequencyType.MONTHLY,
            'Y': FrequencyType.YEARLY,
            'A': FrequencyType.ANNNUAL,
        }
        return freq_map.get(freq_str.upper() if freq_str else '', FrequencyType.DAILY)
