# data/loaders/wrds/generic.py

"""
Generic WRDS data loader implementation.

This loader reads a primary SAS table, converts it to an xarray.Dataset (using
the shared Loader.from_table logic in core/util.py), and then applies a sequence
of atomic postprocessing operations defined in the configuration.
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
    A generic WRDS data loader that loads and assembles multiple SAS tables into a single xarray.Dataset
    based on a configurable YAML/JSON file.
    """

    # Default frequency; can be overridden in subclasses if needed
    FREQUENCY: FrequencyType = FrequencyType.DAILY
    
    # Subclasses must define the LOCAL_SRC attribute
    LOCAL_SRC: str = ""

    def load_data(self, **config) -> xr.Dataset:
        """
        Load the primary SAS table and then apply postprocessing operations
        as defined by the configuration.

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

        Returns:
            xr.Dataset: The assembled dataset.
            
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

        # Ensure the dataset is sliced by the requested date range.
        # Not done here due to performance issues with large datasets
        # ds = self.subset_by_date(ds, config)

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
            num_processes: Number of processes for pyreadstat.
            columns_to_read: Subset of columns to read.
            **config: Additional keyword arguments.

        Returns:
            pd.DataFrame: The raw loaded data.
            
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
        **config
    ) -> pd.DataFrame:
        """
        A 'generic' preprocessing step that child classes can call or override.

        lower-case columns, reset index, convert date column,
        rename identifier column, apply filters, and sort.

        Args:
            df: The initial, raw DataFrame.
            date_col: Name of the SAS date column (e.g., 'date', 'datadate').
            identifier_col: Name of the entity column (e.g., 'permno', 'gvkey').
            filters: Django-style filters (e.g., {"column__gte": value})
            filters_config: Explicit filter configurations (for advanced cases)
            **config: Additional keyword arguments.

        Returns:
            pd.DataFrame: The cleaned/preprocessed DataFrame.
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

        Args:
            sas_date_col: Column of SAS date ints.
            epoch: Base epoch for SAS (default '1960-01-01').

        Returns:
            pd.Series: Date column in datetime format.
            
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
        Convert a frequency string (e.g. 'D', 'W', 'M', 'Y') to a FrequencyType enum.
        Defaults to DAILY if freq_str is unrecognized.

        Args:
            freq_str: One of "D", "W", "M", "Y", etc.

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
        return freq_map.get(freq_str.upper() if freq_str else '', FrequencyType.DAILY)
