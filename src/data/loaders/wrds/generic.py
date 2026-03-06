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
- External table merging at DataFrame level (before xarray conversion)
- Integration with the post-processing framework
"""

import os
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import xarray as xr
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from src.data.loaders.abstracts.base import BaseDataSource
from src.data.core.types import FrequencyType
from src.data.processors import ProcessorsList, ProcessorsDictConfig

def multiprocess_read(src: str, num_processes: int, columns_to_read: Optional[List[str]] = None) -> pd.DataFrame:
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


def _load_external_table(src: str, num_processes: int = 16) -> pd.DataFrame:
    """
    Load an external SAS table for merging.
    
    Args:
        src: Path to the SAS file
        num_processes: Number of processes for parallel reading
        
    Returns:
        DataFrame with lowercase column names
    """
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
        future = executor.submit(multiprocess_read, src, num_processes, None)
        df = future.result()
    
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()
    return df

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
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        external_tables: Optional[List[Dict[str, Any]]] = None,
        **config
    ) -> pd.DataFrame:
        """
        Preprocess the raw DataFrame for standardization and cleanup.
        
        This method performs standard data preprocessing steps:
        1. Normalizes column names to lowercase
        2. Converts date columns to proper datetime format
        3. Renames identifier columns for consistency
        4. Applies date range filtering based on start_date and end_date
        5. Applies additional filters to subset the data
        6. Merges external tables (lookup, replace, timeseries)
        7. Sorts and resets the index
        
        Subclasses can override or extend this method to implement
        data source-specific preprocessing.

        Args:
            df: The initial, raw DataFrame
            date_col: Name of the SAS date column (e.g., 'date', 'datadate')
            identifier_col: Name of the entity column (e.g., 'permno', 'gvkey')
            filters: Django-style filters (e.g., {"column__gte": value})
            filters_config: Explicit filter configurations (for advanced cases)
            start_date: Start date for filtering (e.g., "2020-01-01")
            end_date: End date for filtering (e.g., "2024-01-01")
            external_tables: List of external table merge configurations
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

        # Apply date range filtering based on start_date and end_date
        if start_date or end_date:
            if 'date' in df.columns:
                pre_filter_len = len(df)
                
                # Ensure date column is datetime
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'])
                
                if start_date:
                    start_date_dt = pd.to_datetime(start_date)
                    df = df[df['date'] >= start_date_dt]
                
                if end_date:
                    end_date_dt = pd.to_datetime(end_date)
                    df = df[df['date'] <= end_date_dt]
                
                post_filter_len = len(df)
                print(f"Date range filter [{start_date} to {end_date}]: {pre_filter_len} -> {post_filter_len} rows")
            else:
                print("WARNING: start_date/end_date provided but no 'date' column found in DataFrame")

        # Apply external table merges (lookup, replace, timeseries)
        if external_tables:
            df = self._apply_external_tables(df, external_tables, identifier_col='identifier')

        # Apply additional filters - give precedence to filters_config if both are provided
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
            'A': FrequencyType.ANNUAL,
        }
        return freq_map.get(freq_str.upper() if freq_str else '', FrequencyType.DAILY)

    def _apply_external_tables(
        self, 
        df: pd.DataFrame, 
        external_tables: List[Dict[str, Any]],
        identifier_col: str = 'identifier'
    ) -> pd.DataFrame:
        """
        Apply external table merges to the DataFrame.
        
        This method handles merging external SAS tables into the main DataFrame
        before conversion to xarray. Supports three types of merges:
        
        1. lookup: Merge static lookup values (like company names from msenames)
        2. replace: Replace values in the main DataFrame with values from external table
        3. timeseries: Merge time-series data that needs date alignment
        
        Args:
            df: Main DataFrame to merge into
            external_tables: List of external table configurations
            identifier_col: Column name for the identifier in the main DataFrame
            
        Returns:
            DataFrame with external tables merged
            
        Example external_tables config:
            [
                {
                    "path": "/wrds/crsp/sasdata/a_stock/msenames.sas7bdat",
                    "type": "lookup",
                    "on": "permno",
                    "columns": ["comnam", "exchcd", "shrcd"]
                },
                {
                    "path": "/wrds/crsp/sasdata/a_stock/msedelist.sas7bdat",
                    "type": "replace",
                    "on": "permno",
                    "time_column": "dlstdt",
                    "from_column": "dlret",
                    "to_column": "ret"
                }
            ]
        """
        if not external_tables:
            return df
            
        for table_config in external_tables:
            merge_type = table_config.get("type", "lookup")
            
            if merge_type == "lookup":
                df = self._merge_lookup_table(df, table_config, identifier_col)
            elif merge_type == "replace":
                df = self._merge_replace_values(df, table_config, identifier_col)
            elif merge_type == "timeseries":
                df = self._merge_timeseries_table(df, table_config, identifier_col)
            else:
                print(f"WARNING: Unknown external table merge type: {merge_type}")
                
        return df

    def _merge_lookup_table(
        self, 
        df: pd.DataFrame, 
        config: Dict[str, Any],
        identifier_col: str
    ) -> pd.DataFrame:
        """
        Merge static lookup values from an external table.
        
        This is used for tables like msenames where we want to add columns
        (like company name, exchange code) that don't vary by date.
        
        Args:
            df: Main DataFrame
            config: Lookup table configuration with keys:
                - path: Path to the SAS file
                - on: Column to join on in external table
                - columns: List of columns to merge
            identifier_col: Column name for identifier in main DataFrame
            
        Returns:
            DataFrame with lookup columns added
        """
        path = config.get("path")
        on_col = config.get("on", "permno").lower()
        columns = config.get("columns", [])
        
        if not path or not columns:
            print(f"WARNING: Skipping lookup merge - missing path or columns")
            return df
            
        print(f"Merging lookup table: {path}")
        
        # Load external table
        ext_df = _load_external_table(path)
        
        # Ensure the join column exists
        if on_col not in ext_df.columns:
            print(f"WARNING: Column '{on_col}' not found in {path}")
            return df
        
        # Normalize column names we want
        columns = [c.lower() for c in columns]
        available_cols = [c for c in columns if c in ext_df.columns]
        
        if not available_cols:
            print(f"WARNING: None of the requested columns {columns} found in {path}")
            return df
        
        # Keep only the columns we need plus the join column
        ext_df = ext_df[[on_col] + available_cols].copy()
        
        # Drop duplicates, keeping the last occurrence (most recent)
        ext_df = ext_df.drop_duplicates(subset=on_col, keep='last')
        
        # Map the identifier column name
        # In the main df, identifiers are in 'identifier' column
        # In external df, they're in 'on_col' (e.g., 'permno')
        
        # Get the original identifier values for mapping
        if identifier_col in df.columns:
            # Create mapping from external table
            for col in available_cols:
                mapping = dict(zip(ext_df[on_col], ext_df[col]))
                df[col] = df[identifier_col].map(mapping)
                
        print(f"  Added columns: {available_cols}")
        return df

    def _merge_replace_values(
        self, 
        df: pd.DataFrame, 
        config: Dict[str, Any],
        identifier_col: str
    ) -> pd.DataFrame:
        """
        Replace values in the main DataFrame with values from an external table.
        
        This is used for tables like msedelist where we want to replace return values
        with delisting returns when available.
        
        Args:
            df: Main DataFrame
            config: Replace configuration with keys:
                - path: Path to the SAS file
                - on: Column to join on in external table
                - time_column: Date column in external table
                - from_column: Column in external table to get values from
                - to_column: Column in main DataFrame to replace values in
            identifier_col: Column name for identifier in main DataFrame
            
        Returns:
            DataFrame with replaced values
        """
        path = config.get("path")
        on_col = config.get("on", "permno").lower()
        time_col = config.get("time_column", "dlstdt").lower()
        from_col = config.get("from_column", "dlret").lower()
        to_col = config.get("to_column", "ret").lower()
        
        if not path:
            print(f"WARNING: Skipping replace merge - missing path")
            return df
            
        if to_col not in df.columns:
            print(f"WARNING: Target column '{to_col}' not found in main DataFrame")
            return df
            
        print(f"Merging replace values from: {path}")
        
        # Load external table
        ext_df = _load_external_table(path)
        
        # Check required columns exist
        required = [on_col, from_col]
        if time_col:
            required.append(time_col)
            
        missing = [c for c in required if c not in ext_df.columns]
        if missing:
            print(f"WARNING: Missing columns {missing} in {path}")
            return df
        
        # Convert SAS date if numeric
        if time_col and time_col in ext_df.columns:
            if pd.api.types.is_numeric_dtype(ext_df[time_col]):
                sas_epoch = pd.to_datetime('1960-01-01')
                ext_df[time_col] = sas_epoch + pd.to_timedelta(ext_df[time_col].astype(float), unit='D')
        
        # Keep only needed columns
        cols_to_keep = [on_col, from_col]
        if time_col:
            cols_to_keep.append(time_col)
        ext_df = ext_df[cols_to_keep].copy()
        
        # Drop rows where the replacement value is NaN
        ext_df = ext_df.dropna(subset=[from_col])
        
        if ext_df.empty:
            print(f"  No valid replacement values found")
            return df
        
        # Merge on identifier and date
        # We need to match on both identifier and date
        if 'date' in df.columns and time_col:
            # Create a merge key
            ext_df = ext_df.rename(columns={on_col: identifier_col, time_col: 'date'})
            
            # Merge and replace
            merged = df.merge(
                ext_df[[identifier_col, 'date', from_col]], 
                on=[identifier_col, 'date'], 
                how='left'
            )
            
            # Replace values where external value is not NaN
            mask = merged[from_col].notna()
            original_nans = df[to_col].isna().sum()
            df.loc[mask.values, to_col] = merged.loc[mask, from_col].values
            new_nans = df[to_col].isna().sum()
            
            print(f"  Replaced {mask.sum()} values in '{to_col}'")
            print(f"  NaN count: {original_nans} -> {new_nans}")
        else:
            print(f"  WARNING: Cannot merge - 'date' column not found")
            
        return df

    def _merge_timeseries_table(
        self, 
        df: pd.DataFrame, 
        config: Dict[str, Any],
        identifier_col: str
    ) -> pd.DataFrame:
        """
        Merge time-series data from an external table.
        
        This is used for tables like distributions where we want to add
        time-varying columns that need date alignment.
        
        Args:
            df: Main DataFrame
            config: Timeseries configuration with keys:
                - path: Path to the SAS file
                - on: Column to join on in external table
                - time_column: Date column in external table
                - columns: List of columns to merge
            identifier_col: Column name for identifier in main DataFrame
            
        Returns:
            DataFrame with timeseries columns added
        """
        path = config.get("path")
        on_col = config.get("on", "permno").lower()
        time_col = config.get("time_column", "date").lower()
        columns = config.get("columns", [])
        
        if not path or not columns:
            print(f"WARNING: Skipping timeseries merge - missing path or columns")
            return df
            
        print(f"Merging timeseries table: {path}")
        
        # Load external table
        ext_df = _load_external_table(path)
        
        # Normalize column names
        columns = [c.lower() for c in columns]
        available_cols = [c for c in columns if c in ext_df.columns]
        
        if not available_cols:
            print(f"WARNING: None of the requested columns {columns} found in {path}")
            return df
        
        # Check required columns
        if on_col not in ext_df.columns or time_col not in ext_df.columns:
            print(f"WARNING: Missing join columns in {path}")
            return df
        
        # Convert SAS date if numeric
        if pd.api.types.is_numeric_dtype(ext_df[time_col]):
            sas_epoch = pd.to_datetime('1960-01-01')
            ext_df[time_col] = sas_epoch + pd.to_timedelta(ext_df[time_col].astype(float), unit='D')
        
        # Keep only needed columns
        ext_df = ext_df[[on_col, time_col] + available_cols].copy()
        
        # Rename for merge
        ext_df = ext_df.rename(columns={on_col: identifier_col, time_col: 'date'})
        
        # Merge on identifier and date
        if 'date' in df.columns:
            df = df.merge(
                ext_df,
                on=[identifier_col, 'date'],
                how='left'
            )
            print(f"  Added columns: {available_cols}")
        else:
            print(f"  WARNING: Cannot merge - 'date' column not found")
            
        return df
