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
import pyreadstat
from copy import deepcopy

from src.data.loaders.abstracts.base import BaseDataSource
from src.data.core.util import FrequencyType
from src.data.processors.registry import post_processor  
from src.data.core.util import Loader

class GenericWRDSDataLoader(BaseDataSource):
    """
    A generic WRDS data loader that loads and assembles multiple SAS tables into a single xarray.Dataset
    based on a configurable YAML/JSON file.
    """
    
    DEFAULT_PATHS = {
        "msenames": "/wrds/crsp/sasdata/a_stock/msenames.sas7bdat",
        "delistings": "/wrds/crsp/sasdata/a_stock/msedelist.sas7bdat"
    }

    # Default frequency; can be overridden in subclasses if needed
    FREQUENCY: FrequencyType = FrequencyType.DAILY

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
          - filters: simple filters to apply after load
          - postprocessors: a list of atomic operations to apply (see module docstring)

        Returns:
            xr.Dataset: The assembled dataset.
        """
        user_freq_str = config.get("frequency", None)
        freq_enum = (
            self._parse_frequency(user_freq_str) if user_freq_str else self.FREQUENCY
        )

        # Build cache parameters (including time range).
        params = {"frequency": freq_enum.value}
        for key in ["columns_to_read", "filters", "postprocessors", "start_date", "end_date"]:
            if key in config:
                params[key] = config[key]

        param_hash = deepcopy(params)

        # Load from cache if available.
        cached_ds = self.cache_manager.load_from_cache(self.data_path, param_hash, freq_enum)
        if cached_ds is not None:
            return cached_ds

        # Load and process the primary table.
        df = self._load_local(**config)
        df = self._preprocess_df(df, **config)
        non_data_cols = ["date", "identifier"]
        data_cols = [col for col in df.columns if col not in non_data_cols]
        ds = self._convert_to_xarray(df, data_cols, frequency=freq_enum)
        ds = self._apply_config_postprocessors(ds, deepcopy(config))
        
        # Ensure the dataset is sliced by the requested date range.
        # ds = self.subset_by_date(ds, config) This will throw an error due to ineffecient implementation of a large slice

        # Save the resulting dataset to cache.
        self.cache_manager.save_to_cache(ds, self.data_path, param_hash)
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

        lower-case columns, reset index, convert date column,
        rename identifier column, apply filters, and sort.

        Args:
            df (pd.DataFrame): The initial, raw DataFrame.
            date_col (str): Name of the SAS date column (e.g., 'date', 'datadate').
            identifier_col (str): Name of the entity column (e.g., 'permno', 'gvkey').
            filters (Dict[str, Any]): Simple equality filters to apply after loading.
            **config: Additional arguments.

        Returns:
            pd.DataFrame: The cleaned/preprocessed DataFrame.
        """
        # Normalize column names
        df.columns = df.columns.str.lower()
        df.reset_index(inplace=True, drop=True)

        # Date handling
        if date_col and (date_col_lower := date_col.lower()) in df.columns:
            df['date'] = self.convert_sas_date(df[date_col_lower])

        # Rename the identifier column
        if identifier_col and (id_col_lower := identifier_col.lower()) in df.columns:
            df = df.rename(columns={id_col_lower: 'identifier'})

        # Apply user-defined filters (equalities)
        if filters:
            df = self._apply_filters(df, filters)

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
            sas_date_col (pd.Series): Column of SAS date ints.
            epoch (str): Base epoch for SAS (default '1960-01-01').

        Returns:
            pd.Series: Date column in datetime format.
        """
        sas_epoch = pd.to_datetime(epoch)
        return sas_epoch + pd.to_timedelta(sas_date_col.astype(int), unit='D')
    
    def _apply_config_postprocessors(self, ds: xr.Dataset, config: dict) -> xr.Dataset:
        """
        Iterate over each postprocessing operation defined in the configuration and apply it.
        
        If a postprocessor operation has an external source (via the 'src' parameter), load that file
        (using the provided 'identifier' and optional 'rename' mappings) and pass it into the operation
        as 'external_ds'.
        """
        postprocs = config.get("postprocessors")
        if not postprocs:
            return ds

        for proc in postprocs:
            op_name = proc.get("operation")
            op_params = proc.get("params", {})

            # If an external source is specified, load it and add it to the parameters.
            if "src" in op_params and isinstance(op_params["src"], str):
                external_identifier = op_params.get("identifier")
                if not external_identifier:
                    raise ValueError("Postprocessor operation requires an 'identifier' parameter for external source.")
                external_rename = op_params.get("rename")  
                op_params["external_ds"] = self._load_external_file(
                    op_params["src"],
                    external_identifier,
                    external_rename
                )

            op_func = post_processor.get(op_name)
            if not op_func:
                raise ValueError(f"Postprocessor operation '{op_name}' not found in registry.")
            ds = op_func(ds, op_params)
        return ds

    def _load_external_file(self, src_or_name: str, identifier: str, rename: Optional[List[List[str]]] = None) -> xr.Dataset | pd.DataFrame:
        """
        Load an external SAS file (specified either by a file path or a known source name)
        and convert it to an xarray.Dataset.

        """
        file_path = None

        # If src_or_name is a valid file path, use it; otherwise, try a default mapping.
        if os.path.exists(src_or_name):
            file_path = src_or_name
        elif src_or_name in self.DEFAULT_PATHS:
            file_path = self.DEFAULT_PATHS[src_or_name]
        else:
            raise ValueError(f"Unknown external source: {src_or_name}")

        if not os.path.exists(file_path):
            raise ValueError(f"File for external source '{src_or_name}' not found at {file_path}")

        df, _ = pyreadstat.read_file_multiprocessing(
            pyreadstat.read_sas7bdat,
            file_path,
            num_processes=16
        )
        
        # Normalize columns to lowercase.
        df.columns = df.columns.str.lower()

        # Apply any custom renaming if provided.
        if rename:
            for mapping in rename:
                if len(mapping) != 2:
                    raise ValueError("Each rename mapping must have exactly two elements: [source, destination].")
                src_col, dest_col = mapping
                src_col = src_col.lower()
                dest_col = dest_col.lower()
                if src_col in df.columns:
                    df.rename(columns={src_col: dest_col}, inplace=True)

        # Ensure that the designated identifier column is renamed to "identifier".
        identifier = identifier.lower()
        if identifier in df.columns:
            df.rename(columns={identifier: "identifier"}, inplace=True)
        else:
            raise ValueError(f"Identifier '{identifier}' was not found in external file columns: {df.columns.tolist()}")
        
        # Determine the time column: if 'time' exists (possibly via renaming), use it; else fall back on 'date'.
        time_column = "time" if "time" in df.columns else ("date" if "date" in df.columns else None)
        
        # Convert the DataFrame to an xarray.Dataset using Loader.from_table.
        if time_column:
            df[time_column] = self.convert_sas_date(df[time_column])
            ds = Loader.from_table(df, time_column=time_column, asset_column="identifier")
        else:
            return df
        return ds

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
