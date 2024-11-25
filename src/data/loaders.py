# src/data/loaders.py

import pandas as pd
import numpy as np
import xarray as xr
import yaml
import os
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import hashlib
import json
import pyreadstat

import yfinance as yf  

from abc import ABC, abstractmethod
from .manager import DataLoader
from .registry import register_data_loader
from .data import DatasetDateTimeAccessor
from .data import FrequencyType

class BaseDataSource(DataLoader):
    """
    Base class for handling data sources configuration and path management.
    
    This class provides the foundational logic for accessing different data sources
    defined in the paths.yaml configuration file. It handles both online and offline
    data sources and manages the interaction with the cache system.
    """
    
    def __init__(self):
        """Initialize the data source manager with configuration from paths.yaml."""
        self.config_path = os.path.join(
            os.path.dirname(__file__),
            'config',
            'paths.yaml'
        )
        self.sources = self._load_sources_config()
        self.cache_root = os.path.expanduser(
            os.path.join('~/data', 'cache')
        )
    
    def _load_sources_config(self) -> Dict[str, Any]:
        """Load and parse the data sources configuration file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply filters to the DataFrame.

        Args:
            df: The DataFrame to filter.
            filters: A dictionary where keys are column names and values are filter conditions.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        for column, condition in filters.items():
            if isinstance(condition, tuple) and len(condition) == 2:
                # Condition is a tuple like ('>=', '1959-01-01')
                operator, value = condition
                if operator == '=' or operator == '==':
                    df = df[df[column] == value]
                elif operator == '!=':
                    df = df[df[column] != value]
                elif operator == '>':
                    df = df[df[column] > value]
                elif operator == '>=':
                    df = df[df[column] >= value]
                elif operator == '<':
                    df = df[df[column] < value]
                elif operator == '<=':
                    df = df[df[column] <= value]
                else:
                    raise ValueError(f"Unsupported operator '{operator}' in filter for column '{column}'.")
            else:
                # Condition is a simple equality
                df = df[df[column] == condition]
        return df
    
    def get_cache_path(self, registry_path: str, **params) -> Path:
        """
        Convert a registry path and parameters to a cache file path.
        
        Args:
            registry_path: The path used in the @register_data_loader decorator.
            **params: Parameters used in the data loading function.
        
        Returns:
            Path: The corresponding cache file path.
        """
        # Serialize the parameters to a JSON-formatted string
        params_string = json.dumps(params, sort_keys=True)
        # Generate a hash of the parameters string
        params_hash = hashlib.md5(params_string.encode('utf-8')).hexdigest()
        # Create the cache path using the hash
        cache_path = os.path.join(
            self.cache_root,
            registry_path.lstrip('/'),
            params_hash
        )
        return Path(f"{cache_path}.parquet")
    
    def check_cache(self, cache_path: Path) -> bool:
        """Check if valid cache exists for the given path."""
        return cache_path.exists()
    
    def _convert_to_xarray(self, df: pd.DataFrame, columns, frequency: FrequencyType = FrequencyType.DAILY) -> xr.Dataset:
        """
        Convert pandas DataFrame to xarray Dataset.
        """
        
        # Ensure 'Date' is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
                
        return DatasetDateTimeAccessor.from_table(
            df,
            time_column='date',
            asset_column='identifier',
            feature_columns=columns,
            frequency=frequency
        )
            
    @abstractmethod
    def load_data(self, **kwargs) -> Union[xr.Dataset, xr.DataTree]:
        """Abstract method to load data."""
        pass

@register_data_loader('/market/equities/yahoo')
class YFinanceDataLoader(BaseDataSource):
    """
    Data loader for Yahoo Finance data.
    
    This loader provides access to financial data through the Yahoo Finance API.
    It implements a caching mechanism to store downloaded data locally and
    avoid unnecessary API calls.
    """
    
    def load_data(self, **kwargs) -> xr.Dataset:
        """
        Load market data from Yahoo Finance with caching support.
        
        Args:
            symbols: List of ticker symbols to download.
            start_date: Start date for the data range (YYYY-MM-DD).
            end_date: End date for the data range (YYYY-MM-DD).
            frequency: Data frequency (e.g., '1d' for daily, '1h' for hourly).
            **kwargs: Additional arguments passed to yfinance.
        
        Returns:
            xr.Dataset: Dataset containing prices and returns data.
        """
        symbols = kwargs.get('symbols', [])
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        frequency = kwargs.get('frequency', '1d')
        
        # Collect all parameters into a dictionary
        params = {
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'frequency': frequency,
            **kwargs
        }
        # Generate the cache path based on parameters
        cache_path = self.get_cache_path('/market/equities/yahoo', **params)
        
        # Try to load from cache first
        if self.check_cache(cache_path):
            try:
                return self._load_from_cache(cache_path)
            except Exception as e:
                print(f"No valid cache found for yFinance: {e}. \nFalling back to Yahoo Finance.")
            
        # If no cache or cache failed, load from Yahoo Finance
        loaded_data = self._load_from_yahoo(symbols, start_date, end_date, frequency, cache_path, params)
        
        return loaded_data
        
    def _load_from_cache(self, cache_path: Path) -> xr.Dataset:
        """Load data from cache file."""
        # Load the parquet file into a pandas DataFrame
        df = pd.read_parquet(cache_path)
        return self._convert_to_xarray(df, ['close_prices', 'returns', 'high_prices', 'low_prices', 'open_prices', 'volume'])
        
    def _load_from_yahoo(self, symbols: list, start_date: str, end_date: str, 
                         frequency: str, cache_path: Path, params: dict) -> xr.Dataset:
        """
        Download data from Yahoo Finance and cache it.
        
        Args:
            symbols: List of ticker symbols.
            start_date: Start date string.
            end_date: End date string.
            frequency: Data frequency.
            cache_path: Path where to save the cache file.
            params: Dictionary of parameters for caching.
        """
        # Download data
        df = yf.download(
            tickers=symbols,
            start=start_date,
            end=end_date,
            interval=frequency,
            group_by='ticker',
        )
        
        # Initialize empty list to store individual dataframes
        dfs = []
        
        # Process each symbol separately
        for symbol in symbols:
            # Extract data for this symbol
            if len(symbols) > 1:
                symbol_data = df[symbol].copy()
            else:
                symbol_data = df.copy()
            
            # Reset index to get date as a column
            symbol_data.reset_index(inplace=True)
            
            
            # Rename columns to lowercase
            symbol_data.columns = [col.lower() for col in symbol_data.columns]
            
            # Add identifier column (ticker)
            symbol_data['identifier'] = symbol
            
            # Rename and select columns to match registry requirements
            column_mapping = {
                'date': 'date',
                'open': 'open_prices',
                'high': 'high_prices',
                'low': 'low_prices',  
                'adj close': 'close_prices',  # Using adjusted close as the primary price
                'volume': 'volume'
            }
            symbol_data.rename(columns=column_mapping, inplace=True)
            
            # Calculate returns as difference in adjusted closing prices
            symbol_data['returns'] = symbol_data['close_prices'].diff()
            
            # Select and order columns according to registry
            required_columns = ['date', 'identifier', 'close_prices', 'returns', 
                            'high_prices', 'low_prices', 'open_prices', 'volume']
            symbol_data = symbol_data[required_columns]
            
            dfs.append(symbol_data)
        
        # Combine all symbols into one DataFrame
        result = pd.concat(dfs, ignore_index=True)
        
        # Ensure date is datetime
        result['date'] = pd.to_datetime(result['date'])
        
        # Sort by date and identifier
        result.sort_values(['date', 'identifier'], inplace=True)
        result.reset_index(drop=True, inplace=True)
        
        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path)
        
        # Save metadata
        metadata_path = cache_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(params, f)
        
        return self._convert_to_xarray(result, ['close_prices', 'returns', 'high_prices', 'low_prices', 'open_prices', 'volume'])

@register_data_loader('/market/equities/wrds/compustat')
class CompustatDataLoader(BaseDataSource):
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
        cache_path = self.get_cache_path('/market/equities/wrds/compustat', **params)

        # Try to load from cache first
        if self.check_cache(cache_path):
            try:
                return self._load_from_cache(cache_path)
            except Exception as e:
                print(f"No valid cache found for Compustat: {e}. \nFalling back to loading from source.")

        # If no cache or cache failed, load from source
        loaded_data = self._load_from_source(columns_to_read, filters, self.LOCAL_SRC, cache_path, num_processes, params)

        return loaded_data

    def _load_from_cache(self, cache_path: Path) -> xr.Dataset:
        """Load data from cache file."""
        # Load the parquet file into a pandas DataFrame
        df = pd.read_parquet(cache_path)
        return self._convert_to_xarray(df, list(df.columns.drop(['date', 'identifier'])), frequency=FrequencyType.YEARLY)

    def _load_from_source(self, columns_to_read: List[str], filters: Dict[str, Any], funda_path: str, cache_path: Path, num_processes: int, params: dict) -> xr.Dataset:
        """
        Load data from Compustat source file and cache it.

        Args:
            columns_to_read: List of columns to read.
            filters: Dictionary of filters to apply to the data.
            funda_path: Path to the Compustat 'funda' data file.
            cache_path: Path where to save the cache file.
            num_processes: Number of processes to use in reading the file.
            params: Dictionary of parameters (used for metadata).
        """

        # Load the data using pyreadstat
        df, meta = pyreadstat.read_file_multiprocessing(
            pyreadstat.read_sas7bdat,
            funda_path,
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

        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)

        # Save metadata
        metadata_path = cache_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(params, f)
                  
        return self._convert_to_xarray(df, list(df.columns.drop(['date', 'identifier'])), frequency=FrequencyType.YEARLY)