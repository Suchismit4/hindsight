# src/data/loaders.py

import pandas as pd
import numpy as np
import xarray as xr
import yaml
import os
from typing import Optional, Dict, Any, Union
from pathlib import Path
import hashlib
import json

import yfinance as yf  

from .manager import DataLoader
from .registry import register_data_loader
from .data import DataArrayDateTimeAccessor

class BaseDataSource:
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
    
    def _convert_to_xarray(self, df: pd.DataFrame, columns) -> xr.Dataset:
        """
        Convert pandas DataFrame to xarray Dataset.
        """
        
        # Ensure 'Date' is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        return DataArrayDateTimeAccessor.from_table(
            df,
            time_column='date',
            asset_column='identifier',
            feature_columns=columns,
            frequency='D'
        )
        

@register_data_loader('/market/equities/yahoo', ['close_prices', 'returns', 'high_prices', 'low_prices', 'open_prices', 'volume'])
class YFinanceDataLoader(DataLoader, BaseDataSource):
    """
    Data loader for Yahoo Finance data.
    
    This loader provides access to financial data through the Yahoo Finance API.
    It implements a caching mechanism to store downloaded data locally and
    avoid unnecessary API calls.
    """
    
    def load_data(self, symbols: list, start_date: str, end_date: str, frequency: str = '1d', **kwargs) -> xr.Dataset:
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
        loaded_data = self._load_from_yahoo(symbols, start_date, end_date, frequency, cache_path, params, **kwargs)
        
        return loaded_data
    
    def _load_from_cache(self, cache_path: Path) -> xr.Dataset:
        """Load data from cache file."""
        # Load the parquet file into a pandas DataFrame
        df = pd.read_parquet(cache_path)
        return self._convert_to_xarray(df, ['close_prices', 'returns', 'high_prices', 'low_prices', 'open_prices', 'volume'])
    
    def _load_from_yahoo(self, symbols: list, start_date: str, end_date: str, 
                     frequency: str, cache_path: Path, params: dict, **kwargs) -> xr.Dataset:
        """
        Download data from Yahoo Finance and cache it.
        
        Args:
            symbols: List of ticker symbols.
            start_date: Start date string.
            end_date: End date string.
            frequency: Data frequency.
            cache_path: Path where to save the cache file.
            params: Dictionary of parameters for caching.
            **kwargs: Additional arguments passed to yfinance.
        """
        # Download data
        df = yf.download(
            tickers=symbols,
            start=start_date,
            end=end_date,
            interval=frequency,
            group_by='ticker',
            **kwargs
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