# data/loaders/yfinance/market/historical.py

from src.data.abstracts.base import BaseDataSource
import yfinance as yf
import pandas as pd
import xarray as xr
from pathlib import Path
import xarray_jax
from typing import Dict, Any

class YFinanceEquityHistoricalFetcher(BaseDataSource):
    """
    Data loader for Yahoo Finance data.
    
    This loader provides access to financial data through the Yahoo Finance API.
    It implements a caching mechanism to store downloaded data locally and
    avoid unnecessary API calls.
    """
    
    def load_data(self, **config) -> xr.Dataset:
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
        symbols = config.get('symbols', [])
        start_date = config.get('start_date')
        end_date = config.get('end_date')
        frequency = config.get('frequency', '1d')
            
        # If no cache or cache failed, load from Yahoo Finance
        data = self._load_from_yahoo(symbols, start_date, end_date, frequency)
        
        # Convert to a time-series indexed dataset
        ts_data = self._convert_to_xarray(data, list(data.columns.drop(['date', 'identifier'])))
 
        return ts_data
        
    def _load_from_yahoo(self, symbols: list, start_date: str, end_date: str, 
                         frequency: str) -> xr.Dataset:
        """
        Download data from Yahoo Finance and cache it.
        
        Args:
            symbols: List of ticker symbols.
            start_date: Start date string.
            end_date: End date string.
            frequency: Data frequency.
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
        
        return result

    def _get_cache_params(self, **config) -> Dict[str, Any]:
        """Get parameters used for cache key generation."""
        return {
            'symbols': config.get('symbols', []),
            'start_date': config.get('start_date'),
            'end_date': config.get('end_date'),
            'frequency': config.get('frequency', '1d'),
        }
