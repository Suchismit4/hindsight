# src/data/loaders/crypto/local.py

"""
Local cryptocurrency data loader.

This module provides a data loader for cryptocurrency price data stored as CSV files.
It follows the same pattern as WRDS loaders but is adapted for CSV data from cryptocurrency exchanges.
"""

import pandas as pd
import xarray as xr
import os
import glob
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.data.core.util import FrequencyType
from src.data.loaders.abstracts.base import BaseDataSource


class LocalCryptoDataFetcher(BaseDataSource):
    """
    Data loader for local cryptocurrency CSV files.
    
    Loads cryptocurrency OHLCV data from CSV files stored locally.
    Each CSV file contains data for one cryptocurrency pair (e.g., BTCUSDT.csv).
    
    Expected CSV format:
    - datetime: timestamp in datetime format
    - open_time: Unix timestamp (optional)
    - open: opening price
    - high: highest price
    - low: lowest price  
    - close: closing price
    - volume: trading volume
    
    File naming convention: {SYMBOL}USDT.csv (e.g., BTCUSDT.csv, ETHUSDT.csv)
    """
    
    # Default frequency for crypto data (hourly)
    FREQUENCY: FrequencyType = FrequencyType.HOURLY
    
    # Default data directory
    DEFAULT_DATA_DIR = "/home/suchismit/data/crypto/data"
    
    def __init__(self, data_path: str):
        """
        Initialize the crypto data loader.
        
        Args:
            data_path: Path identifier for the data source
        """
        super().__init__(data_path)
        self.data_directory = self.DEFAULT_DATA_DIR
    
    def load_data(self, **config) -> xr.Dataset:
        """
        Load cryptocurrency data from CSV files.
        
        Configuration parameters:
        - data_directory: Custom data directory path (optional)
        - symbols: List of crypto symbols to load (optional, loads all if not specified)
        - start_date: Start date for filtering
        - end_date: End date for filtering
        - columns_to_read: Specific columns to load (optional)
        
        Args:
            **config: Configuration parameters
            
        Returns:
            xr.Dataset: The assembled dataset with crypto price data
        """
        # Override data directory if provided
        data_dir = config.get('data_directory', self.data_directory)
        
        # Parse frequency (for future extension to different frequencies)
        user_freq_str = config.get("frequency", "H")  # Default to hourly
        freq_enum = self._parse_frequency(user_freq_str)
        
        # Load and combine all CSV files
        df = self._load_csv_files(data_dir, **config)
        
        # Preprocess the data
        df = self._preprocess_df(df, **config)
        
        # Identify data columns vs. metadata columns
        non_data_cols = ["date", "identifier", "datetime", "open_time"]
        data_cols = [col for col in df.columns if col not in non_data_cols]
        
        # Convert to xarray Dataset
        ds = self._convert_to_xarray(df, data_cols, frequency=freq_enum)
        
        return ds
    
    def _load_csv_files(self, data_dir: str, **config) -> pd.DataFrame:
        """
        Load and combine all CSV files from the data directory.
        
        Args:
            data_dir: Directory containing CSV files
            **config: Configuration parameters
            
        Returns:
            Combined DataFrame with all cryptocurrency data
        """
        # Get list of symbols to load
        symbols = config.get('symbols', None)
        columns_to_read = config.get('columns_to_read', None)
        
        # Find all CSV files
        csv_pattern = os.path.join(data_dir, "*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {data_dir}")
        
        combined_data = []
        
        for csv_file in csv_files:
            # Extract symbol from filename (e.g., BTCUSDT.csv -> BTCUSDT)
            filename = os.path.basename(csv_file)
            symbol = filename.replace('.csv', '')
            
            # Filter by symbols if specified
            if symbols and symbol not in symbols:
                continue
            
            try:
                # Read CSV file
                if columns_to_read:
                    # Ensure datetime is always included
                    cols_to_read = list(set(columns_to_read + ['datetime']))
                    df = pd.read_csv(csv_file, usecols=cols_to_read)
                else:
                    df = pd.read_csv(csv_file)
                
                # Add symbol as identifier column
                df['symbol'] = symbol
                
                combined_data.append(df)
                
            except Exception as e:
                print(f"Warning: Failed to load {csv_file}: {e}")
                continue
        
        if not combined_data:
            raise ValueError("No data loaded successfully")
        
        # Combine all dataframes
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        print(f"Loaded data for {len(combined_data)} symbols from {data_dir}")
        
        return combined_df
    
    def _preprocess_df(self, df: pd.DataFrame, **config) -> pd.DataFrame:
        """
        Preprocess the cryptocurrency DataFrame.
        
        Args:
            df: Raw DataFrame from CSV files
            **config: Configuration parameters
            
        Returns:
            Preprocessed DataFrame ready for xarray conversion
        """
        if df.empty:
            return df
        
        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()
        df.reset_index(inplace=True, drop=True)
        
        # Handle datetime column
        if 'datetime' in df.columns:
            df['date'] = pd.to_datetime(df['datetime'])
        else:
            raise ValueError("CSV files must contain 'datetime' column")
        
        # Rename symbol column to identifier for consistency
        if 'symbol' in df.columns:
            df = df.rename(columns={'symbol': 'identifier'})
        else:
            raise ValueError("Symbol identifier missing")
        
        # Apply date range filtering
        start_date = config.get('start_date')
        end_date = config.get('end_date')
        
        if start_date or end_date:
            pre_filter_len = len(df)
            
            if start_date:
                start_date_dt = pd.to_datetime(start_date)
                df = df[df['date'] >= start_date_dt]
            
            if end_date:
                end_date_dt = pd.to_datetime(end_date)
                df = df[df['date'] <= end_date_dt]
            
            post_filter_len = len(df)
            print(f"Date range filter [{start_date} to {end_date}]: {pre_filter_len} -> {post_filter_len} rows")
        
        # Apply additional filters if specified
        filters = config.get('filters')
        if filters:
            df = self.apply_filters(df, filters)
        
        # Sort by identifier and date
        df = df.sort_values(['identifier', 'date']).reset_index(drop=True)
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    @staticmethod
    def _parse_frequency(freq_str: Optional[str]) -> FrequencyType:
        """
        Parse frequency string to FrequencyType enum.
        
        Args:
            freq_str: Frequency string (e.g., 'H', 'D', 'M')
            
        Returns:
            FrequencyType enum value
        """
        if not freq_str:
            return FrequencyType.DAILY
        
        freq_upper = freq_str.upper()
        
        # Map frequency strings to FrequencyType
        freq_map = {
            'H': FrequencyType.HOURLY,
            'HOURLY': FrequencyType.HOURLY,
            'D': FrequencyType.DAILY,
            'DAILY': FrequencyType.DAILY,
            'W': FrequencyType.WEEKLY,
            'WEEKLY': FrequencyType.WEEKLY,
            'M': FrequencyType.MONTHLY,
            'MONTHLY': FrequencyType.MONTHLY,
            'Y': FrequencyType.YEARLY,
            'YEARLY': FrequencyType.YEARLY,
        }
        
        return freq_map.get(freq_upper, FrequencyType.DAILY) 