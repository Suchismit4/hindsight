# data/loaders/yfinance/market/cash_flow.py

from src.data.abstracts.base import BaseDataSource
import yfinance as yf
import pandas as pd
import xarray as xr
import xarray_jax
from typing import Dict, Any


class YFinanceCashFlowFetcher(BaseDataSource):
    """
    Data loader for Yahoo Finance Cash Flow data.
    """

    def load_data(self, **kwargs) -> xr.Dataset:
        """
        Load cash flow data from Yahoo Finance with caching support.

        Args:
            symbols: List of ticker symbols to download.

        Returns:
            xr.Dataset: Dataset containing cash flow data.
        """
        symbols = kwargs.get('symbols', [])

        data = self._load_from_yahoo(symbols)

        ds_data = self._convert_to_xarray(data)
        return ds_data

    def _load_from_yahoo(self, symbols: list) -> pd.DataFrame:
        """
        Download cash flow data from Yahoo Finance.
        """
        dfs = []
        
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            cf = ticker.cashflow.T.reset_index()
            cf.rename(columns={'index': 'date'}, inplace=True)
            cf['identifier'] = symbol
            cf['date'] = pd.to_datetime(cf['date'])
            dfs.append(cf)
        
        result = pd.concat(dfs, ignore_index=True)
        result.sort_values(['date', 'identifier'], inplace=True)
        result.reset_index(drop=True, inplace=True)
        result.set_index(['date', 'identifier'], inplace=True)
        return result

    def _get_cache_params(self, **params) -> Dict[str, Any]:
        """Get parameters used for cache key generation."""
        return {'symbols': params.get('symbols', [])}