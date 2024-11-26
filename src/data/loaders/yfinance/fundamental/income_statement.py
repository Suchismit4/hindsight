# data/loaders/yfinance/market/income_statement.py

from src.data.abstracts.base import BaseDataSource
import yfinance as yf
import pandas as pd
import xarray as xr
import xarray_jax


class YFinanceIncomeStatementFetcher(BaseDataSource):
    """
    Data loader for Yahoo Finance Income Statement data.
    """

    def load_data(self, **kwargs) -> xr.Dataset:
        """
        Load income statement data from Yahoo Finance with caching support.

        Args:
            symbols: List of ticker symbols to download.

        Returns:
            xr.Dataset: Dataset containing income statement data.
        """
        symbols = kwargs.get('symbols', [])
        
        params = {'symbols': symbols}
        cache_path = self.get_cache_path(**params)
        data = self.load_from_cache(cache_path)
        if data is not None:
            return data

        data = self._load_from_yahoo(symbols)
        self.save_to_cache(data, cache_path, params)
        ds_data = self._convert_to_xarray(data)
        return ds_data

    def _load_from_yahoo(self, symbols: list) -> pd.DataFrame:
        """
        Download income statement data from Yahoo Finance.
        """
        dfs = []
        
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            is_data = ticker.financials.T.reset_index()
            is_data.rename(columns={'index': 'date'}, inplace=True)
            is_data['identifier'] = symbol
            is_data['date'] = pd.to_datetime(is_data['date'])
            dfs.append(is_data)
        
        result = pd.concat(dfs, ignore_index=True)
        result.sort_values(['date', 'identifier'], inplace=True)
        result.reset_index(drop=True, inplace=True)
        result.set_index(['date', 'identifier'], inplace=True)
        return result