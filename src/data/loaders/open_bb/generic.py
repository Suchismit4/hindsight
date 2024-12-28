# data/loaders/openbb/generic.py

from src.data.abstracts.base import BaseDataSource
from typing import Dict, Any, List
import xarray as xr
import pandas as pd
from openbb import obb
from src.data.core.struct import FrequencyType

class GenericOpenBBDataFetcher(BaseDataSource):
    """
    Generic data loader for OpenBB data.
    Dynamically calls e.g. obb.equities.price.historical() based on data_path.
    """

    def load_data(self, **config) -> xr.Dataset:
        
        data_path = self.data_path  # e.g. "openbb/equities/price/historical"
        
        if data_path.startswith("openbb/"):
            data_path = data_path[len("openbb/"):]  # => "equities/price/historical"

        dot_path = data_path.replace("/", ".")

        provider   = config.get("provider")
        symbols    = config.get("symbols", [])
        start_date = config.get("start_date")
        end_date   = config.get("end_date")

        if not provider:
            raise ValueError("You must supply a 'provider' in config (e.g. 'yfinance').")

        # Prepare for caching
        cache_params = {
            "provider": provider,
            "symbols": tuple(symbols) if isinstance(symbols, list) else symbols,
            "start_date": start_date,
            "end_date": end_date,
        }
        cache_path = self.get_cache_path(**cache_params)

        # Try loading from cache
        cached_ds = self.load_from_cache(cache_path, request_params=cache_params)
        if cached_ds is not None:
            return cached_ds

        # Resolve the function in OpenBB
        module = obb
        for attr in dot_path.split("."):
            module = getattr(module, attr, None)
            if module is None:
                raise AttributeError(f"OpenBB path '{dot_path}' not found in obb.")

        # Call OpenBB
        df = module(
            symbols,
            provider=provider,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )

        if df is None or df.empty:
            raise ValueError("Something went wrong with OpenBB's endpoints. Empty or NULL dataframe fetched")

        df.reset_index(inplace=True)

        df.rename(columns={'symbol': 'identifier'}, inplace=True)
        
        required_columns = ['date', 'identifier'] + [col for col in df.columns if col not in ['date', 'identifier']]
        df = df[required_columns]
        
         # Sort by date and identifier
        df.sort_values(['date', 'identifier'], inplace=True)

        value_cols = df.columns.drop(["date", "identifier"])
                
        ds = self._convert_to_xarray(
            df, 
            value_cols, 
            frequency=FrequencyType.DAILY,
        )

        self.save_to_cache(ds, cache_path, cache_params)

        return ds
