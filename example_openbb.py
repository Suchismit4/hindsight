# hindsight/main.py

from src import DataManager
import xarray as xr
import xarray_jax as xj
import numpy as np
import pandas as pd
import jax.numpy as jnp
import equinox as eqx
from functools import partial

def main():
    data_manager = DataManager()
    
    paths_info = data_manager.get_available_data_paths()
    
    for dp, info in paths_info.items():
        print(f"|- Data path: {dp}")
        print(f"  |- Provider: {info['provider']}")
        print(f"  |- Sub-providers: {info['sub_providers']}")

    # Load the dataset using the DataManager
    dataset = data_manager.get_data([
    {
        "data_path": "openbb/equity/price/historical",  
        "config": {
            "provider": "yfinance",      
            "symbols": ["AAPL", "TSLA"],
            "start_date": "2015-01-01",
            "end_date": "2021-01-01"
        }
    }
    # {
    #     'data_path': 'wrds/equity/crsp',
    #     'config': {
    #         'num_processes': 16,
    #         'freq': 'D'
    #     }
    # }
    ])

    print(dataset)

if __name__ == "__main__":
    main()