# hindsight/main.py

from src import DataManager
import xarray as xr
import xarray_jax as xj
import numpy as np
import pandas as pd
import jax.numpy as jnp
import equinox as eqx

def main():
    data_manager = DataManager()

    dataset = data_manager.get_data(
        data_type='close_prices',
        symbols=['AAPL', 'MSFT'],
        start_date='2021-01-01',
        end_date='2023-01-31',
        frequency='1d'
    )
    
    print(dataset.dt.sel('2022-01-01'))
    
if __name__ == "__main__":
    main()