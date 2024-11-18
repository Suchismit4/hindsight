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
        symbols=[
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'TSLA', 'BRK.B', 'JPM',
            'JNJ', 'V', 'UNH', 'NVDA', 'HD', 'PG', 'MA', 'DIS', 'PYPL',
            'VZ', 'ADBE', 'NFLX', 'INTC', 'KO', 'PFE', 'T', 'CMCSA',
            'PEP', 'MRK', 'ABT', 'XOM', 'CSCO', 'CVX', 'NKE', 'WMT',
            'ORCL', 'TMO', 'ACN', 'ABBV', 'LLY', 'DHR', 'CRM', 'TXN',
            'COST', 'AVGO', 'QCOM', 'MDT', 'PM', 'AMGN', 'NEE', 'HON',
            'UNP', 'MS', 'BAC', 'BA', 'BMY', 'SBUX', 'C', 'LOW', 'IBM',
            'MMM', 'RTX', 'LMT', 'INTU', 'GE', 'BLK', 'AXP', 'SPGI',
            'GS', 'CAT', 'PLD', 'MDLZ', 'ISRG', 'CVS', 'NOW', 'MO',
            'CHTR', 'BKNG', 'TGT', 'AMAT', 'SYK', 'ZTS', 'DE', 'ADP',
            'GM', 'ADI', 'GILD', 'ADSK', 'F', 'MU', 'ATVI', 'APD',
            'EL', 'CSX', 'BSX', 'PNC', 'EW', 'CI', 'FDX', 'MMC', 'SHW',
            'HUM', 'DUK', 'SO', 'ICE', 'LRCX', 'KLAC', 'TRV', 'SPG',
            'COF', 'AIG', 'ECL', 'WM', 'ITW', 'REGN', 'BIIB', 'BK',
            'MET', 'MCO', 'NSC', 'AON', 'IDXX', 'SLB', 'PRU', 'APTV',
            'D', 'PSA', 'PH', 'DD', 'VLO', 'EXC', 'PEG', 'ROP', 'MAR',
            'SBAC', 'OXY', 'VRSK', 'CTAS', 'TDG', 'FTNT', 'ANET', 'FRC',
        ],
        start_date='2021-01-01',
        end_date='2023-01-31',
        frequency='1d',
    )
    
    print(dataset.dt.sel('2022-02-02')) # a wednesday
    
    print(dataset.dt.sel('2022-01-01')) # why null?
    
if __name__ == "__main__":
    main()
