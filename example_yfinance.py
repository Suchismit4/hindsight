# hindsight/main.py

import xarray as xr
import xarray_jax as xj
import numpy as np
import pandas as pd
import jax.numpy as jnp
import equinox as eqx
from functools import partial

from src import DataManager

def main():
    data_manager = DataManager()
    
    print(data_manager.list_available_data_paths())
    
    # Define data requests with configurations
    data_requests = [
        {
            'data_path': 'yfinance/equities/market/historical',
            'config': {
                'symbols': [
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
                'start_date': '2021-01-01',
                'end_date': '2023-01-31',
                'frequency': '1d',
            }
        }
    ]

    dataset = data_manager.get_data(
        data_requests=data_requests
    )
                
    print(dataset)
    
    # Define a function to compute Exponential Moving Average (EMA)
    # This function will be used with the u_roll method for efficient computation
    # @partial(jax.jit, static_argnames=['window_size'])
    # def ema(i: int, carry, block: jnp.ndarray, window_size: int):
    #     """
    #     Compute the Exponential Moving Average (EMA) for a given window.
        
    #     This function is designed to work with JAX's JIT compilation and
    #     the u_roll method defined in the Tensor class. It computes the EMA
    #     efficiently over a rolling window of data.
        
    #     Args:
    #     i (int): Current index in the time series
    #     state (tuple): Contains current values, carry (previous EMA), and data block
    #     window_size (int): Size of the moving window
        
    #     Returns:
    #     tuple: Updated state (new EMA value, carry, and data block)
    #     """
        
    #     # Initialize the first value
    #     if carry is None:
    #         # Compute the sum of the first window
    #         current_window_sum = block[:window_size].reshape(-1, 
    #                                                          block.shape[1], 
    #                                                          block.shape[2]).sum(axis=0)
        
            
    #         return (current_window_sum * (1/window_size), current_window_sum * (1/window_size))
        
    #     # Get the current price
    #     current_price = block[i]
        
    #     # Compute the new EMA
    #     # EMA = α * current_price + (1 - α) * previous_EMA
    #     # where α = 1 / (window_size)
    #     alpha = 1 / window_size
        
    #     new_ema = alpha * current_price + (1 - alpha) * carry
        
    #     return (new_ema, new_ema)
    
    # rolled = dataset.dt.rolling(dim='time', window=10).reduce(func=ema)

    
if __name__ == "__main__":
    main()
