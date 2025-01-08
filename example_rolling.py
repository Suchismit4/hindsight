# hindsight/example_rolling.py

from src import DataManager
# from src.core.operations import mean

import xarray as xr
import xarray_jax as xj
import numpy as np
import pandas as pd
from functools import partial
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def main():
    
    dm = DataManager()
    
    # Pull in the yfinance data of Apple and Tesla.
    datasets = dm.get_data([{
        "data_path": "openbb/equity/price/historical",
        "config": {
            "provider": "yfinance",
            "symbols": ["AAPL", "TSLA"],
            "start_date": "2022-01-01",
            "end_date": "2024-12-31",
        }
    }])
    
    # Define a function to compute Exponential Moving Average (EMA)
    # This function will be used with the u_roll method for efficient computation
    @partial(jax.jit, static_argnames=['window_size'])
    def ema(i: int, carry, block: jnp.ndarray, window_size: int):
        """
        Compute the Exponential Moving Average (EMA) for a given window.
        
        This function is designed to work with JAX's JIT compilation and
        the u_roll method defined in the Tensor class. It computes the EMA
        efficiently over a rolling window of data.
        
        Args:
        i (int): Current index in the time series
        state (tuple): Contains current values, carry (previous EMA), and data block
        window_size (int): Size of the moving window
        
        Returns:
        tuple: Updated state (new EMA value, carry, and data block)
        """
        
        # Initialize the first value
        if carry is None:
            # Compute the sum of the first window
            current_window_sum = block[:window_size].reshape(-1, 
                                                             block.shape[1], 
                                                             block.shape[2]).sum(axis=0)
        
            
            return (current_window_sum * (1/window_size), current_window_sum * (1/window_size))
        
        # Get the current price
        current_price = block[i]
        
        # Compute the new EMA
        # EMA = α * current_price + (1 - α) * previous_EMA
        # where α = 1 / (window_size)
        alpha = 1 / window_size
        
        new_ema = alpha * current_price + (1 - alpha) * carry
        
        return (new_ema, new_ema)
    
    # Original dataset
    dataset = datasets["openbb/equity/price/historical"]
    
    # Rolling-EMA of "close" over a 200-day window
    ema_dataset = dataset.dt.rolling(dim='time', window=200).reduce(ema)

    # Convert to time-indexed form for plotting
    # -- Original closing prices --
    apple_orig = dataset.sel(asset="AAPL").dt.to_time_indexed()
    tsla_orig  = dataset.sel(asset="TSLA").dt.to_time_indexed()
    # -- EMA-rolled closing prices --
    apple_ema = ema_dataset.sel(asset="AAPL").dt.to_time_indexed()
    tsla_ema  = ema_dataset.sel(asset="TSLA").dt.to_time_indexed()
    
    # Extract the close and the new "ema_close"
    apple_close_orig = apple_orig["close"]
    tsla_close_orig  = tsla_orig["close"]
    
    apple_close_ema  = apple_ema["ema_close"]
    tsla_close_ema   = tsla_ema["ema_close"]
    
    # Create subplots: two rows, one column
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # --- Apple subplot ---
    apple_close_orig.plot.line(
        x="time", ax=ax1, label="AAPL Close", color="blue", linestyle="-"
    )
    apple_close_ema.plot.line(
        x="time", ax=ax1, label="AAPL EMA(10)", color="blue", linestyle="--"
    )
    ax1.set_title("Apple (AAPL) Closing Prices vs. EMA(10)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()

    # --- Tesla subplot ---
    tsla_close_orig.plot.line(
        x="time", ax=ax2, label="TSLA Close", color="red", linestyle="-"
    )
    tsla_close_ema.plot.line(
        x="time", ax=ax2, label="TSLA EMA(10)", color="red", linestyle="--"
    )
    ax2.set_title("Tesla (TSLA) Closing Prices vs. EMA(10)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()

    plt.tight_layout()

    # Save the figure
    plt.savefig("apple_tsla_ema.png")
    plt.close()
    
if __name__ == "__main__":
    main()