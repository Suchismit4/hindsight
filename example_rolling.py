# hindsight/example_rolling.py
# ----------------------------------------
# This script demonstrates how to compute and plot the rolling Exponential Moving Average (EMA)
# for Apple's (AAPL) and Tesla's (TSLA) closing stock prices using data from CRSP.
# The script uses the DataManager to fetch historical price data, applies a rolling window EMA,
# and visualizes both the original closing prices and the EMA for each asset.
# ----------------------------------------

from src import DataManager
from src.data.core.operations import mean, median, mode, ema

import xarray as xr
import xarray_jax as xj
import numpy as np
import pandas as pd
from functools import partial
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def main():
    """
    Main function to fetch historical stock data, compute rolling EMA,
    and plot the original closing prices alongside the EMA for AAPL and TSLA.
    """
    # Initialize the DataManager to handle dataset operations
    dm = DataManager()
    
    # Pull in the CRSP data for Apple and Tesla.
    # Data parameters: symbols, date range, and data provider configuration.
    datasets = dm.get_data([{
        "data_path": "wrds/equity/crsp",
        "config": {
            "freq": "D",
            "start_date": "2015-01-01",
            "end_date": "2024-01-01",
            "filters": {
                "date": [">=", "2015-01-01"]
            },
            "postprocessors": [
                {
                    "operation": "replace",
                    "params": {
                        "src": "delistings",
                        "rename": [["dlstdt", "time"]],
                        "identifier": "permno",
                        "from": "dlret",
                        "to": "ret",
                    }
                },
                {
                    "operation": "merge_2d_table",
                    "params": {
                        "src": "msenames",
                        "identifier": "permno",
                        "ax2": "comnam",
                        "ax1": "asset",
                    }
                }
            ]
        }
    }])

    
    # Extract the original dataset for historical prices.
    dataset = datasets["wrds/equity/crsp"]
    dataset["adj_prc"] = dataset["prc"] / dataset["cfacpr"]
    
    # Compute the rolling Exponential Moving Average (EMA) of the "close" price over a 252-day window.
    # 252 days correspond to roughly one trading year.
    
    import time
    start = time.time()
    ema_dataset = dataset.dt.rolling(dim='time', window=252).reduce(ema)
    print(f"Took: {time.time() - start}")
    
    print(ema_dataset)

    # Convert the xarray datasets to time-indexed Pandas DataFrames for easier plotting.
    # --- Original closing prices ---
    apple_orig = dataset.sel(asset=14593).dt.to_time_indexed()
    tsla_orig  = dataset.sel(asset=93436.).dt.to_time_indexed()
    
    # --- EMA-rolled closing prices ---
    apple_ema = ema_dataset.sel(asset=14593).dt.to_time_indexed()
    tsla_ema  = ema_dataset.sel(asset=93436.).dt.to_time_indexed()
    
    # Extract the closing prices from the original and EMA datasets.
    apple_close_orig = apple_orig["adj_prc"]
    tsla_close_orig  = tsla_orig["adj_prc"]
    
    apple_close_ema  = apple_ema["adj_prc"]
    tsla_close_ema   = tsla_ema["adj_prc"]
    
    # Create subplots with two rows and one column for Apple and Tesla plots.
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # --- Apple subplot ---
    # Plot original closing prices and EMA for AAPL on the first subplot.
    apple_close_orig.plot.line(
        x="time", ax=ax1, label="AAPL Close", color="blue", linestyle="-"
    )
    apple_close_ema.plot.line(
        x="time", ax=ax1, label="AAPL EMA", color="blue", linestyle="--"
    )
    ax1.set_title("Apple (AAPL) Closing Prices vs. EMA")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()

    # --- Tesla subplot ---
    # Plot original closing prices and EMA for TSLA on the second subplot.
    tsla_close_orig.plot.line(
        x="time", ax=ax2, label="TSLA Close", color="red", linestyle="-"
    )
    tsla_close_ema.plot.line(
        x="time", ax=ax2, label="TSLA EMA", color="red", linestyle="--"
    )
    ax2.set_title("Tesla (TSLA) Closing Prices vs. EMA")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()

    # Adjust layout for better spacing between subplots.
    plt.tight_layout()

    # Save the figure to a file.
    plt.savefig("apple_tsla_ema.png")
    plt.close()

if __name__ == "__main__":
    main()
