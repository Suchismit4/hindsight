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
            "start_date": "2015-01-01",
            "end_date":   "2024-01-01",
            "freq": "D",
            "filters": {
                "date__gte": "2015-01-01"
            },
            "processors": {
                # Replacing delisting returns with actual returns
                "replace_values": {
                    "source": "delistings",
                    "rename": [["dlstdt", "time"]],
                    "identifier": "permno",
                    "from_var": "dlret",
                    "to_var": "ret"
                },
                # Merging company names from MSENAMES table
                "merge_table": {
                    "source": "msenames",
                    "identifier": "permno",
                    "column": "comnam",
                    "axis": "asset"
                }
            }
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
    
    # --- Extract the closing prices and returns from the original and EMA datasets ---
    apple_close_orig = apple_orig["adj_prc"]
    tsla_close_orig = tsla_orig["adj_prc"]
    apple_close_orig_returns = apple_orig["ret"]
    tsla_close_orig_returns = tsla_orig["ret"]

    apple_close_ema = apple_ema["adj_prc"]
    tsla_close_ema = tsla_ema["adj_prc"]
    apple_close_ema_returns = apple_ema["ret"] 
    tsla_close_ema_returns = tsla_ema["ret"]   
    
    # --- Create subplots for the closing prices and EMA ---
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # --- Apple subplot ---
    # Plot original closing prices and EMA for AAPL on the first subplot.
    apple_close_orig.plot.line(x="time", ax=ax1, label="AAPL Close", color="blue", linestyle="-")
    apple_close_ema.plot.line(x="time", ax=ax1, label="AAPL EMA", color="blue", linestyle="--")
    ax1.set_title("Apple (AAPL) Closing Prices vs. EMA")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()

    # --- Tesla subplot ---
    # Plot original closing prices and EMA for TSLA on the second subplot.
    tsla_close_orig.plot.line(x="time", ax=ax2, label="TSLA Close", color="red", linestyle="-")
    tsla_close_ema.plot.line(x="time", ax=ax2, label="TSLA EMA", color="red", linestyle="--")
    ax2.set_title("Tesla (TSLA) Closing Prices vs. EMA")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()

    # Adjust layout for better spacing between subplots.
    plt.tight_layout()

    # Save the closing prices figure to a file.
    plt.savefig("apple_tsla_ema.png")
    plt.close()


    # --- Create a new figure for plotting returns ---
    fig, (ax3, ax4) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # --- Apple returns subplot ---
    # Plot original returns and (if applicable) EMA returns for AAPL.
    apple_close_orig_returns.plot.line(x="time", ax=ax3, label="AAPL Returns", color="blue", linestyle="-", alpha=0.15)
    apple_close_ema_returns.plot.line(x="time", ax=ax3, label="AAPL EMA Returns", color="blue", linestyle="--")
    # Add a horizontal line at 0 to indicate zero returns.
    ax3.axhline(y=0, color="black", linewidth=1)
    ax3.set_title("Apple (AAPL) Returns vs. EMA Returns")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Returns")
    ax3.legend()

    # --- Tesla returns subplot ---
    # Plot original returns and (if applicable) EMA returns for TSLA.
    tsla_close_orig_returns.plot.line(x="time", ax=ax4, label="TSLA Returns", color="red", linestyle="-", alpha=0.15)
    tsla_close_ema_returns.plot.line(x="time", ax=ax4, label="TSLA EMA Returns", color="red", linestyle="--")
    # Add a horizontal line at 0 to indicate zero returns.
    ax4.axhline(y=0, color="black", linewidth=1)
    ax4.set_title("Tesla (TSLA) Returns vs. EMA Returns")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Returns")
    ax4.legend()

    # Adjust layout for better spacing between subplots.
    plt.tight_layout()

    # Save the returns figure to a file.
    plt.savefig("apple_tsla_returns.png")
    plt.close()

if __name__ == "__main__":
    main()
