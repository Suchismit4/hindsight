# hindsight/example_load.py
# ----------------------------------------
# This script demonstrates how to load financial datasets using the DataManager,
# perform basic operations on the data, and visualize the adjusted prices from
# the CRSP dataset alongside the closing prices from Yahoo Finance for Apple (AAPL).
# ----------------------------------------

from src import DataManager
import xarray as xr
import xarray_jax as xj
import numpy as np
import pandas as pd
import jax.numpy as jnp
import equinox as eqx
from functools import partial
import matplotlib.pyplot as plt

def main():
    """
    Main function to load financial datasets, compute adjusted prices,
    and plot CRSP adjusted prices vs. Yahoo Finance closing prices.
    """
    # Initialize the DataManager to handle data retrieval
    data_manager = DataManager()

    # Load datasets as specified in a YAML configuration file
    dataset_collection = data_manager.get_data("data_requests.yaml")  # post-processing not applied here

    # Extract the CRSP dataset and compute adjusted prices
    crsp_dataset = dataset_collection["wrds/equity/crsp"]
    crsp_dataset["adj_prc"] = crsp_dataset["prc"] / crsp_dataset["cfacpr"]

    # Extract the Yahoo Finance dataset for historical equity prices
    yfinance_dataset = dataset_collection["openbb/equity/price/historical"]
    
    # Select specific assets and convert their data to time-indexed form for plotting
    # Select CRSP data for asset with permno=14593
    crsp_time_series = crsp_dataset.sel(asset=14593).dt.to_time_indexed()
    # Select Yahoo Finance data for Apple (AAPL)
    yf_time_series = yfinance_dataset.sel(asset="AAPL").dt.to_time_indexed()

    # Extract the adjusted price and closing price series
    crsp_adj_price = crsp_time_series["adj_prc"]
    yf_close_price = yf_time_series["close"]

    # Create subplots: two rows, one column for side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # Plot CRSP adjusted prices on the first subplot
    crsp_adj_price.plot.line(x="time", ax=ax1, label="CRSP (permno=14593)", color="blue")
    ax1.set_title("CRSP Adjusted Price")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price")
    ax1.legend()

    # Plot Yahoo Finance closing prices on the second subplot
    yf_close_price.plot.line(x="time", ax=ax2, label="AAPL (Yahoo Finance)", color="red")
    ax2.set_title("Yahoo Finance Close Price")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Price")
    ax2.legend()

    # Adjust layout for better spacing between subplots
    plt.tight_layout()

    # Save the figure to a file
    plt.savefig("crsp_vs_yfinance_subplots.png")
    plt.close()

if __name__ == "__main__":
    main()
