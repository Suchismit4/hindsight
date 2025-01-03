# hindsight/main.py

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
    data_manager = DataManager()

    dataset = data_manager.get_data("data_requests.yaml")  # un post-proc.

    crsp_ds = dataset["wrds/equity/crsp"]
    crsp_ds["adj_prc"] = crsp_ds["prc"] / crsp_ds["cfacpr"]

    yfinance_ds = dataset["openbb/equity/price/historical"]
    
    # Time-index them
    crsp_sel = crsp_ds.sel(asset=14593).dt.to_time_indexed()
    yf_sel = yfinance_ds.sel(asset="AAPL").dt.to_time_indexed()

    crsp_adj = crsp_sel["adj_prc"]
    yf_close = yf_sel["close"]

    # Create subplots: two rows, one column
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # Plot CRSP data on the first (top) subplot
    crsp_adj.plot.line(x="time", ax=ax1, label="CRSP (permno=14593)", color="blue")
    ax1.set_title("CRSP Adjusted Price")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price")
    ax1.legend()

    # Plot YF data on the second (bottom) subplot
    yf_close.plot.line(x="time", ax=ax2, label="AAPL (YF)", color="red")
    ax2.set_title("YFinance Close")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Price")
    ax2.legend()

    # Adjust spacing, if needed
    plt.tight_layout()

    # Save the figure
    plt.savefig("crsp_vs_yfinance_subplots.png")
    plt.close()
    
if __name__ == "__main__":
    main()