# hindsight/example_rolling.py
# ----------------------------------------
# This script demonstrates how to compute and plot the rolling Exponential Moving Average (EMA)
# for Apple's (AAPL) and Tesla's (TSLA) closing stock prices using data from CRSP.
# The script calculates multiple EMAs with different window sizes and compares them.
# ----------------------------------------

from src import DataManager
from src.data.core.operations import mean, median, mode, ema
import matplotlib.pyplot as plt
import time

def main():
    """
    Main function to fetch historical stock data, compute rolling EMAs with different window sizes,
    and plot the original closing prices alongside the EMAs for AAPL and TSLA.
    """
    # Initialize the DataManager to handle dataset operations
    dm = DataManager()
    
    # Pull in the CRSP data for Apple and Tesla.
    # Data parameters: symbols, date range, and data provider configuration.
    datasets = dm.get_data([{"data_path": "wrds/equity/crsp",
        "config": {
            "start_date": "2000-01-01",
            "end_date": "2024-01-01",
            "freq": "D",
            "filters": {
                "date__gte": "2000-01-01"
            },
            "processors": {
                "replace_values": {
                    "source": "delistings",
                    "rename": [["dlstdt", "time"]],
                    "identifier": "permno",
                    "from_var": "dlret",
                    "to_var": "ret"
                },
                "merge_table": [
                    {
                        "source": "msenames",
                        "identifier": "permno",
                        "column": "comnam",
                        "axis": "asset"
                    },
                    {
                        "source": "msenames",
                        "identifier": "permno",
                        "column": "exchcd",
                        "axis": "asset"
                    }
                ],
                "set_permco_coord":  True,
                "fix_market_equity": True
            }
        }}])


    
    # Extract the original dataset for historical prices.
    dataset = datasets["wrds/equity/crsp"]
    dataset["adj_prc"] = dataset["prc"] / dataset["cfacpr"]
    
    # Window sizes for EMAs (in trading days)
    window_sizes = [30, 60, 252]  # 30 days, 60 days, 252 days (roughly 1 trading year)
    
    # Compute EMAs for different window sizes and track computation time
    ema_datasets = {}
    for window in window_sizes:
        print(f"Computing {window}-day EMA...")
        start_time = time.time()
        ema_datasets[window] = dataset.dt.rolling(dim='time', window=window).reduce(ema)
        elapsed = time.time() - start_time
        print(f"Computed {window}-day EMA in {elapsed:.2f} seconds")
    
    # Convert the xarray datasets to time-indexed form for easier plotting
    # --- Original closing prices ---
    apple_orig = dataset.sel(asset=14593).dt.to_time_indexed()
    tsla_orig = dataset.sel(asset=93436.).dt.to_time_indexed()
    
    # --- EMA-rolled closing prices for different window sizes ---
    apple_emas = {window: ema_datasets[window].sel(asset=14593).dt.to_time_indexed() for window in window_sizes}
    tsla_emas = {window: ema_datasets[window].sel(asset=93436.).dt.to_time_indexed() for window in window_sizes}
    
    # --- Create subplots for the closing prices and EMAs ---
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    
    # --- Apple subplot ---
    # Plot original closing prices and EMAs for AAPL
    apple_orig["adj_prc"].plot.line(x="time", ax=ax1, label="AAPL Close", color="blue", linestyle="-")
    
    colors = ["darkblue", "royalblue", "lightblue"]
    for i, window in enumerate(window_sizes):
        apple_emas[window]["adj_prc"].plot.line(
            x="time", 
            ax=ax1, 
            label=f"AAPL {window}-day EMA", 
            color=colors[i], 
            linestyle="--"
        )
    
    ax1.set_title("Apple (AAPL) Closing Prices vs. EMAs")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    
    # --- Tesla subplot ---
    # Plot original closing prices and EMAs for TSLA
    tsla_orig["adj_prc"].plot.line(x="time", ax=ax2, label="TSLA Close", color="red", linestyle="-")
    
    colors = ["darkred", "red", "salmon"]
    for i, window in enumerate(window_sizes):
        tsla_emas[window]["adj_prc"].plot.line(
            x="time", 
            ax=ax2, 
            label=f"TSLA {window}-day EMA", 
            color=colors[i], 
            linestyle="--"
        )
    
    ax2.set_title("Tesla (TSLA) Closing Prices vs. EMAs")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()
    
    # Adjust layout for better spacing between subplots
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("apple_tsla_multiple_emas.png")
    plt.close()
    
    # --- Create a figure for returns ---
    fig, (ax3, ax4) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    
    # --- Apple returns subplot ---
    apple_orig["ret"].plot.line(x="time", ax=ax3, label="AAPL Returns", color="blue", linestyle="-", alpha=0.15)
    
    colors = ["darkblue", "royalblue", "lightblue"]
    for i, window in enumerate(window_sizes):
        apple_emas[window]["ret"].plot.line(
            x="time", 
            ax=ax3, 
            label=f"AAPL {window}-day EMA Returns", 
            color=colors[i], 
            linestyle="--"
        )
    
    ax3.axhline(y=0, color="black", linewidth=1)
    ax3.set_title("Apple (AAPL) Returns vs. EMA Returns")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Returns")
    ax3.legend()
    
    # --- Tesla returns subplot ---
    tsla_orig["ret"].plot.line(x="time", ax=ax4, label="TSLA Returns", color="red", linestyle="-", alpha=0.15)
    
    colors = ["darkred", "red", "salmon"]
    for i, window in enumerate(window_sizes):
        tsla_emas[window]["ret"].plot.line(
            x="time", 
            ax=ax4, 
            label=f"TSLA {window}-day EMA Returns", 
            color=colors[i], 
            linestyle="--"
        )
    
    ax4.axhline(y=0, color="black", linewidth=1)
    ax4.set_title("Tesla (TSLA) Returns vs. EMA Returns")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Returns")
    ax4.legend()
    
    # Adjust layout for better spacing between subplots
    plt.tight_layout()
    
    # Save the returns figure
    plt.savefig("apple_tsla_multiple_ema_returns.png")
    plt.close()

if __name__ == "__main__":
    main()
