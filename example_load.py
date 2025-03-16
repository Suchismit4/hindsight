# hindsight/example_load.py
# ----------------------------------------
# This script demonstrates how to load financial datasets using the DataManager,
# showing both YAML-based configuration loading and direct API loading with
# Django-style post-processors. It also performs basic operations on the data,
# and visualizes the adjusted prices for comparison.
# ----------------------------------------

from src import DataManager
import matplotlib.pyplot as plt

def main():
    """
    Main function to load financial datasets, compute adjusted prices,
    and plot CRSP adjusted prices vs. Yahoo Finance closing prices.
    Shows two different ways to load data: from YAML config and direct API calls.
    """
    # Initialize the DataManager to handle data retrieval
    data_manager = DataManager()

    # Method 1: Load datasets as specified in a YAML configuration file
    print("Loading data from YAML configuration file...")
    dataset_collection = data_manager.get_data("data_requests.yaml")

    # Method 2: Load datasets directly with Django-style post-processors
    print("\nLoading data directly with Django-style post-processors...")
    direct_datasets = data_manager.get_data([
        {
            "data_path": "wrds/equity/crsp",
            "config": {
                "start_date": "2000-01-01",
                "end_date": "2024-01-01",
                "freq": "M",
                "filters": {
                    # Django-style filter syntax
                    "date__gte": "2000-01-01",
                    "exchcd": 1,               # NYSE only
                    "shrcd__in": [10, 11]      # Common stocks only
                },
                # Django-style post-processor configuration
                "processors": {
                    # Set PERMNO and PERMCO as coordinates for easier selection
                    "set_permno_coord": True,
                    "set_permco_coord": True,
                    
                    # Fix market equity values for companies with multiple securities
                    "fix_market_equity": True,
                    
                    # Replacing delisting returns with actual returns
                    "replace_values": {
                        "source": "delistings",
                        "rename": [["dlstdt", "time"]],
                        "identifier": "permno",
                        "from_var": "dlret",
                        "to_var": "ret"
                    },
                    
                    # Merging company metadata from MSENAMES table
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
                    ]
                }
            }
        }
    ])

    # Compare both loading methods by printing the shapes
    print("\nComparison of loading methods:")
    print(f"YAML-based dataset shape: {dataset_collection['wrds/equity/crsp'].dims}")
    print(f"Direct API dataset shape: {direct_datasets[0].dims}")
    
    # Continue with the original visualization example using the YAML-loaded data
    print("\nProceeding with visualization using YAML-loaded data...")

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
