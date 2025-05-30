"""
Simplified verification script for GlobalTimeMask length comparison.

This script loads data, determines its date range, gets the dataset's mask 
and the corresponding global mask slice, and prints their lengths.
"""
import os
import sys
import pandas as pd
import numpy as np
import jax.numpy as jnp
import xarray as xr

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src import DataManager
from src.data.core import get_global_mask

def verify_mask_lengths():
    """Loads data and prints lengths of dataset mask and global mask slice."""
    print("=== Comparing Mask Lengths ===")

    # --- 1. Load sample data --- 
    print("Loading sample CRSP data...")
    dm = DataManager()
    # Keep the try/except for data loading itself, as it depends on external factors
    try:
        ds = dm.get_data({
            "data_sources": [{
                "data_path": "wrds/equity/crsp",
                "config": {
                    "start_date": "2020-01-01", 
                    "end_date": "2021-12-31",
                    "freq": "D",
                    "filters": {
                        "date__gte": "2020-01-01"
                    },
                    "processors": {"set_permco_coord": True}
                }
            }]
        })['wrds/equity/crsp']
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Ensure WRDS connection is available or use simulated data.")
        return # Exit if data loading fails
        
    # --- 2. Get Dataset Mask Info ---
    print("--- Dataset Mask Info ---")
    dataset_mask_coord = ds.coords['mask']
    dataset_indices_coord = ds.coords['mask_indices']
    print(f"Dataset mask length (from coords): {len(dataset_mask_coord.values)}")
    print(f"Dataset indices length (from coords): {len(dataset_indices_coord.values)}")
    # Count valid business days in dataset mask
    dataset_valid_days = np.sum(dataset_mask_coord.values)
    print(f"Number of valid days in dataset mask: {dataset_valid_days}")

    # --- 3. Get Global Mask Slice Info ---
    print("\n--- Global Mask Slice Info ---")
    print("Retrieving corresponding slice from GlobalTimeMask...")

    # Determine date range from the dataset's time coordinate
    # Flatten the 3D time coordinate and find min/max
    dataset_time_flat = ds['time'].values.ravel()

    # Find min/max, which might be NaT
    min_date_np = dataset_time_flat.min().astype('datetime64[D]')
    max_date_np = dataset_time_flat.max().astype('datetime64[D]')
    print(f"Dataset date range (raw min/max): {min_date_np} to {max_date_np}")

    # Check for NaT before converting to ordinals
    if np.isnat(min_date_np) or np.isnat(max_date_np):
        print("Error: Cannot determine valid date range for slicing due to NaT values.")
        return

    # Convert valid dates to integer ordinals (days since epoch)
    epoch = np.datetime64('1970-01-01', 'D')
    start_ordinal = (min_date_np - epoch).astype('timedelta64[D]').astype(np.int32)
    end_ordinal = (max_date_np - epoch).astype('timedelta64[D]').astype(np.int32)
    print(f"Converted ordinals: {start_ordinal} to {end_ordinal}")

    global_mask_instance = get_global_mask()
    try:
        # Get the boolean mask slice and the padded indices slice using ordinals
        global_mask_slice_bool, global_mask_slice_indices = global_mask_instance.get_mask_slice(start_ordinal, end_ordinal)

        print(f"Global mask slice length (boolean): {len(global_mask_slice_bool)}")
        print(f"Global mask slice length (indices): {len(global_mask_slice_indices)}")

        # Count valid business days in global mask slice
        global_valid_days = np.sum(global_mask_slice_bool)
        print(f"Number of valid days in global mask slice: {global_valid_days}")

        # --- 4. Comparison ---
        print("\n--- Comparison Results ---")

        # Compare lengths
        length_match = len(dataset_mask_coord.values) == len(global_mask_slice_bool)
        print(f"Length comparison (Dataset mask vs Global slice bool): {'Match' if length_match else 'Mismatch!'}")
        length_match_idx = len(dataset_indices_coord.values) == len(global_mask_slice_indices)
        print(f"Length comparison (Dataset indices vs Global slice indices): {'Match' if length_match_idx else 'Mismatch!'}")

        # Compare number of valid days
        valid_days_match = dataset_valid_days == global_valid_days
        print(f"Valid day count comparison: {'Match' if valid_days_match else 'Mismatch!'}")

        # Compare boolean mask patterns
        bool_mask_match = np.array_equal(dataset_mask_coord.values, global_mask_slice_bool)
        print(f"Boolean mask pattern comparison: {'Match' if bool_mask_match else 'Mismatch!'}")

        # Compare derived boolean masks from indices
        dataset_indices_bool = dataset_indices_coord.values != -1
        global_indices_bool = np.array(global_mask_slice_indices) != -1 # Convert JAX array for numpy op
        indices_bool_match = np.array_equal(dataset_indices_bool, global_indices_bool)
        print(f"Derived boolean mask from indices comparison: {'Match' if indices_bool_match else 'Mismatch!'}")

        # Optional: Detailed index comparison (can be verbose)
        # if not indices_bool_match:
        #     print("Detailed Index Mismatch:")
        #     mismatches = np.where(dataset_indices_bool != global_indices_bool)[0]
        #     print(f"  Indices where derived bool masks differ: {mismatches[:10]}...")

    except Exception as e:
        print(f"Error during GlobalTimeMask slicing or comparison: {e}")

if __name__ == "__main__":
    verify_mask_lengths() 