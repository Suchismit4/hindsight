# hindsight/main.py

from src import DataManager
import xarray as xr
import xarray_jax as xj
import numpy as np
import pandas as pd
import jax.numpy as jnp
import equinox as eqx
from functools import partial

def main():
    data_manager = DataManager()

    # Specify the columns you want to read from the Compustat dataset
    columns_to_read = [
        'gvkey', 'datadate', 'at', 'pstkl', 'txditc',
        'pstkrv', 'seq', 'pstk', 'indfmt', 'datafmt',
        'popsrc', 'consol'
    ]

    # Load the dataset using the DataManager
    dataset = data_manager.get_data([
        {
            'data_path': 'wrds/equities/compustat',
            'config': {
                'columns_to_read': columns_to_read,
                'num_processes': 16,
                'filters': {
                    'indfmt': 'INDL',
                    'datafmt': 'STD',
                    'popsrc': 'D',
                    'consol': 'C',
                    'datadate': ('>=', '1959-01-01')
                },
            }
        }
    ])

    # Access the Compustat dataset
    compustat_ds = dataset['/market/equities/wrds/compustat'].ds

    # Print the dataset structure
    print(compustat_ds)

if __name__ == "__main__":
    main()