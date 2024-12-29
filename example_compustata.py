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
    print("Loading data with config")
    dataset = data_manager.get_data([
        {
            'data_path': 'wrds/equity/compustat',
            'config': {
                'columns_to_read': columns_to_read,
                'freq': 'A',
                'num_processes': 16,
                'filters': {
                    'indfmt': 'INDL',
                    'datafmt': 'STD',
                    'popsrc': 'D',
                    'consol': 'C',
                    'date': ('>=', '1959-01-01')
                },
            }
        }
    ])

    # Access the Compustat dataset
    print("Converting to ds")
    compustat_ds = dataset['/market/equities/wrds/compustat'].ds

    # Remove duplicate entries for annual data
    df = compustat_ds.to_dataset().to_dataframe()
    print(f"df type: {type(df)}")
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', ascending=False, inplace=True)

    # drop based on other row values, create a function
    # df_cleaned = df.drop_duplicates(subset='identifier', keep='last')

    # Add lag parameter for time shifts 
    df_cleaned['lag'] = 0

    # Return back to DataTree
    compustat_ds = xr.DataTree(
        dataset=xr.Dataset.from_dataframe(df_cleaned)
    ).ds

    # Print the dataset structure
    print(compustat_ds)

if __name__ == "__main__":
    main()