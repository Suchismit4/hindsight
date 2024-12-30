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

    dataset = data_manager.get_data("data_requests.yaml")

    # Access the dataset
    print(dataset)

if __name__ == "__main__":
    main()