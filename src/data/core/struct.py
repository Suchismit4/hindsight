# src/data/core/struct.py


import numpy as np
import pandas as pd
import xarray as xr
import xarray_jax as xj
import jax.numpy as jnp
import equinox as eqx
from typing import Union, Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
import equinox as eqx

from .util import Rolling

class DateTimeAccessorBase:
    """
    Base class for managing time series operations on panel data structures.

    Provides methods to infer data dimensions, create metadata, and align datasets.

    Attributes:
        _obj (Union[xr.Dataset, xr.DataArray]): The xarray object being accessed.
    """

    def __init__(self, xarray_obj: Union[xr.Dataset, xr.DataArray]) -> None:
        """
        Initializes the DateTimeAccessorBase with an xarray object.

        Parameters:
            xarray_obj (Union[xr.Dataset, xr.DataArray]): The xarray object to be accessed.
        """
        self._obj = xarray_obj

    def sel(self, time):
        """
        Selects data corresponding to the given time(s) using TimeSeriesIndex.

        Parameters:
            time: The timestamp(s) to select.

        Returns:
            xarray object: The selected data.
        """
        ts_index = self._obj.coords['time'].attrs['indexes']['time']
        return self._obj.isel(**ts_index.sel(time))

    def to_time_indexed(self):
        """
        Converts multi-dimensional data into time-indexed format without leaving
        inconsistent multi-index coordinates.
        """
        ds = self._obj

        # Rename original 3D 'time' to avoid collision
        ds = ds.rename({'time': 'time_3d'})
        
        # Stack year/month/day into a single dimension
        ds_flat = ds.stack(stacked_time=("year", "month", "day"))
        
        # Prepare to clean up old coordinates before assigning new time
        vars_to_drop = ['time', 'year', 'month', 'day', 'time_3d', 'stacked_time']
        
        # Remove old coordinates FIRST to prevent inconsistency
        ds_flat = ds_flat.drop_vars(vars_to_drop, errors="ignore")
        
        # Rename dimension and assign new time coordinate
        ds_flat = ds_flat.rename_dims({"stacked_time": "time"})
        ds_flat = ds_flat.assign_coords(
            time=("time", ds["time_3d"].values.ravel())
        )
        
        return ds_flat
    
    def rolling(self, dim: str, window: int) -> 'Rolling':
        """
        Creates a Rolling object for applying rolling window operations.

        Parameters:
            dim (str): The dimension over which to apply the rolling window.
            window (int): The size of the rolling window.

        Returns:
            Rolling: An instance of the Rolling class.
        """
        # Extract mask and indices from the Dataset 
        mask = jnp.array(self._obj.coords['mask'].values)  # Shape: (T,)
        indices = jnp.array(self._obj.coords['mask_indices'].values)  # Shape: (T,)
        return Rolling(self._obj, dim, window, mask, indices)

# Register accessors for xarray objects
@xr.register_dataset_accessor('dt')
class DatasetDateTimeAccessor(DateTimeAccessorBase):
    """
    Extends DateTimeAccessorBase to work with xarray.Dataset objects.
    """
    pass

@xr.register_dataarray_accessor('dt')
class DataArrayDateTimeAccessor(DateTimeAccessorBase):
    """
    Extends DateTimeAccessorBase to work with xarray.DataArray objects.
    """
    pass
