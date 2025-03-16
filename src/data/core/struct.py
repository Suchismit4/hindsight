# src/data/core/struct.py


import numpy as np
import xarray as xr
import jax.numpy as jnp
import equinox as eqx
from typing import Union, Dict, List, Optional, Tuple, Any
import equinox as eqx
import functools


from src.data.core.operations import TimeSeriesOps
from .util import Rolling
import warnings

warnings.simplefilter("ignore", xr.core.extensions.AccessorRegistrationWarning)

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

    def shift(self, periods: int = 1) -> Union[xr.Dataset, xr.DataArray]:
        """
        Shift the data by a specified number of business days, skipping weekends and holidays.
        
        Parameters:
            periods (int): Number of periods to shift. Positive values shift forward in time,
                        negative values shift backward.
        
        Returns:
            Union[xr.Dataset, xr.DataArray]: A new xarray object with shifted data.
        """
        obj = self._obj
        
        # Handle Dataset case by applying shift to each DataArray
        if isinstance(obj, xr.Dataset):
            # Extract mask and mask_indices from the dataset
            mask_indices = obj.coords.get('mask_indices', None)
            
            if mask_indices is None:
                # Can't perform business day shifting without mask and mask_indices
                return obj.copy()
            
            # Shift each data variable
            shifted_vars = {}
            for var_name, da in obj.data_vars.items():
                if np.issubdtype(da.dtype, np.number):
                    if 'year' not in da.coords:
                        shifted_vars[var_name] = da
                    else:
                        # Stack time dimensions for efficient processing
                        stacked_da = da.stack(time_index=("year", "month", "day"))
                        stacked_da = stacked_da.transpose("time_index", ...)
                        
                        # Convert to JAX arrays for efficient computation
                        indices_array = jnp.array(mask_indices.values)
                        # Replace -1 with 0 for valid indexing, but maintain mask for filtering
                        indices_array = jnp.where(indices_array == -1, 0, indices_array).astype(jnp.int32)
                        data = jnp.asarray(stacked_da.values)
                        
                        # Apply the shift operation
                        shifted_data = TimeSeriesOps.shift(data, indices_array, periods)
                        
                        # Reconstruct the DataArray and unstack
                        shifted_da = stacked_da.copy(data=shifted_data)
                        shifted_vars[var_name] = shifted_da.unstack("time_index")
                else:
                    # Non-numeric data doesn't get shifted
                    shifted_vars[var_name] = da
            
            # Create a new dataset with the shifted variables and original coordinates
            return xr.Dataset(shifted_vars, coords=obj.coords, attrs=obj.attrs)
        
        # For DataArrays
        # Extract mask and mask_indices from the dataset
        mask_indices = obj.coords.get('mask_indices', None)
            
        if mask_indices is None:
            raise ValueError("No mask found and tried to shift a DataArray.")
        
        # If the DataArray is not numeric, simply return it.
        if not np.issubdtype(self.obj.dtype, np.number):
            return self.obj
        
        stacked_obj = self.obj.stack(time_index=("year", "month", "day"))
        stacked_obj = stacked_obj.transpose("time_index", ...)
        
        # Convert to JAX arrays for efficient computation
        indices_array = jnp.array(mask_indices.values)
        # Replace -1 with 0 for valid indexing, but maintain mask for filtering
        indices_array = jnp.where(indices_array == -1, 0, indices_array).astype(jnp.int32)
        data = jnp.asarray(stacked_obj.values)
        
        # Apply the shift operation
        shifted_data = TimeSeriesOps.shift(data, indices_array, periods)
                    
        # Reconstruct the DataArray and unstack
        shifted_da = stacked_obj.copy(data=shifted_data)
        unstacked_da = shifted_da.unstack("time_index")
        
        # For other cases (no time dimension or periods=0), return a copy
        return unstacked_da

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
