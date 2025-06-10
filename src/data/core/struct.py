"""
Time series data structures and accessors for panel data in Hindsight.

This module provides custom accessors for xarray datasets and data arrays that facilitate
financial time series operations. The central components include:

1. DateTimeAccessorBase: Base class implementing time series functionality
2. DatasetDateTimeAccessor: Accessor for xarray.Dataset objects
3. DataArrayDateTimeAccessor: Accessor for xarray.DataArray objects

These accessors enable financial-specific operations like:
- Time-based selection via business day indices
- Converting between multi-dimensional and time-indexed formats
- Rolling window operations with business day awareness
- Shifting data by business days (skipping weekends/holidays)

The accessors are registered with xarray using the 'dt' namespace.
"""

import numpy as np
import xarray as xr
import jax.numpy as jnp
import equinox as eqx
from typing import Union, Dict, List, Optional, Tuple, Any
import xarray_jax

from src.data.core.operations import TimeSeriesOps
from .util import Rolling
import warnings

# Suppress xarray accessor registration warnings
# warnings.simplefilter("ignore", xr.core.extensions.AccessorRegistrationWarning)

class DateTimeAccessorBase:
    """
    Base class for managing time series operations on panel data structures.

    Provides methods to perform time-based selection, convert between data formats,
    and apply time series operations with business day awareness.

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

        This method uses the TimeSeriesIndex stored in the time coordinate's attributes
        to efficiently select data for the specified timestamps.

        Parameters:
            time: The timestamp(s) to select.

        Returns:
            Union[xr.Dataset, xr.DataArray]: The selected data.
        """
        ts_index = self._obj.coords['time'].attrs['indexes']['time']
        return self._obj.isel(**ts_index.sel(time))

    def to_time_indexed(self):
        """
        Converts multi-dimensional data into time-indexed format.

        Transforms data with separate year/month/day/hour dimensions into a single time dimension,
        ensuring no inconsistent multi-index coordinates remain.

        Returns:
            Union[xr.Dataset, xr.DataArray]: The time-indexed data.
        """
        ds = self._obj

        # Determine time dimensions to stack based on what's present
        time_dims = ["year", "month", "day"]
        if "hour" in ds.dims:
            time_dims.append("hour")

        # Rename original time coordinate to avoid collision
        if len(time_dims) == 3:
            ds = ds.rename({'time': 'time_3d'})
            old_time_name = 'time_3d'
        else:
            ds = ds.rename({'time': 'time_4d'})
            old_time_name = 'time_4d'
        
        # Stack time dimensions into a single dimension
        ds_flat = ds.stack(stacked_time=tuple(time_dims))
        
        # Prepare to clean up old coordinates before assigning new time
        vars_to_drop = ['time'] + time_dims + [old_time_name, 'stacked_time']
        
        # Remove old coordinates FIRST to prevent inconsistency
        ds_flat = ds_flat.drop_vars(vars_to_drop, errors="ignore")
        
        # Rename dimension and assign new time coordinate
        ds_flat = ds_flat.rename_dims({"stacked_time": "time"})
        ds_flat = ds_flat.assign_coords(
            time=("time", ds[old_time_name].values.ravel())
        )
        
        return ds_flat
    
    def rolling(self, dim: str, window: int, mask: Optional[jnp.ndarray] = None, mask_indices: Optional[jnp.ndarray] = None) -> 'Rolling':
        """
        Creates a Rolling object for applying rolling window operations.

        The returned Rolling object properly handles business days when computing
        rolling operations, accounting for weekends and holidays.

        Parameters:
            dim (str): The dimension over which to apply the rolling window.
            window (int): The size of the rolling window.
            mask (Optional[jnp.ndarray]): Boolean mask indicating valid business days.
                If None, will attempt to extract from the object's coordinates.
            mask_indices (Optional[jnp.ndarray]): Indices mapping business days to positions.
                If None, will attempt to extract from the object's coordinates.

        Returns:
            Rolling: An instance of the Rolling class that allows applying 
                    window operations with methods like .mean(), .sum(), etc.
                    
        Raises:
            ValueError: If unable to obtain the required mask and indices for
                        the rolling operation.
        """
        obj = self._obj
        
        # Use provided mask and indices if available
        if mask is None or mask_indices is None:
            # If not provided, try to extract from the object's coordinates
            if 'mask' in obj.coords and mask is None:
                mask = jnp.array(obj.coords['mask'].values)
            if 'mask_indices' in obj.coords and mask_indices is None:
                mask_indices = jnp.array(obj.coords['mask_indices'].values)
            
            # For DataArrays, also try to get from parent Dataset if available
            if isinstance(obj, xr.DataArray) and hasattr(self, '_parent_obj') and self._parent_obj is not None:
                parent_ds = self._parent_obj
                if 'mask' in parent_ds.coords and mask is None:
                    mask = jnp.array(parent_ds.coords['mask'].values)
                if 'mask_indices' in parent_ds.coords and mask_indices is None:
                    mask_indices = jnp.array(parent_ds.coords['mask_indices'].values)
            
            # If still not available, raise an error
            if mask is None or mask_indices is None:
                raise ValueError(
                    "Rolling operation requires mask and mask_indices. "
                    "Either provide them as parameters or ensure they exist "
                    "as coordinates on the object."
                )
        
        return Rolling(obj, dim, window, mask=mask, indices=mask_indices)

    def shift(self, periods: int = 1, mask_indices: Optional[jnp.ndarray] = None) -> Union[xr.Dataset, xr.DataArray]:
        """
        Shift the data by a specified number of business days, skipping weekends and holidays.
        
        Parameters:
            periods (int): Number of periods to shift. Positive values shift forward in time,
                        negative values shift backward.
            mask_indices (Optional[jnp.ndarray]): Indices mapping business days to positions.
                If None, will attempt to extract from the object's coordinates.
        
        Returns:
            Union[xr.Dataset, xr.DataArray]: A new xarray object with shifted data.
            
        Raises:
            ValueError: If no mask indices are found in the dataset coordinates.
        """
        obj = self._obj
        
        # Handle Dataset case by applying shift to each DataArray
        if isinstance(obj, xr.Dataset):
            # Extract mask and mask_indices from the dataset
            if mask_indices is None:
                mask_indices_coord = obj.coords.get('mask_indices', None)
                if mask_indices_coord is not None:
                    mask_indices = jnp.array(mask_indices_coord.values)
            
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
                        # Determine time dimensions to stack based on what's present
                        time_dims = ["year", "month", "day"]
                        if "hour" in da.dims:
                            time_dims.append("hour")
                        
                        # Stack time dimensions for efficient processing
                        stacked_da = da.stack(time_index=tuple(time_dims))
                        stacked_da = stacked_da.transpose("time_index", ...)
                        
                        # Convert to JAX arrays for efficient computation
                        indices_array = mask_indices
                        # Replace -1 with 0 for valid indexing, but maintain mask for filtering
                        indices_array = jnp.where(indices_array == -1, 0, indices_array).astype(jnp.int32)
                        
                        # Get JAX-compatible data without converting to numpy via .values
                        # Access the underlying data array directly to avoid TracerArrayConversionError
                        data = stacked_da.data
                        
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
        # Use provided mask_indices or extract from coordinates
        if mask_indices is None:
            mask_indices_coord = obj.coords.get('mask_indices', None)
            if mask_indices_coord is not None:
                mask_indices = jnp.array(mask_indices_coord.values)
            
        if mask_indices is None:
            raise ValueError("No mask found and tried to shift a DataArray.")
        
        # If the DataArray is not numeric, simply return it.
        if not np.issubdtype(obj.dtype, np.number):
            return obj
        
        # Determine time dimensions to stack based on what's present
        time_dims = ["year", "month", "day"]
        if "hour" in obj.dims:
            time_dims.append("hour")
        
        stacked_obj = obj.stack(time_index=tuple(time_dims))
        stacked_obj = stacked_obj.transpose("time_index", ...)
        
        # Convert to JAX arrays for efficient computation
        indices_array = mask_indices
        # Replace -1 with 0 for valid indexing, but maintain mask for filtering
        indices_array = jnp.where(indices_array == -1, 0, indices_array).astype(jnp.int32)
        
        # Get JAX-compatible data without converting to numpy via .values
        # Access the underlying data array directly to avoid TracerArrayConversionError
        data = stacked_obj.data
        
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
    Accessor for xarray.Dataset objects providing time series functionality.
    
    This class inherits all methods from DateTimeAccessorBase and is registered
    with xarray using the 'dt' namespace.
    
    Example:
        ```python
        # Access time series functionality on an xarray Dataset
        dataset.dt.sel(time='2022-01-01')
        dataset.dt.rolling(dim='time', window=20).mean()
        ```
    """
    
    def get_var(self, var_name: str) -> xr.DataArray:
        """
        Retrieve a DataArray from the Dataset with a reference back to its parent.
        
        This method stores a reference to the parent Dataset in the DataArray's attrs,
        enabling mask/indices access for rolling operations.
        
        Parameters:
            var_name (str): The name of the variable to extract.
            
        Returns:
            xr.DataArray: The DataArray with a reference to its parent Dataset.
            
        Raises:
            KeyError: If the variable doesn't exist in the Dataset.
        """
        if var_name not in self._obj.data_vars:
            raise KeyError(f"Variable '{var_name}' not found in Dataset")
            
        # Get the DataArray and copy it to avoid modifying the original
        data_array = self._obj[var_name].copy()
        
        # Store reference to parent Dataset in attrs
        data_array.attrs['_parent_dataset'] = self._obj
        
        return data_array

@xr.register_dataarray_accessor('dt')
class DataArrayDateTimeAccessor(DateTimeAccessorBase):
    """
    Accessor for xarray.DataArray objects providing time series functionality.
    
    This class inherits all methods from DateTimeAccessorBase and is registered
    with xarray using the 'dt' namespace.
    
    Example:
        ```python
        # Access time series functionality on an xarray DataArray
        data_array.dt.sel(time='2022-01-01')
        data_array.dt.rolling(dim='time', window=20).mean()
        ```
    """
    def __init__(self, xarray_obj: xr.DataArray) -> None:
        """
        Initialize the DataArrayDateTimeAccessor with a DataArray and attempt
        to locate its parent Dataset for mask/indices access.
        
        Parameters:
            xarray_obj (xr.DataArray): The xarray DataArray to access.
        """
        super().__init__(xarray_obj)
        
        # Try to identify the parent Dataset
        # First check if DataArray has a parent dataset reference in attrs
        self._parent_obj = xarray_obj.attrs.get('_parent_dataset', None)
        
        # If not found in attrs, try to find it through coordinates (legacy approach)
        if self._parent_obj is None and hasattr(xarray_obj, 'name') and xarray_obj.name is not None:
            # Find the DataArray's parent Dataset
            # This is a heuristic approach and might not always work
            # Ideally DataArrays would track their parent Dataset
            for var in xarray_obj.coords.values():
                if hasattr(var, 'attrs') and '_parent_dataset' in var.attrs:
                    parent_ds = var.attrs['_parent_dataset']
                    if isinstance(parent_ds, xr.Dataset) and xarray_obj.name in parent_ds.data_vars:
                        self._parent_obj = parent_ds
                        break
