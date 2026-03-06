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
from typing import Union, Dict, List, Optional, Tuple, Any, Sequence
import xarray_jax

from src.data.core.operations import TimeSeriesOps
from .rolling import Rolling
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
                If None, will attempt to compute on-demand.
            mask_indices (Optional[jnp.ndarray]): Indices mapping business days to positions.
                If None, will attempt to compute on-demand.

        Returns:
            Rolling: An instance of the Rolling class that allows applying 
                    window operations with methods like .mean(), .sum(), etc.
                    
        Raises:
            ValueError: If unable to obtain or compute the required mask and indices for
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
            
            # If still not available, compute on-demand
            if mask is None or mask_indices is None:
                try:
                    if isinstance(obj, xr.Dataset):
                        computed_mask, computed_indices = obj.dt.compute_mask()
                    elif isinstance(obj, xr.DataArray) and hasattr(self, '_parent_obj') and self._parent_obj is not None:
                        computed_mask, computed_indices = self._parent_obj.dt.compute_mask()
                    else:
                        raise ValueError("Cannot compute mask for this object")
                    
                    if mask is None:
                        mask = computed_mask
                    if mask_indices is None:
                        mask_indices = computed_indices
                except Exception as e:
                    raise ValueError(
                        f"Rolling operation requires mask and mask_indices. "
                        f"Could not obtain or compute them: {e}"
                    )
        
        return Rolling(obj, dim, window, mask=mask, indices=mask_indices)

    def compute_mask(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute mask and mask_indices on-demand from the dataset or data array.
        
        This method creates a per-asset mask indicating valid (non-NaN) data points,
        along with a stable "compressed timeline" mapping showing which original positions
        contain valid data for each asset. This computation is JIT-compatible.
        
        For Datasets: Computes mask from the full dataset structure
        For DataArrays: Computes mask from the DataArray's own data (no parent needed)
        
        Returns:
            tuple[jnp.ndarray, jnp.ndarray]:
                - mask: Boolean array (T, N) where T=time_index, N=assets. True indicates valid data.
                - mask_indices: Integer array (T, N) where entries are the original time index of the
                  t-th valid observation for asset a, or -1 if no valid observation at that position.
        
        Raises:
            ValueError: If unable to extract numeric data or determine time structure
            
        Notes:
            - This computation is fully JIT-compatible using JAX operations
            - Masks adapt dynamically to any slicing or transformations on the dataset
            - The indices are sorted such that valid positions come first per asset,
              with invalid positions (-1) padded at the bottom
        """
        obj = self._obj
        
        # Handle both Dataset and DataArray
        if isinstance(obj, xr.Dataset):
            data_source = obj
            # Determine time dimensions
            time_dims = [d for d in ("year", "month", "day") if d in data_source.dims]
            if "hour" in data_source.dims:
                time_dims.append("hour")
            
            if not time_dims:
                raise ValueError("Dataset must have time dimensions (year, month, day, or hour)")
            
            # Find the first numeric variable to use as reference
            stacked_obj = None
            for var_name, da in data_source.data_vars.items():
                if np.issubdtype(da.dtype, np.number):
                    stacked_obj = da.stack(time_index=tuple(time_dims))
                    stacked_obj = stacked_obj.transpose('time_index', 'asset')
                    break
            
            if stacked_obj is None:
                raise ValueError("No numeric variables found in dataset. Cannot compute mask.")
                
        elif isinstance(obj, xr.DataArray):
            # For DataArray, use it directly if it has the right structure
            # Check if it has asset dimension
            if 'asset' not in obj.dims:
                raise ValueError(
                    "DataArray must have 'asset' dimension. "
                    "For single-asset data, use dataset.dt.compute_mask() instead."
                )
            
            # Determine time dimensions
            time_dims = [d for d in ("year", "month", "day") if d in obj.dims]
            if "hour" in obj.dims:
                time_dims.append("hour")
            
            if not time_dims:
                raise ValueError("DataArray must have time dimensions (year, month, day, or hour)")
            
            # Stack time dimensions
            stacked_obj = obj.stack(time_index=tuple(time_dims))
            stacked_obj = stacked_obj.transpose('time_index', 'asset')
            
        else:
            raise TypeError(f"Unsupported xarray object type: {type(obj)}")
        
        # Extract data array with shape (T, N) and convert to JAX
        data_arr = jnp.asarray(stacked_obj.values, dtype=jnp.float64)
        
        # Create asset-specific mask: True where data is not NaN (JIT-compatible)
        mask = ~jnp.isnan(data_arr)  # Shape: (T, N)
        
        # Create stable compressed timeline per asset using JAX (JIT-compatible)
        # For each asset, positions of valid rows with -1 padding for invalid
        T = data_arr.shape[0]
        N = data_arr.shape[1]
        pos = jnp.arange(T)[:, None]  # (T, 1)
        
        # Where mask is True, use position; where False, use sentinel value for sorting
        # Use T (beyond all valid indices) as the sentinel so invalid go to bottom
        valid_pos_per_asset = jnp.where(mask, pos, -1)  # (T, N), invalid -> -1
        
        # For sorting: valid positions get their value, invalid get (T + index in invalid range)
        # This ensures valid positions sort to top while maintaining relative order
        max_pos = T
        sort_keys = jnp.where(mask, pos, max_pos + jnp.arange(T)[:, None])  # (T, N)
        
        # Argsort to get the permutation that sorts each asset's valid items to top
        order = jnp.argsort(sort_keys, axis=0)  # (T, N)
        
        # Gather sorted positions per asset using the order indices
        # This is JAX-compatible indexing
        sorted_pos = jnp.take_along_axis(valid_pos_per_asset, order, axis=0)  # (T, N)
        
        # Convert to proper types for output
        mask = jnp.asarray(mask, dtype=jnp.bool_)
        sorted_pos = jnp.asarray(sorted_pos, dtype=jnp.int64)
        
        return mask, sorted_pos

    def shift(self, periods: int = 1, mask_indices: Optional[jnp.ndarray] = None) -> Union[xr.Dataset, xr.DataArray]:
        """
        Shift the data by a specified number of business days, skipping weekends and holidays.
        
        Parameters:
            periods (int): Number of periods to shift. Positive values shift forward in time,
                        negative values shift backward.
            mask_indices (Optional[jnp.ndarray]): Indices mapping business days to positions.
                If None, will attempt to extract from the object's coordinates or compute on-demand.
        
        Returns:
            Union[xr.Dataset, xr.DataArray]: A new xarray object with shifted data.
            
        Raises:
            ValueError: If no mask indices are found and cannot be computed (DataArray case).
        """
        obj = self._obj

        if periods == 0:
            return obj.copy()

        if isinstance(obj, xr.Dataset):
            return self._shift_dataset(obj, periods, mask_indices)
        elif isinstance(obj, xr.DataArray):
            return self._shift_dataarray(obj, periods, mask_indices, allow_fallback=True)
        else:
            raise TypeError(f"Unsupported xarray object type: {type(obj)}")

    @staticmethod
    def _infer_time_dims(obj: Union[xr.Dataset, xr.DataArray]) -> List[str]:
        dims = [d for d in ("year", "month", "day") if d in obj.dims]
        if "hour" in obj.dims:
            dims.append("hour")
        return dims

    @staticmethod
    def _prepare_indices(indices: jnp.ndarray) -> jnp.ndarray:
        indices = jnp.asarray(indices)
        if indices.ndim == 1:
            indices = indices[:, None]
        return jnp.where(indices < 0, 0, indices).astype(jnp.int32)

    def _resolve_dataset_indices(
        self,
        dataset: xr.Dataset,
        provided: Optional[jnp.ndarray]
    ) -> Optional[jnp.ndarray]:
        if provided is not None:
            return jnp.asarray(provided)

        coord = dataset.coords.get("mask_indices", None)
        if coord is not None:
            return jnp.asarray(coord.values)

        try:
            _, computed = self.compute_mask()
            return computed
        except Exception:
            return None

    def _shift_dataset(
        self,
        dataset: xr.Dataset,
        periods: int,
        mask_indices: Optional[jnp.ndarray],
    ) -> xr.Dataset:
        indices = self._resolve_dataset_indices(dataset, mask_indices)
        if indices is None:
            return dataset.copy()

        time_dims_dataset = self._infer_time_dims(dataset)
        if not time_dims_dataset:
            return dataset.copy()

        prepared_indices = self._prepare_indices(indices)
        asset_cols = prepared_indices.shape[1]

        shifted_vars: Dict[str, xr.DataArray] = {}

        for name, da in dataset.data_vars.items():
            if not np.issubdtype(da.dtype, np.number):
                shifted_vars[name] = da
                continue

            if 'year' not in da.coords:
                # Preserve legacy behaviour for variables without business-day layout
                shifted_vars[name] = da
                continue

            var_asset = da.sizes.get("asset")
            if var_asset is None or var_asset == 0:
                shifted_vars[name] = da
                continue

            if var_asset != asset_cols:
                # Fall back to per-variable mask computation when asset counts diverge
                try:
                    _, var_indices = da.dt.compute_mask()
                    indices_to_use = self._prepare_indices(var_indices)
                except Exception:
                    shifted_vars[name] = da
                    continue
            else:
                indices_to_use = prepared_indices

            shifted_vars[name] = self._shift_dataarray(
                da,
                periods,
                indices_to_use,
                allow_fallback=False,
            )

        return xr.Dataset(shifted_vars, coords=dataset.coords, attrs=dataset.attrs)

    def _shift_dataarray(
        self,
        array: xr.DataArray,
        periods: int,
        mask_indices: Optional[jnp.ndarray],
        allow_fallback: bool,
    ) -> xr.DataArray:
        if not np.issubdtype(array.dtype, np.number):
            return array

        time_dims = self._infer_time_dims(array)
        if not time_dims:
            if allow_fallback and "time" in array.dims:
                return array.shift(time=periods)
            return array.copy()

        resolved_indices = mask_indices
        if resolved_indices is None:
            coord = array.coords.get("mask_indices", None)
            if coord is not None:
                resolved_indices = coord.values

        if resolved_indices is None:
            parent_ds = getattr(self, "_parent_obj", None)
            if isinstance(parent_ds, xr.Dataset):
                coord = parent_ds.coords.get("mask_indices", None)
                if coord is not None:
                    resolved_indices = coord.values

        if resolved_indices is None:
            if allow_fallback:
                _, computed = self.compute_mask()
                resolved_indices = computed
            else:
                return array

        prepared_indices = self._prepare_indices(resolved_indices)
        return self._shift_dataarray_core(array, periods, prepared_indices, time_dims)

    def _shift_dataarray_core(
        self,
        array: xr.DataArray,
        periods: int,
        indices: jnp.ndarray,
        time_dims: Sequence[str],
    ) -> xr.DataArray:
        stacked = array.stack(time_index=tuple(time_dims)).transpose("time_index", ...)
        data = jnp.asarray(stacked.data)

        original_shape = data.shape
        asset_dim = indices.shape[1]

        if original_shape[1] != asset_dim:
            if asset_dim == 0:
                return array.copy()
            remainder = int(np.prod(original_shape[1:]) // asset_dim)
            reshaped = data.reshape(original_shape[0], asset_dim, remainder)
            flat = reshaped.reshape(original_shape[0], asset_dim * remainder)
            expanded_indices = jnp.repeat(indices, remainder, axis=1)
            shifted_flat = TimeSeriesOps.shift(flat, expanded_indices, periods)
            shifted = shifted_flat.reshape(original_shape[0], asset_dim, remainder).reshape(original_shape)
        else:
            shifted = TimeSeriesOps.shift(data, indices, periods)

        result = stacked.copy(data=shifted).unstack("time_index")
        result.attrs.update(array.attrs)
        return result

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
