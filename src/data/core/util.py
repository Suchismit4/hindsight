# src/data/core/util.py

import numpy as np
import xarray as xr
import xarray_jax as xj
import jax.numpy as jnp
import equinox as eqx
from typing import Union, Dict, List, Optional, Tuple, Any, Callable
import functools
from enum import Enum
import pandas as pd
import jax

# Import the core operations
from src.data.core.operations import TimeSeriesOps

class FrequencyType(Enum):
    """
    Enumeration of supported data frequencies.
    Used for proper alignment in cross-frequency operations.
    """
    DAILY     = 'D'
    WEEKLY    = 'W'
    MONTHLY   = 'M'
    YEARLY    = 'Y'
    ANNUAL    = 'Y'

class TimeSeriesIndex:
    """
    A class to map timestamps to indices in a multi-dimensional time coordinate.

    Attributes:
        time_coord (xr.DataArray): The time coordinate DataArray.
        shape (Tuple[int, ...]): The shape of the time coordinate array.
        time_to_index (pd.Series): A mapping from timestamps to flat indices.
    """
    
    def __init__(self, time_coord: xr.DataArray):
        self.time_coord = time_coord
        self.shape = time_coord.shape # (Y, Q, M, D, I), N, J

        # Flatten the time coordinate values and create a mapping to flat indices
        times = pd.Series(time_coord.values.ravel())
        valid_times = times[~pd.isnull(times)]
        
        self.time_to_index = pd.Series(
            np.arange(len(times))[~pd.isnull(times)],
            index=valid_times
        )

    def sel(self, labels, method=None, tolerance=None):
        """
        Selects indices corresponding to the given labels.

        Parameters:
            labels: The timestamp(s) to select.
            method: Method for selection (not used here).
            tolerance: Tolerance for inexact matches (not used here).

        Returns:
            dict: A dictionary mapping dimension names to indices.
        """
        if isinstance(labels, pd.DatetimeIndex):
            labels_array = labels.to_numpy()
        elif isinstance(labels, (list, np.ndarray)):
            labels_array = pd.to_datetime(labels).to_numpy()
        else:
            labels_array = pd.to_datetime([labels]).to_numpy()
        
        # Initialize a list to collect flat indices
        flat_indices = []
        for label in labels_array:
            try:
                locs = self.time_to_index.index.get_loc(label)
                if isinstance(locs, slice):
                    indices = self.time_to_index.iloc[locs].values
                elif isinstance(locs, np.ndarray):
                    indices = self.time_to_index.iloc[locs].values
                elif isinstance(locs, int):
                    indices = [self.time_to_index.iloc[locs]]
                else:
                    raise KeyError(f"Date {label} not found in index")
                flat_indices.extend(indices)
            except KeyError:
                raise KeyError(f"Date {label} not found in index")
        
        if not flat_indices:
            raise KeyError(f"Dates {labels_array} not found in index")
        
        flat_indices = np.array(flat_indices)
        multi_indices = np.unravel_index(flat_indices.astype(int), self.shape)
        dim_names = self.time_coord.dims

        return dict(zip(dim_names, multi_indices))

class Loader:

    @classmethod
    def from_table(
        cls,
        data: pd.DataFrame,
        time_column: str = 'time',
        asset_column: str = 'asset',
        feature_columns: Optional[List[str]] = None,
        frequency: FrequencyType = FrequencyType.DAILY,
    ) -> xr.Dataset:
        """
        Creates an xarray Dataset from a table (Pandas DataFrame), with fixed-size time dimensions.
        This function performs various transformations and coordinate assignments to build a 4D structure
        (years, months, days, assets). It ensures that each dimension is set according to the specified
        frequency, and then loads feature values into the corresponding positions in a new xarray Dataset.

        Parameters:
            data : pd.DataFrame
                The input data table containing time, asset, and feature columns.
            time_column : str
                Name of the time column in the data.
            asset_column : str
                Name of the asset column in the data.
            feature_columns : list of str, optional
                List of feature columns whose values should be placed in the xarray Dataset.
                If None, all columns except time_column, year, month, day, and asset_column will be treated as features.
            frequency : FrequencyType
                The frequency of the data. Must be one of YEARLY, MONTHLY, or DAILY.

        Returns:
            xr.Dataset
                The resulting Dataset with the dimensions (year, month, day, asset) and the given
                features as variables. It also attaches a time coordinate computed from the year, month,
                and day values, and includes a custom index for time management.
        """
        
        # We create a copy of the original data to avoid any unintended modifications
        data = data.copy()
        
        # We convert the specified time column into datetime format
        # This is necessary to ensure that we can extract year, month, and day correctly.
        data[time_column] = pd.to_datetime(data[time_column], errors='coerce')
        
        # If any rows have invalid datetime values, we raise an error, as it would break subsequent operations
        if data[time_column].isnull().any():
            raise ValueError(f"The '{time_column}' column contains invalid datetime values.")
        
        # We add year, month, and day columns to the DataFrame depending on the frequency
        # For YEARLY frequency, month and day are set to 1.
        # For MONTHLY frequency, only day is set to 1.
        # For DAILY frequency, we keep actual day values.
        if frequency == FrequencyType.YEARLY:
            data['year'] = data[time_column].dt.year
            data['month'] = 1
            data['day'] = 1
        elif frequency == FrequencyType.MONTHLY:
            data['year'] = data[time_column].dt.year
            data['month'] = data[time_column].dt.month
            data['day'] = 1
        elif frequency == FrequencyType.DAILY:
            data['year'] = data[time_column].dt.year
            data['month'] = data[time_column].dt.month
            data['day'] = data[time_column].dt.day
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        # Gather unique values for each dimension (year, month, day, asset)
        # The data is sorted so we can properly create coordinate arrays in ascending order
        years = np.sort(data['year'].unique())
        months = np.sort(data['month'].unique()) if 'month' in data.columns else np.array([1])
        days = np.sort(data['day'].unique()) if 'day' in data.columns else np.array([1])
        assets = np.sort(data[asset_column].unique())
        
        # Determine feature columns if not explicitly provided
        # We exclude the time, year, month, day, and asset columns from feature consideration.
        if feature_columns is None:
            exclude_cols = [time_column, 'year', 'month', 'day', asset_column]
            feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Ensure that all specified feature columns are actually present in the DataFrame
        missing_features = [fc for fc in feature_columns if fc not in data.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found: {missing_features}")
        
        # Define arrays to map coordinate values to their indices.
        # For each row in 'data', we locate the positions of year, month, day, and asset in their respective sorted arrays.
        y_idx = np.searchsorted(years, data['year'].to_numpy())
        m_idx = np.searchsorted(months, data['month'].to_numpy())
        d_idx = np.searchsorted(days, data['day'].to_numpy())
        a_idx = np.searchsorted(assets, data[asset_column].to_numpy())
        
        # Create a flat 1D index that represents the combination of (year, month, day, asset).
        # We compute this by combining indices in a single integer using multiplication and addition, ensuring each dimension is accounted for.
        n_months = len(months)
        n_days = len(days)
        n_assets = len(assets)
        idx_1d = ((y_idx * n_months + m_idx) * n_days + d_idx) * n_assets + a_idx
        
        # Define the shape of the final 4D data arrays we will store. This is based on the length of each dimension.
        shape_data = (len(years), len(months), len(days), len(assets))
        
        # Separate numeric and object feature columns to build arrays appropriately.
        numeric_cols = [fc for fc in feature_columns if data[fc].dtype.kind in 'bifc']
        object_cols = [fc for fc in feature_columns if data[fc].dtype.kind not in 'bifc']
        
        # Create numeric feature arrays, initialized with np.nan.
        # We choose float64 for numeric columns so that NaN can be assigned without issues.
        feature_arrays = {}
        for fc in numeric_cols:
            arr = np.empty(shape_data, dtype='float64')
            arr.fill(np.nan)
            feature_arrays[fc] = arr
        
        # Create object feature arrays for non-numeric columns, also initialized with np.nan.
        # This allows for storing arbitrary object types.
        for fc in object_cols:
            arr = np.empty(shape_data, dtype=object)
            arr.fill(np.nan)
            feature_arrays[fc] = arr
        
        # Store the actual values of each feature column in a dictionary for quick access.
        # This way, we can assign them in a vectorized manner later.
        feature_vals = {}
        for fc in feature_columns:
            feature_vals[fc] = data[fc].to_numpy()
        
        # Assign the feature values into their corresponding positions in the created arrays.
        # We leverage idx_1d to place each value where it belongs in the 4D structure.
        for fc in feature_columns:
            fa_flat = feature_arrays[fc].ravel()
            fa_flat[idx_1d] = feature_vals[fc]
        
        # Here we create the actual time coordinate as a 3D array (years x months x days).
        # We build a meshgrid for year, month, day, then convert each position to a valid datetime.
        # This will serve as the actual "time" dimension in the final dataset.
        yr_mesh, mo_mesh, dd_mesh = np.meshgrid(years, months, days, indexing='ij')
        flat_years = yr_mesh.ravel()
        flat_months = mo_mesh.ravel()
        flat_days = dd_mesh.ravel()
        time_index_flat = pd.to_datetime(
            {
                'year': flat_years,
                'month': flat_months,
                'day': flat_days
            },
            errors='coerce'
        )
        time_data = time_index_flat.values.reshape((len(years), len(months), len(days)))
        
        # Build an xarray DataArray for the time coordinate, setting up the coordinate dims as year, month, day.
        time_coord = xr.DataArray(
            data=time_data,
            coords={
                'year': years,
                'month': months,
                'day': days
            },
            dims=['year', 'month', 'day']
        )
        
        # Create a custom time index (TimeSeriesIndex) object or a placeholder for more advanced time handling.
        ts_index = TimeSeriesIndex(time_coord)
        
        # Temporarily, we build an xarray Dataset, attaching all the coordinates we need.
        ds = xr.Dataset(
            coords={
                'year': years,
                'month': months,
                'day': days,
                'asset': assets,
                'time': (['year', 'month', 'day'], time_data),
                'mask': ('time', np.zeros(len(time_index_flat.values))),
                'mask_indices': ('time', np.zeros(len(time_index_flat.values)))
            }
        )

        # Add feature variables to the Dataset
        for fc in feature_columns:
            arr = feature_arrays[fc]
            if np.issubdtype(arr.dtype, np.number):
                ds[fc] = xr.DataArray(
                    data=jnp.array(arr),
                    dims=['year', 'month', 'day', 'asset']
                )
            else:
                ds[fc] = xr.DataArray(
                    data=arr,
                    dims=['year', 'month', 'day', 'asset']
                )

        
        # Create a stacked DataArray for mask and indices
        first_var = list(ds.data_vars)[0]
        stacked_obj = ds[first_var].stack(time_index=("year", "month", "day"))
        
        # Extract date tuples for mask creation
        time_tuples = stacked_obj.coords["time_index"].values  # Shape: (T, 3)
        _dates = np.array([[*date] for date in time_tuples])  # Convert to list of tuples
        
        # Create business day mask and indices
        
        dates = pd.to_datetime(
            {
                'year': _dates[:, 0],
                'month': _dates[:, 1],
                'day': _dates[:, 2]
            }, 
            errors='coerce'
        )
        
        # Create mask and indices
        is_valid_date = ~dates.isna()
        is_business_day = dates.dt.dayofweek < 5  # Monday=0 to Friday=4
        mask = is_valid_date & is_business_day

        mask = mask.to_numpy(dtype=bool)
                
        # For positions where mask is True, store the original index.
        valid_positions = np.flatnonzero(mask)  # positions of valid business days
        valid_positions_sorted = np.sort(valid_positions)
        
        indices = -1 * np.ones(len(_dates), dtype=int)
        num_valid = len(valid_positions_sorted)
        indices[:num_valid] = valid_positions_sorted

        # Debug to test for values on NaNs        
        # print(indices[10])
        # print(_dates[indices[10]])
        # print(ds.dt.to_time_indexed()['close'].values[0, indices[:100]])
 
        ds.coords['mask'] = ('time', mask)                # Shape: (T,)
        ds.coords['mask_indices'] = ('time', indices) 
                
        # We have successfully built the Dataset. At this point, the structure
        # is fully set up with time, asset, and feature dimensions and coordinates.
        return ds

class Rolling(eqx.Module):
    """
    Custom Rolling class to apply rolling window operations using JAX.
    """
    
    obj:    Union[xr.DataArray, xr.Dataset]
    dim:    str
    window: int
    
    mask: jnp.ndarray 
    indices: jnp.ndarray
    
    def __init__(self, 
                 obj: Union[xr.DataArray, xr.Dataset], 
                 dim: str, 
                 window: int,
                 mask: jnp.ndarray, 
                 indices: jnp.ndarray):
        """
        Initializes the Rolling object.

        Args:
            obj (Union[xr.DataArray, xr.Dataset]): The xarray object to apply rolling on.
            dim (str): The dimension over which to apply the rolling window.
            window (int): The size of the rolling window.
        """
        self.obj = obj
        self.dim = dim
        self.window = window

        self.mask = mask
        self.indices = jnp.where(indices == -1, 0, indices).astype(jnp.int32)

    @eqx.filter_jit
    def reduce(
        self, 
        func: Callable[[int, Any, jnp.ndarray, int], Tuple[jnp.ndarray, Any]],
        overlap_factor: Optional[float] = None,
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Applies the rolling function using the provided callable.
        
        Args:
            func: A JIT-compatible function with signature
                    (i, carry, block, window_size) > (value, new_carry).
            overlap_factor: Optional overlap factor for block-based rolling.

        Returns:
            A new xarray object with the same structure as self.obj
            but with rolled computations applied.
        """
        
        if isinstance(self.obj, xr.Dataset) and len(self.obj.data_vars) > 0:
            # pick the first variable
            a_var = list(self.obj.data_vars)[0]
            var_dims = self.obj[a_var].dims
        elif isinstance(self.obj, xr.DataArray):
            var_dims = self.obj.dims
        else:
            var_dims = ()

        is_multidim_time = set(["year", "month", "day"]).issubset(var_dims)
        user_dim_exists = (self.dim in var_dims)
        wants_time = (self.dim == "time")
        
        if not user_dim_exists and not (wants_time and is_multidim_time):
            # dimension is not in var_dims, and we are not in the "flatten time" case
            raise ValueError(
                f"Dimension '{self.dim}' not found in the xarray object. "
                "Nor is it a recognized multi-dim time (year, month, day)."
            )
        
        if isinstance(self.obj, xr.Dataset):
            rolled_data = { }
            
            # For each data array, we will reduce it with the fn
            for var_name, da in self.obj.data_vars.items():
                # Recursively call Rolling on the DataArray
                # TODO: Find a better strategy to avoid non-linear and untrackable call maps.
                rolled_da = Rolling(da, self.dim, self.window, self.mask, self.indices).reduce(
                    func=func,
                    overlap_factor=overlap_factor
                )
                rolled_data[f"{func.__name__}_{var_name}"] = rolled_da
                
            # Rebuild a new Dataset
            new_ds = xr.Dataset(
                data_vars=rolled_data,
                coords=self.obj.coords,
                attrs=self.obj.attrs
            )

            # Attached Time based indexing object.
            if "time" in new_ds.coords and "time" in self.obj.coords:
                old_time_attrs = self.obj.coords["time"].attrs
                new_ds.coords["time"].attrs.update(old_time_attrs)
                
            return new_ds
            
        elif isinstance(self.obj, xr.DataArray):
            
            if wants_time and is_multidim_time:
                stacked_obj = self.obj.stack(time_index=("year", "month", "day"))
                stacked_obj = stacked_obj.transpose("time_index", ...)

                # Extract data as JAX array and expand dims to (T, assets, 1)
                data = jnp.asarray(stacked_obj.data)[..., None]  # Shape: (T, assets, 1)
            
                # Extract valid data based on mask_indices 
                valid_data = data[self.indices, ...]  # Shape: (T, assets, 1)
                
                
                # Perform rolling
                rolled_result = TimeSeriesOps.u_roll(
                    data=valid_data,
                    window_size=self.window,
                    func=func,
                    overlap_factor=overlap_factor
                )
                
                T_full = data.shape[0]
                
                # Initialize a full array with NaNs
                rolled_full = jnp.full((T_full, *rolled_result.shape[1:]), jnp.nan, dtype=rolled_result.dtype)
                
                # Insert rolled results back into their original positions
                rolled_full = rolled_full.at[self.indices].set(rolled_result)
                
                # Remove the extra dimension added earlier
                rolled_full = rolled_full[..., 0]  # Shape: (T_full, assets, 1)
                # jax.debug.breakpoint()
    
                # Reconstruct the DataArray with rolled data
                rolled_da = stacked_obj.copy(data=rolled_full)
                
                # Unstack back to original multi-dimensional time
                unstacked_da = rolled_da.unstack("time_index")
    
                return unstacked_da

            else: 
                raise ValueError('Asset cross-sectional rolling not supported yet.')
        else:
            raise TypeError("Unsupported xarray object type.")
    
            
