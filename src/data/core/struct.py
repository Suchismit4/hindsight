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
from enum import Enum

from .computations import Rolling

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
        Creates a Dataset from a table (DataFrame), with fixed-size time dimensions.

        Parameters:
            data (pd.DataFrame): The input data table.
            time_column (str): Name of the time column in the data.
            asset_column (str): Name of the asset column in the data.
            feature_columns (list of str, optional): List of value columns.
            frequency (FrequencyType): The frequency of the data (ANNUAL, MONTHLY, DAILY).

        Returns:
            xr.Dataset: The resulting Dataset with dimensions adjusted based on frequency.
        """
        import time

        print("Entering the conversion phase...")
        overall_start = time.time()

        # We create a copy to avoid modifying the original data
        block_start = time.time()
        data = data.copy()
        block_end = time.time()
        print(f"[timeit] data.copy() took {block_end - block_start:.6f} seconds")

        # We convert the time column to datetime
        block_start = time.time()
        data[time_column] = pd.to_datetime(data[time_column], errors='coerce')
        block_end = time.time()
        print(f"[timeit] pd.to_datetime() took {block_end - block_start:.6f} seconds")

        # We check for invalid datetime values
        block_start = time.time()
        if data[time_column].isnull().any():
            raise ValueError(f"The '{time_column}' column contains invalid datetime values.")
        block_end = time.time()
        print(f"[timeit] Checking invalid datetime values took {block_end - block_start:.6f} seconds")

        # We assign year, month, and day columns based on frequency
        block_start = time.time()
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
        block_end = time.time()
        print(f"[timeit] Assigning year/month/day columns took {block_end - block_start:.6f} seconds")

        # We gather the sorted unique coordinates for each dimension
        block_start = time.time()
        years = np.sort(data['year'].unique())
        months = np.sort(data['month'].unique()) if 'month' in data.columns else np.array([1])
        days = np.sort(data['day'].unique()) if 'day' in data.columns else np.array([1])
        assets = np.sort(data[asset_column].unique())
        block_end = time.time()
        print(f"[timeit] Gathering and sorting unique coords took {block_end - block_start:.6f} seconds")

        # We determine the feature columns if not explicitly given
        block_start = time.time()
        if feature_columns is None:
            exclude_cols = [time_column, 'year', 'month', 'day', asset_column]
            feature_columns = [col for col in data.columns if col not in exclude_cols]
        block_end = time.time()
        print(f"[timeit] Determining feature columns took {block_end - block_start:.6f} seconds")

        # We verify that the feature columns exist in the data
        block_start = time.time()
        missing_features = [fc for fc in feature_columns if fc not in data.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found: {missing_features}")
        block_end = time.time()
        print(f"[timeit] Verifying feature columns existence took {block_end - block_start:.6f} seconds")

        # We define arrays that map coordinate values to their indices
        block_start = time.time()
        y_idx = np.searchsorted(years, data['year'].to_numpy())
        m_idx = np.searchsorted(months, data['month'].to_numpy())
        d_idx = np.searchsorted(days, data['day'].to_numpy())
        a_idx = np.searchsorted(assets, data[asset_column].to_numpy())
        block_end = time.time()
        print(f"[timeit] Defining coordinate to index mappings took {block_end - block_start:.6f} seconds")

        # We now create a "flat" 1D index for each row i
        block_start = time.time()
        n_months = len(months)
        n_days = len(days)
        n_assets = len(assets)
        idx_1d = ((y_idx * n_months + m_idx) * n_days + d_idx) * n_assets + a_idx
        block_end = time.time()
        print(f"[timeit] Creating flat index took {block_end - block_start:.6f} seconds")

        # We define the 4D shape for the data: (years, months, days, assets)
        block_start = time.time()
        shape_data = (len(years), len(months), len(days), len(assets))
        block_end = time.time()
        print(f"[timeit] Defining shape_data took {block_end - block_start:.6f} seconds")

        # We create arrays for each feature
        block_start = time.time()
        feature_arrays = {}
        
        # Separate numeric from non-numeric columns to reduce branching in the loop
        numeric_cols = [fc for fc in feature_columns if data[fc].dtype.kind in 'bifc']
        object_cols = [fc for fc in feature_columns if data[fc].dtype.kind not in 'bifc']
        
        # Create numeric feature arrays
        for fc in numeric_cols:
            arr = np.empty(shape_data, dtype='float64')
            arr.fill(np.nan)
            feature_arrays[fc] = arr

        # Create object (non-numeric) feature arrays
        for fc in object_cols:
            arr = np.empty(shape_data, dtype=object)
            arr.fill(np.nan) 
            feature_arrays[fc] = arr
        
        block_end = time.time()
        print(f"[timeit] Creating feature arrays took {block_end - block_start:.6f} seconds")

        # We store each feature's values in a dict for easy access
        block_start = time.time()
        feature_vals = {}
        for fc in feature_columns:
            feature_vals[fc] = data[fc].to_numpy()
        block_end = time.time()
        print(f"[timeit] Storing feature column values took {block_end - block_start:.6f} seconds")

        # We assign feature values in a vectorized manner
        block_start = time.time()
        for fc in (feature_columns):
            fa_flat = feature_arrays[fc].ravel()
            fa_flat[idx_1d] = feature_vals[fc]
        block_end = time.time()
        print(f"[timeit] Vectorized assignment of feature values took {block_end - block_start:.6f} seconds")

        # We create the time coordinate as a 3D array (years x months x days)
        block_start = time.time()
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
        block_end = time.time()
        print(f"[timeit] Creating time coordinate arrays took {block_end - block_start:.6f} seconds")

        # We build an xarray DataArray for the time coordinate
        block_start = time.time()
        time_coord = xr.DataArray(
            data=time_data,
            coords={
                'year': years,
                'month': months,
                'day': days
            },
            dims=['year', 'month', 'day']
        )
        block_end = time.time()
        print(f"[timeit] Building time DataArray took {block_end - block_start:.6f} seconds")

        # We create a dummy TimeSeriesIndex (or your real custom index)
        block_start = time.time()
        ts_index = TimeSeriesIndex(time_coord)
        block_end = time.time()
        print(f"[timeit] Creating TimeSeriesIndex took {block_end - block_start:.6f} seconds")

        # We build the final xarray.Dataset, attaching all coordinates
        block_start = time.time()
        ds = xr.Dataset(
            coords={
                'year': years,
                'month': months,
                'day': days,
                'asset': assets,
                'time': (['year', 'month', 'day'], time_data),
            }
        )
        block_end = time.time()
        print(f"[timeit] Building initial xarray.Dataset took {block_end - block_start:.6f} seconds")

        # We add each feature's data as a variable in the dataset
        block_start = time.time()
        for fc in feature_columns:
            ds[fc] = xr.DataArray(
                data=feature_arrays[fc],
                dims=['year', 'month', 'day', 'asset']
            )
        block_end = time.time()
        print(f"[timeit] Adding feature variables to Dataset took {block_end - block_start:.6f} seconds")

        # We attach the custom time index to the time coordinate
        block_start = time.time()
        ds.coords['time'].attrs['indexes'] = {'time': ts_index}
        block_end = time.time()
        print(f"[timeit] Attaching custom time index took {block_end - block_start:.6f} seconds")

        print("Conversion to xarray.Dataset is complete.")
        overall_end = time.time()
        print(f"[timeit] Overall function execution took {overall_end - overall_start:.6f} seconds")
        return ds

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
        Converts multi-dimensional data into time-indexed format.

        Returns:
            Union[xr.DataArray, xr.Dataset]: The time-indexed data.
        """
        ds = self._obj
        ds = ds.rename({'time': 'time_3d'})  # We rename the old 3D 'time' coordinate to avoid collision.
        ds_flat = ds.stack(stacked_time=("year", "month", "day"))  # We create a single dimension from (year, month, day).
        ds_flat = ds_flat.rename_dims({"stacked_time": "time"})  # We rename the stacked dimension to 'time'.
        if "stacked_time" in ds_flat.coords:
            ds_flat = ds_flat.rename_vars({"stacked_time": "time"})
        ds_flat = ds_flat.drop_vars("time_3d", errors="ignore")
        ds_flat = ds_flat.assign_coords(time=("time", ds["time_3d"].values.ravel()))  # We flatten the 3D times to 1D.
        return ds_flat
    
    def rolling(self, dim: str, window: int) -> Rolling:
        """
        Creates a Rolling object for applying rolling window operations.

        Parameters:
            dim (str): The dimension over which to apply the rolling window.
            window (int): The size of the rolling window.

        Returns:
            Rolling: An instance of the Rolling class.
        """
        return Rolling(self._obj, dim, window)

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
