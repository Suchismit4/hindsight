"""
Core utility classes and functions for time series data handling in Hindsight.

This module provides foundational utilities for working with financial time series data, including:

1. FrequencyType: Enumeration of supported data frequencies (daily, weekly, monthly, yearly)
2. TimeSeriesIndex: Maps timestamps to multi-dimensional indices for efficient access
3. Loader: Functions for loading and transforming financial data into xarray datasets
4. Rolling: Custom implementation of rolling window operations for time series data

These utilities form the backbone of Hindsight's data handling capabilities,
enabling efficient storage, access, and manipulation of financial panel data.
"""

import os
import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Union, Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import pandas as pd
import pyreadstat
import operator # For JAX assertions

# Import the core operations
from src.data.core.operations import TimeSeriesOps

def prepare_for_jit(dataset: xr.Dataset) -> Tuple[xr.Dataset, Dict[str, xr.DataArray]]:
    """
    Separates non-numeric data variables from an xarray Dataset to make it JIT-compatible.

    This function identifies DataArrays with non-numeric dtypes (like object or string)
    and removes them, returning a new Dataset containing only numeric data and a
    context dictionary holding the removed variables.

    Args:
        dataset: The input xarray Dataset potentially containing non-numeric DataArrays.

    Returns:
        A tuple containing:
            - jit_ready_dataset: A new Dataset containing only numeric DataArrays.
            - context: A dictionary holding the removed non-numeric DataArrays, keyed by name.
    """
    non_numeric_names = [
        name for name, da in dataset.data_vars.items()
        if not np.issubdtype(da.dtype, np.number)
    ]

    if not non_numeric_names:
        # If no non-numeric variables, return the original dataset and empty context
        return dataset, {}

    # Store the non-numeric variables in the context dictionary
    context = {name: dataset[name] for name in non_numeric_names}

    # Create the JIT-ready dataset by dropping the non-numeric variables
    jit_ready_dataset = dataset.drop_vars(non_numeric_names)

    return jit_ready_dataset, context

def restore_from_jit(processed_dataset: xr.Dataset, context: Dict[str, xr.DataArray]) -> xr.Dataset:
    """
    Restores non-numeric data variables back into a processed xarray Dataset.

    Args:
        processed_dataset: The Dataset after JIT computations (should contain numeric vars).
        context: The dictionary containing the non-numeric DataArrays removed by prepare_for_jit.

    Returns:
        A new Dataset with the non-numeric data variables merged back in.
    """
    # Start with the processed dataset
    restored_dataset = processed_dataset.copy()

    # Merge the non-numeric variables back from the context
    restored_dataset = restored_dataset.update(context)
    # update returns None, need to use merge or assign
    # restored_dataset = processed_dataset.merge(xr.Dataset(context)) # Alternative using merge

    # A more direct way using assign might be better if coords don't clash
    restored_dataset = processed_dataset.assign(**context)


    return restored_dataset

class FrequencyType(Enum):
    """
    Enumeration of supported data frequencies.
    
    Used for proper alignment and handling in cross-frequency operations and
    for correct time dimension creation in datasets.
    
    Attributes:
        DAILY: Daily frequency ('D')
        WEEKLY: Weekly frequency ('W')
        MONTHLY: Monthly frequency ('M')
        YEARLY/ANNUAL: Annual frequency ('Y')
    """
    DAILY     = 'D'
    WEEKLY    = 'W'
    MONTHLY   = 'M'
    YEARLY    = 'Y'
    ANNUAL    = 'Y'

class TimeSeriesIndex:
    """
    A class to map timestamps to indices in a multi-dimensional time coordinate.
    
    This provides efficient access to time-indexed data by converting between
    timestamps and their corresponding indices in the dataset, enabling fast
    time-based selection and alignment.

    Attributes:
        time_coord (xr.DataArray): The time coordinate DataArray.
        shape (Tuple[int, ...]): The shape of the time coordinate array.
        time_to_index (pd.Series): A mapping from timestamps to flat indices.
    """
    
    def __init__(self, time_coord: xr.DataArray):
        """
        Initialize a TimeSeriesIndex from a time coordinate DataArray.
        
        Parameters:
            time_coord (xr.DataArray): The time coordinate array, typically 
                with dimensions (year, month, day).
        """
        self.time_coord = time_coord
       
        # Use np.ravel with C order to obtain a flat view of the time coordinate.
        # This ensures that the flattening order is consistent with np.unravel_index later.
        self._flat_times = time_coord.values.ravel(order="C")
        self.shape = time_coord.shape  # the original shape
        
        # Create a Series from the flattened times.
        times = pd.Series(self._flat_times)
        valid_times = times[~pd.isnull(times)]
        self.time_to_index = pd.Series(
            np.arange(len(self._flat_times))[~pd.isnull(times)],
            index=valid_times
        )


    def sel(self, labels, method=None, tolerance=None):
        """
        Selects indices corresponding to the given time labels.

        This method translates time-based selections into multi-dimensional indices
        that can be used with .isel() for efficient data access.

        Parameters:
            labels: The timestamp(s) to select. This can be:
                    - A single timestamp (string or datetime-like)
                    - A list or array of timestamps
                    - A pandas DatetimeIndex
                    - A slice with both start and stop defined (e.g. slice('2020-01-01', '2020-12-31'))
            method: Method for selection (not used here).
            tolerance: Tolerance for inexact matches (not used here).

        Returns:
            dict: A dictionary mapping dimension names to multi-dimensional indices.

        Raises:
            ValueError: If a slice without start and stop is provided.
            KeyError: If specified timestamps are not found in the index.

        Note:
            This implementation assumes that the flattened time coordinate (derived via ravel(order="C"))
            corresponds to the same ordering that np.unravel_index will use.
        """
        # Handle the case where labels is a slice.
        if isinstance(labels, slice):
            if labels.start is None or labels.stop is None:
                raise ValueError("Slice must have both start and stop defined.")
            start = pd.to_datetime(labels.start)
            stop = pd.to_datetime(labels.stop)
            # Use slice_locs on the Series index (which is assumed sorted in C order)
            start_loc, stop_loc = self.time_to_index.index.slice_locs(start, stop)
            flat_indices = self.time_to_index.iloc[start_loc:stop_loc].values
        else:
            # Convert the input into an array of timestamps.
            if isinstance(labels, pd.DatetimeIndex):
                labels_array = labels.to_numpy()
            elif isinstance(labels, (list, np.ndarray)):
                labels_array = pd.to_datetime(labels).to_numpy()
            else:
                labels_array = pd.to_datetime([labels]).to_numpy()

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

        # Convert the flat indices back to multi-dimensional indices using the original shape.
        multi_indices = np.unravel_index(flat_indices.astype(int), self.shape)
        dim_names = self.time_coord.dims
        
        return dict(zip(dim_names, multi_indices))

class Loader:
    """
    Utility class for loading and transforming financial data into xarray datasets.
    
    This class provides methods to load data from various sources (such as SAS datasets)
    and convert them into the standardized xarray Dataset format used throughout Hindsight.
    It handles date conversions, dimension creation, and proper data organization.
    """
    
    DEFAULT_PATHS = {
        "msenames": "/wrds/crsp/sasdata/a_stock/msenames.sas7bdat",
        "delistings": "/wrds/crsp/sasdata/a_stock/msedelist.sas7bdat"
    }
    
    @classmethod
    def load_external_proc_file(cls, src_or_name: str, identifier: str, rename: Optional[List[List[str]]] = None) -> Union[xr.Dataset, pd.DataFrame]:
        """
        Load an external SAS file and convert it to an xarray.Dataset.
        
        Parameters:
            src_or_name (str): Either a file path or a known source name from DEFAULT_PATHS.
            identifier (str): Column name to use as the identifier (will be renamed to "identifier").
            rename (Optional[List[List[str]]]): Optional list of [source, destination] column name pairs to rename.
            
        Returns:
            Union[xr.Dataset, pd.DataFrame]: Data as an xarray Dataset if time column is found, otherwise as DataFrame.
            
        Raises:
            ValueError: If source not found or identifier column not present.
        """
        
        file_path = None

        # If src_or_name is a valid file path, use it; otherwise, try a default mapping.
        if os.path.exists(src_or_name):
            file_path = src_or_name
        elif src_or_name in cls.DEFAULT_PATHS:
            file_path = cls.DEFAULT_PATHS[src_or_name]
        else:
            raise ValueError(f"Unknown external source: {src_or_name}")

        if not os.path.exists(file_path):
            raise ValueError(f"File for external source '{src_or_name}' not found at {file_path}")

        df, _ = pyreadstat.read_file_multiprocessing(
            pyreadstat.read_sas7bdat,
            file_path,
            num_processes=16
        )
        
        # Normalize columns to lowercase.
        df.columns = df.columns.str.lower()

        # Apply any custom renaming if provided.
        if rename:
            for mapping in rename:
                if len(mapping) != 2:
                    raise ValueError("Each rename mapping must have exactly two elements: [source, destination].")
                src_col, dest_col = mapping
                src_col = src_col.lower()
                dest_col = dest_col.lower()
                if src_col in df.columns:
                    df.rename(columns={src_col: dest_col}, inplace=True)

        # Ensure that the designated identifier column is renamed to "identifier".
        identifier = identifier.lower()
        if identifier in df.columns:
            df.rename(columns={identifier: "identifier"}, inplace=True)
        else:
            raise ValueError(f"Identifier '{identifier}' was not found in external file columns: {df.columns.tolist()}")
        
        # Determine the time column: if 'time' exists (possibly via renaming), use it; else fall back on 'date'.
        time_column = "time" if "time" in df.columns else ("date" if "date" in df.columns else None)
        
        # Convert the DataFrame to an xarray.Dataset using Loader.from_table.
        if time_column:
            df[time_column] = cls.convert_sas_date(df[time_column])
            ds = Loader.from_table(df, time_column=time_column, asset_column="identifier")
        else:
            return df
        return ds

    @staticmethod
    def convert_sas_date(sas_date_col: pd.Series, epoch: str = '1960-01-01') -> pd.Series:
        """
        Convert a numeric SAS date column to a proper Pandas datetime.

        Parameters:
            sas_date_col (pd.Series): Column of SAS date ints.
            epoch (str): Base epoch for SAS (default '1960-01-01').

        Returns:
            pd.Series: Date column in datetime format.
        """
        sas_epoch = pd.to_datetime(epoch)
        return sas_epoch + pd.to_timedelta(sas_date_col.astype(int), unit='D')
    
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
        
        This function transforms tabular data into a multi-dimensional xarray structure with
        year, month, day, and asset dimensions. It handles frequency-specific time dimension
        creation, coordinate assignments, and proper indexing for time series operations.

        Parameters:
            data (pd.DataFrame): Input data table with time, asset, and feature columns.
            time_column (str): Name of the time column in the data.
            asset_column (str): Name of the asset column in the data.
            feature_columns (Optional[List[str]]): List of feature columns. If None, all columns 
                except time_column, year, month, day, and asset_column will be treated as features.
            frequency (FrequencyType): The frequency of the data (YEARLY, MONTHLY, or DAILY).

        Returns:
            xr.Dataset: Dataset with dimensions (year, month, day, asset) and the given
                features as variables, with time coordinate and business day masks.
                
        Raises:
            ValueError: If frequency is not supported or required columns are missing.
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

        # Add feature variables to the Dataset (initial creation without mask/indices)
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

        # Add the TimeSeriesIndexing
        ds.coords['time'].attrs['indexes'] = {'time': ts_index}
        
        # We are going to select a known asset and variable to create the mask and indices
        # This is a temporary fix to avoid the problem of true NaNs and filling with 0s
        # discuss with prof.
        asset = 14593
        var = 'prc'
        
        stacked_obj = None
        
        for var_name, da in ds.data_vars.items():
            if np.issubdtype(da.dtype, np.number):
                stacked_obj = da.sel(asset=asset).stack(time_index=("year", "month", "day"))
                break
        
        if stacked_obj is None:
            raise ValueError("No numeric variable found in the dataset. Failed to create mask and indices.")
        
        # Create a stacked DataArray for mask and indices
        # first_var = 'ret'
        # stacked_obj = ds[first_var].stack(time_index=("year", "month", "day"))
        
        # Extract date tuples for mask creation
        time_tuples = stacked_obj.coords["time_index"].values  # Shape: (T, 3)
        _dates = np.array([[*date] for date in time_tuples])  # Convert to list of tuples
        
        # TODO: A problem is that there are true NaNs and filling with 0s doesnt work for a
        # a certain number of computations and produces errors. A temp fix to mask all NaNs
        # discuss with prof.
        # Create business day mask and indices
        
        # dates = pd.to_datetime(
        #     {
        #         'year': _dates[:, 0],
        #         'month': _dates[:, 1],
        #         'day': _dates[:, 2]
        #     }, 
        #     errors='coerce'
        # )
        
        # # Create mask and indices
        # is_valid_date = ~dates.isna()
        # is_business_day = dates.dt.dayofweek < 5  # Monday=0 to Friday=4
        # mask = is_valid_date & is_business_day

        # mask = mask.to_numpy(dtype=bool)
                
        # # For positions where mask is True, store the original index.
        # valid_positions = np.flatnonzero(mask)  # positions of valid business days
        # valid_positions_sorted = np.sort(valid_positions)
        
        # indices = -1 * np.ones(len(_dates), dtype=int)
        # num_valid = len(valid_positions_sorted)
        # indices[:num_valid] = valid_positions_sorted

        # # Reshape mask and indices to match the time dimensions (year, month, day)
        # time_shape = (len(years), len(months), len(days))
        # # mask_3d = mask.reshape(time_shape)
        # # indices_3d = indices.reshape(time_shape)

        data_arr = stacked_obj.values # (T, assets)
        # print(data_arr.shape)
        mask = ~(np.isnan(data_arr)) # (T,)
        # print(mask.shape)
        # raise Exception("Stop here")

        valid_pos = np.flatnonzero(mask)
        T = mask.shape[0]
        indices = -1 * np.ones(T, dtype=int)
        indices[:valid_pos.shape[0]] = valid_pos

        # Only assign mask and indices at the Dataset level with the flattened 'time' coordinate
        ds = ds.assign_coords({
             'mask': ('time', mask),                # Shape: (T,)
             'mask_indices': ('time', indices)      # Shape: (T,)
        })
                
        # We have successfully built the Dataset. At this point, the structure
        # is fully set up with time, asset, and feature dimensions and coordinates.
        return ds

    @classmethod
    def load_simulated_data(
        cls,
        num_assets: int,
        num_timesteps: int,
        num_vars: int,
        freq: FrequencyType = FrequencyType.DAILY,
        start_date: str = '2000-01-01'
    ) -> xr.Dataset:
        """
        Generates a simulated xarray Dataset with numeric data.

        Args:
            num_assets: Number of assets to simulate.
            num_timesteps: Number of time steps to simulate.
            num_vars: Number of data variables (features) to simulate.
            freq: Frequency of the time series data (default: DAILY).
            start_date: Start date for the time series (default: '2000-01-01').

        Returns:
            An xarray Dataset containing simulated numeric data.
        """
        # Generate asset identifiers
        assets = [f"asset_{i+1}" for i in range(num_assets)]

        # Generate date range based on frequency
        # Note: pd.date_range uses calendar days/months/years.
        # For true business day frequency, more complex logic is needed,
        # but for simulation, this is often sufficient.
        time_index = pd.date_range(start=start_date, periods=num_timesteps, freq=freq.value)

        # Create MultiIndex
        multi_index = pd.MultiIndex.from_product([assets, time_index], names=['asset', 'time'])

        # Create DataFrame
        df = pd.DataFrame(index=multi_index)

        # Populate with random numeric data
        for i in range(num_vars):
            var_name = f"var_{i+1}"
            # Use standard normal distribution for simulation
            df[var_name] = np.random.randn(len(multi_index))

        # Reset index to make 'asset' and 'time' columns
        df = df.reset_index()

        # Call from_table to create the Dataset
        simulated_ds = cls.from_table(
            data=df,
            time_column='time',
            asset_column='asset',
            frequency=freq
        )

        return simulated_ds


# Type alias for clarity (optional)
DateLikeNp = Union[np.datetime64, np.ndarray] # Expecting np.datetime64[D]

class Rolling(eqx.Module):
    """
    Custom Rolling class to apply rolling window operations using JAX.
    
    This class enables efficient rolling window calculations on time series data,
    with proper handling of business days, weekends, and holidays. It uses JAX for
    accelerated computation of window operations.
    
    Attributes:
        obj (Union[xr.DataArray, xr.Dataset]): The xarray object to apply rolling on.
        dim (str): The dimension over which to apply the rolling window.
        window (int): The size of the rolling window.
        mask (jnp.ndarray): Boolean mask indicating valid business days.
        indices (jnp.ndarray): Indices mapping business days to positions.
    """
    
    obj:    Union[xr.DataArray, xr.Dataset]
    dim:    str = eqx.field(static=True)
    window: int = eqx.field(static=True)
    
    # JAX arrays are leaves by default, no need to mark mask/indices unless they are static data (which they shouldn't be)
    mask: jnp.ndarray 
    indices: jnp.ndarray
    
    def __init__(self, 
                 obj: Union[xr.DataArray, xr.Dataset], 
                 dim: str, 
                 window: int,
                 mask: Optional[jnp.ndarray] = None, 
                 indices: Optional[jnp.ndarray] = None):
        """
        Initializes the Rolling object.

        Parameters:
            obj (Union[xr.DataArray, xr.Dataset]): The xarray object to apply rolling on.
            dim (str): The dimension over which to apply the rolling window.
            window (int): The size of the rolling window.
            mask (Optional[jnp.ndarray]): Boolean mask indicating valid business days.
                Required when obj is a DataArray, optional for Dataset.
            indices (Optional[jnp.ndarray]): Indices mapping business days to positions.
                Required when obj is a DataArray, optional for Dataset.
        """
        self.obj = obj
        self.dim = dim
        self.window = window

        # Different handling for Dataset vs DataArray
        if isinstance(obj, xr.Dataset):
            # For Datasets, we can auto-extract mask and indices from coordinates if not provided
            if mask is None and 'mask' in obj.coords:
                mask = obj.coords['mask'].values
            if indices is None and 'mask_indices' in obj.coords:
                indices = obj.coords['mask_indices'].values
                
            # Ensure we have valid mask and indices at this point
            if mask is None or indices is None:
                raise ValueError("Dataset rolling operation requires 'mask' and 'mask_indices' coordinates "
                                "or explicit mask/indices parameters.")
        elif isinstance(obj, xr.DataArray):
            # For DataArrays, mask and indices MUST be provided explicitly, as we don't 
            # try to attach these coordinates to individual DataArrays anymore 
            if mask is None or indices is None:
                raise ValueError("DataArray rolling operation requires explicit mask and indices parameters. "
                                "These should typically come from the parent Dataset's coordinates.")
        
        # Store as JAX arrays with type conversion
        self.mask = jnp.asarray(mask, dtype=jnp.bool_)
        self.indices = jnp.where(
            jnp.asarray(indices) == -1, 
            0, 
            jnp.asarray(indices)
        ).astype(jnp.int32)
        
    def reduce(self, 
               func: Callable[[int, Any, jnp.ndarray, int], Tuple[jnp.ndarray, Any]],
               overlap_factor: Optional[float] = None,
               **func_kwargs
              ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Apply a rolling window reduction function to the data.
        
        This method handles both Dataset and DataArray objects, applying the rolling
        operation to each numeric variable while preserving non-numeric variables.

        Parameters:
            func (Callable): The reduction function to apply to each window.
                The function should take (window_size, initial_state, data, axis) and
                return (result, final_state).
            overlap_factor (Optional[float]): If specified, requires at least this
                fraction of the window to contain valid data.
            **func_kwargs: Additional keyword arguments to pass to the function.

        Returns:
            Union[xr.DataArray, xr.Dataset]: The result of applying the rolling operation.
            
        Raises:
            TypeError: If the xarray object type is not supported.
        """
        if isinstance(self.obj, xr.Dataset):
            rolled_data = {}
            for var_name, da in self.obj.data_vars.items():
                if np.issubdtype(da.dtype, np.number):
                    # Pass the mask and indices explicitly when creating Rolling for a DataArray
                    # This ensures the DataArray has access to these values even though they're
                    # only stored at the Dataset level
                    rolled_data[var_name] = Rolling(
                        da, 
                        self.dim, 
                        self.window, 
                        mask=self.mask,  # Explicitly pass mask from the Dataset
                        indices=self.indices  # Explicitly pass indices from the Dataset
                    ).reduce(func, overlap_factor=overlap_factor, **func_kwargs)
                else:
                    rolled_data[var_name] = da
            return xr.Dataset(rolled_data, coords=self.obj.coords, attrs=self.obj.attrs)
        elif isinstance(self.obj, xr.DataArray):
            return self._reduce_dataarray(func, overlap_factor, **func_kwargs)
        else:
            raise TypeError("Unsupported xarray object type.")
    
    def _reduce_dataarray(self, 
                          func: Callable[[int, Any, jnp.ndarray, int], Tuple[jnp.ndarray, Any]],
                          overlap_factor: Optional[float] = None,
                          **func_kwargs
                         ) -> xr.DataArray:
        """
        JIT-compiled implementation of rolling reduction for DataArrays.
        
        This method handles the actual JAX-based computation, converting data to
        JAX arrays, applying the rolling operation, and reconstructing the DataArray.

        Parameters:
            func (Callable): The reduction function to apply to each window.
            overlap_factor (Optional[float]): Minimum fraction of valid data required.
            **func_kwargs: Additional keyword arguments to pass to the function.

        Returns:
            xr.DataArray: The result of the rolling operation.
            
        Raises:
            ValueError: If the dimension is not supported.
        """
        # If the DataArray is not numeric, simply return it.
        if not np.issubdtype(self.obj.dtype, np.number):
            return self.obj

        # For time-based rolling, we expect a multi-dimensional time (year, month, day).
        if self.dim == "time" and set(["year", "month", "day"]).issubset(self.obj.dims):
            # Stack the time dimensions.
            stacked_obj = self.obj.stack(time_index=("year", "month", "day"))
            stacked_obj = stacked_obj.transpose("time_index", ...)

            # Convert to a JAX array and add a trailing singleton dimension.
            data = jnp.asarray(stacked_obj.data)[..., None]  # Expected shape: (T, assets, 1)

            # Select valid data based on self.indices.
            valid_data = data[self.indices, ...]  # Shape: (T, assets, 1)

            # Apply the rolling function.
            rolled_result = TimeSeriesOps.u_roll(
                data=valid_data,
                window_size=self.window,
                func=func,
                overlap_factor=overlap_factor,
                **func_kwargs
            )

            T_full = data.shape[0]
            # Prepare a full array (with the original time dimension) filled with NaNs.
            rolled_full = jnp.full((T_full, *rolled_result.shape[1:]), jnp.nan, dtype=rolled_result.dtype)
            # Insert the rolled results into their proper positions.
            rolled_full = rolled_full.at[self.indices].set(rolled_result)
            # Remove the extra dimension.
            rolled_full = rolled_full[..., 0]  # Final shape: (T_full, assets)

            # Reconstruct the DataArray with the rolled data.
            rolled_da = stacked_obj.copy(data=rolled_full)
            # Unstack back to the original multi-dimensional time coordinates.
            unstacked_da = rolled_da.unstack("time_index")
            return unstacked_da
        else:
            print(f'warning:{self.obj.dims} cross-sectional rolling not supported yet.')
            return self.obj
        
        