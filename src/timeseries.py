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

class FrequencyType(Enum):
    """
    Enumeration of supported data frequencies.
    Used for proper alignment in cross-frequency operations.
    """
    DAILY     = 'D'
    WEEKLY    = 'W'
    MONTHLY   = 'M'
    QUARTERLY = 'Q'
    YEARLY    = 'Y'

@dataclass(frozen=True)
class DataDimensions:
    """
    Holds the positions of each dimension in the data array for panel data operations.

    Attributes:
        time_dim (int): Position of time dimension in array.
        asset_dim (int): Position of asset dimension in array.
        char_dim (Optional[int]): Position of characteristics dimension in array, if any.
    """
    time_dim: int
    asset_dim: int
    char_dim: Optional[int] = None

@dataclass(frozen=True)
class TimeIndexMetadata:
    """
    Stores metadata information about the time index for alignment purposes.

    Attributes:
        time_coords (np.ndarray): Time axis coordinates for alignment.
        dims (Dict[str, int]): Mapping of dimension names to their positions.
        frequency (FrequencyType): Data sampling frequency.
        shape (Tuple[int, ...]): Shape of the data array.
    """
    time_coords: np.ndarray
    dims: Dict[str, int]
    frequency: FrequencyType
    shape: Tuple[int, ...]


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
        
        flat_indices = self.time_to_index.reindex(labels_array)
        if flat_indices.isnull().any():
            missing = labels_array[pd.isnull(flat_indices)]
            raise KeyError(f"Dates {missing} not found in index")
                
        multi_indices = np.unravel_index(flat_indices.values.astype(int), self.shape)
        dim_names = self.time_coord.dims
        
        return dict(zip(dim_names, multi_indices))


class TimeSeriesOps:
    """
    Core operations for multi-dimensional panel data processing.
    Handles arrays with Time x Assets x Characteristics structure.
    """

    @staticmethod
    def merge_panel_data(
        data1: xr.DataArray,
        data2: xr.DataArray
    ) -> xr.DataArray:
        """
        Merges two panel data arrays along the asset dimension, aligning assets and time indices.
        If data arrays have a 'feature' dimension, combines along that dimension.
        """
        raise NotImplementedError("Merging is not supported yet.")


class DateTimeAccessorBase:
    """
    Base class for managing time series operations on panel data structures.

    Provides methods to infer data dimensions, create metadata, and align datasets.

    Attributes:
        _obj (Union[xr.Dataset, xr.DataArray]): The xarray object being accessed.
        _dims (DataDimensions): Dimensional information of the data array.
        _metadata (TimeIndexMetadata): Metadata about the time index.
    """

    def __init__(self, xarray_obj: Union[xr.Dataset, xr.DataArray]) -> None:
        """
        Initializes the DateTimeAccessorBase with an xarray object.

        Parameters:
            xarray_obj (Union[xr.Dataset, xr.DataArray]): The xarray object to be accessed.
        """
        self._obj = xarray_obj
        self._dims = self._infer_dimensions()
        self._metadata = self._create_metadata()
        
    @classmethod
    def from_table(
        cls,
        data: pd.DataFrame,
        time_column: str = 'time',
        asset_column: str = 'asset',
        feature_columns: Optional[List[str]] = None
    ) -> xr.DataArray:
        """
        Creates a DataArray from a table (DataFrame), ensuring the time dimension is multi-dimensional
        with fixed sizes for Year (Y), Quarter (Q), Month (M), Day (D), and Intraday (I).

        Parameters:
            data (pd.DataFrame): The input data table.
            time_column (str): Name of the time column in the data.
            asset_column (str): Name of the asset column in the data.
            feature_columns (list of str): List of feature columns.

        Returns:
            xr.DataArray: The resulting DataArray with dimensions ('time', 'asset', 'feature'),
                          and data stored as a JAX array.
        """
        data = data.copy()
        data[time_column] = pd.to_datetime(data[time_column])
        
        # Extract time components
        data['year'] = data[time_column].dt.year
        data['quarter'] = data[time_column].dt.quarter
        data['month'] = data[time_column].dt.month
        data['day'] = data[time_column].dt.day
        data['intraday'] = 0  # Since no intraday data, set to 0

        # Define time dimensions with fixed sizes
        years = np.sort(data['year'].unique())
        quarters = np.array([1, 2, 3, 4])  # 4 quarters
        months = np.arange(1, 4)  # Months 1 to 3 within each quarter
        days = np.arange(1, 32)  # Days 1 to 31
        intraday_levels = np.array([0])  # Single intraday level
        
        # Create mappings for months within quarters
        # Quarter 1: Months 1-3, Quarter 2: Months 4-6, etc.
        quarter_month_map = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}
        
        # Map actual months to month indices within quarters
        data['month_in_quarter'] = data.apply(
            lambda row: quarter_month_map[row['quarter']].index(row['month'] % 12 or 12) + 1,
            axis=1
        )
        
        # Update months to be indices within quarters
        data['month'] = data['month_in_quarter']
        data.drop(columns=['month_in_quarter'], inplace=True)

        # Prepare asset coordinates
        assets = np.sort(data[asset_column].unique())

        # If feature_columns is None, select all columns except time_column and asset_column
        # Prepare feature columns
        if feature_columns is None:
            feature_columns = [
                col for col in data.columns
                if col not in [time_column, asset_column, 'year', 'quarter', 'month', 'day', 'intraday']
            ]

        # Create a MultiIndex from all dimensions
        index = pd.MultiIndex.from_product(
            [years, quarters, months, days, intraday_levels, assets],
            names=['year', 'quarter', 'month', 'day', 'intraday', asset_column]
        )

        # Set data index to the multi-dimensional time and asset dimensions
        data.set_index(['year', 'quarter', 'month', 'day', 'intraday', asset_column], inplace=True)
        # Reindex data to align with the complete index
        data = data.reindex(index)

        # Prepare data values array
        data_values = data[feature_columns].values

        # Reshape data values to match the dimensions
        shape = (
            len(years),
            len(quarters),
            len(months),
            len(days),
            len(intraday_levels),
            len(assets),
            len(feature_columns)
        )
        data_values = data_values.reshape(shape)

        # Convert data values to JAX array
        data_values = jnp.array(data_values)
        
        # Create coordinates dictionary for xarray DataArray
        coords = {
            'year': years,
            'quarter': quarters,
            'month': months,
            'day': days,
            'intraday': intraday_levels,
            'asset': assets,
            'feature': feature_columns
        }

        # Create time coordinate DataArray 
        time_coord = xr.DataArray(
            times, # 
            dims=['time'],
            coords={'time': times}
        )

        # Create TimeSeriesIndex
        ts_index = TimeSeriesIndex(time_coord)

        # Create DataArray with coordinates and dimensions
        da = xr.DataArray(
            data_values,
            coords={
                'time': times,
                'asset': assets,
                'feature': feature_columns
            },
            dims=['time', 'asset', 'feature']
        )

        # Attach TimeSeriesIndex to time coordinate
        da.coords['time'].attrs['indexes'] = {'time': ts_index}

        return da
        

        
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
        # Since data is already time-indexed, we might not need to stack.
        # This method can be customized if needed.
        return self._obj

    def _infer_dimensions(self) -> DataDimensions:
        """
        Determines the positions of time, asset, and characteristic dimensions in the data array.

        Returns:
            DataDimensions: An object containing the positions of each dimension.
        """
        dims = list(self._obj.dims)  # List of dimension names

        # Find time dimension
        time_dim = None
        for i, dim in enumerate(dims):
            # Check if the coordinate values are datetime-like
            coord_values = self._obj.coords.get(dim, None)
            if coord_values is not None and np.issubdtype(coord_values.dtype, np.datetime64):
                time_dim = i
                break
        if time_dim is None:
            raise ValueError("No time dimension found in data")

        # Find asset dimension
        asset_dim = None
        for i, dim in enumerate(dims):
            if dim.lower() in ['asset', 'assets']:
                asset_dim = i
                break
        if asset_dim is None:
            # Assume the largest non-time dimension is the asset dimension
            sizes = [(idx, size) for idx, size in enumerate(self._obj.sizes.values()) if idx != time_dim]
            if not sizes:
                raise ValueError("No asset dimension found in data")
            asset_dim = max(sizes, key=lambda x: x[1])[0]

        # Identify characteristics dimension if present
        remaining_dims = set(range(len(dims))) - {time_dim, asset_dim}
        char_dim = remaining_dims.pop() if remaining_dims else None

        return DataDimensions(time_dim, asset_dim, char_dim)

    def _create_metadata(self) -> TimeIndexMetadata:
        """
        Creates metadata information for the time index.

        Returns:
            TimeIndexMetadata: An instance containing metadata about the time index.
        """
        # Extract time coordinates
        time_dim_name = self._obj.dims[self._dims.time_dim]
        time_coords = self._obj.coords[time_dim_name].values

        # Create dimension mapping
        dims = {dim_name: idx for idx, dim_name in enumerate(self._obj.dims)}

        # Determine data frequency
        inferred_freq = pd.infer_freq(pd.DatetimeIndex(time_coords))
        if inferred_freq is not None:
            # Map inferred frequency to FrequencyType
            freq_map = {
                'D': FrequencyType.DAILY,
                'W': FrequencyType.WEEKLY,
                'M': FrequencyType.MONTHLY,
                'Q': FrequencyType.QUARTERLY,
                'A': FrequencyType.ANNUAL,
                'Y': FrequencyType.ANNUAL
            }
            frequency = freq_map.get(inferred_freq[0], FrequencyType.DAILY)
        else:
            frequency = FrequencyType.DAILY  # Default frequency

        # Get shape of the data array
        shape = self._obj.shape

        return TimeIndexMetadata(
            time_coords=time_coords,
            dims=dims,
            frequency=frequency,
            shape=shape
        )

    def align_with(
        self,
        other: Union[xr.DataArray, xr.Dataset],
        method: str = 'outer',
        freq_method: str = 'ffill'
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Aligns two panel datasets across time, assets, and characteristics.

        Parameters:
            other (Union[xr.DataArray, xr.Dataset]): The dataset to align with.
            method (str): Join method for alignment ('outer', 'inner', 'left', 'right').
            freq_method (str): Method for frequency alignment ('ffill', 'bfill', 'mean').

        Returns:
            Union[xr.DataArray, xr.Dataset]: The aligned dataset.
        """
        if not isinstance(other, (xr.DataArray, xr.Dataset)):
            raise TypeError("Can only align with xarray objects")

        # Use TimeSeriesOps.merge_panel_data to align the data
        merged_data = TimeSeriesOps.merge_panel_data(self._obj, other)

        return merged_data

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
