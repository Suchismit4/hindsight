# src/data/data.py


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

from .operations import Rolling

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
        
        # TODO: Fix infer_dimensions() to infer the time from the attributes.
        # self._dims = self._infer_dimensions()
        # self._metadata = self._create_metadata()
        
    @classmethod
    def from_table(
        cls,
        data: pd.DataFrame,
        time_column: str = 'time',
        asset_column: str = 'asset',
        feature_columns: Optional[List[str]] = None,
        frequency: Optional[str] = 'D'
    ) -> xr.DataArray:
        """
        Creates a DataArray from a table (DataFrame), with fixed-size time dimensions.

        Parameters:
            data (pd.DataFrame): The input data table.
            time_column (str): Name of the time column in the data.
            asset_column (str): Name of the asset column in the data.
            feature_columns (list of str, optional): List of value columns.

        Returns:
            xr.DataArray: The resulting DataArray with dimensions:
                - year: Unique years in data
                - month: 12 months
                - day: 31 days
                - asset: Unique assets
                - feature: Data features (if multiple value columns)
        """
        # Make a copy to avoid modifying the original DataFrame
        data = data.copy()
        # Convert the time column to datetime
        data[time_column] = pd.to_datetime(data[time_column])

        # Extract time components
        dates = data[time_column]
        data['year'] = dates.dt.year
        data['month'] = dates.dt.month
        data['day'] = dates.dt.day

        # Prepare the unique ranges for each time component
        years = np.sort(data['year'].unique())
        months = np.arange(1, 13)  # Fixed-size months (1 to 12)
        days = np.arange(1, 32)    # Fixed-size days (1 to 31)

        # Prepare asset coordinates
        assets = np.sort(data[asset_column].unique())

        # Determine feature columns if not provided
        if feature_columns is None:
            time_cols = [time_column, 'year', 'month', 'day', asset_column]
            feature_columns = [col for col in data.columns if col not in time_cols]

        # Create the MultiIndex for reindexing
        index_components = [years, months, days, assets]
        index_names = ['year', 'month', 'day', asset_column]
        full_index = pd.MultiIndex.from_product(index_components, names=index_names)

        # Set DataFrame index and reindex to include all possible combinations
        data.set_index(index_names, inplace=True)
        data = data.reindex(full_index)

        # Create the time coordinate array
        # Extract unique time combinations to avoid duplicates
        unique_time = full_index.droplevel(asset_column).unique()
        time_index = pd.to_datetime({
            'year': unique_time.get_level_values('year'),
            'month': unique_time.get_level_values('month'),
            'day': unique_time.get_level_values('day')
        }, errors='coerce')

        # Reshape the time data to match the dimensions without 'asset'
        shape_time = (len(years), len(months), len(days))
        time_data = time_index.values.reshape(shape_time)

        # Create the time coordinate DataArray
        time_coord = xr.DataArray(
            data=time_data,
            coords={
                'year': years,
                'month': months,
                'day': days,
            },
            dims=['year', 'month', 'day']
        )

        # Create the TimeSeriesIndex
        ts_index = TimeSeriesIndex(time_coord)

        # Prepare the data values
        shape_data = (len(years), len(months), len(days), len(assets))
        if len(feature_columns) == 1:
            # Single value column
            var_data = data[feature_columns[0]].values.reshape(shape_data)
            da = xr.DataArray(
                data=var_data,
                coords={
                    'year': years,
                    'month': months,
                    'day': days,
                    'asset': assets,
                    'time': (['year', 'month', 'day'], time_data)
                },
                dims=['year', 'month', 'day', 'asset'],
                name=feature_columns[0]
            )
        else:
            # Multiple value columns
            var_data = data[feature_columns].values.reshape(shape_data + (len(feature_columns),))
            da = xr.DataArray(
                data=var_data,
                coords={
                    'year': years,
                    'month': months,
                    'day': days,
                    'asset': assets,
                    'feature': feature_columns,
                    'time': (['year', 'month', 'day'], time_data)
                },
                dims=['year', 'month', 'day', 'asset', 'feature']
            )

        # Attach the TimeSeriesIndex to the time coordinate
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
        # data = self._obj.stack(time=('year', 'quarter', 'month', 'day'))
        time_values = data.coords['time'].values
        
        if isinstance(self._obj, xr.Dataset):
            return xr.Dataset({var: ('time', data[var].values) for var in self._obj.data_vars}, coords={'time': time_values})
        return xr.DataArray(data.values, coords={'time': time_values}, dims=['time'], name=data.name)
    
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

    # def align_with(
    #     self,
    #     other: Union[xr.DataArray, xr.Dataset],
    #     method: str = 'outer',
    #     freq_method: str = 'ffill'
    # ) -> Union[xr.DataArray, xr.Dataset]:
    #     """
    #     Aligns two panel datasets across time, assets, and characteristics.

    #     Parameters:
    #         other (Union[xr.DataArray, xr.Dataset]): The dataset to align with.
    #         method (str): Join method for alignment ('outer', 'inner', 'left', 'right').
    #         freq_method (str): Method for frequency alignment ('ffill', 'bfill', 'mean').

    #     Returns:
    #         Union[xr.DataArray, xr.Dataset]: The aligned dataset.
    #     """
    #     if not isinstance(other, (xr.DataArray, xr.Dataset)):
    #         raise TypeError("Can only align with xarray objects")

    #     # Use TimeSeriesOps.merge_panel_data to align the data
    #     merged_data = TimeSeriesOps.merge_panel_data(self._obj, other)

    #     return merged_data
    
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
