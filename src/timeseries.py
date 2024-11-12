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
        feature_columns: Optional[List[str]] = None,
        frequency: Optional[str] = None
    ) -> xr.DataArray:
        """
        Creates a DataArray from a table (DataFrame), with a fixed-size time dimension structure.
        All months have 31 days, all quarters have 3 months, etc. Invalid dates or missing
        observations are filled with NaN/NAT values.

        Parameters:
            data (pd.DataFrame): The input data table.
            time_column (str): Name of the time column in the data.
            asset_column (str): Name of the asset column in the data.
            feature_columns (list of str): List of feature columns.
            frequency (str): Data frequency ('D', 'M', 'Q', 'Y'). Will be inferred if not provided.

        Returns:
            xr.DataArray: The resulting DataArray with fixed-size dimensions:
                - year: Unique years in data
                - quarter: 4 quarters
                - month: 12 months
                - day: 31 days
                - intraday: 1 slot (for future extension)
                - asset: Unique assets
                - feature: Data features
        """
        data = data.copy()
        data[time_column] = pd.to_datetime(data[time_column])

        # Infer or validate frequency
        if frequency is None:
            inferred_freq = pd.infer_freq(data[time_column].sort_values())
            if inferred_freq is None:
                raise ValueError("Could not infer frequency from the time column. Please specify the frequency explicitly.")
            frequency = inferred_freq[0]
        else:
            frequency = frequency[0].upper()

        freq_to_components = {
            'D': ['year', 'quarter', 'month', 'day'],
            'M': ['year', 'quarter', 'month'],
            'Q': ['year', 'quarter'],
            'A': ['year'],
            'Y': ['year'],
        }

        if frequency not in freq_to_components:
            raise ValueError(f"Unsupported frequency '{frequency}'. Supported frequencies are D, M, Q, A/Y.")

        time_components = freq_to_components[frequency]

        # Extract relevant time components
        data['year'] = data[time_column].dt.year
        if 'quarter' in time_components:
            data['quarter'] = data[time_column].dt.quarter
        if 'month' in time_components:
            data['month'] = data[time_column].dt.month
        if 'day' in time_components:
            data['day'] = data[time_column].dt.day
        data['intraday'] = 0  # Since no intraday data

        # Define ranges for time components
        time_ranges = {}
        time_ranges['year'] = np.sort(data['year'].unique())
        time_ranges['quarter'] = np.array([1, 2, 3, 4]) if 'quarter' in time_components else np.array([1])
        time_ranges['month'] = np.arange(1, 13) if 'month' in time_components else np.array([1])
        time_ranges['day'] = np.arange(1, 32) if 'day' in time_components else np.array([1])
        time_ranges['intraday'] = np.array([0])

        # Prepare asset coordinates
        assets = np.sort(data[asset_column].unique())

        # Prepare feature columns
        if feature_columns is None:
            time_cols = [time_column, 'year', 'quarter', 'month', 'day', 'intraday']
            feature_columns = [col for col in data.columns if col not in time_cols + [asset_column]]

        # Create MultiIndex for data reindexing (including assets)
        index_components = [time_ranges[comp] for comp in ['year', 'quarter', 'month', 'day', 'intraday']]
        index_components.append(assets)
        index_names = ['year', 'quarter', 'month', 'day', 'intraday', asset_column]
        index = pd.MultiIndex.from_product(index_components, names=index_names)

        # Set DataFrame index and reindex
        data.set_index(index_names, inplace=True)
        data = data.reindex(index)

        # --- Fix starts here ---
        # Create time index without assets
        index_time_components = [time_ranges[comp] for comp in ['year', 'quarter', 'month', 'day', 'intraday']]
        index_time_names = ['year', 'quarter', 'month', 'day', 'intraday']
        index_time = pd.MultiIndex.from_product(index_time_components, names=index_time_names)

        # Create time_dict from index_time
        time_dict = {comp: index_time.get_level_values(comp) for comp in ['year', 'month', 'day'] if comp in time_components}
        if 'month' not in time_components:
            time_dict['month'] = 1
        if 'day' not in time_components:
            time_dict['day'] = 1

        time_index = pd.to_datetime(time_dict, errors='coerce')

        # Create the time coordinate
        shape = tuple(len(time_ranges[comp]) for comp in ['year', 'quarter', 'month', 'day', 'intraday'])
        time_coord = time_index.values.reshape(shape)
        # --- Fix ends here ---

        # Prepare data values
        data_shape = shape + (len(assets),)
        data_values = data[feature_columns].values.reshape(data_shape + (len(feature_columns),))
        data_values = jnp.array(data_values)

        # Create coordinates dictionary
        coords = {comp: time_ranges[comp] for comp in ['year', 'quarter', 'month', 'day', 'intraday']}
        coords['asset'] = assets
        coords['feature'] = feature_columns

        # Create DataArray
        dims = ['year', 'quarter', 'month', 'day', 'intraday', 'asset', 'feature']
        da = xr.DataArray(
            data_values,
            coords=coords,
            dims=dims
        )

        # Add time coordinate
        da.coords['time'] = (dims[:-2], time_coord)

        # Create TimeSeriesIndex
        time_coord_da = xr.DataArray(
            time_coord,
            coords={k: da.coords[k] for k in dims[:-2]},
            dims=dims[:-2]
        )
        ts_index = TimeSeriesIndex(time_coord_da)
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
