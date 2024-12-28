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
        # Make a copy to avoid modifying the original DataFrame
        data = data.copy()
        # Convert the time column to datetime
        data[time_column] = pd.to_datetime(data[time_column], errors='coerce')

        # Check for NaT values in the time column
        if data[time_column].isnull().any():
            raise ValueError(f"The '{time_column}' column contains invalid datetime values.")

        # Extract time components based on frequency
        dates = data[time_column]

        if frequency == FrequencyType.YEARLY:
            data['year'] = dates.dt.year
            data['month'] = 1
            data['day'] = 1
            months = np.array([1])
            days = np.array([1])
        elif frequency == FrequencyType.MONTHLY:
            data['year'] = dates.dt.year
            data['month'] = dates.dt.month
            data['day'] = 1
            months = np.arange(1, 13)
            days = np.array([1])
        elif frequency == FrequencyType.DAILY:
            data['year'] = dates.dt.year
            data['month'] = dates.dt.month
            data['day'] = dates.dt.day
            months = np.arange(1, 13)
            days = np.arange(1, 32)
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

        # Prepare the unique ranges for each time component
        years = np.sort(data['year'].unique())

        # Prepare asset coordinates
        assets = np.sort(data[asset_column].unique())

        # Determine feature columns if not provided
        if feature_columns is None:
            time_cols = [time_column, 'year', 'month', 'day', asset_column]
            feature_columns = [col for col in data.columns if col not in time_cols]

        # Check if feature columns are present
        missing_features = [col for col in feature_columns if col not in data.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found in data: {missing_features}")
        
        data = data[['year', 'month', 'day', time_column, asset_column, *feature_columns]]

        # Create the MultiIndex for reindexing
        index_components = [years, months, days, assets]
        index_names = ['year', 'month', 'day', asset_column]
        full_index = pd.MultiIndex.from_product(index_components, names=index_names)
        
        dup_subset = ['year','month','day','identifier']
        pre_dup_mask = data.duplicated(subset=dup_subset, keep=False)
        pre_dup_data = data[pre_dup_mask].sort_values(dup_subset)

        print("==== DUPLICATE ROWS BEFORE SETTING INDEX ====") #DEBUG
        print(pre_dup_data)
        
        counts = (
            data
            .reset_index(drop=True)    # Make sure we don't have a MultiIndex yet
            .groupby(['year','month','day','identifier'])  
            .size()  
            .sort_values(ascending=False)
        )

        # Show only those with duplicates (DEBUG)
        counts_dup = counts[counts > 1]
        print("==== MULTIINDEX GROUPS THAT APPEAR MORE THAN ONCE ====")
        print(counts_dup)

        # Set DataFrame index and reindex to include all possible combinations
        data.set_index(index_names, inplace=True)
        
    
        # OVERLOOK (DEBUG)
        print("The multi-index is not unique. Identifying duplicate index entries:")
        # Find duplicated index entries
        duplicated_indices = data.index[data.index.duplicated(keep=False)]
        print(duplicated_indices.unique())
        print(data[data.index.isin(duplicated_indices)])
        duplicated_mask = data.index.duplicated(keep=False)
        duplicated_data = data[duplicated_mask]
        print("==== DUPLICATE ROWS ====")
        print(duplicated_data)
        i = 0
        for idx, group in duplicated_data.groupby(level=[0, 1, 2, 3]): 
            print("MultiIndex:", idx)
            print(group)
            print("-----")
            i += 1
            if (i > 3):
                break

        # quit(0)
        # OVERLOOK STOP
        
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

        # Initialize an empty dataset with the coordinates
        ds = xr.Dataset(
            coords={
                'year': years,
                'month': months,
                'day': days,
                'asset': assets,
                'time': (['year', 'month', 'day'], time_data)
            }
        )

        # Add each feature as a separate variable in the dataset
        shape_data = (len(years), len(months), len(days), len(assets))
        for feature in feature_columns:
            var_data = data[feature].values.reshape(shape_data)
            ds[feature] = xr.DataArray(
                data=var_data,
                dims=['year', 'month', 'day', 'asset']
            )

        # Attach the TimeSeriesIndex to the time coordinate
        ds.coords['time'].attrs['indexes'] = {'time': ts_index}

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
        data = self._obj

        time_values = data.coords['time'].values
        
        if isinstance(self._obj, xr.Dataset):
            return xr.Dataset({var: ('time', data[var].values) for var in self._obj.data_vars}, coords={'time': time_values})
        return xr.DataArray(data.values, coords={'time': time_values}, dims=['time'], name=data.name)
    
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
