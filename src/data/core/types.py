"""
Core type definitions for data-layer time structures.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


class FrequencyType(Enum):
    """Supported frequencies for tabular-to-xarray conversion."""

    HOURLY = "H"
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    YEARLY = "Y"
    ANNUAL = "Y"


class TimeSeriesIndex:
    """
    Map datetime labels to multi-dimensional time indices.

    The index is built from the flattened `time` coordinate and converted back to
    multi-dimensional indices with `np.unravel_index`.
    """

    def __init__(self, time_coord: xr.DataArray):
        self.time_coord = time_coord
        self.shape = time_coord.shape

        # Flatten once and build datetime -> flat index mapping.
        flat_times = time_coord.values.ravel(order="C")
        times = pd.Series(flat_times)
        valid_times = times[~pd.isnull(times)]

        self.time_to_index = pd.Series(
            np.arange(len(flat_times))[~pd.isnull(times)],
            index=valid_times,
        )

    def sel(self, labels: Any, method: Any = None, tolerance: Any = None):
        del method, tolerance  # Kept for xarray-compatible signature.

        if isinstance(labels, slice):
            # Slice path keeps chronological span from start to stop.
            if labels.start is None or labels.stop is None:
                raise ValueError("Slice must include both start and stop.")
            start = pd.to_datetime(labels.start)
            stop = pd.to_datetime(labels.stop)
            start_loc, stop_loc = self.time_to_index.index.slice_locs(start, stop)
            flat_indices = self.time_to_index.iloc[start_loc:stop_loc].values
        else:
            # Normalize all non-slice inputs into datetime array form.
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
                except KeyError as exc:
                    raise KeyError(f"Date {label} not found in index") from exc

                if isinstance(locs, slice):
                    indices = self.time_to_index.iloc[locs].values
                elif isinstance(locs, np.ndarray):
                    indices = self.time_to_index.iloc[locs].values
                elif isinstance(locs, int):
                    indices = [self.time_to_index.iloc[locs]]
                else:
                    raise KeyError(f"Date {label} not found in index")
                flat_indices.extend(indices)

            if not flat_indices:
                raise KeyError(f"Dates {labels_array} not found in index")
            flat_indices = np.array(flat_indices)

        # Convert flat positions back into original calendar dims.
        multi_indices = np.unravel_index(flat_indices.astype(int), self.shape)
        return dict(zip(self.time_coord.dims, multi_indices))
