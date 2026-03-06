"""
Tabular-to-xarray conversion helpers used by data loaders.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from src.data.core.types import FrequencyType, TimeSeriesIndex


def convert_sas_date(sas_date_col: pd.Series, epoch: str = "1960-01-01") -> pd.Series:
    """Convert numeric SAS date values to pandas datetimes."""
    # Fail early on bad values so date conversion errors are explicit.
    problematic = sas_date_col.isna() | np.isinf(sas_date_col)
    if problematic.any():
        sample = sas_date_col[problematic].head(10).tolist()
        raise ValueError(f"Invalid SAS date values encountered: {sample}")
    sas_epoch = pd.to_datetime(epoch)
    return sas_epoch + pd.to_timedelta(sas_date_col.astype(int), unit="D")


def from_table(
    data: pd.DataFrame,
    time_column: str = "time",
    asset_column: str = "asset",
    feature_columns: Optional[List[str]] = None,
    frequency: FrequencyType = FrequencyType.DAILY,
) -> xr.Dataset:
    """
    Build a multi-dimensional xarray Dataset from a flat table.

    Output dimensions are `(year, month, day, hour, asset)`.
    """
    data = data.copy()
    # Normalize time field once. All downstream indexing depends on valid datetimes.
    data[time_column] = pd.to_datetime(data[time_column], errors="coerce")
    if data[time_column].isnull().any():
        raise ValueError(f"The '{time_column}' column contains invalid datetime values.")

    # Build calendar components based on declared sampling frequency.
    if frequency == FrequencyType.YEARLY:
        data["year"] = data[time_column].dt.year
        data["month"] = 1
        data["day"] = 1
        data["hour"] = 0
    elif frequency == FrequencyType.MONTHLY:
        data["year"] = data[time_column].dt.year
        data["month"] = data[time_column].dt.month
        data["day"] = 1
        data["hour"] = 0
    elif frequency == FrequencyType.DAILY:
        data["year"] = data[time_column].dt.year
        data["month"] = data[time_column].dt.month
        data["day"] = data[time_column].dt.day
        data["hour"] = 0
    elif frequency == FrequencyType.HOURLY:
        data["year"] = data[time_column].dt.year
        data["month"] = data[time_column].dt.month
        data["day"] = data[time_column].dt.day
        data["hour"] = data[time_column].dt.hour
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")

    years = np.sort(data["year"].unique())
    months = np.sort(data["month"].unique())
    days = np.sort(data["day"].unique())
    hours = np.sort(data["hour"].unique())
    assets = np.sort(data[asset_column].unique())

    if feature_columns is None:
        # Auto-pick feature columns by excluding structural fields.
        exclude_cols = [time_column, "year", "month", "day", "hour", asset_column]
        feature_columns = [col for col in data.columns if col not in exclude_cols]

    missing_features = [fc for fc in feature_columns if fc not in data.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found: {missing_features}")

    # Map each row to one flattened slot in (year, month, day, hour, asset).
    y_idx = np.searchsorted(years, data["year"].to_numpy())
    m_idx = np.searchsorted(months, data["month"].to_numpy())
    d_idx = np.searchsorted(days, data["day"].to_numpy())
    h_idx = np.searchsorted(hours, data["hour"].to_numpy())
    a_idx = np.searchsorted(assets, data[asset_column].to_numpy())

    n_months = len(months)
    n_days = len(days)
    n_hours = len(hours)
    n_assets = len(assets)

    idx_1d = (((y_idx * n_months + m_idx) * n_days + d_idx) * n_hours + h_idx) * n_assets + a_idx
    shape_data = (len(years), len(months), len(days), len(hours), len(assets))

    feature_arrays = {}
    for fc in feature_columns:
        if data[fc].dtype.kind in "bifc":
            arr = np.empty(shape_data, dtype=np.float64)
        else:
            arr = np.empty(shape_data, dtype=object)
        # Use NaN as default fill so sparse panels stay explicit.
        arr.fill(np.nan)
        feature_arrays[fc] = arr

    for fc in feature_columns:
        # Vectorized scatter from flat table into panel tensor.
        feature_arrays[fc].ravel()[idx_1d] = data[fc].to_numpy()

    # Build full calendar coordinate tensor (year, month, day, hour).
    yr_mesh, mo_mesh, dd_mesh, hh_mesh = np.meshgrid(years, months, days, hours, indexing="ij")
    time_index_flat = pd.to_datetime(
        {
            "year": yr_mesh.ravel(),
            "month": mo_mesh.ravel(),
            "day": dd_mesh.ravel(),
            "hour": hh_mesh.ravel(),
        },
        errors="coerce",
    )
    time_data = time_index_flat.values.reshape((len(years), len(months), len(days), len(hours)))

    time_coord = xr.DataArray(
        data=time_data,
        coords={"year": years, "month": months, "day": days, "hour": hours},
        dims=["year", "month", "day", "hour"],
    )
    ts_index = TimeSeriesIndex(time_coord)

    ds = xr.Dataset(
        coords={
            "year": years,
            "month": months,
            "day": days,
            "hour": hours,
            "asset": assets,
            "time": (["year", "month", "day", "hour"], time_data),
            "time_flat": np.arange(len(time_index_flat.values)),
        }
    )

    for fc in feature_columns:
        # Keep all variables on the same panel shape for consistent downstream ops.
        ds[fc] = xr.DataArray(
            data=feature_arrays[fc],
            dims=["year", "month", "day", "hour", "asset"],
        )

    # Attach custom selector index used by `.dt.sel(...)`.
    ds.coords["time"].attrs["indexes"] = {"time": ts_index}
    return ds


def load_simulated_data(
    num_assets: int,
    num_timesteps: int,
    num_vars: int,
    freq: FrequencyType = FrequencyType.DAILY,
    start_date: str = "2000-01-01",
) -> xr.Dataset:
    """Generate a lightweight synthetic panel dataset for tests and notebooks."""
    assets = [f"asset_{i + 1}" for i in range(num_assets)]
    time_index = pd.date_range(start=start_date, periods=num_timesteps, freq=freq.value)
    multi_index = pd.MultiIndex.from_product([assets, time_index], names=["asset", "time"])
    df = pd.DataFrame(index=multi_index).reset_index()
    for i in range(num_vars):
        df[f"var_{i + 1}"] = np.random.randn(len(df))
    return from_table(
        data=df,
        time_column="time",
        asset_column="asset",
        frequency=freq,
    )
