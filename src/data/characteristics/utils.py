"""
Utility functions for financial characteristics computation.

This module provides helper functions for computing financial characteristics,
including operations like winsorization, lagging, and other common transformations
that are used across multiple characteristic computations.
"""

import xarray as xr
import numpy as np
from typing import Dict, Any, Optional, Union, List, Callable, Tuple

def winsorize(data: xr.DataArray, dim: str, q_low: float = 0.01, q_high: float = 0.99) -> xr.DataArray:
    """
    Winsorize a DataArray along a specified dimension.
    
    Winsorization replaces extreme values with less extreme values to reduce
    the impact of outliers. This function caps values below the q_low percentile
    and above the q_high percentile.
    
    Args:
        data: Input DataArray
        dim: Dimension along which to compute percentiles
        q_low: Lower percentile (0.01 = 1%)
        q_high: Upper percentile (0.99 = 99%)
        
    Returns:
        Winsorized DataArray
    """
    # Calculate percentiles along the specified dimension
    low_values = data.quantile(q_low, dim=dim)
    high_values = data.quantile(q_high, dim=dim)
    
    # Broadcast percentiles to match input data dimensions
    low_broadcast = low_values.broadcast_like(data)
    high_broadcast = high_values.broadcast_like(data)
    
    # Apply lower and upper caps
    result = xr.where(data < low_broadcast, low_broadcast, data)
    result = xr.where(result > high_broadcast, high_broadcast, result)
    
    return result

def lag(data: xr.DataArray, dim: str, periods: int = 1) -> xr.DataArray:
    """
    Shift data along a dimension by a specified number of periods.
    
    This is equivalent to pandas .shift() for DataArrays. Positive periods
    shifts forward in time (introduces NaNs at the beginning), while negative
    periods shifts backward in time (introduces NaNs at the end).
    
    Args:
        data: Input DataArray
        dim: Dimension along which to shift
        periods: Number of periods to shift (positive = forward, negative = backward)
        
    Returns:
        Shifted DataArray
    """
    return data.shift({dim: periods})

def growth_rate(data: xr.DataArray, dim: str, periods: int = 1) -> xr.DataArray:
    """
    Compute growth rate over a specified number of periods.
    
    Formula: (x_t - x_{t-periods}) / x_{t-periods}
    
    Args:
        data: Input DataArray
        dim: Dimension along which to compute growth rate
        periods: Number of periods to look back
        
    Returns:
        Growth rate DataArray
    """
    lagged_data = lag(data, dim, periods)
    return (data - lagged_data) / lagged_data

def annualize(data: xr.DataArray, periods_per_year: int) -> xr.DataArray:
    """
    Annualize a DataArray of rates.
    
    For multiplicative quantities like returns, compounds the rate.
    For additive quantities like growth rates, scales by the number of periods.
    
    Args:
        data: Input DataArray
        periods_per_year: Number of periods in a year (e.g., 12 for monthly, 252 for daily)
        
    Returns:
        Annualized DataArray
    """
    return data * periods_per_year

def compound(data: xr.DataArray, dim: str, periods: int = None) -> xr.DataArray:
    """
    Compound a DataArray of returns over a specified number of periods.
    
    Formula: prod(1 + r) - 1
    
    Args:
        data: Input DataArray of returns
        dim: Dimension along which to compound
        periods: Number of periods to compound. If None, compounds over the entire dimension.
        
    Returns:
        Compounded returns DataArray
    """
    if periods is None:
        # Compound over the entire dimension
        return (1 + data).prod(dim=dim) - 1
    else:
        # Compound over the specified number of periods
        return (1 + data).rolling({dim: periods}).reduce(lambda x: x.prod(dim=dim)) - 1

def rolling_mean(data: xr.DataArray, dim: str, window: int, min_periods: int = None) -> xr.DataArray:
    """
    Compute rolling mean over a specified window.
    
    Args:
        data: Input DataArray
        dim: Dimension along which to compute rolling mean
        window: Window size
        min_periods: Minimum number of observations required. If None, defaults to window.
        
    Returns:
        Rolling mean DataArray
    """
    min_periods = min_periods or window
    return data.rolling({dim: window}, min_periods=min_periods).mean()

def calendar_year_to_fiscal_year(time_var: xr.DataArray, fiscal_year_end_month: int) -> xr.DataArray:
    """
    Convert calendar year-month to fiscal year-month.
    
    For companies with fiscal year end in month M, the fiscal year is the calendar
    year if the calendar month is > M, otherwise it's the calendar year - 1.
    
    Args:
        time_var: DataArray with time dimension containing year-month information
        fiscal_year_end_month: Month when the fiscal year ends (1-12)
        
    Returns:
        DataArray with fiscal year
    """
    # Extract year and month from time variable
    if isinstance(time_var, xr.DataArray):
        year = time_var.dt.year
        month = time_var.dt.month
    else:
        # Assume pandas datetime
        year = time_var.year
        month = time_var.month
    
    # Determine fiscal year
    fiscal_year = xr.where(month > fiscal_year_end_month, year, year - 1)
    
    return fiscal_year

def ff_fama_french_dates(time_var: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Create Fama-French dates for portfolio formation.
    
    Following Fama-French methodology:
    - For July-December dates, ffyear = calendar year
    - For January-June dates, ffyear = calendar year - 1
    - ffmonth ranges from 1 (July) to 12 (June)
    
    Args:
        time_var: DataArray with time dimension
        
    Returns:
        Tuple of (ffyear, ffmonth) DataArrays
    """
    if isinstance(time_var, xr.DataArray):
        year = time_var.dt.year
        month = time_var.dt.month
    else:
        # Assume pandas datetime
        year = time_var.year
        month = time_var.month
    
    # Create ffyear (Fama French year)
    jan_june_mask = month <= 6
    ffyear = xr.where(jan_june_mask, year - 1, year)
    
    # Create ffmonth (Fama French month)
    ffmonth = xr.where(jan_june_mask, month + 6, month - 6)
    
    return ffyear, ffmonth 