# src/data/processors/processors.py

"""
XArray Dataset processors for financial data.

This module contains processors that operate on xarray Datasets to implement
various financial data transformations, coordinate handling, and specialized 
calculations commonly used in financial research.

NOTE: External table merging (msenames, msedelist, etc.) is now handled at the
DataFrame level before xarray conversion. See GenericWRDSDataLoader.
"""

import xarray as xr
import numpy as np
from typing import Any, Dict
from src.data.processors.registry import post_processor


@post_processor
def set_permno(ds: xr.Dataset, params: Dict[str, Any]) -> xr.Dataset:
    """
    Set PERMNO as a coordinate in the dataset.
    
    This processor extracts PERMNO values from a variable and sets them
    as coordinates associated with the asset dimension. This is useful
    for entity identification in financial analysis.
    
    Args:
        ds: The xarray Dataset to process
        params: Optional parameters (not used)
            
    Returns:
        Updated xarray Dataset with PERMNO as a coordinate
        
    Raises:
        ValueError: If 'permno' variable is not present in the dataset
    """
    # Validate presence of permno variable
    if 'permno' not in ds.variables:
        raise ValueError("Dataset must contain a 'permno' variable to set as coordinate")
    
    # Handle possible NaN values safely by using nanmax explicitly
    permno_values = ds["permno"].values
    
    # Reduce over all dimensions except 'asset' to get a single PERMNO per asset
    # This handles both 4D (year, month, day, asset) and 5D (..., hour, asset) cases
    dims_to_reduce = tuple(i for i, d in enumerate(ds["permno"].dims) if d != 'asset')
    permno_asset = np.nanmax(permno_values, axis=dims_to_reduce)
    
    # Drop the original variable since we're converting to coordinate
    ds = ds.drop_vars("permno")
    
    # Set the extracted values as a coordinate on the asset dimension
    ds = ds.assign_coords(permno=("asset", permno_asset))
    
    return ds


@post_processor
def set_permco(ds: xr.Dataset, params: Dict[str, Any]) -> xr.Dataset:
    """
    Set PERMCO as a coordinate in the dataset.
    
    This processor extracts PERMCO (permanent company identifier) values 
    from a variable and sets them as coordinates associated with the asset 
    dimension. This is useful for company-level analysis.
    
    Args:
        ds: The xarray Dataset to process
        params: Optional parameters (not used)
            
    Returns:
        Updated xarray Dataset with PERMCO as a coordinate
        
    Raises:
        ValueError: If 'permco' variable is not present in the dataset
    """
    # Validate presence of permco variable
    if 'permco' not in ds.variables:
        raise ValueError("Dataset must contain a 'permco' variable to set as coordinate")
    
    # Handle possible NaN values safely by using nanmax explicitly
    permco_values = ds["permco"].values
    
    # Reduce over all dimensions except 'asset' to get a single PERMCO per asset
    # This handles both 4D (year, month, day, asset) and 5D (..., hour, asset) cases
    dims_to_reduce = tuple(i for i, d in enumerate(ds["permco"].dims) if d != 'asset')
    permco_asset = np.nanmax(permco_values, axis=dims_to_reduce)
    
    # Drop the original variable since we're converting to coordinate
    ds = ds.drop_vars("permco")
    
    # Set the extracted values as a coordinate on the asset dimension
    ds = ds.assign_coords(permco=("asset", permco_asset))
    
    return ds


@post_processor
def ps(ds: xr.Dataset, params: Dict[str, Any]) -> xr.Dataset:
    """
    Calculate preferred stock (ps) value following a standard hierarchy of variables.
    
    This processor implements a common financial data cleaning operation where
    preferred stock values are determined based on availability in a specific
    hierarchy: pstkrv → pstkl → pstk → 0.
    
    Args:
        ds: The xarray Dataset to process
        params: Optional parameters (not used)
            
    Returns:
        Updated xarray Dataset with ps and txditc variables
        
    Raises:
        ValueError: If required variables are missing
    """
    # Validate required variables
    required_vars = ['pstkrv', 'pstkl', 'pstk']
    missing_vars = [var for var in required_vars if var not in ds]
    if missing_vars:
        raise ValueError(f"Dataset missing required variables for ps calculation: {missing_vars}")
    
    # Use immutable patterns with xr.where for NaN handling
    # Follow the hierarchy: pstkrv → pstkl → pstk → 0
    ds['ps'] = xr.where(ds['pstkrv'].isnull(), ds['pstkl'], ds['pstkrv'])
    ds['ps'] = xr.where(ds['ps'].isnull(), ds['pstk'], ds['ps'])
    ds['ps'] = xr.where(ds['ps'].isnull(), 0, ds['ps'])
    
    # Handle txditc (defaulting to 0 if missing)
    if 'txditc' in ds:
        ds['txditc'] = ds['txditc'].fillna(0)
    else:
        # Create a zeros array with the same shape as other variables
        template_var = list(ds.data_vars.values())[0]
        ds['txditc'] = xr.zeros_like(template_var)
    
    return ds


@post_processor
def fix_mke(ds: xr.Dataset, params: Dict[str, Any]) -> xr.Dataset:
    """
    Fix market equity values in a thread-safe, JAX-compatible way.
    
    This processor implements a standard procedure for fixing market equity (ME)
    values for companies with multiple securities. It follows these steps:
    1. Calculate ME for each security as abs(price) * shares outstanding
    2. Group by PERMCO to identify all securities for each company
    3. For each company, apply the market equity sum to the security with the max ME
    
    This ensures market equity is properly aggregated at the company level.
    
    Args:
        ds: The xarray Dataset to process
        params: Optional parameters (not used)
            
    Returns:
        Updated xarray Dataset with fixed market equity values
        
    Raises:
        ValueError: If required variables are missing or PERMCO is not set as a coordinate
    """
    # Validate required components
    if 'permco' not in ds.coords:
        raise ValueError("PERMCO must be set as a coordinate. Try adding set_permco_coord BEFORE this processor.")
        
    if 'prc' not in ds or 'shrout' not in ds:
        raise ValueError("Dataset must contain 'prc' and 'shrout' variables to calculate market equity")
    
    # Compute trivial market equity (price * shares outstanding)
    ds['me'] = abs(ds['prc']) * ds['shrout']

    # Replace NaNs with zeros for aggregation to avoid thread-safety issues
    # This gives clean zeros where data is missing rather than propagating NaNs
    me_filled = ds['me'].fillna(0)
    
    # Group by permco without using skipna to avoid thread-safety issues
    # For sum, we can just fillna(0) and sum normally
    summe = me_filled.groupby(ds.permco).sum(dim="asset")
    
    # For max, we use a masked approach to handle NaNs safely
    # First create a mask of non-NaN values
    me_mask = ~np.isnan(ds['me'])
    
    # Create a version of me with -inf where NaN for safe max finding
    me_for_max = ds['me'].fillna(-np.inf)
    maxme = me_for_max.groupby(ds.permco).max(dim="asset")
    
    # Align with original dimensions
    maxme_broadcast = maxme.sel(permco=ds.permco)
    summe_broadcast = summe.sel(permco=ds.permco)
    
    # Create a mask for the max values
    # Only mark as max where the value is valid (not NaN in original)
    is_max_permno = (me_mask & (ds['me'] == maxme_broadcast))
    
    # Apply the aggregation using a JAX-friendly approach
    # This avoids the thread-safety issues with NaN handling
    # Only replace values for the permno with max market equity
    ds['me'] = xr.where(
        is_max_permno, 
        summe_broadcast,
        ds['me']
    )

    return ds
