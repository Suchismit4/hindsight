# src/data/processors.py

"""
XArray Dataset processors for financial data.

This module contains processors that operate on xarray Datasets to implement
various financial data transformations, coordinate handling, and specialized 
calculations commonly used in financial research.
"""

import xarray as xr
import numpy as np
import pandas as pd
from typing import Any, Dict, Union, List, Optional, Tuple
from src.data.processors.registry import post_processor
from src.data.core.util import Loader


@post_processor
def merge_2d_table(ds: xr.Dataset, params: Dict[str, Any]) -> xr.Dataset:
    """
    Merge values from an external dataset (loaded via "external_ds") into the primary dataset.
    
    This processor takes values from a lookup table and adds them as a new variable
    in the dataset, using an identifier column for mapping.
    
    Args:
        ds: The xarray Dataset to process
        params: Parameters with keys:
            - external_ds: External X-array dataset to merge
            - ax1: Primary axis dimension name (usually 'asset')
            - ax2: The column/variable name to merge
            - identifier: Optional column name in external_ds to use as key (defaults to 'identifier')
            
    Returns:
        Updated xarray Dataset with new variable
        
    Raises:
        ValueError: If required parameters are missing or incompatible
    """
    # Extract required parameters
    external_ds = params.get("external_ds")
    ax1 = params.get('ax1')
    ax2 = params.get('ax2')

    # Validate parameters
    if external_ds is None or ax1 is None or ax2 is None:
        raise ValueError("Required parameters missing for merge_2d_table: external_ds, ax1, ax2")
        
    # Create a mapping from the external dataset using the identifier column
    external_ds = external_ds[["identifier", ax2]].drop_duplicates(subset="identifier", keep='last')
    
    # Use the ax1 coordinate (coming from the 'identifier' column) to build the mapping
    ax1_ids = ds.coords[ax1].values
    ax2_mapping = dict(zip(external_ds["identifier"], external_ds[ax2]))
    
    # Map values for each identifier in the dataset
    values = [ax2_mapping.get(ax1_i, "") for ax1_i in ax1_ids]
            
    # Attach the values as a new DataArray with the ax1 dimension
    ds[ax2] = xr.DataArray(
        values,
        dims=[ax1],
        coords={ax1: ax1_ids}
    )
    
    return ds


@post_processor
def replace(ds: xr.Dataset, params: Dict[str, Any]) -> xr.Dataset:
    """
    Replace values in one variable with values from another dataset.
    
    This processor is commonly used for handling delisting returns or other
    replacement operations where values from a different source should take
    precedence.
    
    Args:
        ds: The xarray Dataset to process
        params: Parameters with keys:
            - external_ds: External dataset with replacement values
            - from: Source variable name to get values from
            - to: Target variable name to update
            - rename: Optional list of column name pairs to align dimensions
            - identifier: Optional identifier column name for loading external dataset (defaults to 'permno')
            
    Returns:
        Updated xarray Dataset with replaced values
        
    Raises:
        ValueError: If required parameters are missing or incompatible
    """
    # Extract required parameters
    external_ds = params.get("external_ds")
    replace_frm = params.get('from')
    replace_to = params.get('to')
    rename_pairs = params.get('rename', [])
    identifier_col = params.get('identifier', 'permno')
    
    # Validate parameters
    if external_ds is None or replace_frm is None or replace_to is None:
        raise ValueError("Required parameters missing for replace: external_ds, from, to")
    
    # Check if the target variable exists in the dataset
    if replace_to not in ds:
        raise ValueError(f"Target variable '{replace_to}' not found in dataset")
        
    # Reindex the dataset to match the primary dataset
    external_ds = external_ds.reindex_like(ds, method=None)
    
    # Only replace non-NaN values from the source
    valid_replace_frm = ~np.isnan(external_ds[replace_frm])
    ds[replace_to] = xr.where(valid_replace_frm, external_ds[replace_frm], ds[replace_to])
    
    return ds

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
    
    # Reduce over time dimensions using nanmax to avoid NaN warnings
    # This gives us a single PERMNO value per asset
    permno_asset = np.nanmax(permno_values, axis=tuple(range(3)))
    
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
    
    # Reduce over time dimensions using nanmax to avoid NaN warnings
    # This gives us a single PERMCO value per asset
    permco_asset = np.nanmax(permco_values, axis=tuple(range(3)))
    
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