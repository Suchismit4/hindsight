# src/data/processors.py

import xarray as xr
import numpy as np
import pandas as pd
from typing import Any, Dict
from src.data.processors.registry import post_processor


@post_processor
def merge_2d_table(ds: xr.Dataset, params: Dict[str, Any]) -> xr.Dataset:
    """
    Merge values from an external dataset (loaded via "src") into the primary dataset.
    """
    external_ds = params.get("external_ds")
    ax1 = params.get('ax1')
    ax2 = params.get('ax2')

    external_ds = external_ds[["identifier", ax2]].drop_duplicates(subset='identifier', keep='last')
    
    if external_ds is None or ax1 is None or ax2 is None:
        raise ValueError("Something went wrong...")

    # Use the ax1 coordinate (coming from the 'identifier' column) to build the mapping.
    ax1_ids = ds.coords[ax1].values
    ax2_mapping = dict(zip(external_ds['identifier'], external_ds[ax2]))
    
    values = [ax2_mapping.get(ax1_i, "") for ax1_i in ax1_ids]
            
    # Attach the values as a new DataArray with the ax1
    ds[ax2] = xr.DataArray(
        values,
        dims=[ax1],
        coords={ax1: ax1_ids}
    )
    
    return ds

@post_processor
def replace(ds: xr.Dataset, params: Dict[str, Any]) -> xr.Dataset:
    
    external_ds = params.get("external_ds")
    replace_frm = params.get('from')
    replace_to = params.get('to')
        
    # Reindex the dataset to match the primary dataset.
    external_ds = external_ds.reindex_like(ds, method=None)
        
    valid_replace_frm = ~np.isnan(external_ds[replace_frm])
    current_values = ds[replace_to].copy()
    ds[replace_to] = xr.where(valid_replace_frm, external_ds.dlret, current_values)
    ds.attrs['patched'] = 1
    ds.attrs[f'num_{replace_frm}-{replace_to}_patched'] = int(valid_replace_frm.sum())
    
    return ds



