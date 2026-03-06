"""
Helpers for making xarray datasets JIT-safe.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import xarray as xr


def prepare_for_jit(dataset: xr.Dataset) -> Tuple[xr.Dataset, Dict[str, xr.DataArray]]:
    """
    Remove non-numeric variables before JAX/JIT computation.

    Returns:
    - Dataset with numeric vars only
    - Context dict containing removed non-numeric variables
    """
    non_numeric_names = [
        name
        for name, da in dataset.data_vars.items()
        if not np.issubdtype(da.dtype, np.number)
    ]

    if not non_numeric_names:
        return dataset, {}

    # Keep removed vars in a side context and run JIT only on numeric payload.
    context = {name: dataset[name] for name in non_numeric_names}
    jit_ready_dataset = dataset.drop_vars(non_numeric_names)
    return jit_ready_dataset, context


def restore_from_jit(
    processed_dataset: xr.Dataset, context: Dict[str, xr.DataArray]
) -> xr.Dataset:
    """Restore non-numeric variables removed by `prepare_for_jit`."""
    if not context:
        return processed_dataset
    return processed_dataset.assign(**context)  # type: ignore[arg-type]
