"""
Rolling-window helper backed by JAX operations.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from src.data.core.operations import TimeSeriesOps


class Rolling:
    """Rolling window adapter for xarray Dataset/DataArray objects."""

    def __init__(
        self,
        obj: Union[xr.DataArray, xr.Dataset],
        dim: str,
        window: int,
        mask: Optional[jnp.ndarray] = None,
        indices: Optional[jnp.ndarray] = None,
    ):
        self.obj = obj
        self.dim = dim
        self.window = window

        if isinstance(obj, xr.Dataset):
            # Fast path: reuse precomputed mask state if present on coords.
            if mask is None and "mask" in obj.coords:
                mask = obj.coords["mask"].values
            if indices is None and "mask_indices" in obj.coords:
                indices = obj.coords["mask_indices"].values
            if mask is None or indices is None:
                # Fallback: compute per-asset valid timeline from current dataset.
                computed_mask, computed_indices = obj.dt.compute_mask()
                if mask is None:
                    mask = computed_mask
                if indices is None:
                    indices = computed_indices
        elif isinstance(obj, xr.DataArray):
            # Keep DataArray behavior strict so callers pass parent-derived state explicitly.
            if mask is None or indices is None:
                raise ValueError(
                    "DataArray rolling requires explicit mask/indices from parent dataset."
                )
        else:
            raise TypeError(f"Unsupported xarray object type: {type(obj)}")

        if mask is None or indices is None:
            raise ValueError("Rolling operation could not obtain valid mask and indices")

        self.mask = jnp.asarray(mask, dtype=jnp.bool_)
        self.indices = jnp.asarray(indices, dtype=jnp.int32)

    def reduce(
        self,
        func: Callable[[int, Any, jnp.ndarray, int], Tuple[jnp.ndarray, Any]],
        overlap_factor: Optional[float] = None,
        **func_kwargs,
    ) -> Union[xr.DataArray, xr.Dataset]:
        if isinstance(self.obj, xr.Dataset):
            # Apply rolling only to numeric variables and pass through non-numeric variables.
            rolled_data = {}
            for var_name, da in self.obj.data_vars.items():
                if np.issubdtype(da.dtype, np.number):
                    rolled_data[var_name] = Rolling(
                        da,
                        self.dim,
                        self.window,
                        mask=self.mask,
                        indices=self.indices,
                    ).reduce(func, overlap_factor=overlap_factor, **func_kwargs)
                else:
                    rolled_data[var_name] = da
            return xr.Dataset(rolled_data, coords=self.obj.coords, attrs=self.obj.attrs)

        if isinstance(self.obj, xr.DataArray):
            return self._reduce_dataarray(func, overlap_factor, **func_kwargs)

        raise TypeError("Unsupported xarray object type.")

    def _reduce_dataarray(
        self,
        func: Callable[[int, Any, jnp.ndarray, int], Tuple[jnp.ndarray, Any]],
        overlap_factor: Optional[float] = None,
        **func_kwargs,
    ) -> xr.DataArray:
        if not np.issubdtype(self.obj.dtype, np.number):
            return self.obj

        expected_time_dims = ["year", "month", "day"]
        if "hour" in self.obj.dims:
            expected_time_dims.append("hour")

        # Only run this path for time rolling on panel-style dimensions.
        if self.dim != "time" or not set(expected_time_dims).issubset(self.obj.dims):
            return self.obj

        # Stack calendar dimensions into one axis: (T, N, ...).
        stacked_obj = self.obj.stack(time_index=tuple(expected_time_dims)).transpose(
            "time_index", "asset", ...
        )

        # Keep last dimension as singleton so TimeSeriesOps.u_roll sees (T, N, C).
        data = jnp.asarray(stacked_obj.data, dtype=jnp.float64)[..., None]
        idx = self.indices
        idx_valid = idx >= 0
        idx_safe = jnp.where(idx_valid, idx, 0)

        def _gather_one(col_t1, idx_t):
            # Build compressed per-asset timeline (invalid rows become zeroed padding).
            gathered = jnp.take(col_t1, idx_t, axis=0)
            return jnp.where((idx_t >= 0)[:, None], gathered, 0.0)

        # Vectorize gather across asset axis.
        compressed = jax.vmap(_gather_one, in_axes=(1, 1), out_axes=1)(data, idx_safe)

        # Run the rolling kernel on compressed timelines.
        rolled_result = TimeSeriesOps.u_roll(
            data=compressed,
            window_size=self.window,
            func=func,
            overlap_factor=overlap_factor,
            **func_kwargs,
        )

        t_full = data.shape[0]

        def _scatter_one(tidx_t, upd_t1):
            valid = tidx_t >= 0
            tidx_safe = jnp.where(valid, tidx_t, 0)

            vals = jnp.zeros((t_full, 1), dtype=upd_t1.dtype)
            wts = jnp.zeros((t_full,), dtype=jnp.float64)

            # Accumulate updates back to original timeline positions.
            vals = vals.at[tidx_safe].add(jnp.where(valid[:, None], upd_t1, 0.0))
            wts = wts.at[tidx_safe].add(valid.astype(jnp.float64))
            return vals, wts

        # Scatter each asset independently, then normalize by write counts.
        vals_cols, wts_cols = jax.vmap(_scatter_one, in_axes=(1, 1), out_axes=(1, 1))(
            idx, rolled_result
        )

        rolled_full = jnp.where(
            (wts_cols > 0)[..., None], vals_cols / wts_cols[..., None], jnp.nan
        )
        rolled_full = rolled_full[..., 0]

        rolled_da = stacked_obj.copy(data=rolled_full)
        # Restore original calendar dimensions.
        return rolled_da.unstack("time_index")

    def mean(self, **kwargs) -> Union[xr.DataArray, xr.Dataset]:
        from src.data.core.operations.standard import mean

        return self.reduce(mean, **kwargs)

    def sum(self, **kwargs) -> Union[xr.DataArray, xr.Dataset]:
        from src.data.core.operations.standard import sum_func

        return self.reduce(sum_func, **kwargs)

    def std(self, **kwargs) -> Union[xr.DataArray, xr.Dataset]:
        del kwargs
        raise NotImplementedError("Rolling.std is not implemented yet.")

    def median(self, **kwargs) -> Union[xr.DataArray, xr.Dataset]:
        from src.data.core.operations.standard import median

        return self.reduce(median, **kwargs)
