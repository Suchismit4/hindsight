"""
Concrete processor implementations for the data handling pipeline.

This module contains the actual processor implementations that follow the 
ProcessorContract interface. These processors handle common data transformations
such as cross-sectional z-scoring, forward filling, and formula evaluation.

Each processor is designed to work with xarray datasets and supports both
stateful (fit/transform) and stateless operation modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import jax

import sys

from src.data.ast.manager import FormulaManager
from src.data.core import prepare_for_jit, restore_from_jit
from src.data.ast.functions import get_function_context
from .core import ProcessorContract, ProcessorState


@dataclass
class Processor(ProcessorContract):
    """
    Base processor implementation with common functionality.
    
    This class provides the basic structure for processors following the
    ProcessorContract interface. All concrete processors should inherit from
    this class.
    
    Parameters
    ----------
    name : str
        Unique name identifier for this processor instance
    """
    name: str

    def fit(self, ds: xr.Dataset) -> ProcessorState:
        """
        Learn parameters from the input dataset.
        
        Default implementation raises NotImplementedError. Subclasses must override.
        
        Parameters
        ----------
        ds : xr.Dataset
            Input dataset to learn parameters from
            
        Returns
        -------
        ProcessorState
            Opaque state object containing learned parameters
            
        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses
        """
        raise NotImplementedError

    def transform(
        self,
        ds: xr.Dataset,
        state: Optional[ProcessorState] = None,
    ) -> xr.Dataset:
        """
        Apply the transformation to the input dataset.
        
        Default implementation raises NotImplementedError. Subclasses must override.
        
        Parameters
        ----------
        ds : xr.Dataset
            Input dataset to transform
        state : ProcessorState, optional
            State object returned by ``fit``. Processors should gracefully handle
            ``None`` when they can operate in stateless mode.
            
        Returns
        -------
        xr.Dataset
            Transformed dataset
            
        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses
        """
        raise NotImplementedError


@dataclass
class CSZScoreState:
    """
    Container for per-variable normalization statistics.
    
    Attributes
    ----------
    variables :
        Tuple of variable names the state covers.
    means :
        Mapping ``var -> xr.DataArray`` holding per-asset means.
    stds :
        Mapping ``var -> xr.DataArray`` holding per-asset standard deviations.
    """
    variables: Tuple[str, ...]
    means: Dict[str, xr.DataArray]
    stds: Dict[str, xr.DataArray]


@dataclass
class CSZScore(Processor):
    """
    Cross-sectional z-score normalization processor.
    
    The processor estimates per-asset means and standard deviations from the
    training data and reuses those statistics when transforming inference data.
    The learned state is a lightweight dataclass instead of an xr.Dataset so it
    can be stored without carrying time coordinates that would later conflict.
    """
    vars: Optional[List[str]] = None
    out_suffix: str = "_csz"
    eps: float = 1e-8

    def fit(self, ds: xr.Dataset) -> CSZScoreState:
        """
        Learn per-asset statistics for each configured variable.
        """
        vars_to_use = self._select_variables(ds)
        asset_dim = self._asset_dim(ds)

        means: Dict[str, xr.DataArray] = {}
        stds: Dict[str, xr.DataArray] = {}

        for var in vars_to_use:
            if var not in ds:
                continue

            reduce_dims = [dim for dim in ds[var].dims if dim != asset_dim]
            if reduce_dims:
                mu = ds[var].mean(dim=reduce_dims, skipna=True)
                sd = ds[var].std(dim=reduce_dims, skipna=True)
            else:
                mu = ds[var].copy(deep=True)
                sd = xr.zeros_like(mu)

            means[var] = mu.astype(np.float64)
            stds[var] = sd.astype(np.float64).fillna(0.0)

        return CSZScoreState(
            variables=tuple(vars_to_use),
            means=means,
            stds=stds,
        )

    def transform(
        self,
        ds: xr.Dataset,
        state: Optional[ProcessorState] = None,
    ) -> xr.Dataset:
        """
        Apply learned statistics to normalize variables.
        """
        resolved_state = self._ensure_state(ds, state)
        asset_dim = self._asset_dim(ds)

        out = ds.copy()
        target_assets = ds.coords.get(asset_dim)

        for var in resolved_state.variables:
            if var not in ds:
                continue

            mu = resolved_state.means.get(var)
            sd = resolved_state.stds.get(var)
            if mu is None or sd is None:
                continue

            if target_assets is not None and asset_dim in mu.dims:
                mu = mu.reindex({asset_dim: target_assets}, copy=False)
            if target_assets is not None and asset_dim in sd.dims:
                sd = sd.reindex({asset_dim: target_assets}, copy=False)

            normalized = (ds[var] - mu) / (sd + self.eps)
            out[f"{var}{self.out_suffix}"] = normalized

        return out

    def _select_variables(self, ds: xr.Dataset) -> List[str]:
        vars_candidate = self.vars or [
            name for name, da in ds.data_vars.items() if np.issubdtype(da.dtype, np.number)
        ]
        asset_dim = self._asset_dim(ds)
        return [name for name in vars_candidate if name in ds and asset_dim in ds[name].dims]

    @staticmethod
    def _asset_dim(ds: xr.Dataset) -> str:
        if "asset" not in ds.dims:
            raise ValueError("CSZScore requires datasets with an 'asset' dimension")
        return "asset"

    def _ensure_state(
        self,
        ds: xr.Dataset,
        state: Optional[ProcessorState],
    ) -> CSZScoreState:
        if state is None:
            return self.fit(ds)
        if isinstance(state, CSZScoreState):
            return state
        if isinstance(state, xr.Dataset):
            return self._migrate_legacy_state(state)
        raise TypeError(f"Unsupported CSZScore state type: {type(state)!r}")

    def _migrate_legacy_state(self, state_ds: xr.Dataset) -> CSZScoreState:
        """
        Convert the historical xr.Dataset-based state into the new dataclass form.
        """
        prefix_mu = f"{self.name}_mu__"
        prefix_sd = f"{self.name}_sd__"

        means: Dict[str, xr.DataArray] = {}
        stds: Dict[str, xr.DataArray] = {}
        variables: List[str] = []

        for key, da in state_ds.data_vars.items():
            if key.startswith(prefix_mu):
                var = key[len(prefix_mu):]
                means[var] = da.astype(np.float64)
                variables.append(var)
            elif key.startswith(prefix_sd):
                var = key[len(prefix_sd):]
                stds[var] = da.astype(np.float64).fillna(0.0)
                if var not in variables:
                    variables.append(var)

        return CSZScoreState(
            variables=tuple(variables),
            means=means,
            stds=stds,
        )


@dataclass
class PerAssetFFill(Processor):
    """
    Per-asset forward fill processor.
    
    This processor performs forward-fill (last observation carried forward) for each
    asset independently along chronological time. It is stateless and writes results
    in-place by default.
    
    Parameters
    ----------
    name : str, default "ffill"
        Processor name identifier
    vars : list of str, optional
        List of variables to forward fill. If None, applies to all numeric variables.
        
    Notes
    -----
    The processor is stateless; ``fit`` simply returns ``None``.
    """
    name: str = "ffill"
    vars: Optional[List[str]] = None

    def fit(self, ds: xr.Dataset) -> None:
        """Stateless processor – returns ``None``."""
        return None

    def transform(
        self,
        ds: xr.Dataset,
        state: Optional[ProcessorState] = None,
    ) -> xr.Dataset:
        """
        Apply forward fill transformation per asset.
        
        Forward fills missing values for each variable and asset independently
        along the time dimension.
        
        Parameters
        ----------
        ds : xr.Dataset
            Input dataset to forward fill
        state : ProcessorState, optional
            Unused for this stateless processor
            
        Returns
        -------
        xr.Dataset
            Dataset with forward-filled values
        """
        vars_to_use = self.vars or [
            k for k, v in ds.data_vars.items() if np.issubdtype(v.dtype, np.number)
        ]
        time_dims = self._time_dims(ds)

        out = ds.copy()
        for var in vars_to_use:
            if var not in out:
                continue
            values = out[var]
            for dim in time_dims:
                values = values.ffill(dim=dim)
            out[var] = values
        return out

    @staticmethod
    def _time_dims(ds: xr.Dataset) -> List[str]:
        ordered_dims = ["year", "month", "day", "hour", "time"]
        return [dim for dim in ordered_dims if dim in ds.dims]


@dataclass
class FormulaEval(Processor):
    """
    Formula evaluation processor using the AST system.
    
    This processor compiles and evaluates a set of YAML-defined formulas using the
    FormulaManager and AST system. It can optionally use JAX JIT compilation for
    performance optimization.
    
    Parameters
    ----------
    name : str
        Processor name identifier
    formula_configs : dict
        Dictionary mapping formula_name -> list of config dicts
        Example: {"sma": [{"window": 100}, {"window": 200}], "rsi": [{"window": 14}]}
    static_context : dict, optional
        Dictionary of constants/functions to provide to formulas
        Example: {"price": "close"} plus get_function_context()
    use_jit : bool, default True
        Whether to wrap evaluation in jax.jit for performance
    defs_dir : str, optional
        Optional directory to load custom formula YAML files
    assign_in_place : bool, default True
        If True, merge results into dataset; else results are namespaced with prefix
    prefix : str, optional
        Optional variable name prefix when assign_in_place=False
        
    Notes
    -----
    The compiled callable is cached on the instance (not serialized in state) for
    performance. The processor builds a single compiled function that:
    1) Prepares dataset for JIT compilation
    2) Calls FormulaManager.evaluate_bulk with context
    3) Returns an xr.Dataset of computed results
    """
    formula_configs: Dict[str, List[Dict[str, Any]]]
    static_context: Optional[Dict[str, Any]] = None
    use_jit: bool = True
    defs_dir: Optional[str] = None
    assign_in_place: bool = True
    prefix: Optional[str] = None

    # compiled callable cached on the instance (not serialized in state)
    _compiled: Optional[Callable[[xr.Dataset], xr.Dataset]] = field(default=None, init=False, repr=False)
    _manager: Optional[FormulaManager] = field(default=None, init=False, repr=False)

    def _build_compiled(self) -> Callable[[xr.Dataset], xr.Dataset]:
        """
        Build a single compiled function for formula evaluation.
        
        Creates a FormulaManager instance, compiles all formulas, and returns
        a function that can evaluate the configured formulas on datasets.
        
        Returns
        -------
        callable
            Compiled function that takes xr.Dataset and returns xr.Dataset of results
        """
        # Create manager and load definitions only once
        mgr = FormulaManager(definitions_dir=self.defs_dir) if self.defs_dir else FormulaManager()
        mgr.compile_all_formulas_as_functions()  # primes internal caches
        self._manager = mgr

        # Static context: builtins + user statics
        base_ctx = {"price": "close", **get_function_context()}
        if self.static_context:
            base_ctx.update(self.static_context)

        # A pure Python closure that will be optionally wrapped by jax.jit
        def _eval(ds_in: xr.Dataset) -> xr.Dataset:
            # Strip non-jittable objects
            ds_jit, nonnum_ctx = prepare_for_jit(ds_in)
            ctx = {"_dataset": ds_jit, **base_ctx}
            out = mgr.evaluate_bulk(self.formula_configs, ctx, validate_inputs=True, jit_compile=True)
            # Bring coords and dtypes back
            return restore_from_jit(out, nonnum_ctx)

        return jax.jit(_eval) if self.use_jit else _eval

    def fit(self, ds: xr.Dataset) -> None:
        """
        Stateless fit that ensures compiled function is ready.
        
        This processor is stateless in terms of learned parameters, but it does
        compile the formula evaluation function during fit for efficiency.
        
        Parameters
        ----------
        ds : xr.Dataset
            Input dataset (used to trigger compilation)
            
        Returns
        -------
        None
        """
        if self._compiled is None:
            self._compiled = self._build_compiled()
        return None

    def transform(
        self,
        ds: xr.Dataset,
        state: Optional[ProcessorState] = None,
    ) -> xr.Dataset:
        """
        Run the compiled formula evaluator and merge results.
        
        Evaluates all configured formulas on the input dataset and merges the
        results according to the processor configuration.
        
        Parameters
        ----------
        ds : xr.Dataset
            Input dataset to evaluate formulas on
        state : ProcessorState, optional
            Unused placeholder to satisfy the processor contract.
            
        Returns
        -------
        xr.Dataset
            Dataset with formula results merged in
        """
        if self._compiled is None:
            # if called without fit, build on-the-fly
            self._compiled = self._build_compiled()

        res: xr.Dataset = self._compiled(ds)

        # Optionally namespace variable names
        if not self.assign_in_place and self.prefix:
            res = res.rename({k: f"{self.prefix}{k}" for k in res.data_vars})

        # Merge into ds; xarray handles alignment by coords
        return ds.merge(res, compat="override")


@dataclass
class CrossSectionalSort(Processor):
    """
    Cross-sectional sorting processor.
    
    Sorts assets into portfolios based on a signal variable at each time step.
    Supports scoping (e.g., calculating breakpoints on a subset of assets like NYSE)
    and assigning those breakpoints to the entire universe.
    
    Parameters
    ----------
    signal : str
        Name of the signal variable to sort on.
    n_bins : int
        Number of quantile bins to create.
    name : str, default "sort"
        Processor name identifier.
    scope : str, optional
        Name of the boolean mask variable to define the scope for breakpoint calculation.
        If None, all assets are used.
    labels : list, optional
        List of labels for the bins. If None, integer labels (0 to n_bins-1) are used.
    """
    signal: str = ""
    n_bins: int = 0
    name: str = "sort"
    scope: Optional[str] = None
    labels: Optional[List[Any]] = None

    def fit(self, ds: xr.Dataset) -> None:
        """Stateless processor."""
        return None

    def transform(
        self,
        ds: xr.Dataset,
        state: Optional[ProcessorState] = None,
    ) -> xr.Dataset:
        """
        Apply cross-sectional sorting.
        """
        if self.signal not in ds:
            raise ValueError(f"Signal variable '{self.signal}' not found in dataset")
        
        if self.scope and self.scope not in ds:
            raise ValueError(f"Scope variable '{self.scope}' not found in dataset")

        # Define the core function for apply_ufunc
        def _sort_step(signal_data, scope_mask=None):
            # signal_data: 1D array of signal values for one time step
            # scope_mask: 1D boolean array (or None)
            
            # Create result array initialized with NaNs
            # We use float to accommodate NaNs, will cast to object/int later if needed
            result = np.full_like(signal_data, np.nan, dtype=np.float64)
            
            # Identify valid data (not NaN)
            valid_mask = ~np.isnan(signal_data)
            
            # If scope is provided, combine with valid mask
            if scope_mask is not None:
                # Ensure scope_mask is boolean and handle NaNs in it
                scope_mask = np.nan_to_num(scope_mask, nan=0).astype(bool)
                calc_mask = valid_mask & scope_mask
            else:
                calc_mask = valid_mask
                
            # If no valid data to calculate breakpoints, return all NaNs
            if not np.any(calc_mask):
                return result
            
            # Get data for breakpoint calculation
            calc_data = signal_data[calc_mask]
            
            # Calculate quantiles
            # We use linspace to get n_bins+1 edges from 0 to 1
            # e.g., for 2 bins: [0, 0.5, 1.0]
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            breakpoints = np.nanquantile(calc_data, quantiles)
            
            # Handle edge case where all values are the same or breakpoints are not unique
            # For standard factor construction, we usually want unique bins.
            # But np.digitize requires monotonic increasing.
            # Let's ensure strict monotonicity for digitize, or use pd.cut style logic.
            # For simplicity and robustness, we'll use a searchsorted approach or digitize.
            
            # Make breakpoints unique to avoid empty bins if many values are identical?
            # FF usually implies strict quantiles.
            # Let's stick to standard quantiles.
            
            # Apply bins to ALL valid data (not just the scope)
            # We only assign bins to assets that have a valid signal
            target_data = signal_data[valid_mask]
            
            # np.digitize returns indices 1..N for bins defined by edges.
            # We want 0..N-1.
            # breakpoints has N+1 items.
            # bins: [b0, b1), [b1, b2), ...
            # right=False (default) -> bins include left edge, exclude right.
            # But we want to include the max value in the last bin.
            
            # Adjust breakpoints for robust inclusion
            breakpoints[0] -= 1e-9 # Extend lower bound slightly
            breakpoints[-1] += 1e-9 # Extend upper bound slightly
            
            # Digitize
            bins = np.digitize(target_data, breakpoints) - 1
            
            # Clip to ensure range 0..n_bins-1 (handle precision issues)
            bins = np.clip(bins, 0, self.n_bins - 1)
            
            # Assign back to result
            result[valid_mask] = bins
            
            return result

        # Prepare arguments for apply_ufunc
        args = [ds[self.signal]]
        input_core_dims = [[self._asset_dim(ds)]]
        
        if self.scope:
            args.append(ds[self.scope])
            input_core_dims.append([self._asset_dim(ds)])
            
        # Apply the function
        # vectorize=True allows the function to work on numpy arrays
        # dask='parallelized' allows it to work with dask chunks if needed (though we use eager execution mostly)
        out_da = xr.apply_ufunc(
            _sort_step,
            *args,
            input_core_dims=input_core_dims,
            output_core_dims=[[self._asset_dim(ds)]],
            vectorize=True, # Loop over non-core dims (time) in python if needed, or let numpy handle it
            dask='parallelized',
            output_dtypes=[np.float64]
        )
        
        # Assign result to dataset
        # Use a descriptive name
        out_name = f"{self.signal}_port"
        ds[out_name] = out_da
        
        return ds

    @staticmethod
    def _asset_dim(ds: xr.Dataset) -> str:
        if "asset" not in ds.dims:
            raise ValueError("CrossSectionalSort requires datasets with an 'asset' dimension")
        return "asset"


@dataclass
class PortfolioReturns(Processor):
    """
    Portfolio returns aggregator using pure xarray operations.
    
    Calculates the returns of portfolios defined by grouping variables.
    Supports value-weighting and equal-weighting.
    
    Parameters
    ----------
    groupby : list of str
        List of variable names to group by (e.g. ['sz_port', 'bm_port']).
    returns_var : str, default "ret"
        Name of the returns variable.
    weights_var : str, optional
        Name of the weights variable (e.g. "me_lag"). If None, equal weighting is used.
    name : str, default "port_ret"
        Processor name identifier.
    """
    groupby: List[str] = field(default_factory=list)
    returns_var: str = "ret"
    weights_var: Optional[str] = None
    name: str = "port_ret"

    @staticmethod
    def _asset_dim(ds: xr.Dataset) -> str:
        if "asset" not in ds.dims:
            raise ValueError("PortfolioReturns requires datasets with an 'asset' dimension")
        return "asset"

    def fit(self, ds: xr.Dataset) -> None:
        """Stateless processor."""
        return None

    def transform(
        self,
        ds: xr.Dataset,
        state: Optional[ProcessorState] = None,
    ) -> xr.Dataset:
        """
        Calculate portfolio returns using xarray operations.
        """
        if self.returns_var not in ds:
            raise ValueError(f"Returns variable '{self.returns_var}' not found")
        
        for g in self.groupby:
            if g not in ds:
                raise ValueError(f"Grouping variable '{g}' not found")
            
        ret = ds[self.returns_var]
        
        # Prepare weights
        if self.weights_var:
            if self.weights_var not in ds:
                raise ValueError(f"Weights variable '{self.weights_var}' not found")
            weights = ds[self.weights_var]
        else:
            # Equal weighting: weights = 1 where ret is valid
            weights = xr.ones_like(ret)
            
        # Ensure weights are valid where returns are valid
        valid_mask = ~np.isnan(ret) & ~np.isnan(weights)
        
        # Apply mask
        ret_masked = ret.where(valid_mask)
        weights_masked = weights.where(valid_mask)
        
        # Create a single grouping variable
        # If multiple groupby vars, we need to combine them or use multi-dimensional groupby
        # Xarray's groupby works best with a single variable.
        # We can construct a composite key.
        
        if len(self.groupby) == 1:
            group_var = ds[self.groupby[0]]
            # We treat this as a special case of the general logic for simplicity
            # or just implement the loop here.
            
            # Get max bin
            max_bin = int(group_var.max().item())
            num_groups = max_bin + 1
            
            results = []
            asset_dim = self._asset_dim(ds)
            
            for g_id in range(num_groups):
                mask_g = (group_var == g_id) & valid_mask
                
                num_g = (ret * weights).where(mask_g).sum(dim=asset_dim)
                den_g = weights.where(mask_g).sum(dim=asset_dim)
                
                res_g = num_g / den_g.where(den_g != 0)
                results.append(res_g)
                
            # Concat
            # Dimension name should be the group variable name + "_dim"
            dim_name = f"{self.groupby[0]}_dim"
            port_ret = xr.concat(results, dim=dim_name)
            port_ret = port_ret.assign_coords({dim_name: np.arange(num_groups)})
            
        else:
            # Multiple grouping variables.
            # We need to stack or combine them.
            # Approach: Create a temporary composite coordinate.
            # But we want the output to have dimensions corresponding to the groups.
            # e.g. (time, sz_port, bm_port)
            
            # Let's use a composite integer ID if possible, assuming integer bins.
            # Or use string composition for generality.
            
            # Check if variables are effectively discrete integers
            # We'll assume they are discrete categories.
            
            # Create a stacked DataArray of the grouping variables
            # We want to group by the combination of (g1, g2, ...) at each time step.
            # Actually, we want to aggregate over 'asset' for each unique combination of (g1, g2...)
            
            # Ideally: ds.groupby(g1, g2).mean() -> Not supported directly in xarray yet.
            
            # Workaround:
            # 1. Stack the grouping variables into a single variable
            #    e.g. tuple(v1, v2) or just use a MultiIndex?
            #    Xarray doesn't support grouping by MultiIndex easily in this way on DataArrays.
            
            # 2. Use arithmetic combination if integer-like.
            #    This is efficient.
            #    We need to know the cardinality or just use a safe multiplier.
            #    But we don't know max values easily without compute.
            
            # 3. Use string formatting? Slow.
            
            # 4. Iterative groupby?
            #    Group by g1, then for each group, group by g2?
            #    Can be done but complex to reconstruct.
            
            # Let's try the MultiIndex approach via stacking.
            # We can stack the dataset over 'asset' and then groupby?
            # No, we want to preserve time.
            
            # Best approach for "Tensor" style:
            # If we assume the grouping vars are integer bins 0..N-1:
            # We can compute a linear index.
            # To do this safely, we need the max value of each group var.
            # We can compute max() eagerly.
            
            # Let's try to infer max bins.
            # If they are from CrossSectionalSort, they are 0..n_bins-1.
            # We can check attrs or just compute max.
            
            # Let's use a robust method:
            # Create a temporary Dataset with just the grouping vars.
            # Stack them?
            
            # Let's go with the linear index method, assuming they are integers.
            # It's fast and "tensor-like".
            
            group_arrays = [ds[g].fillna(-1).astype(int) for g in self.groupby]
            
            # Calculate strides
            # We need to know the range of each.
            # We'll compute max on the fly.
            maxs = [da.max().item() for da in group_arrays]
            
            # If any max is huge, this fails. But for portfolios it's usually small (2, 3, 5, 10).
            # Check bounds?
            if any(m > 1000 for m in maxs):
                 # Fallback to something else or raise warning?
                 # For now assume small integer bins.
                 pass
                 
            # Compute strides
            # ID = g0 + g1*stride0 + g2*stride0*stride1 ...
            # We want the last one to be contiguous? Doesn't matter.
            
            composite_id = xr.zeros_like(group_arrays[0])
            current_stride = 1
            
            # We also need to track the shape for unstacking later
            dims_shape = []
            for m in maxs:
                dims_shape.append(int(m) + 1)
            
            # Compute composite ID matching MultiIndex.from_product order (C-order)
            # The last variable should vary fastest (stride 1)
            # The first variable should vary slowest
            
            composite_id = xr.zeros_like(group_arrays[0])
            current_stride = 1
            
            for i in reversed(range(len(group_arrays))):
                da = group_arrays[i]
                dim_size = dims_shape[i]
                
                composite_id = composite_id + da * current_stride
                current_stride *= dim_size
                
            # Mask where any group var was NaN (value -1)
            for g in self.groupby:
                valid_mask = valid_mask & ds[g].notnull()
                
            # Apply updated mask
            weighted_ret = (ret * weights).where(valid_mask)
            weights_masked = weights.where(valid_mask)
            composite_id_masked = composite_id.where(valid_mask)
            
            # Iterate over all possible composite IDs to compute weighted sums
            # This preserves time dimensions by only reducing 'asset'
            
            # We know the range is 0 to current_stride - 1
            num_groups = current_stride
            results = []
            
            # We need to handle the single group case similarly or just use this general logic
            # If single group, composite_id is just the group var
            
            for g_id in range(num_groups):
                # Mask for this group
                # We use == comparison.
                mask_g = (composite_id_masked == g_id)
                
                # Sum over asset dimension
                # We need to know the asset dim name.
                asset_dim = self._asset_dim(ds)
                
                # Check if mask has any True values?
                # If we sum over asset, we get time series.
                # If a group is empty at a time step, sum is 0.
                
                num_g = weighted_ret.where(mask_g).sum(dim=asset_dim)
                den_g = weights_masked.where(mask_g).sum(dim=asset_dim)
                
                res_g = num_g / den_g.where(den_g != 0)
                results.append(res_g)
                
            # Concat along a new dimension 'composite_id'
            port_ret_full = xr.concat(results, dim="composite_id")
            port_ret_full = port_ret_full.assign_coords(composite_id=np.arange(num_groups))
            
            # Create MultiIndex for unstacking
            import pandas as pd
            ranges = [range(s) for s in dims_shape]
            multi_idx = pd.MultiIndex.from_product(ranges, names=self.groupby)
            
            # Assign coordinates
            port_ret_full = port_ret_full.assign_coords(composite_id=multi_idx)
            
            # Unstack
            port_ret = port_ret_full.unstack("composite_id")
            
        # Rename dimensions to avoid conflict with input variables
        rename_dims = {g: f"{g}_dim" for g in self.groupby if g in port_ret.dims}
        port_ret = port_ret.rename(rename_dims)
        
        # Name the output
        out_name = f"port_ret_{'_'.join(self.groupby)}"
        port_ret.name = out_name
        
        ds[out_name] = port_ret
        
        return ds
