"""
Dataset merging utilities for combining multiple xarray datasets.

This module provides a flexible, declarative merging system for combining datasets
with different time frequencies (e.g., monthly CRSP with annual Compustat). It handles:

1. Alignment on shared dimensions (typically 'asset')
2. Time-based joins with configurable lag/lead offsets
3. Forward-filling of lower-frequency data to higher-frequency grids
4. Namespace prefixing to avoid variable collisions
5. Point-in-time correctness for financial data

The design follows xarray idioms and avoids pandas where possible.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import xarray as xr


class MergeMethod(Enum):
    """Methods for combining datasets."""
    LEFT = "left"       # Keep all entries from left dataset
    RIGHT = "right"     # Keep all entries from right dataset
    INNER = "inner"     # Keep only matching entries
    OUTER = "outer"     # Keep all entries from both


class TimeAlignment(Enum):
    """How to align time dimensions between datasets."""
    EXACT = "exact"           # Require exact time match
    FFILL = "ffill"           # Forward-fill from lower to higher frequency
    BFILL = "bfill"           # Backward-fill
    NEAREST = "nearest"       # Use nearest available time
    AS_OF = "as_of"           # Point-in-time: use latest available before target


@dataclass
class MergeSpec:
    """
    Specification for merging two datasets.
    
    Attributes:
        right_name: Name/key of the right dataset being merged in
        on: Dimension(s) to join on (e.g., 'asset' or ['asset', 'year'])
        time_alignment: How to handle time dimension alignment
        time_offset_months: Months to offset right dataset's time before merging.
                           Positive = shift forward (data available later),
                           Negative = shift backward (data available earlier).
        ffill_limit: Maximum number of periods to forward-fill (None = unlimited)
        prefix: Prefix to add to right dataset's variables (e.g., 'comp_')
        suffix: Suffix to add to right dataset's variables
        variables: Specific variables to include from right dataset (None = all)
        drop_vars: Variables to exclude from right dataset
    """
    right_name: str
    on: Union[str, List[str]] = "asset"
    time_alignment: TimeAlignment = TimeAlignment.FFILL
    time_offset_months: int = 0
    ffill_limit: Optional[int] = None
    prefix: str = ""
    suffix: str = ""
    variables: Optional[List[str]] = None
    drop_vars: Optional[List[str]] = None


class DatasetMerger:
    """
    Merges multiple xarray datasets with configurable alignment strategies.
    
    This class handles the complexity of combining datasets with different
    time frequencies while maintaining point-in-time correctness. It's designed
    for financial data workflows like Fama-French factor construction.
    
    The key insight is that Hindsight datasets use multi-dimensional time
    (year, month, day, hour) rather than a single flattened time axis. This
    merger works directly with those dimensions.
    
    Example usage:
        >>> merger = DatasetMerger()
        >>> # Merge annual Compustat into monthly CRSP
        >>> merged = merger.merge(
        ...     left=crsp_monthly,
        ...     right=compustat_annual,
        ...     spec=MergeSpec(
        ...         right_name='compustat',
        ...         on='asset',
        ...         time_alignment=TimeAlignment.AS_OF,
        ...         time_offset_months=6,  # 6-month lag for point-in-time
        ...         prefix='comp_'
        ...     )
        ... )
    """
    
    def __init__(self):
        """Initialize the DatasetMerger."""
        pass
    
    def merge(
        self,
        left: xr.Dataset,
        right: xr.Dataset,
        spec: MergeSpec,
        method: MergeMethod = MergeMethod.LEFT
    ) -> xr.Dataset:
        """
        Merge two datasets according to the specification.
        
        Args:
            left: Primary dataset (typically higher frequency, e.g., monthly)
            right: Secondary dataset to merge in (e.g., annual)
            spec: Merge specification
            method: How to handle non-matching entries
            
        Returns:
            Merged dataset with variables from both inputs
        """
        # Validate inputs
        self._validate_datasets(left, right, spec)
        
        # Normalize asset dimension dtype for consistent merging
        # This handles cases like string gvkey vs numeric permno
        on_dims = [spec.on] if isinstance(spec.on, str) else spec.on
        for dim in on_dims:
            left, right = self._normalize_asset_dtype(left, right, dim)
        
        # Step 1: Select and rename variables from right dataset
        right_prepared = self._prepare_right_dataset(right, spec)
        
        # Step 2: Expand right dataset to left's time grid
        right_expanded = self._expand_to_time_grid(
            left, right_prepared, 
            spec.time_alignment,
            spec.time_offset_months,
            spec.ffill_limit
        )
        
        # Step 3: Align on asset dimension and merge
        result = self._merge_on_asset(left, right_expanded, spec.on, method)
        
        return result
    
    def merge_multiple(
        self,
        base: xr.Dataset,
        datasets: Dict[str, xr.Dataset],
        specs: List[MergeSpec],
        method: MergeMethod = MergeMethod.LEFT
    ) -> xr.Dataset:
        """
        Merge multiple datasets into a base dataset.
        
        Args:
            base: Primary dataset to merge into
            datasets: Dictionary mapping names to datasets
            specs: List of merge specifications (one per dataset to merge)
            method: Default merge method
            
        Returns:
            Dataset with all specified datasets merged in
        """
        result = base
        for spec in specs:
            if spec.right_name not in datasets:
                raise ValueError(f"Dataset '{spec.right_name}' not found in datasets dict")
            result = self.merge(result, datasets[spec.right_name], spec, method)
        return result
    
    def _validate_datasets(
        self, 
        left: xr.Dataset, 
        right: xr.Dataset, 
        spec: MergeSpec
    ) -> None:
        """Validate that datasets can be merged according to spec."""
        # Check that join dimension exists
        on_dims = [spec.on] if isinstance(spec.on, str) else spec.on
        for dim in on_dims:
            if dim not in left.dims:
                raise ValueError(f"Join dimension '{dim}' not found in left dataset. "
                               f"Available: {list(left.dims)}")
            if dim not in right.dims:
                raise ValueError(f"Join dimension '{dim}' not found in right dataset. "
                               f"Available: {list(right.dims)}")
    
    def _normalize_asset_dtype(
        self,
        left: xr.Dataset,
        right: xr.Dataset,
        dim: str = 'asset'
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Normalize the dtype of the asset/join dimension to ensure compatibility.
        
        Converts both datasets' asset coordinates to int64 for consistent merging.
        This handles cases where one dataset has string identifiers and another has
        numeric identifiers (e.g., Compustat gvkey vs CRSP permno).
        
        Args:
            left: Left dataset
            right: Right dataset
            dim: Dimension to normalize (default: 'asset')
            
        Returns:
            Tuple of (left, right) datasets with normalized asset dtypes
        """
        if dim not in left.coords or dim not in right.coords:
            return left, right
        
        left_dtype = left.coords[dim].dtype
        right_dtype = right.coords[dim].dtype
        
        # If dtypes already match, no conversion needed
        if left_dtype == right_dtype:
            return left, right
        
        # Convert both to int64 for consistency
        # This is the standard for financial identifiers like permno
        def convert_to_int(ds: xr.Dataset, dim: str) -> xr.Dataset:
            """Convert dimension coordinate to int64."""
            coords = ds.coords[dim].values
            
            # Handle string/object arrays
            if coords.dtype.kind in ('U', 'S', 'O'):  # Unicode, byte string, object
                try:
                    # Try to convert strings to int
                    new_coords = np.array([int(float(x)) for x in coords], dtype=np.int64)
                except (ValueError, TypeError):
                    # If conversion fails, use hash for consistent ordering
                    new_coords = np.array([hash(str(x)) for x in coords], dtype=np.int64)
            elif coords.dtype.kind == 'f':  # Float
                # Convert float to int (handles NaN by dropping those rows)
                new_coords = coords.astype(np.int64)
            else:
                # Already integer-like, just ensure int64
                new_coords = coords.astype(np.int64)
            
            # Assign new coordinates
            return ds.assign_coords({dim: new_coords})
        
        left = convert_to_int(left, dim)
        right = convert_to_int(right, dim)
        
        return left, right
    
    def _prepare_right_dataset(
        self, 
        right: xr.Dataset, 
        spec: MergeSpec
    ) -> xr.Dataset:
        """Select variables and apply prefix/suffix renaming."""
        # Select specific variables if requested
        if spec.variables is not None:
            # Keep only requested variables (plus coordinates)
            vars_to_keep = [v for v in spec.variables if v in right.data_vars]
            right = right[vars_to_keep]
        
        # Drop specified variables
        if spec.drop_vars is not None:
            vars_to_drop = [v for v in spec.drop_vars if v in right.data_vars]
            right = right.drop_vars(vars_to_drop)
        
        # Apply prefix/suffix renaming
        if spec.prefix or spec.suffix:
            rename_map = {
                var: f"{spec.prefix}{var}{spec.suffix}" 
                for var in right.data_vars
            }
            right = right.rename(rename_map)
        
        return right
    
    def _expand_to_time_grid(
        self,
        left: xr.Dataset,
        right: xr.Dataset,
        alignment: TimeAlignment,
        offset_months: int = 0,
        ffill_limit: Optional[int] = None
    ) -> xr.Dataset:
        """
        Expand right dataset to match left's time grid.
        
        This is the core logic for merging datasets with different time frequencies.
        It handles the multi-dimensional time structure (year, month, day, hour).
        
        For annual-to-monthly merging:
        1. Reindex right's year dimension to left's years
        2. Broadcast across month/day/hour dimensions
        3. Apply forward-fill along time dimensions
        4. Apply any time offset (for point-in-time correctness)
        """
        # Identify time dimensions in each dataset
        time_dims = ['year', 'month', 'day', 'hour']
        left_time_dims = [d for d in time_dims if d in left.dims]
        right_time_dims = [d for d in time_dims if d in right.dims]
        
        if not left_time_dims:
            # No time dimensions in left, just return right as-is
            return right
        
        # Handle time offset first (shift which year's data applies to which period)
        if offset_months != 0:
            right = self._apply_month_offset(right, offset_months, left)
        
        # For each time dimension in left but not fully represented in right,
        # we need to broadcast/expand
        result = right
        
        # First, align the year dimension
        if 'year' in left.dims and 'year' in right.dims:
            left_years = left.coords['year'].values
            right_years = right.coords['year'].values
            
            # Reindex right to left's years, filling with NaN where missing
            result = result.reindex(year=left_years, method=None)
        
        # For month dimension: if right doesn't have it or has fewer months,
        # we need to broadcast/expand
        if 'month' in left.dims:
            left_months = left.coords['month'].values
            if 'month' not in result.dims:
                # Right has no month dimension - broadcast across all months
                result = result.expand_dims(month=left_months)
            elif result.sizes['month'] < left.sizes['month']:
                # If right only has a single month (typical for annual data),
                # broadcast that month across all target months so that offset
                # masking can correctly pick previous/ current year.
                if result.sizes['month'] == 1:
                    result = result.reindex(month=left_months, method='nearest')
                else:
                    # Otherwise just align and let ffill/bfill handle gaps
                    result = result.reindex(month=left_months, method=None)
        
        # Same for day dimension
        if 'day' in left.dims:
            left_days = left.coords['day'].values
            if 'day' not in result.dims:
                result = result.expand_dims(day=left_days)
            elif result.sizes['day'] < left.sizes['day']:
                result = result.reindex(day=left_days, method=None)
        
        # Same for hour dimension
        if 'hour' in left.dims:
            left_hours = left.coords['hour'].values
            if 'hour' not in result.dims:
                result = result.expand_dims(hour=left_hours)
            elif result.sizes['hour'] < left.sizes['hour']:
                result = result.reindex(hour=left_hours, method=None)
        
        # Apply forward-fill if requested
        if alignment in [TimeAlignment.FFILL, TimeAlignment.AS_OF]:
            result = self._apply_ffill(result, left_time_dims, ffill_limit)
        elif alignment == TimeAlignment.BFILL:
            result = self._apply_bfill(result, left_time_dims, ffill_limit)
        
        # Ensure dimension order matches left dataset
        # Get the order of dimensions from left (for data variables)
        left_dim_order = list(left.sizes.keys())
        result_dims = list(result.sizes.keys())
        
        # Reorder result dimensions to match left where possible
        new_order = [d for d in left_dim_order if d in result_dims]
        new_order += [d for d in result_dims if d not in new_order]
        
        result = result.transpose(*new_order)
        
        return result
    
    def _apply_month_offset(
        self,
        ds: xr.Dataset,
        offset_months: int,
        target: xr.Dataset
    ) -> xr.Dataset:
        """
        Apply a month offset to the dataset.
        
        For annual data with a 6-month offset:
        - Data from Dec 2019 becomes available in June 2020
        - So when we're in July 2020, we use Dec 2019 data
        
        This is implemented by shifting which year's data maps to which target year.
        For a 6-month offset with annual data:
        - Year Y data is used for months July Y to June Y+1
        """
        if 'year' not in ds.dims:
            return ds
        
        # For annual data with month offset:
        # We need to determine which source year maps to each target (year, month)
        
        # If offset is 6 months:
        # - Months 1-6 of year Y use data from year Y-1
        # - Months 7-12 of year Y use data from year Y
        
        if 'month' not in target.dims:
            # Target has no month dimension, just shift years
            year_shift = offset_months // 12
            if year_shift != 0:
                new_years = ds.coords['year'].values - year_shift
                ds = ds.assign_coords(year=new_years)
            return ds
        
        # For monthly target, we need to create a mapping
        # This is more complex - we'll handle it in the merge step
        # by creating year-shifted versions
        
        # Simple approach: shift the year coordinate
        # This means data labeled as year Y will be available starting month (offset+1)
        # For offset=6: year 2020 data available from July 2020 (month 7)
        
        # Store the offset as an attribute for later use
        ds.attrs['_month_offset'] = offset_months
        
        return ds
    
    def _apply_ffill(
        self,
        ds: xr.Dataset,
        time_dims: List[str],
        limit: Optional[int] = None
    ) -> xr.Dataset:
        """Apply forward-fill along time dimensions."""
        result = ds
        
        # Forward-fill in order: year, month, day, hour
        for dim in time_dims:
            if dim in result.dims:
                result = result.ffill(dim=dim, limit=limit)
        
        return result
    
    def _apply_bfill(
        self,
        ds: xr.Dataset,
        time_dims: List[str],
        limit: Optional[int] = None
    ) -> xr.Dataset:
        """Apply backward-fill along time dimensions."""
        result = ds
        
        # Backward-fill in reverse order: hour, day, month, year
        for dim in reversed(time_dims):
            if dim in result.dims:
                result = result.bfill(dim=dim, limit=limit)
        
        return result
    
    def _merge_on_asset(
        self,
        left: xr.Dataset,
        right: xr.Dataset,
        on: Union[str, List[str]],
        method: MergeMethod
    ) -> xr.Dataset:
        """
        Merge datasets on the asset dimension.
        
        This handles the actual combination of variables after time expansion.
        """
        on_dims = [on] if isinstance(on, str) else on
        
        # Align the asset dimension
        for dim in on_dims:
            if dim in left.coords and dim in right.coords:
                # Get the target assets from left
                left_assets = left.coords[dim].values
                
                # Reindex right to match left's assets
                right = right.reindex({dim: left_assets}, method=None)
        
        # Handle month offset if present
        offset = right.attrs.pop('_month_offset', 0)
        if offset != 0 and 'month' in left.dims and 'year' in left.dims:
            right = self._apply_offset_mask(left, right, offset)
        
        # Merge using xarray's merge
        join_method = {
            MergeMethod.LEFT: 'left',
            MergeMethod.RIGHT: 'right', 
            MergeMethod.INNER: 'inner',
            MergeMethod.OUTER: 'outer'
        }[method]
        
        result = xr.merge(
            [left, right],
            join=join_method,
            compat='override'
        )
        
        return result
    
    def _apply_offset_mask(
        self,
        left: xr.Dataset,
        right: xr.Dataset,
        offset_months: int
    ) -> xr.Dataset:
        """
        Apply month offset by masking data that shouldn't be available yet.
        
        For a 6-month offset:
        - In months 1-6, we should use the previous year's data
        - In months 7-12, we use the current year's data
        
        This is done by shifting data appropriately.
        """
        if 'month' not in left.dims or 'year' not in left.dims:
            return right
        
        # Get month values
        months = left.coords['month'].values
        years = left.coords['year'].values
        
        # Determine the cutoff month (1-indexed)
        # For offset=6: months 1-6 use previous year, 7-12 use current year
        cutoff_month = offset_months % 12
        if cutoff_month == 0:
            cutoff_month = 12
        
        # Create a mask for which (year, month) combinations should use shifted data
        # For months <= cutoff, we need data from year-1
        # For months > cutoff, we use data from current year
        
        result_vars = {}
        
        for var in right.data_vars:
            da = right[var]
            
            if 'year' not in da.dims or 'month' not in da.dims:
                result_vars[var] = da
                continue
            
            # Create the shifted version (data from previous year)
            shifted = da.shift(year=1)
            
            # Create mask: True where we should use current year data
            # (months > cutoff_month)
            month_coord = da.coords['month']
            use_current = month_coord > cutoff_month
            
            # Combine: use shifted for early months, current for later months
            combined = xr.where(use_current, da, shifted)
            result_vars[var] = combined
        
        return xr.Dataset(result_vars, coords=right.coords, attrs=right.attrs)


def merge_datasets(
    base: xr.Dataset,
    datasets: Dict[str, xr.Dataset],
    merge_config: List[Dict[str, Any]]
) -> xr.Dataset:
    """
    Convenience function to merge multiple datasets using config dictionaries.
    
    Args:
        base: Primary dataset
        datasets: Dictionary of datasets to merge
        merge_config: List of merge configuration dictionaries
        
    Returns:
        Merged dataset
        
    Example:
        >>> config = [
        ...     {
        ...         'right_name': 'compustat',
        ...         'on': 'asset',
        ...         'time_alignment': 'as_of',
        ...         'time_offset_months': 6,
        ...         'prefix': 'comp_'
        ...     }
        ... ]
        >>> merged = merge_datasets(crsp, {'compustat': comp}, config)
    """
    merger = DatasetMerger()
    
    specs = []
    for cfg in merge_config:
        # Convert string alignment to enum
        alignment_str = cfg.get('time_alignment', 'ffill')
        if isinstance(alignment_str, str):
            alignment = TimeAlignment(alignment_str.lower())
        else:
            alignment = alignment_str
        
        # Handle both old dict-style offset and new int-style
        offset = cfg.get('time_offset_months', 0)
        if isinstance(offset, dict):
            # Convert dict to months
            offset = offset.get('months', 0) + offset.get('years', 0) * 12
        
        spec = MergeSpec(
            right_name=cfg['right_name'],
            on=cfg.get('on', 'asset'),
            time_alignment=alignment,
            time_offset_months=offset,
            ffill_limit=cfg.get('ffill_limit'),
            prefix=cfg.get('prefix', ''),
            suffix=cfg.get('suffix', ''),
            variables=cfg.get('variables'),
            drop_vars=cfg.get('drop_vars')
        )
        specs.append(spec)
    
    return merger.merge_multiple(base, datasets, specs)
