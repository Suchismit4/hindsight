"""
Global cache manager for content-addressable hierarchical caching.

This module provides the core cache management infrastructure for the pipeline
system, implementing content-addressable storage with automatic dependency tracking
and cache invalidation.
"""

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import xarray as xr

from .stages import CacheStage
from .metadata import MetadataManager, CacheMetadata


class GlobalCacheManager:
    """
    Content-addressable cache manager for the entire pipeline.
    
    This manager implements a hierarchical caching system where each cache entry
    is identified by a content-addressable key computed from its configuration
    and parent dependencies. This ensures automatic cache reuse when pipeline
    stages overlap across different specifications.
    
    Key Features:
    - Content-addressable storage (cache key = hash of config + parent keys)
    - Hierarchical cache levels (L1-L4)
    - Automatic dependency tracking
    - Cache invalidation with downstream propagation
    - Get-or-compute pattern for transparent caching
    
    Attributes:
        cache_root: Root directory for all cache storage
        metadata_manager: Manager for cache metadata operations
    """
    
    def __init__(self, cache_root: Optional[str] = None):
        """
        Initialize the global cache manager.
        
        Args:
            cache_root: Root directory for cache storage. 
                       Defaults to ~/data/hindsight_cache
        """
        if cache_root is None:
            cache_root = Path.home() / "data" / "hindsight_cache"
        
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        
        self.metadata_manager = MetadataManager(self.cache_root)
    
    def get_or_compute(
        self,
        stage: CacheStage,
        config: Dict[str, Any],
        parent_keys: List[str],
        compute_fn: Callable[[], Any],
        force_recompute: bool = False
    ) -> Tuple[Any, str]:
        """
        Get cached result or compute and cache it.
        
        This is the main interface for using the cache system. It implements
        the get-or-compute pattern: try to load from cache, and if not found,
        compute the result and save it to cache.
        
        Args:
            stage: Cache stage for this computation
            config: Configuration dictionary for this computation
            parent_keys: List of cache keys this computation depends on
            compute_fn: Function to call if cache miss occurs
            force_recompute: If True, bypass cache and recompute
            
        Returns:
            Tuple of (result, cache_key)
            
        Example:
            >>> def load_data():
            ...     return dataset
            >>> data, key = cache_mgr.get_or_compute(
            ...     stage=CacheStage.L1_RAW,
            ...     config={"provider": "wrds", "dataset": "crsp"},
            ...     parent_keys=[],
            ...     compute_fn=load_data
            ... )
        """
        # Compute content-addressable key
        cache_key = self._compute_key(stage, config, parent_keys)
        
        # Try to load from cache (unless force recompute)
        if not force_recompute:
            cache_path = self._get_cache_path(stage, cache_key)
            
            if cache_path.exists():
                try:
                    result = self._load(cache_path, stage)
                    self.metadata_manager.update_access(cache_key, stage)
                    print(f"Cache hit: {stage.value}/{cache_key[:12]}...")
                    return result, cache_key
                except Exception as e:
                    print(f"Warning: Failed to load cache {cache_key}: {e}")
                    # Fall through to recompute
        
        # Cache miss or load failed - compute
        print(f"Cache miss: {stage.value}/{cache_key[:12]}... - computing...")
        result = compute_fn()
        
        # Save to cache
        try:
            cache_path = self._save(result, stage, cache_key)
            
            # Save metadata
            size_bytes = cache_path.stat().st_size if cache_path.exists() else 0
            self.metadata_manager.save_metadata(
                cache_key=cache_key,
                stage=stage,
                config=config,
                parent_keys=parent_keys,
                size_bytes=size_bytes
            )
            
            print(f"Cached: {stage.value}/{cache_key[:12]}... ({size_bytes / 1024 / 1024:.2f} MB)")
            
        except Exception as e:
            print(f"Warning: Failed to save cache {cache_key}: {e}")
            # Continue with result even if caching fails
        
        return result, cache_key
    
    def _compute_key(
        self,
        stage: CacheStage,
        config: Dict[str, Any],
        parent_keys: List[str]
    ) -> str:
        """
        Compute content-addressable hash for a cache entry.
        
        The key is computed from:
        1. The stage name
        2. The configuration (sorted JSON)
        3. The parent keys (sorted)
        
        This ensures that identical configurations with identical dependencies
        produce the same cache key, enabling automatic cache reuse.
        
        Args:
            stage: Cache stage
            config: Configuration dictionary
            parent_keys: List of parent cache keys
            
        Returns:
            16-character hexadecimal cache key
        """
        # Create deterministic string representation
        components = [
            stage.value,
            json.dumps(self._normalize_config(config), sort_keys=True),
            "|".join(sorted(parent_keys))
        ]
        content = "|".join(components)
        
        # Compute SHA256 hash and take first 16 characters
        hash_obj = hashlib.sha256(content.encode('utf-8'))
        return hash_obj.hexdigest()[:16]
    
    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize configuration for consistent hashing.
        
        Converts values to JSON-serializable types and handles special cases.
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            Normalized configuration
        """
        normalized = {}
        
        for key, value in config.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized[key] = value
            elif isinstance(value, (list, tuple)):
                normalized[key] = [self._normalize_value(v) for v in value]
            elif isinstance(value, dict):
                normalized[key] = self._normalize_config(value)
            else:
                # Convert other types to string representation
                normalized[key] = str(value)
        
        return normalized
    
    def _normalize_value(self, value: Any) -> Any:
        """Normalize a single value for JSON serialization."""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        elif isinstance(value, dict):
            return self._normalize_config(value)
        elif isinstance(value, (list, tuple)):
            return [self._normalize_value(v) for v in value]
        else:
            return str(value)
    
    def _get_cache_path(self, stage: CacheStage, cache_key: str) -> Path:
        """
        Get the filesystem path for a cache entry.
        
        Args:
            stage: Cache stage
            cache_key: Cache key
            
        Returns:
            Path to the cache file
        """
        stage_dir = self.cache_root / stage.value
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        # Use .nc for xarray datasets, .pkl for other objects
        return stage_dir / f"{cache_key}.nc"
    
    def _save(self, result: Any, stage: CacheStage, cache_key: str) -> Path:
        """
        Save result to cache.
        
        Args:
            result: Object to cache
            stage: Cache stage
            cache_key: Cache key
            
        Returns:
            Path to saved cache file
        """
        cache_path = self._get_cache_path(stage, cache_key)
        
        if isinstance(result, xr.Dataset):
            # Save xarray Dataset as NetCDF after sanitizing unsupported attrs
            ds_to_save, extras = self._sanitize_dataset_for_netcdf(result)
            ds_to_save.to_netcdf(cache_path, mode='w', format='NETCDF4', engine='netcdf4')
            self._save_dataset_extras(stage, cache_key, extras)
        elif isinstance(result, dict) and all(isinstance(v, xr.Dataset) for v in result.values()):
            # Save dict of datasets as a combined NetCDF with groups
            # For now, pickle it (can optimize later)
            pkl_path = cache_path.with_suffix('.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            return pkl_path
        else:
            # Save other objects as pickle
            pkl_path = cache_path.with_suffix('.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            return pkl_path
        
        return cache_path
    
    def _sanitize_dataset_for_netcdf(self, ds: xr.Dataset) -> Tuple[xr.Dataset, Dict[str, Any]]:
        """
        Remove/convert attributes that are not serializable by NetCDF.
        
        Notably, xarray cannot serialize custom objects in attrs (e.g., 'indexes').
        We strip problematic attrs at the dataset and variable levels.
        """
        extras: Dict[str, Any] = {
            "dataset_attrs": {},
            "coord_attrs": {},
            "var_attrs": {},
        }
        def _is_netcdf_attr_value(v: Any) -> bool:
            # Allowed: str, numbers, lists/tuples of basic types, bytes
            from numbers import Number
            if isinstance(v, (str, bytes, Number)) or v is None:
                return True
            if isinstance(v, (list, tuple)):
                return all(_is_netcdf_attr_value(x) for x in v)
            try:
                import numpy as np  # noqa: WPS433
                if isinstance(v, np.ndarray):
                    return True
            except Exception:
                pass
            return False
        
        ds_clean = ds.copy(deep=False)
        # Clean dataset attrs
        safe_attrs = {}
        for k, v in ds_clean.attrs.items():
            if k == "indexes":
                extras["dataset_attrs"][k] = v
                continue
            if _is_netcdf_attr_value(v):
                safe_attrs[k] = v
        ds_clean.attrs = safe_attrs
        
        # Clean variable attrs
        for var_name, da in ds_clean.data_vars.items():
            if not hasattr(da, "attrs"):
                continue
            safe = {}
            for k, v in da.attrs.items():
                if k == "indexes":
                    extras["var_attrs"].setdefault(var_name, {})[k] = v
                    continue
                if _is_netcdf_attr_value(v):
                    safe[k] = v
            da.attrs = safe
        
        # Clean coordinate attrs
        for coord_name, coord in ds_clean.coords.items():
            if not hasattr(coord, "attrs"):
                continue
            safe = {}
            for k, v in coord.attrs.items():
                if k == "indexes":
                    extras["coord_attrs"].setdefault(coord_name, {})[k] = v
                    continue
                if _is_netcdf_attr_value(v):
                    safe[k] = v
            coord.attrs = safe
        return ds_clean, extras

    def _save_dataset_extras(self, stage: CacheStage, cache_key: str, extras: Dict[str, Any]) -> None:
        """Persist stripped dataset attributes for restoration after load."""
        if not extras:
            return
        # Only save if there is something meaningful
        has_data = any(extras.get(section) for section in ("dataset_attrs", "coord_attrs", "var_attrs"))
        if not has_data:
            return
        extras_path = self._get_cache_path(stage, cache_key).with_suffix(".attrs.pkl")
        extras_path.parent.mkdir(parents=True, exist_ok=True)
        with open(extras_path, "wb") as f:
            pickle.dump(extras, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_dataset_extras(self, stage: CacheStage, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load persisted dataset extras if available."""
        extras_path = self._get_cache_path(stage, cache_key).with_suffix(".attrs.pkl")
        if not extras_path.exists():
            return None
        with open(extras_path, "rb") as f:
            return pickle.load(f)
    
    def _restore_dataset_attrs(self, ds: xr.Dataset, extras: Optional[Dict[str, Any]]) -> xr.Dataset:
        """Reapply previously stripped attributes (e.g., indexes) after loading."""
        if not extras:
            return ds
        # Restore dataset attrs
        for k, v in extras.get("dataset_attrs", {}).items():
            ds.attrs[k] = v
        # Restore coordinate attrs
        for coord_name, attrs in extras.get("coord_attrs", {}).items():
            if coord_name in ds.coords:
                ds.coords[coord_name].attrs.update(attrs)
        # Restore variable attrs
        for var_name, attrs in extras.get("var_attrs", {}).items():
            if var_name in ds.data_vars:
                ds[var_name].attrs.update(attrs)
        return ds
    
    def _load(self, cache_path: Path, stage: CacheStage) -> Any:
        """
        Load result from cache.
        
        Args:
            cache_path: Path to cache file
            stage: Cache stage
            
        Returns:
            Loaded object
        """
        if cache_path.suffix == '.nc':
            # Load xarray Dataset from NetCDF
            ds = xr.load_dataset(cache_path)
            extras = self._load_dataset_extras(stage, cache_path.stem)
            return self._restore_dataset_attrs(ds, extras)
        elif cache_path.suffix == '.pkl':
            # Load pickled object
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unknown cache file format: {cache_path.suffix}")
    
    def invalidate(self, cache_key: str, stage: CacheStage, recursive: bool = True) -> int:
        """
        Invalidate a cache entry and optionally its dependents.
        
        Args:
            cache_key: Cache key to invalidate
            stage: Cache stage
            recursive: If True, also invalidate all downstream dependents
            
        Returns:
            Number of cache entries invalidated
        """
        count = 0
        
        # Delete the cache file
        cache_path = self._get_cache_path(stage, cache_key)
        if cache_path.exists():
            cache_path.unlink()
            count += 1
        
        # Also try .pkl version
        pkl_path = cache_path.with_suffix('.pkl')
        if pkl_path.exists():
            pkl_path.unlink()
        # Remove extras (attrs) if present
        extras_path = cache_path.with_suffix('.attrs.pkl')
        if extras_path.exists():
            extras_path.unlink()
        
        # Delete metadata
        self.metadata_manager.delete_metadata(cache_key, stage)
        
        print(f"Invalidated: {stage.value}/{cache_key[:12]}...")
        
        # Recursively invalidate dependents
        if recursive:
            dependents = self.metadata_manager.find_dependents(cache_key)
            for dep_key, dep_stage in dependents:
                count += self.invalidate(dep_key, dep_stage, recursive=True)
        
        return count
    
    def get_or_compute_with_state(
        self,
        stage: CacheStage,
        config: Dict[str, Any],
        parent_keys: List[str],
        compute_fn: Callable[[], Tuple[Any, Dict[str, Any]]],
        force_recompute: bool = False
    ) -> Tuple[Any, Dict[str, Any], str]:
        """
        Get cached result with learned states or compute and cache both.
        
        This is a specialized version of get_or_compute for preprocessing stages
        that need to cache both the transformed dataset and learned processor states.
        
        Args:
            stage: Cache stage for this computation
            config: Configuration dictionary for this computation
            parent_keys: List of cache keys this computation depends on
            compute_fn: Function that returns (result, learned_states)
            force_recompute: If True, bypass cache and recompute
            
        Returns:
            Tuple of (result, learned_states, cache_key)
        """
        # Compute content-addressable key
        cache_key = self._compute_key(stage, config, parent_keys)
        
        # Try to load from cache (unless force recompute)
        if not force_recompute:
            cache_path = self._get_cache_path(stage, cache_key)
            
            # Try .pkl first (tuple of dataset + states)
            pkl_path = cache_path.with_suffix('.pkl')
            if pkl_path.exists():
                try:
                    cached_tuple = self._load(pkl_path, stage)
                    if isinstance(cached_tuple, tuple) and len(cached_tuple) == 2:
                        result, learned_states = cached_tuple
                        self.metadata_manager.update_access(cache_key, stage)
                        print(f"Cache hit: {stage.value}/{cache_key[:12]}... - loading...")
                        return result, learned_states, cache_key
                except Exception as e:
                    print(f"Warning: Failed to load cache {cache_key[:12]}...: {e}")
        
        # Cache miss - compute
        print(f"Cache miss: {stage.value}/{cache_key[:12]}... - computing...")
        result, learned_states = compute_fn()
        
        # Save to cache as tuple
        try:
            cache_tuple = (result, learned_states)
            cache_path = self._save(cache_tuple, stage, cache_key)
            
            # Record metadata
            size_bytes = cache_path.stat().st_size if cache_path.exists() else 0
            self.metadata_manager.save_metadata(
                cache_key=cache_key,
                stage=stage,
                config=self._normalize_config(config),
                parent_keys=parent_keys,
                size_bytes=size_bytes
            )
            
            print(f"Cached: {stage.value}/{cache_key[:12]}... ({size_bytes / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"Warning: Failed to save cache {cache_key[:12]}...: {e}")
        
        return result, learned_states, cache_key
    
    def get_stats(self, stage: Optional[CacheStage] = None) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Args:
            stage: Optional stage to filter by
            
        Returns:
            Dictionary with cache statistics
        """
        entries = self.metadata_manager.list_all_entries(stage)
        
        total_size = sum(e.size_bytes for e in entries)
        total_accesses = sum(e.access_count for e in entries)
        
        stats = {
            'total_entries': len(entries),
            'total_size_mb': total_size / 1024 / 1024,
            'total_accesses': total_accesses,
            'avg_size_mb': (total_size / len(entries) / 1024 / 1024) if entries else 0,
            'avg_accesses': (total_accesses / len(entries)) if entries else 0,
        }
        
        if stage is None:
            # Add per-stage breakdown
            by_stage = {}
            for s in CacheStage:
                stage_entries = [e for e in entries if e.stage == s.value]
                stage_size = sum(e.size_bytes for e in stage_entries)
                by_stage[s.value] = {
                    'count': len(stage_entries),
                    'size_mb': stage_size / 1024 / 1024
                }
            stats['by_stage'] = by_stage
        
        return stats

