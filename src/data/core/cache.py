"""
src/data/core/cache.py

An intelligent CacheManager that stores each cache as an object with metadata.
The metadata includes the source, frequency, instructions (other parameters),
and the date range (start_date and end_date) of the cached data. When loading,
the cache manager will check for a cache whose date range is a superset of the
requested range.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import xarray as xr
from datetime import datetime
import pandas as pd

from src.data.core.util import FrequencyType, TimeSeriesIndex


class CacheManager:
    def __init__(self, cache_root: Optional[str] = None):
        self.cache_root = cache_root or os.path.expanduser("~/data/cache")
        
    def make_serializable(self, obj: Any) -> Any:
        """
        Recursively convert an object into something that is JSON serializable.
        If an object is not natively serializable, convert it to its string representation.
        """
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        elif isinstance(obj, dict):
            return {k: self.make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.make_serializable(v) for v in obj]
        # For known non-serializable types, e.g. xarray.Dataset, we simply return a string
        elif isinstance(obj, xr.Dataset):
            return f"<xr.Dataset: {len(obj.variables)} variables>"
        else:
            return str(obj)
        
    def normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize parameters to ensure consistent naming. In particular, if a key
        'freq' is provided, it will be renamed to 'frequency'.
        """
        norm = {}
        for k, v in params.items():
            key = "frequency" if k == "freq" else k
            norm[key] = v
        return norm

    def get_cache_dir(self, data_path: str) -> Path:
        sub_dir = data_path.strip("/")
        cache_dir = os.path.join(self.cache_root, sub_dir)
        os.makedirs(cache_dir, exist_ok=True)
        return Path(cache_dir)

    def compute_cache_key(self, params: Dict[str, Any]) -> str:
        """
        Compute a cache key based on request parameters except the time range.
        (That is, exclude 'start_date' and 'end_date'.) The parameters are normalized
        (e.g. renaming 'freq' to 'frequency') and then recursively converted to a JSON-friendly
        object.
        """
        norm_params = self.normalize_params(params)
        key_params = {k: v for k, v in norm_params.items() if k not in ("start_date", "end_date")}
        serializable_params = self.make_serializable(key_params)
        params_string = json.dumps(serializable_params, sort_keys=True)
        return hashlib.md5(params_string.encode("utf-8")).hexdigest()

    def load_from_cache(
        self, data_path: str, params: Dict[str, Any], frequency: FrequencyType
    ) -> Optional[xr.Dataset]:
        """
        Search the cache directory for a file with the same cache key and matching frequency.
        Also check that the cached date range is a superset of the requested start_date and end_date.
        If found, load the dataset and subset its time coordinate accordingly.
        """
        norm_params = self.normalize_params(params)
        cache_dir = self.get_cache_dir(data_path)
        cache_key = self.compute_cache_key(norm_params)
        requested_start = norm_params.get("start_date")
        requested_end = norm_params.get("end_date")
        requested_freq = norm_params.get("frequency")

        # Look for candidate metadata files matching the computed cache key.
        candidate_files = list(cache_dir.glob(f"{cache_key}_*.json"))
        
        for metadata_file in candidate_files:
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                if metadata.get("frequency") != requested_freq:
                    print(f"{data_path}: Skipping this cache... Frequency mismatch")
                    continue
                if requested_start and requested_end:
                    cache_start = metadata.get("start_date")
                    cache_end = metadata.get("end_date")
                    if not cache_start or not cache_end:
                        continue
                base_name = metadata_file.with_suffix("")  # remove .json extension
                netcdf_path = base_name.with_suffix(".nc")
                if not netcdf_path.exists():
                    continue
                ds = xr.load_dataset(netcdf_path)
                # Restore custom time index if present.
                if "time" in ds.coords:
                    ts_index = TimeSeriesIndex(ds.coords["time"])
                    ds.coords["time"].attrs["indexes"] = {"time": ts_index}
                # # TODO: Subset the dataset if needed.
                # if requested_start and requested_end and "time" in ds.coords:
                #     ds = ds.dt.sel(time=slice(requested_start, requested_end))
                print(f"{data_path}: Loaded a cache from {netcdf_path}")
                return ds
            except Exception as e:
                print(f"Error loading cache from {metadata_file}: {e}")
                continue
        print(f"{data_path}: No matching cache found.")
        return None

    def save_to_cache(self, ds: xr.Dataset, data_path: str, params: Dict[str, Any]) -> None:
        """
        Save the dataset to cache. Extract the time range from the dataset (if available)
        and include it in the metadata. The file name is based on the computed cache key and
        the date range.
        """
        norm_params = self.normalize_params(params)
        cache_dir = self.get_cache_dir(data_path)
        cache_key = self.compute_cache_key(norm_params)

        if "time" in ds.coords:
            time_values = pd.to_datetime(ds.coords["time"].values)
            start_date = str(time_values.min().date())
            end_date = str(time_values.max().date())
        else:
            start_date = norm_params.get("start_date", "")
            end_date = norm_params.get("end_date", "")

        base_filename = f"{cache_key}_{start_date}_{end_date}"
        base_path = cache_dir / base_filename

        netcdf_path = base_path.with_suffix(".nc")
        metadata_path = base_path.with_suffix(".json")
        netcdf_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove nonserializable attributes.
        time_indexes = None
        if "time" in ds.coords and "indexes" in ds.coords["time"].attrs:
            time_indexes = ds.coords["time"].attrs.pop("indexes")
        try:
            ds.to_netcdf(path=netcdf_path, mode="w", format="NETCDF4", engine="netcdf4")
            metadata = {
                "src": data_path,
                "frequency": norm_params.get("frequency"),
                "start_date": start_date,
                "end_date": end_date,
                "instructions": self.make_serializable({k: v for k, v in norm_params.items() if k not in ("start_date", "end_date")}),
                "timestamp": datetime.now().isoformat(),
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"{data_path}: Saved cache to {netcdf_path}")
        except Exception as e:
            raise Exception(f"Failed to save dataset to cache: {e}")
        finally:
            if time_indexes is not None:
                ds.coords["time"].attrs["indexes"] = time_indexes
