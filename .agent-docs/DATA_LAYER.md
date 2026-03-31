# Data Layer

The data layer converts raw tabular data (CSV, SAS, API) into xarray Datasets with the canonical `(year, month, day, hour, asset)` dimension layout, and provides caching, provider registration, and a custom `.dt` accessor for time-aware operations.

---

## Public API Signatures

### `from_table()` — Core Converter

```python
# src/data/loaders/table.py
def from_table(
    data: pd.DataFrame,
    time_column: str = "time",
    asset_column: str = "asset",
    feature_columns: Optional[List[str]] = None,
    frequency: FrequencyType = FrequencyType.DAILY,
) -> xr.Dataset
```

Converts a flat `pd.DataFrame` into an `xr.Dataset` with multi-dimensional time coords.

**Behavior by frequency:**

| Frequency | `year` | `month` | `day` | `hour` |
|-----------|--------|---------|-------|--------|
| `HOURLY`  | from data | from data | from data | from data |
| `DAILY`   | from data | from data | from data | `0` (fixed) |
| `MONTHLY` | from data | from data | `1` (fixed) | `0` (fixed) |
| `YEARLY`  | from data | `1` (fixed) | `1` (fixed) | `0` (fixed) |

A `time` coordinate of dtype `datetime64[ns]` is computed and attached.  
A `TimeSeriesIndex` is stored in `ds.coords['time'].attrs['indexes']` for `.dt.sel()` lookups.

---

### `DataManager` — User-Facing Data Loading

```python
# src/data/managers/data_manager.py
class DataManager:
    def __init__(self)
    def _get_raw_data(self, data_requests: List[Dict]) -> Dict[str, xr.Dataset]
    def load_from_config(self, config_path: str) -> Dict[str, xr.Dataset]
    def load_from_built_config(self, config: DataConfig) -> Dict[str, xr.Dataset]
    def load_builtin(self, config_name: str, start_date=None, end_date=None) -> Dict[str, xr.Dataset]
```

`DataManager` reads from `_PROVIDER_REGISTRY` at init time. It orchestrates cache checks, loader dispatch, and post-processor application.

---

### `CacheManager` — Two-Level NetCDF Cache

```python
# src/data/core/cache.py
class CacheManager:
    def __init__(self, cache_root: Optional[str] = None)  # defaults to ~/data/cache
    def fetch(
        self,
        relative_path: str,
        parameters: Dict,
        data_loader: Callable,
    ) -> Optional[xr.Dataset]
    def cache(self, dataset: xr.Dataset, relative_path: str, parameters: Dict) -> None
```

**Cache levels:**

| Level | Content | Storage |
|-------|---------|---------|
| L1 | Raw data (pre-processors) | NetCDF + JSON metadata |
| L2 | Post-processed data | NetCDF + JSON metadata |

Cache key = MD5 hash of parameters (excluding date range). Date range is encoded in the filename.

---

### `.dt` Accessor — Multi-Dimensional Time Operations

```python
# src/data/core/struct.py (registered on both xr.Dataset and xr.DataArray)

class DateTimeAccessorBase:
    def sel(self, time) -> Union[xr.Dataset, xr.DataArray]
        # Timestamp-based selection via TimeSeriesIndex

    def to_time_indexed(self) -> Union[xr.Dataset, xr.DataArray]
        # Flatten multi-dim time to a single 'time' dimension (for plotting/export)

    def rolling(self, dim: str, window: int, mask=None, mask_indices=None) -> Rolling
        # Business-day-aware rolling window backed by JAX u_roll

    def compute_mask(self) -> Tuple[jnp.ndarray, jnp.ndarray]
        # Returns (mask, sorted_pos) for valid business days per asset

    def shift(self, periods: int = 1, mask_indices=None) -> Union[xr.Dataset, xr.DataArray]
        # Business-day-aware temporal shift
```

**Important**: The `.dt` accessor **overrides** xarray's built-in `.dt` accessor. It provides financial-time-aware operations on the multi-dimensional layout.

---

## Provider Registration Pattern

Providers self-register when their module is imported. The pattern:

```python
# src/data/loaders/wrds/__init__.py (example)
from src.data.core.provider import Provider, register_provider

wrds_provider = Provider(
    name="wrds",
    website="https://wrds-www.wharton.upenn.edu/",
    description="WRDS data provider",
    fetcher_dict={
        "equity/crsp": CRSPDataFetcher,
        "equity/compustat": CompustatDataFetcher,
    },
)
register_provider(wrds_provider)
```

**Registered providers:**

| Provider | Module | Datasets |
|----------|--------|----------|
| `wrds` | `src.data.loaders.wrds` | `equity/crsp`, `equity/compustat` |
| `crypto` | `src.data.loaders.crypto` | `spot/binance` |
| `open_bb` | `src.data.loaders.open_bb` | Dynamic from `obb.coverage` |

`DataManager` reads the global `_PROVIDER_REGISTRY` dict to dispatch `load_data()` calls.

---

## How to Add a New Loader

1. **Create a fetcher class** that subclasses `BaseDataSource`:

```python
# src/data/loaders/my_source/fetcher.py
from src.data.loaders.abstracts.base import BaseDataSource

class MyFetcher(BaseDataSource):
    def load_data(self, **config) -> xr.Dataset:
        df = self._load_raw(**config)
        return self._convert_to_xarray(df, columns=..., frequency=...)
```

2. **Register a provider** in the package `__init__.py`:

```python
# src/data/loaders/my_source/__init__.py
from src.data.core.provider import Provider, register_provider

provider = Provider(
    name="my_source",
    website="https://...",
    description="My data source",
    fetcher_dict={"dataset_name": MyFetcher},
)
register_provider(provider)
```

3. **Import the package** in `src/data/loaders/__init__.py` to trigger registration:

```python
from src.data.loaders import my_source  # triggers register_provider()
```

---

## Data-Level Post-Processors

```python
# src/data/processors/
def apply_processors(ds: xr.Dataset, processors) -> Tuple[xr.Dataset, List]
```

Registered via `@post_processor` decorator. Django-style shortcut config supported:

| Shortcut | Processor | Purpose |
|----------|-----------|---------|
| `set_permno_coord` | `set_permno` | Set PERMNO as asset coordinate |
| `set_permco_coord` | `set_permco` | Set PERMCO as asset coordinate |
| `ps` | `ps` | Compute preferred stock |
| `fix_market_equity` | `fix_mke` | Fix market equity calculation |

---

## Known Pitfalls

| Pitfall | Location | Detail |
|---------|----------|--------|
| **NaN in `u_roll`** | `src/data/core/operations/__init__.py` | `_prepare_blocks` calls `jnp.nan_to_num(data)` converting NaN→0 before rolling. This silently changes semantics for functions expecting NaN propagation. |
| **`Rolling.std()` not implemented** | `src/data/core/rolling.py` | Raises `NotImplementedError`. Blocks Bollinger Bands and volatility calculations. |
| **Cache side-effect on import** | `src/data/__init__.py:70` | `initialize_cache_directories()` runs at import time, creating filesystem dirs. |
| **`_parent_dataset` in DataArray attrs** | `src/data/core/struct.py` | Storing `xr.Dataset` reference in DataArray attrs can cause memory leaks and serialization issues. |
| **OpenBB loader broken** | `src/data/loaders/open_bb/generic.py` | Calls `self.get_cache_path()` / `self.save_to_cache()` which don't exist on `BaseDataSource`. |
| **Compustat hardcoded CCM path** | `src/data/loaders/wrds/compustat.py` | `/wrds/crsp/sasdata/a_ccm/ccmxpf_linktable.sas7bdat` is hardcoded. Not configurable. |

---

## See also

- [ARCHITECTURE.md](./ARCHITECTURE.md) — Overall system data-flow and dimension contract
- [FORMULA_SYSTEM.md](./FORMULA_SYSTEM.md) — Formula definitions, registered functions, FormulaManager
- [PIPELINE_SYSTEM.md](./PIPELINE_SYSTEM.md) — Pipeline processors that consume datasets from this layer
