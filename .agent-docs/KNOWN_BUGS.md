# Known Bugs

Active bugs in the Hindsight codebase, anchored to current source files and behavior observed in tests/lint output.

---

## Bug Table


| #   | File:line                                                                      | Symptom                                                                                                                                                                                                                                                                                   | Workaround                                                                                                                                                                                                                                                           |
| --- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `src/data/core/operations/__init__.py` : `u_roll` / `_prepare_blocks`          | **u_roll NaN issue.** `_prepare_blocks` calls `jnp.nan_to_num(data)` converting NaN to 0 before the JAX rolling kernel. This silently changes semantics for any function that expects NaN propagation (e.g., a rolling mean over NaN values returns 0-influenced results instead of NaN). | Use `jnp.nan`* variants (e.g., `jnp.nanmean`) inside new rolling kernels explicitly. Do not assume NaN propagation through `u_roll`. When writing tests, compare results against known NaN-aware reference implementations.                                          |
| 2   | `src/data/core/rolling.py` : `Rolling.std()`                                   | **Rolling.std issue.** `Rolling.std()` raises `NotImplementedError`. Any workflow requiring rolling standard deviation (Bollinger Bands, volatility, risk metrics) cannot use the `.dt.rolling().std()` API.                                                                              | No direct workaround in the public API. Implement a `std` kernel in `src/data/core/operations/standard.py` following the existing kernel pattern `(i, carry, block, window_size) -> Tuple[value, carry]`, then wire it into `Rolling.std()`.                         |
| 3   | `src/data/loaders/open_bb/generic.py` : `GenericOpenBBDataFetcher.load_data()` | **OpenBB broken methods.** The `load_data()` method calls `self.get_cache_path()` and `self.save_to_cache()`, which do not exist on `BaseDataSource`. Any attempt to load data through the OpenBB provider raises `AttributeError`.                                                       | Do not use the OpenBB provider until these methods are either implemented on `BaseDataSource` or removed from `GenericOpenBBDataFetcher`. Use alternative loaders (WRDS, crypto) instead.                                                                            |
| 4   | `src/data/managers/config_schema.py`                                           | **config_schema leak.** The module is marked deprecated (emits `DeprecationWarning` on import), but `DataManager` still imports from it. Every `import src` triggers the deprecation warning.                                                                                             | Suppress the warning at call site with `warnings.filterwarnings("ignore", category=DeprecationWarning, module="config_schema")`, or migrate `DataManager` to use `src.pipeline.spec` instead of `config_schema`. Do not import from `config_schema` in any new code. |
| 5   | `src/data/loaders/wrds/compustat.py`                                           | **Hardcoded CCM path.** The CCM linktable path `/wrds/crsp/sasdata/a_ccm/ccmxpf_linktable.sas7bdat` is hardcoded in `CompustatDataFetcher`. The path is not configurable via YAML or constructor parameter, making the loader non-portable across environments.                           | Ensure the hardcoded path exists in your environment, or modify the source to accept the CCM path as a constructor parameter or YAML config key.                                                                                                                     |


---

## Pre-Existing Lint/Warning Issues (Not Bugs)

These are not functional bugs but are flagged by tools and should not be confused with the above:


| Location                         | Issue                                                     | Impact                                                  |
| -------------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| `src/backtester/core.py:436,457` | flake8 F821: undefined `execution_timestamp`              | Backtester module; does not affect data/pipeline layers |
| `src/data/ast/functions.py:186`  | flake8 F824: unused `nonlocal` declaration                | No runtime impact                                       |
| `src/data/ast/parser.py:207`     | `DeprecationWarning`: `ast.Num` deprecated in Python 3.14 | Works in Python 3.12; will break in 3.14+               |
| `src/data/loaders/table.py:166`  | `FutureWarning`: `'M'` freq deprecated in pandas          | Replace `'M'` with `'ME'` when upgrading pandas         |


---

## See also

- [KNOWN_GAPS.md](./KNOWN_GAPS.md) — Missing functionality (distinct from broken functionality)
- [DATA_LAYER.md](./DATA_LAYER.md) — Data layer pitfalls including u_roll NaN behavior and cache side effects
- [ARCHITECTURE.md](./ARCHITECTURE.md) — System overview and layer boundaries

