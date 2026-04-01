# YAML Reference

Comprehensive key reference for every YAML configuration surface in Hindsight. Keys are extracted from `examples/ff3_model.yaml`, `examples/pipeline_specs/*.yaml`, `src/data/ast/definitions/*.yaml`, `src/data/configs/*.yaml`, and `src/pipeline/spec/schema.py`.

---

## Pipeline Spec — Top-Level Keys

Source: `src/pipeline/spec/schema.py` (`PipelineSpec`), `examples/ff3_model.yaml`, `examples/pipeline_specs/*.yaml`


| Key                | Type                        | Default      | Description                          | Example                                    |
| ------------------ | --------------------------- | ------------ | ------------------------------------ | ------------------------------------------ |
| `spec_version`     | `str`                       | —            | Schema version identifier            | `"1.0"`                                    |
| `name`             | `str`                       | **required** | Unique pipeline name                 | `"ff3_model"`                              |
| `version`          | `str`                       | **required** | Pipeline version string              | `"2.0"`                                    |
| `time_range`       | `dict`                      | **required** | Date bounds for data loading         | `{start: "2010-01-01", end: "2021-12-31"}` |
| `time_range.start` | `str`                       | **required** | Start date (inclusive), ISO 8601     | `"2010-01-01"`                             |
| `time_range.end`   | `str`                       | **required** | End date (inclusive), ISO 8601       | `"2021-12-31"`                             |
| `data`             | `dict[str, DataSourceSpec]` | **required** | Named data source specifications     | See Data Source section                    |
| `merge_base`       | `str`                       | `None`       | Base source name for ordered merges  | `"crsp"`                                   |
| `merges`           | `list[dict]`                | `[]`         | Ordered merge configuration list     | See Merges section                         |
| `features`         | `FeaturesSpec`              | `None`       | Feature engineering specification    | See Features section                       |
| `preprocessing`    | `PreprocessingSpec`         | `None`       | Preprocessing pipeline stages        | See Preprocessing section                  |
| `model`            | `ModelSpec`                 | `None`       | Model training specification         | See Model section                          |
| `metadata`         | `dict`                      | `{}`         | Arbitrary metadata for documentation | See Metadata section                       |


---

## Data Source Keys

Source: `src/pipeline/spec/schema.py` (`DataSourceSpec`)


| Key               | Type                | Default      | Description                             | Example                           |
| ----------------- | ------------------- | ------------ | --------------------------------------- | --------------------------------- |
| `provider`        | `str`               | **required** | Data provider name                      | `"wrds"`, `"crypto"`, `"open_bb"` |
| `dataset`         | `str`               | **required** | Dataset name within provider            | `"equity/crsp"`, `"spot/binance"` |
| `frequency`       | `str`               | `None`       | Data frequency code                     | `"H"`, `"D"`, `"M"`, `"Y"`        |
| `columns`         | `list[str]`         | `[]`         | Column subset to load from source       | `["date", "permno", "ret"]`       |
| `filters`         | `dict`              | `{}`         | Django-style filters at DataFrame level | `{shrcd__in: [10, 11]}`           |
| `external_tables` | `list[dict]`        | `[]`         | Side-table merge configs (pre-xarray)   | See External Tables section       |
| `processors`      | `dict | list[dict]` | `{}`         | Source-level post-processing transforms | See Source Processors section     |


### External Tables Keys


| Key           | Type        | Default      | Description                          | Example                                           |
| ------------- | ----------- | ------------ | ------------------------------------ | ------------------------------------------------- |
| `path`        | `str`       | **required** | Filesystem path to SAS/CSV table     | `"/wrds/crsp/sasdata/a_stock/msedelist.sas7bdat"` |
| `type`        | `str`       | **required** | Merge strategy                       | `"replace"`, `"lookup"`                           |
| `on`          | `str`       | **required** | Join column                          | `"permno"`                                        |
| `time_column` | `str`       | —            | Time column for temporal merge       | `"dlstdt"`                                        |
| `from_column` | `str`       | —            | Source column in external table      | `"dlret"`                                         |
| `to_column`   | `str`       | —            | Target column to replace             | `"ret"`                                           |
| `columns`     | `list[str]` | —            | Columns to include from lookup table | `["exchcd", "shrcd"]`                             |


### Source Processors Keys (dict format)


| Key                       | Type         | Default      | Description                           | Example                                                         |
| ------------------------- | ------------ | ------------ | ------------------------------------- | --------------------------------------------------------------- |
| `transforms`              | `list[dict]` | —            | List of transform operations          | See below                                                       |
| `transforms[].type`       | `str`        | **required** | Transform type                        | `"set_coordinates"`, `"fix_market_equity"`, `"preferred_stock"` |
| `transforms[].coord_type` | `str`        | —            | Coordinate type for `set_coordinates` | `"permco"`, `"permno"`                                          |


---

## Merge Keys


| Key                  | Type         | Default      | Description                            | Example                                                 |
| -------------------- | ------------ | ------------ | -------------------------------------- | ------------------------------------------------------- |
| `right_name`         | `str`        | **required** | Name of the right data source          | `"compustat"`                                           |
| `on`                 | `str`        | **required** | Join dimension                         | `"asset"`                                               |
| `time_alignment`     | `str`        | —            | Temporal alignment strategy            | `"exact"`, `"ffill"`, `"bfill"`, `"nearest"`, `"as_of"` |
| `time_offset_months` | `int`        | —            | Months to offset for as-of alignment   | `6`                                                     |
| `ffill_limit`        | `int | null` | `null`       | Forward-fill limit in periods          | `3`                                                     |
| `prefix`             | `str`        | `""`         | Prefix for merged variables            | `"comp_"`                                               |
| `suffix`             | `str`        | `""`         | Suffix for merged variables            | `"_ann"`                                                |
| `variables`          | `list[str]`  | —            | Variables to include from right source | `["seq", "txditc", "ps"]`                               |
| `drop_vars`          | `list[str]`  | `[]`         | Variables to drop after merge          | `["gvkey"]`                                             |


---

## Features Keys

Source: `src/pipeline/spec/schema.py` (`FeaturesSpec`, `FormulaOperationSpec`)


| Key                       | Type                         | Default      | Description                                  | Example                 |
| ------------------------- | ---------------------------- | ------------ | -------------------------------------------- | ----------------------- |
| `features.operations`     | `list[FormulaOperationSpec]` | `[]`         | Ordered list of formula operation groups     | See below               |
| `operations[].name`       | `str`                        | **required** | Unique operation name (for caching and deps) | `"momentum_indicators"` |
| `operations[].depends_on` | `list[str]`                  | `[]`         | Operation names this depends on              | `["ff3_base_features"]` |
| `operations[].formulas`   | `dict`                       | **required** | Named formula configurations                 | See below               |


### Formula Configs

Formulas can be either **named references** (to `FormulaManager` definitions) or **inline expressions**.

**Named formula — list of config overrides:**


| Key                       | Type         | Description                                  | Example                             |
| ------------------------- | ------------ | -------------------------------------------- | ----------------------------------- |
| `{formula_name}`          | `list[dict]` | List of override dictionaries per invocation | `sma: [{window: 20}, {window: 50}]` |
| `{formula_name}[].window` | `number`     | Override for the `window` variable           | `20`                                |


**Inline formula — expression key:**


| Key                         | Type  | Description                      | Example                  |
| --------------------------- | ----- | -------------------------------- | ------------------------ |
| `{formula_name}.expression` | `str` | Inline formula expression string | `"$seq + $txditc - $ps"` |


---

## Preprocessing Keys

Source: `src/pipeline/spec/schema.py` (`PreprocessingSpec`)


| Key                    | Type         | Default         | Description                               | Example                     |
| ---------------------- | ------------ | --------------- | ----------------------------------------- | --------------------------- |
| `preprocessing.mode`   | `str`        | `"independent"` | Pipeline mode                             | `"independent"`, `"append"` |
| `preprocessing.shared` | `list[dict]` | `[]`            | Shared (stateless) processors             | See Processor Configs       |
| `preprocessing.learn`  | `list[dict]` | `[]`            | Learn (stateful fit/transform) processors | See Processor Configs       |
| `preprocessing.infer`  | `list[dict]` | `[]`            | Infer (transform-only) processors         | See Processor Configs       |


### Common Processor Config Keys

Every processor has:


| Key    | Type  | Default      | Description               | Example                               |
| ------ | ----- | ------------ | ------------------------- | ------------------------------------- |
| `type` | `str` | **required** | Processor type identifier | `"sort"`, `"port_ret"`, `"cs_zscore"` |
| `name` | `str` | **required** | Unique instance name      | `"size_sort"`                         |


### `per_asset_ffill` Processor


| Key    | Type        | Default      | Description               | Example               |
| ------ | ----------- | ------------ | ------------------------- | --------------------- |
| `type` | `str`       | —            | `"per_asset_ffill"`       |                       |
| `name` | `str`       | —            | Instance name             | `"forward_fill"`      |
| `vars` | `list[str]` | `None` (all) | Variables to forward-fill | `["close", "volume"]` |


### `cs_zscore` Processor


| Key          | Type        | Default      | Description                     | Example               |
| ------------ | ----------- | ------------ | ------------------------------- | --------------------- |
| `type`       | `str`       | —            | `"cs_zscore"`                   |                       |
| `name`       | `str`       | —            | Instance name                   | `"normalizer"`        |
| `vars`       | `list[str]` | **required** | Variables to normalize          | `["close", "volume"]` |
| `out_suffix` | `str`       | `"_norm"`    | Suffix for output variables     | `"_norm"`             |
| `eps`        | `float`     | `1e-8`       | Epsilon for numerical stability | `1e-8`                |


### `formula_eval` Processor


| Key               | Type   | Default      | Description                           | Example                 |
| ----------------- | ------ | ------------ | ------------------------------------- | ----------------------- |
| `type`            | `str`  | —            | `"formula_eval"`                      |                         |
| `name`            | `str`  | —            | Instance name                         | `"features"`            |
| `formula_configs` | `dict` | **required** | Formula name → list of override dicts | `{sma: [{window: 20}]}` |
| `use_jit`         | `bool` | `True`       | Apply `jax.jit` for performance       | `true`                  |
| `defs_dir`        | `str`  | `None`       | Custom definitions directory          | `"./my_formulas/"`      |
| `assign_in_place` | `bool` | `True`       | Attach results to input dataset       | `true`                  |


### `sort` (CrossSectionalSort) Processor


| Key         | Type          | Default      | Description                                      | Example            |
| ----------- | ------------- | ------------ | ------------------------------------------------ | ------------------ |
| `type`      | `str`         | —            | `"sort"`                                         |                    |
| `name`      | `str`         | —            | Instance name                                    | `"size_sort"`      |
| `signal`    | `str`         | **required** | Signal variable to sort on                       | `"me_june"`        |
| `n_bins`    | `int`         | **required** | Number of equal quantile bins                    | `2`                |
| `scope`     | `str`         | `None`       | Boolean mask for breakpoint calculation          | `"is_nyse"`        |
| `quantiles` | `list[float]` | `None`       | Custom interior breakpoints (overrides `n_bins`) | `[0.3, 0.7]`       |
| `labels`    | `list`        | `None`       | Custom bin labels                                | `["Small", "Big"]` |


### `port_ret` (PortfolioReturns) Processor


| Key           | Type        | Default      | Description                            | Example                              |
| ------------- | ----------- | ------------ | -------------------------------------- | ------------------------------------ |
| `type`        | `str`       | —            | `"port_ret"`                           |                                      |
| `name`        | `str`       | —            | Instance name                          | `"ff_port_ret"`                      |
| `groupby`     | `list[str]` | **required** | Grouping variables (portfolio labels)  | `["me_june_port", "beme_june_port"]` |
| `returns_var` | `str`       | `"ret"`      | Returns variable name                  | `"ret"`                              |
| `weights_var` | `str`       | `None`       | Weights variable (None = equal-weight) | `"me_lag1"`                          |


### `factor_spread` (FactorSpread) Processor


| Key                           | Type   | Default      | Description                             | Example                                  |
| ----------------------------- | ------ | ------------ | --------------------------------------- | ---------------------------------------- |
| `type`                        | `str`  | —            | `"factor_spread"`                       |                                          |
| `name`                        | `str`  | —            | Instance name                           | `"ff_factors"`                           |
| `source`                      | `str`  | **required** | Source portfolio returns variable       | `"port_ret_me_june_port_beme_june_port"` |
| `factors`                     | `dict` | **required** | Factor definitions (one per output)     | See below                                |
| `factors.{name}.long`         | `dict` | **required** | `.sel()` kwargs for long leg            | `{me_june_port_dim: 0}`                  |
| `factors.{name}.short`        | `dict` | **required** | `.sel()` kwargs for short leg           | `{me_june_port_dim: 1}`                  |
| `factors.{name}.average_over` | `str`  | `None`       | Dimension to average before subtraction | `"beme_june_port_dim"`                   |


---

## Model Keys

Source: `src/pipeline/spec/schema.py` (`ModelSpec`)


| Key                    | Type        | Default     | Description                           | Example                                         |
| ---------------------- | ----------- | ----------- | ------------------------------------- | ----------------------------------------------- |
| `model.adapter`        | `str`       | `"sklearn"` | Adapter type                          | `"sklearn"`                                     |
| `model.type`           | `str`       | `""`        | Model class name                      | `"RandomForestRegressor"`, `"LinearRegression"` |
| `model.params`         | `dict`      | `{}`        | Model constructor hyperparameters     | `{n_estimators: 50, max_depth: 14}`             |
| `model.features`       | `list[str]` | `[]`        | Input variable names                  | `["close_norm", "volume_norm"]`                 |
| `model.target`         | `str`       | `""`        | Target variable name                  | `"close"`                                       |
| `model.walk_forward`   | `dict`      | `{}`        | Walk-forward SegmentConfig parameters | See below                                       |
| `model.adapter_params` | `dict`      | `{}`        | Additional adapter parameters         | See below                                       |
| `model.runner_params`  | `dict`      | `{}`        | ModelRunner parameters                | See below                                       |


### Walk-Forward Keys


| Key                | Type  | Default      | Description                           | Example                 |
| ------------------ | ----- | ------------ | ------------------------------------- | ----------------------- |
| `train_span_hours` | `int` | **required** | Training window size in hours         | `720` (30 days)         |
| `infer_span_hours` | `int` | **required** | Inference window size in hours        | `24` (1 day)            |
| `step_hours`       | `int` | **required** | Step size between segments in hours   | `24`                    |
| `gap_hours`        | `int` | `0`          | Gap between train_end and infer_start | `0`                     |
| `start`            | `str` | data bounds  | Optional start datetime               | `"2020-01-22 18:00:00"` |
| `end`              | `str` | data bounds  | Optional end datetime                 | `"2023-12-01 00:00:00"` |


### Adapter Parameters


| Key          | Type   | Default   | Description                              | Example        |
| ------------ | ------ | --------- | ---------------------------------------- | -------------- |
| `output_var` | `str`  | `"score"` | Name of prediction output variable       | `"close_pred"` |
| `use_proba`  | `bool` | `false`   | Use `predict_proba` instead of `predict` | `false`        |


### Runner Parameters


| Key              | Type  | Default  | Description                           | Example             |
| ---------------- | ----- | -------- | ------------------------------------- | ------------------- |
| `overlap_policy` | `str` | `"last"` | How overlapping segments are resolved | `"last"`, `"first"` |


---

## Metadata Keys


| Key                    | Type        | Default | Description                         | Example                               |
| ---------------------- | ----------- | ------- | ----------------------------------- | ------------------------------------- |
| `metadata.description` | `str`       | —       | Human-readable pipeline description | `"Baseline crypto momentum strategy"` |
| `metadata.author`      | `str`       | —       | Author name                         | `"Hindsight Team"`                    |
| `metadata.tags`        | `list[str]` | —       | Tags for categorization             | `["crypto", "momentum"]`              |


---

## Data Config Keys (Built-In Configs)

Source: `src/data/configs/*.yaml`

These are data-loading-only configs used by `DataManager.load_builtin()`.


| Key                                         | Type         | Description                     | Example                                             |
| ------------------------------------------- | ------------ | ------------------------------- | --------------------------------------------------- |
| `data.name`                                 | `str`        | Config name                     | `"equity-standard"`                                 |
| `data.start_date`                           | `str`        | Start date                      | `"2000-01-01"`                                      |
| `data.end_date`                             | `str`        | End date                        | `"2024-01-01"`                                      |
| `data.cache_path`                           | `str`        | Cache directory                 | `"~/data/cache/"`                                   |
| `data.sources`                              | `dict`       | Named source configurations     | See below                                           |
| `data.sources.{name}.provider`              | `str`        | Provider name                   | `"wrds"`, `"crypto"`                                |
| `data.sources.{name}.dataset`               | `str`        | Dataset name                    | `"crsp"`, `"binance_spot"`                          |
| `data.sources.{name}.frequency`             | `str`        | Data frequency                  | `"monthly"`, `"H"`                                  |
| `data.sources.{name}.schema.time_var`       | `str`        | Time column name                | `"date"`, `"datadate"`                              |
| `data.sources.{name}.schema.identifier_var` | `str`        | Asset identifier column         | `"permno"`, `"gvkey"`                               |
| `data.sources.{name}.columns`               | `list[str]`  | Columns to load                 | `["gvkey", "at", "seq"]`                            |
| `data.sources.{name}.filters`               | `dict`       | Django-style filters            | `{indfmt: "INDL"}`                                  |
| `data.sources.{name}.external_tables`       | `list[dict]` | Same structure as pipeline spec | See External Tables above                           |
| `data.sources.{name}.processors.transforms` | `list[dict]` | Post-processing transforms      | `[{type: "set_coordinates", coord_type: "permno"}]` |


---

## Formula Definition Keys (AST Definitions)

Source: `src/data/ast/definitions/schema.yaml`, `technical.yaml`, `marketchars.yaml`, `cross_sectional.yaml`, `composite.yaml`, `ts_dependence.yaml`


| Key                                   | Type         | Required | Description                                        | Example                                           |
| ------------------------------------- | ------------ | -------- | -------------------------------------------------- | ------------------------------------------------- |
| `{formula_name}`                      | `dict`       | —        | Top-level key is the formula name                  | `rsi:`, `sma:`, `me_cs_rank:`                     |
| `description`                         | `str`        | Yes      | Human-readable formula description                 | `"Relative Strength Index"`                       |
| `expression`                          | `str`        | Yes      | Formula expression in CFG syntax                   | `"sma($price, $window)"`                          |
| `return_type`                         | `str`        | Yes      | Expected return type                               | `"dataarray"`, `"scalar"`, `"array"`, `"dataset"` |
| `variables`                           | `dict`       | Yes      | Variable definitions (keyed by name)               | See below                                         |
| `variables.{name}.type`               | `str`        | Yes      | Variable type                                      | `"dataarray"`, `"number"`, `"array"`, `"boolean"` |
| `variables.{name}.description`        | `str`        | Yes      | Variable description                               | `"Price data"`                                    |
| `variables.{name}.default`            | `any`        | No       | Default value                                      | `14`, `0.85`, `null`                              |
| `variables.{name}.validation`         | `dict`       | No       | Validation rules                                   | `{min: 1, max: 100}`                              |
| `variables.{name}.validation.min`     | `number`     | No       | Minimum allowed value                              | `1`                                               |
| `variables.{name}.validation.max`     | `number`     | No       | Maximum allowed value                              | `200`                                             |
| `variables.{name}.generator`          | `str`        | No       | Reference to a module alias for dynamic generation | `"alma_generator"`                                |
| `functions`                           | `dict`       | No       | Function definitions used in expression            | See below                                         |
| `functions.{name}.description`        | `str`        | Yes      | Function description                               | `"Exponential Moving Average"`                    |
| `functions.{name}.args`               | `list[dict]` | Yes      | Argument specifications                            | See below                                         |
| `functions.{name}.args[].name`        | `str`        | Yes      | Argument name                                      | `"data"`                                          |
| `functions.{name}.args[].type`        | `str`        | Yes      | Argument type                                      | `"dataarray"`, `"number"`                         |
| `functions.{name}.args[].description` | `str`        | Yes      | Argument description                               | `"Input data"`                                    |
| `functions.{name}.args[].optional`    | `bool`       | No       | Whether argument is optional                       | `false`                                           |
| `modules`                             | `dict`       | No       | External module definitions for pre-computation    | See below                                         |
| `modules.{alias}.module_path`         | `str`        | Yes      | Python module path                                 | `"src.data.generators.weights"`                   |
| `modules.{alias}.function_name`       | `str`        | Yes      | Function name in module                            | `"alma_weights"`                                  |
| `modules.{alias}.description`         | `str`        | No       | Module function description                        | `"Generate ALMA weights"`                         |
| `modules.{alias}.cache_result`        | `bool`       | No       | Whether to cache result                            | `true`                                            |
| `tags`                                | `list[str]`  | No       | Categorization tags                                | `["technical", "momentum"]`                       |
| `notes`                               | `str`        | No       | Implementation notes and details                   | `"RSI values range 0-100"`                        |


---

## Django-Style Filter Syntax

Filters use a `{field}__{operator}` suffix convention:


| Suffix     | Operator              | Example                                       |
| ---------- | --------------------- | --------------------------------------------- |
| (none)     | exact equality        | `{indfmt: "INDL"}`                            |
| `__in`     | membership            | `{shrcd__in: [10, 11]}`                       |
| `__not_in` | exclusion             | `{exchcd__not_in: [4]}`                       |
| `__gt`     | greater than          | `{price__gt: 5.0}`                            |
| `__lt`     | less than             | `{volume__lt: 1000}`                          |
| `__gte`    | greater than or equal | `{date__gte: "2010-01-01"}`                   |
| `__lte`    | less than or equal    | `{date__lte: "2021-12-31"}`                   |
| `__range`  | between (inclusive)   | `{date__range: ["2010-01-01", "2021-12-31"]}` |


---

## See also

- [PIPELINE_SYSTEM.md](./PIPELINE_SYSTEM.md) — Processor stage model and how processors are configured
- [FORMULA_SYSTEM.md](./FORMULA_SYSTEM.md) — Formula expression syntax and registered functions
- [DATA_LAYER.md](./DATA_LAYER.md) — Data loading, `from_table()` conversion, cache system

