# Pipeline System Architecture: Deep Dive

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [Core Components](#core-components)
4. [Data Flow: End-to-End](#data-flow-end-to-end)
5. [DataHandler Integration](#datahandler-integration)
6. [ModelAdapter Integration](#modeladapter-integration)
7. [Caching System](#caching-system)
8. [Processor Registry](#processor-registry)
9. [Under the Hood: Detailed Mechanics](#under-the-hood-detailed-mechanics)
10. [Future Extensions](#future-extensions)

---

## Overview

The pipeline specification system is a **declarative abstraction layer** built on top of Hindsight's existing infrastructure. It provides YAML-based configuration for end-to-end machine learning workflows while leveraging battle-tested components like `DataHandler`, `ModelAdapter`, and `FormulaManager`.

**Key Principle**: The system **wraps, not replaces**. Every component in the spec system delegates to existing, proven infrastructure.

```
┌──────────────────────────────────────────────────────────────┐
│                    YAML Specification                         │
│  Declarative "what" - user intent, configuration             │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                   Pipeline Executor                           │
│  Orchestration "when" - caching, ordering, coordination      │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│              Existing Infrastructure                          │
│  Implementation "how" - DataHandler, ModelAdapter, etc.      │
└──────────────────────────────────────────────────────────────┘
```

---

## Design Philosophy

### 1. Abstraction, Not Replacement

The YAML spec system does **not** duplicate or replace existing infrastructure:

- **DataHandler**: Used as-is for all preprocessing
- **ModelAdapter**: Used as-is for all model training
- **FormulaManager**: Used as-is for all feature engineering
- **DataManager**: Used as-is for all data loading

What the spec system **adds**:
- Hierarchical caching (L2-L5)
- Content-addressable cache keys
- Declarative YAML interface
- Automatic dependency tracking
- Cross-pipeline cache reuse

### 2. Separation of Concerns

The system maintains clear boundaries:

| Layer | Responsibility | Example |
|-------|---------------|---------|
| **YAML Spec** | What to compute | "Normalize close and volume" |
| **Executor** | When and where to compute | "Check cache, if miss compute, then cache" |
| **Infrastructure** | How to compute | "Use CSZScore processor on specified vars" |

### 3. Composability

Every stage is independently cacheable and reusable:

```
Pipeline A: Data → Features → Preprocessing → Model A
Pipeline B: Data → Features → Preprocessing → Model B
                    ↑
              Cache reused here
```

---

## Core Components

### Component Hierarchy

```
PipelineExecutor
├── GlobalCacheManager (L2-L5 caching)
├── DataManager (data loading)
├── FormulaManager (feature engineering)
├── ProcessorRegistry (preprocessing)
└── ModelAdapter (model training)
```

### File Structure

```
src/pipeline/
├── spec/
│   ├── schema.py          # YAML schema definitions
│   ├── parser.py          # YAML → Python objects
│   ├── executor.py        # Pipeline orchestration
│   ├── result.py          # Execution results
│   └── processor_registry.py  # Dynamic processor creation
├── cache/
│   ├── manager.py         # Cache operations
│   ├── metadata.py        # JSON-based metadata
│   └── stages.py          # Cache level definitions
├── data_handler/
│   ├── handler.py         # DataHandler (preprocessing)
│   ├── processors.py      # Processor implementations
│   └── config.py          # Handler configuration
└── model/
    ├── adapter.py         # ModelAdapter interface
    └── runner.py          # Walk-forward validation
```

---

## Data Flow: End-to-End

### High-Level Flow

```
YAML Spec → Parser → Executor → Infrastructure → Results
```

### Detailed Flow with Caching

```
1. YAML Parsing
   ├─ Load YAML file
   ├─ Validate schema
   ├─ Create PipelineSpec object
   └─ Return spec

2. Data Loading (L2)
   ├─ Build data config from spec
   ├─ Compute cache key: hash(data_config)
   ├─ Check cache
   │  ├─ HIT: Load from cache
   │  └─ MISS: Load via DataManager
   ├─ Apply data-level processors
   ├─ Save to cache (xr.Dataset)
   └─ Return (dataset, cache_key)

3. Feature Engineering (L3)
   ├─ Topological sort of operations
   ├─ For each operation:
   │  ├─ Compute cache key: hash(formula_config, parent_keys)
   │  ├─ Check cache
   │  │  ├─ HIT: Load from cache
   │  │  └─ MISS: Compute via FormulaManager
   │  ├─ Save to cache (xr.Dataset)
   │  └─ Return (dataset, cache_key)
   └─ Merge all operations

4. Preprocessing (L4)
   ├─ Build preprocessing config
   ├─ Compute cache key: hash(preprocessing_config, L3_key)
   ├─ Check cache
   │  ├─ HIT: Load (dataset, handler) from cache
   │  └─ MISS: Create DataHandler
   │     ├─ Instantiate processors via ProcessorRegistry
   │     ├─ Create HandlerConfig
   │     ├─ Create DataHandler(base=data, config=config)
   │     ├─ handler.build()  # Fit learn processors
   │     ├─ Get handler.view(View.LEARN)  # Apply all transforms
   │     └─ Return (dataset, handler)
   ├─ Save to cache (dataset, handler)
   └─ Return (preprocessed_data, cache_key, handler)

5. Model Training (L5)
   ├─ Build walk-forward configuration (train/infer/step/gap)
   ├─ Compute cache key: hash(model_config + walk_forward, L4_key)
   ├─ Check cache
   │  ├─ HIT: Load (predictions, ModelRunnerResult) from cache
   │  └─ MISS: Train via ModelRunner
   │     ├─ Dynamically import model class and create adapter factory
   │     ├─ Generate SegmentPlan (`make_plan`) from walk-forward config
   │     ├─ Instantiate `ModelRunner` with preprocessing handler + factory
   │     ├─ Runner iterates segments with tqdm progress, fitting fresh adapter per segment
   │     ├─ Predictions aggregated and unstacked back to Dataset
   │     └─ Return (predictions_dataset, runner_result)
   ├─ Save to cache (predictions_dataset, runner_result)
   └─ Return (predictions, cache_key, runner_result)
```

---

## DataHandler Integration

### What is DataHandler?

`DataHandler` is Hindsight's core preprocessing orchestrator. It manages:
- **Processor pipelines**: Shared, learn, and infer stages
- **State management**: Learned parameters (e.g., normalization stats)
- **View management**: Different data views (RAW, LEARN, INFER)
- **Caching**: Intermediate results

### DataHandler Architecture

```
DataHandler
├── base: xr.Dataset (raw input data)
├── config: HandlerConfig
│   ├── shared: List[Processor]  # Transform-only, no fitting
│   ├── learn: List[Processor]   # Fit on train, transform on all
│   ├── infer: List[Processor]   # Transform-only on inference
│   └── mode: PipelineMode (INDEPENDENT or APPEND)
├── cache: Dict
│   ├── shared_view: xr.Dataset
│   ├── learn_view: xr.Dataset
│   ├── infer_view: xr.Dataset
│   └── learn_states: List[Any]  # Fitted processor states
└── methods:
    ├── build()         # Execute pipeline, fit processors
    ├── view(View)      # Get specific view
    └── fetch(View, cols)  # Get specific columns from view
```

### Pipeline Modes

#### INDEPENDENT Mode
```
                    ┌─────────────┐
                    │  Raw Data   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Shared    │  (transform-only)
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
       ┌──────▼──────┐          ┌──────▼──────┐
       │    Learn    │          │    Infer    │
       │ (fit+trans) │          │  (trans)    │
       └─────────────┘          └─────────────┘
         LEARN View               INFER View
```

#### APPEND Mode
```
       ┌─────────────┐
       │  Raw Data   │
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │   Shared    │
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │    Infer    │
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │    Learn    │  (sees infer outputs)
       └─────────────┘
         LEARN View
```

### How Spec System Uses DataHandler

#### Step 1: Parse YAML Preprocessing Spec

```yaml
preprocessing:
  mode: "independent"
  shared:
    - type: "per_asset_ffill"
      name: "forward_fill"
      vars: ["close", "volume"]
  learn:
    - type: "cs_zscore"
      name: "normalizer"
      vars: ["close", "volume", "sma_ww20"]
      out_suffix: "_norm"
  infer: []
```

#### Step 2: Instantiate Processors

```python
# In executor._compute_preprocessing()

from src.pipeline.spec.processor_registry import ProcessorRegistry

# ProcessorRegistry dynamically creates processor instances
shared_processors = [
    ProcessorRegistry.create_processor({
        'type': 'per_asset_ffill',
        'name': 'forward_fill',
        'vars': ['close', 'volume']
    })
]
# Returns: [PerAssetFFill(name='forward_fill', vars=['close', 'volume'])]

learn_processors = [
    ProcessorRegistry.create_processor({
        'type': 'cs_zscore',
        'name': 'normalizer',
        'vars': ['close', 'volume', 'sma_ww20'],
        'out_suffix': '_norm'
    })
]
# Returns: [CSZScore(name='normalizer', vars=[...], out_suffix='_norm')]
```

#### Step 3: Create HandlerConfig

```python
from src.pipeline.data_handler import HandlerConfig, PipelineMode

# Map YAML mode string to PipelineMode enum
mode_map = {
    'independent': PipelineMode.INDEPENDENT,
    'append': PipelineMode.APPEND,
}
mode = mode_map['independent']  # PipelineMode.INDEPENDENT

# Create config
handler_config = HandlerConfig(
    shared=shared_processors,     # [PerAssetFFill(...)]
    learn=learn_processors,       # [CSZScore(...)]
    infer=[],                     # []
    mode=mode,                    # PipelineMode.INDEPENDENT
    feature_cols=[],              # Not used in preprocessing-only mode
    label_cols=[]                 # Not used in preprocessing-only mode
)
```

#### Step 4: Create and Build DataHandler

```python
from src.pipeline.data_handler import DataHandler

# Create handler with input data and config
handler = DataHandler(base=features_data, config=handler_config)

# Build the pipeline
handler.build()
```

**What happens in `handler.build()`:**

1. **Apply shared processors** (transform-only):
   ```python
   shared_view = base_data
   for processor in shared_processors:
       shared_view = processor.transform(shared_view)
   # Result: close and volume are forward-filled
   ```

2. **Apply learn processors** (fit + transform):
   ```python
   learn_view = shared_view
   learn_states = []
   for processor in learn_processors:
       # Fit on the data
       processor.fit(learn_view)
       # Transform the data
       learn_view = processor.transform(learn_view)
       # Store learned state
       learn_states.append(processor.get_state())
   # Result: close_norm, volume_norm, sma_ww20_norm created
   # States: {mean: [...], std: [...]} for each variable
   ```

3. **Cache results**:
   ```python
   handler.cache['shared_view'] = shared_view
   handler.cache['learn_view'] = learn_view
   handler.cache['learn_states'] = learn_states
   ```

#### Step 5: Get Preprocessed Data

```python
from src.pipeline.data_handler import View

# Get the LEARN view (shared + learn transformations applied)
preprocessed_data = handler.view(View.LEARN)

# This dataset now contains:
# - Original variables: open, high, low, close, volume, etc.
# - Forward-filled: close, volume (in-place)
# - Normalized: close_norm, volume_norm, sma_ww20_norm (new variables)
```

#### Step 6: Cache Handler and Data

```python
# The executor caches BOTH the dataset and the handler
cache_manager.save(
    stage=CacheStage.L4_PREPROCESSED,
    key=cache_key,
    data=preprocessed_data,      # xr.Dataset
    state=handler                # DataHandler with learned states
)

# On cache hit, both are restored:
preprocessed_data, handler = cache_manager.load(...)
```

### Why Cache the Handler?

The `DataHandler` contains critical information:

1. **Learned States**: Normalization parameters, fill strategies, etc.
   ```python
   handler.cache['learn_states'][0]  # CSZScore state
   # {'mean': array([...]), 'std': array([...])}
   ```

2. **Pipeline Configuration**: Exact processors and order
   ```python
   handler.config.shared  # [PerAssetFFill(...)]
   handler.config.learn   # [CSZScore(...)]
   ```

3. **Inference Capability**: Can transform new data consistently
   ```python
   # Later, on new data:
   new_preprocessed = handler.view(View.INFER)
   # Uses cached learn_states, no refitting
   ```

---

## ModelAdapter Integration

### What is ModelAdapter?

`ModelAdapter` is Hindsight's interface for machine learning models. It provides:
- **Uniform API**: Same interface for sklearn, LightGBM, PyTorch, etc.
- **Data handling**: Automatic stacking/unstacking of xarray data
- **Prediction wrapping**: Converts model outputs back to xarray
- **State management**: Fitted model parameters

### ModelAdapter Architecture

```
ModelAdapter (Abstract Base Class)
├── model: Any (sklearn, lightgbm, pytorch, etc.)
├── handler: DataHandler (for data stacking)
├── output_var: str (prediction variable name)
└── methods:
    ├── fit(ds, features, label)      # Train model
    ├── predict(ds, features)         # Generate predictions
    ├── stack_features(ds, features)  # xarray → numpy
    └── unstack_predictions(preds)    # numpy → xarray

SklearnAdapter (Concrete Implementation)
├── Inherits from ModelAdapter
├── Wraps any sklearn-compatible model
├── Handles 2D numpy arrays
└── Supports classification and regression
```

### How Spec System Uses ModelAdapter

#### Step 1: Parse YAML Model Spec

```yaml
model:
  adapter: "sklearn"
  type: "RandomForestRegressor"
  params:
    n_estimators: 50
    max_depth: 14
    random_state: 42
  features:
    - "close_norm"
    - "volume_norm"
    - "sma_ww20_norm"
    - "sma_ww50_norm"
    - "ema_ww12_norm"
    - "ema_ww26_norm"
  target: "close"
  adapter_params:
    output_var: "close_pred"
    use_proba: false
```

#### Step 2: Dynamically Import Model Class

```python
# In executor._compute_model()

import importlib

adapter_type = "sklearn"  # From spec
model_type = "RandomForestRegressor"  # From spec

# Try common sklearn modules
sklearn_modules = [
    'sklearn.linear_model',
    'sklearn.ensemble',
    'sklearn.tree',
    'sklearn.svm',
    'sklearn.neighbors',
]

model_class = None
for module_name in sklearn_modules:
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, model_type):
            model_class = getattr(module, model_type)
            break
    except ImportError:
        continue

# Result: model_class = sklearn.ensemble.RandomForestRegressor
```

#### Step 3: Instantiate Model

```python
# Get params from spec
params = {
    'n_estimators': 50,
    'max_depth': 14,
    'random_state': 42
}

# Create model instance
model = model_class(**params)
# Result: RandomForestRegressor(n_estimators=50, max_depth=14, ...)
```

#### Step 4: Create DataHandler for Adapter

```python
from src.pipeline.data_handler import DataHandler, HandlerConfig, PipelineMode

# The adapter needs a DataHandler for consistent data stacking
# We create a minimal one (no processors, just for stacking)
handler_config = HandlerConfig(
    shared=[],
    learn=[],
    infer=[],
    mode=PipelineMode.INDEPENDENT,
    feature_cols=features,  # ['close_norm', 'volume_norm', ...]
    label_cols=[target]     # ['close']
)

handler = DataHandler(base=preprocessed_data, config=handler_config)
handler.build()
```

**Why does the adapter need a DataHandler?**

The adapter uses the handler's stacking logic to convert xarray → numpy:

```python
# xarray.Dataset with dimensions (time, asset)
# → numpy array with shape (n_samples, n_features)

# The handler knows how to:
# 1. Extract specified variables
# 2. Stack time and asset dimensions
# 3. Handle NaNs consistently
# 4. Unstack predictions back to xarray
```

#### Step 5: Create ModelAdapter

```python
from src.pipeline.model.adapter import SklearnAdapter

# Get adapter params
adapter_params = {
    'output_var': 'close_pred',
    'use_proba': False
}

# Create adapter
adapter = SklearnAdapter(
    model=model,                      # RandomForestRegressor instance
    handler=handler,                  # DataHandler for stacking
    output_var='close_pred',          # Name for predictions
    use_proba=False                   # Regression, not classification
)
```

#### Step 6: Fit Model

```python
features = ['close_norm', 'volume_norm', 'sma_ww20_norm', ...]
target = 'close'

# Fit the model
adapter.fit(
    ds=preprocessed_data,  # xr.Dataset
    features=features,     # List of feature variable names
    label=target           # Target variable name
)
```

**What happens in `adapter.fit()`:**

1. **Stack features**:
   ```python
   # Extract feature variables from dataset
   feature_arrays = [preprocessed_data[var] for var in features]
   
   # Stack using handler
   X = handler.stack_features(preprocessed_data, features)
   # Shape: (n_samples, n_features)
   # Example: (2260928, 6) for 253 assets × 8928 timesteps
   ```

2. **Stack target**:
   ```python
   y = handler.stack_features(preprocessed_data, [target])
   # Shape: (n_samples, 1)
   ```

3. **Remove NaNs**:
   ```python
   # Find rows with no NaNs
   valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
   X_clean = X[valid_mask]
   y_clean = y[valid_mask].ravel()
   ```

4. **Fit model**:
   ```python
   model.fit(X_clean, y_clean)
   # RandomForestRegressor is now fitted
   ```

#### Step 7: Generate Predictions

```python
# Generate predictions
pred_da = adapter.predict(
    ds=preprocessed_data,
    features=features
)
```

**What happens in `adapter.predict()`:**

1. **Stack features** (same as fit):
   ```python
   X = handler.stack_features(preprocessed_data, features)
   ```

2. **Predict**:
   ```python
   predictions = model.predict(X)
   # Shape: (n_samples,)
   ```

3. **Unstack to xarray**:
   ```python
   # Convert back to xarray.DataArray
   pred_da = handler.unstack_predictions(
       predictions,
       template=preprocessed_data[target],  # Use target shape as template
       output_var='close_pred'
   )
   # Result: xr.DataArray with dims (time, asset)
   ```

4. **Return**:
   ```python
   return pred_da  # xr.DataArray named 'close_pred'
   ```

#### Step 8: Add Predictions to Dataset

```python
# Create output dataset with predictions
output_ds = preprocessed_data.copy()
output_ds['close_pred'] = pred_da

# Now output_ds contains:
# - All original variables
# - All preprocessed variables (*_norm)
# - Predictions (close_pred)
```

#### Step 9: Cache Adapter and Predictions

```python
# Cache both predictions and fitted adapter
cache_manager.save(
    stage=CacheStage.L5_MODEL,
    key=cache_key,
    data=output_ds,      # xr.Dataset with predictions
    state=adapter        # Fitted SklearnAdapter
)

# On cache hit, both are restored:
predictions, adapter = cache_manager.load(...)

# The adapter can be used for inference on new data:
new_predictions = adapter.predict(new_data, features)
```

### ModelRunner Execution (Current)

While the above steps describe the adapter mechanics, the **pipeline executor now wraps them inside `ModelRunner`**:

1. **Walk-forward plan**: build `SegmentConfig` from `model.walk_forward` (train/infer/span/step/gap/start/end) and generate a `SegmentPlan` with `make_plan`.
2. **Factory per segment**: create a factory that instantiates a fresh adapter (and underlying sklearn model) for each segment. The preprocessing `DataHandler`—with learned states cached at L4—is reused so stacking/unstacking remain consistent.
3. **Progress reporting**: `ModelRunner.run()` iterates `plan` with `tqdm(desc="Walk-forward segments", unit="segment")`, printing live segment progress.
4. **Segment loop**:
   - fit learn processors on train slice (via handler),
   - transform inference slice using learned states,
   - fit adapter using factory-provided model,
   - predict on inference slice,
   - scatter predictions back into global buffer according to `runner_params.overlap_policy` (`"last"` by default).
5. **Results**: returns `ModelRunnerResult(pred_ds, segment_states, attrs)`, which the cache stores alongside the predictions dataset. On cache hit both predictions and runner metadata are restored.

This keeps the YAML spec thin (declarative) while respecting the library’s existing walk-forward architecture (`example.py`).

---

## Caching System

### Cache Architecture

```
GlobalCacheManager
├── cache_dir: Path (e.g., /home/user/data/hindsight_cache/)
├── metadata_manager: MetadataManager
└── methods:
    ├── get_or_compute(stage, config, parent_keys, compute_fn)
    ├── get_or_compute_with_state(...)  # For handlers/adapters
    ├── compute_key(stage, config, parent_keys)
    └── get_stats()

Cache Directory Structure:
cache_dir/
├── l2_post/
│   ├── a40d6ea463d5443f.nc          # Dataset
│   ├── a40d6ea463d5443f.attrs.pkl   # Non-NetCDF attrs
│   └── a40d6ea463d5443f.meta.json   # Metadata
├── l3_features/
│   ├── 26d8ca3930ff1665.nc
│   ├── 26d8ca3930ff1665.attrs.pkl
│   └── 26d8ca3930ff1665.meta.json
├── l4_prep/
│   ├── d167e13d1b36a8f2.pkl         # (dataset, DataHandler)
│   └── d167e13d1b36a8f2.meta.json
└── l5_model/
    ├── 8b08547aa0704c1e.pkl         # (predictions, ModelRunnerResult)
    └── 8b08547aa0704c1e.meta.json
```

- **Attribute preservation:** when datasets carry objects NetCDF cannot serialize (e.g., `coords['time'].attrs['indexes']` holding a `TimeSeriesIndex`), the cache manager strips them before writing the `.nc`, stores them in the companion `.attrs.pkl`, and automatically restores them on load. Callers see the exact original dataset structure, including dt-accessor metadata.

### Content-Addressable Keys

Cache keys are computed as:

```python
def compute_key(stage, config, parent_keys):
    """
    Compute content-addressable cache key.
    
    Args:
        stage: CacheStage enum (L2, L3, L4, L5)
        config: Dict of configuration for this stage
        parent_keys: List of cache keys from dependencies
        
    Returns:
        str: 16-character hex hash
    """
    # Serialize config to stable JSON
    config_json = json.dumps(config, sort_keys=True)
    
    # Combine with parent keys
    combined = f"{stage.value}:{config_json}:{':'.join(parent_keys)}"
    
    # Hash
    hash_obj = hashlib.sha256(combined.encode())
    return hash_obj.hexdigest()[:16]
```

**Example:**

```python
# L2 (Data Loading)
L2_key = hash("l2_post:{'provider':'crypto','dataset':'spot/binance',...}:")
# Result: "a40d6ea463d5443f"

# L3 (Features)
L3_key = hash("l3_features:{'formulas':{'sma':[{...}],...}}:a40d6ea463d5443f")
# Result: "26d8ca3930ff1665"

# L4 (Preprocessing)
L4_key = hash("l4_prep:{'shared':[...],'learn':[...]}:26d8ca3930ff1665")
# Result: "d167e13d1b36a8f2"

# L5 (Model)
L5_key = hash("l5_model:{'type':'RandomForest',...}:d167e13d1b36a8f2")
# Result: "8b08547aa0704c1e"
```

### Cache Reuse Logic

```python
def get_or_compute(stage, config, parent_keys, compute_fn):
    """
    Get from cache or compute if missing.
    
    Flow:
    1. Compute cache key from config + parent_keys
    2. Check if key exists in cache
    3. If HIT: Load and return
    4. If MISS: Call compute_fn(), save, return
    """
    # Compute key
    key = compute_key(stage, config, parent_keys)
    
    # Check cache
    if exists(cache_dir / stage.value / f"{key}.nc"):
        print(f"Cache hit: {stage.value}/{key[:16]}...")
        data = load_from_cache(key)
        metadata_manager.update_access(key)  # Update access time
        return data, key
    
    # Cache miss - compute
    print(f"Cache miss: {stage.value}/{key[:16]}... - computing...")
    data = compute_fn()
    
    # Save to cache
    save_to_cache(key, data)
    metadata_manager.save(key, {
        'stage': stage.value,
        'config': config,
        'parent_keys': parent_keys,
        'created_at': datetime.now(),
        'size_bytes': get_size(data)
    })
    
    return data, key
```

### State Caching (L4 and L5)

For stages that produce both data and state (handlers, adapters):

```python
def get_or_compute_with_state(stage, config, parent_keys, compute_fn):
    """
    Get from cache or compute, handling state objects.
    
    Returns:
        Tuple[xr.Dataset, Any, str]: (data, state, cache_key)
    """
    key = compute_key(stage, config, parent_keys)
    
    # Check cache
    data_path = cache_dir / stage.value / f"{key}.nc"
    state_path = cache_dir / stage.value / f"{key}_state.pkl"
    
    if data_path.exists() and state_path.exists():
        print(f"Cache hit: {stage.value}/{key[:16]}...")
        data = xr.open_dataset(data_path)
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
        return data, state, key
    
    # Cache miss - compute
    print(f"Cache miss: {stage.value}/{key[:16]}... - computing...")
    data, state = compute_fn()  # Returns (dataset, handler/adapter)
    
    # Save both
    data.to_netcdf(data_path)
    with open(state_path, 'wb') as f:
        pickle.dump(state, f)
    
    return data, state, key
```

### Cache Invalidation

Caches are automatically invalidated when:

1. **Config changes**: Different parameters → different key
2. **Parent changes**: Different input data → different key
3. **Code changes**: Manual cache clear required

```python
# Example: Changing a parameter invalidates downstream caches

# Original
config_A = {'type': 'cs_zscore', 'vars': ['close']}
L4_key_A = hash(config_A + L3_key)  # "d167e13d1b36a8f2"

# Modified
config_B = {'type': 'cs_zscore', 'vars': ['close', 'volume']}
L4_key_B = hash(config_B + L3_key)  # "f8a2c9d4e1b7063a" (different!)

# L5 caches that depend on L4_key_A are not reused
```

---

## Processor Registry

### What is ProcessorRegistry?

A factory for dynamically creating processor instances from YAML configuration.

### Implementation

```python
# src/pipeline/spec/processor_registry.py

class ProcessorRegistry:
    """
    Factory for creating processor instances from configuration.
    """
    
    # Map YAML type names to processor classes
    _registry = {
        'cs_zscore': CSZScore,
        'per_asset_ffill': PerAssetFFill,
        'formula_eval': FormulaEval,
        # Add more as needed
    }
    
    @classmethod
    def create_processor(cls, config: Dict[str, Any]) -> Processor:
        """
        Create a processor instance from configuration.
        
        Args:
            config: Dict with 'type' and processor-specific params
            
        Returns:
            Processor instance
            
        Example:
            config = {
                'type': 'cs_zscore',
                'name': 'normalizer',
                'vars': ['close', 'volume'],
                'out_suffix': '_norm'
            }
            processor = ProcessorRegistry.create_processor(config)
            # Returns: CSZScore(name='normalizer', vars=[...], ...)
        """
        processor_type = config.get('type')
        if processor_type not in cls._registry:
            raise ValueError(f"Unknown processor type: {processor_type}")
        
        # Get processor class
        processor_class = cls._registry[processor_type]
        
        # Extract params (everything except 'type')
        params = {k: v for k, v in config.items() if k != 'type'}
        
        # Instantiate
        return processor_class(**params)
```

### Usage in Executor

```python
# In executor._compute_preprocessing()

from src.pipeline.spec.processor_registry import ProcessorRegistry

# YAML config
preprocessing_spec = {
    'shared': [
        {'type': 'per_asset_ffill', 'name': 'ffill', 'vars': ['close']}
    ],
    'learn': [
        {'type': 'cs_zscore', 'name': 'norm', 'vars': ['close'], 'out_suffix': '_norm'}
    ]
}

# Create processors
shared_processors = [
    ProcessorRegistry.create_processor(cfg)
    for cfg in preprocessing_spec['shared']
]
# Result: [PerAssetFFill(name='ffill', vars=['close'])]

learn_processors = [
    ProcessorRegistry.create_processor(cfg)
    for cfg in preprocessing_spec['learn']
]
# Result: [CSZScore(name='norm', vars=['close'], out_suffix='_norm')]
```

---

## Under the Hood: Detailed Mechanics

### Complete Execution Trace

Let's trace a complete pipeline execution with detailed internal operations.

#### YAML Spec

```yaml
spec_version: "1.0"
name: "example_pipeline"

time_range:
  start: "2023-01-01"
  end: "2023-12-31"

data:
  crypto_prices:
    provider: "crypto"
    dataset: "spot/binance"
    frequency: "H"

features:
  operations:
    - name: "indicators"
      formulas:
        sma: [{window: 20}]
        rsi: [{window: 14}]

preprocessing:
  mode: "independent"
  shared:
    - type: "per_asset_ffill"
      vars: ["close"]
  learn:
    - type: "cs_zscore"
      vars: ["close", "sma_ww20"]
      out_suffix: "_norm"

model:
  adapter: "sklearn"
  type: "LinearRegression"
  params: {fit_intercept: true}
  features: ["close_norm", "sma_ww20_norm"]
  target: "close"
```

#### Execution Trace

```python
# 1. PARSE YAML
# --------------
spec = SpecParser.load_from_yaml("example_pipeline.yaml")
# Result: PipelineSpec object with all fields populated

# 2. CREATE EXECUTOR
# ------------------
executor = PipelineExecutor(
    cache_manager=GlobalCacheManager(cache_dir="/path/to/cache"),
    data_manager=DataManager(),
    formula_manager=FormulaManager()
)

# 3. EXECUTE PIPELINE
# -------------------
result = executor.execute(spec)

# 3.1. DATA LOADING (L2)
# ----------------------
# Build config
data_config = {
    'provider': 'crypto',
    'dataset': 'spot/binance',
    'frequency': 'H',
    'time_range': {'start': '2023-01-01', 'end': '2023-12-31'}
}

# Compute cache key
L2_key = cache_manager.compute_key(
    stage=CacheStage.L2_POSTPROCESSED,
    config=data_config,
    parent_keys=[]
)
# Result: "a40d6ea463d5443f"

# Check cache
if not cache_manager.exists(L2_key):
    # MISS - Load data
    raw_data = data_manager.load_builtin(
        provider='crypto',
        dataset='spot/binance',
        start='2023-01-01',
        end='2023-12-31'
    )
    # Result: xr.Dataset with dims (year, month, day, hour, asset)
    #         Variables: open, high, low, close, volume
    #         Shape: (1, 12, 31, 24, 253)
    
    # Save to cache
    cache_manager.save(L2_key, raw_data)

# Load from cache
dataset = cache_manager.load(L2_key)

# 3.2. FEATURE ENGINEERING (L3)
# -----------------------------
# For operation "indicators"
formula_config = {
    'name': 'indicators',
    'formulas': {
        'sma': [{'window': 20}],
        'rsi': [{'window': 14}]
    }
}

# Compute cache key
L3_key = cache_manager.compute_key(
    stage=CacheStage.L3_FEATURES,
    config=formula_config,
    parent_keys=[L2_key]
)
# Result: "26d8ca3930ff1665"

# Check cache
if not cache_manager.exists(L3_key):
    # MISS - Compute features
    features_data = formula_manager.evaluate_bulk(
        formula_names={'sma': [{'window': 20}], 'rsi': [{'window': 14}]},
        context={'_dataset': dataset}
    )
    # Internal operations:
    # 1. Parse formula "sma" with window=20
    # 2. Call sma(dataset, window=20)
    #    - dataset.dt.rolling(dim='time', window=20).reduce(mean)
    #    - Compute mask on-demand from dataset
    #    - Apply rolling mean with mask
    #    - Return xr.DataArray named "sma_ww20"
    # 3. Parse formula "rsi" with window=14
    # 4. Call rsi(dataset, window=14)
    #    - Compute gain and loss
    #    - Apply EMA
    #    - Return xr.DataArray named "rsi"
    # 5. Merge into dataset
    
    # Result: dataset with new variables sma_ww20, rsi
    
    # Save to cache
    cache_manager.save(L3_key, features_data)

# Load from cache
features_data = cache_manager.load(L3_key)

# 3.3. PREPROCESSING (L4)
# -----------------------
preprocessing_config = {
    'mode': 'independent',
    'shared': [{'type': 'per_asset_ffill', 'vars': ['close']}],
    'learn': [{'type': 'cs_zscore', 'vars': ['close', 'sma_ww20'], 'out_suffix': '_norm'}]
}

# Compute cache key
L4_key = cache_manager.compute_key(
    stage=CacheStage.L4_PREPROCESSED,
    config=preprocessing_config,
    parent_keys=[L3_key]
)
# Result: "d167e13d1b36a8f2"

# Check cache
if not cache_manager.exists(L4_key):
    # MISS - Create DataHandler
    
    # Step 1: Instantiate processors
    shared_processors = [
        PerAssetFFill(name='ffill', vars=['close'])
    ]
    learn_processors = [
        CSZScore(name='norm', vars=['close', 'sma_ww20'], out_suffix='_norm')
    ]
    
    # Step 2: Create config
    handler_config = HandlerConfig(
        shared=shared_processors,
        learn=learn_processors,
        infer=[],
        mode=PipelineMode.INDEPENDENT
    )
    
    # Step 3: Create handler
    handler = DataHandler(base=features_data, config=handler_config)
    
    # Step 4: Build pipeline
    handler.build()
    # Internal operations:
    # 1. Apply shared processors:
    #    - PerAssetFFill.transform(features_data)
    #    - For each asset, forward-fill 'close' variable
    #    - Result: shared_view with filled close values
    # 2. Apply learn processors:
    #    - CSZScore.fit(shared_view)
    #      - Compute mean and std for 'close' and 'sma_ww20'
    #      - Across assets (cross-sectional)
    #      - Store in processor state
    #    - CSZScore.transform(shared_view)
    #      - Create 'close_norm' = (close - mean) / std
    #      - Create 'sma_ww20_norm' = (sma_ww20 - mean) / std
    #    - Result: learn_view with normalized variables
    # 3. Cache views:
    #    - handler.cache['shared_view'] = shared_view
    #    - handler.cache['learn_view'] = learn_view
    #    - handler.cache['learn_states'] = [cszscore_state]
    
    # Step 5: Get preprocessed data
    preprocessed_data = handler.view(View.LEARN)
    # Result: dataset with close_norm, sma_ww20_norm
    
    # Save to cache (data + handler)
    cache_manager.save_with_state(L4_key, preprocessed_data, handler)

# Load from cache
preprocessed_data, handler = cache_manager.load_with_state(L4_key)

# 3.4. MODEL TRAINING (L5)
# ------------------------
model_config = {
    'adapter': 'sklearn',
    'type': 'LinearRegression',
    'params': {'fit_intercept': True},
    'features': ['close_norm', 'sma_ww20_norm'],
    'target': 'close'
}

# Compute cache key
L5_key = cache_manager.compute_key(
    stage=CacheStage.L5_MODEL,
    config=model_config,
    parent_keys=[L4_key]
)
# Result: "8b08547aa0704c1e"

# Check cache
if not cache_manager.exists(L5_key):
    # MISS - Train model
    
    # Step 1: Import model class
    from sklearn.linear_model import LinearRegression
    
    # Step 2: Instantiate model
    model = LinearRegression(fit_intercept=True)
    
    # Step 3: Create handler for adapter
    adapter_handler = DataHandler(
        base=preprocessed_data,
        config=HandlerConfig(
            shared=[],
            learn=[],
            infer=[],
            mode=PipelineMode.INDEPENDENT,
            feature_cols=['close_norm', 'sma_ww20_norm'],
            label_cols=['close']
        )
    )
    adapter_handler.build()
    
    # Step 4: Create adapter
    adapter = SklearnAdapter(
        model=model,
        handler=adapter_handler,
        output_var='close_pred'
    )
    
    # Step 5: Fit model
    adapter.fit(
        ds=preprocessed_data,
        features=['close_norm', 'sma_ww20_norm'],
        label='close'
    )
    # Internal operations:
    # 1. Stack features:
    #    X = adapter_handler.stack_features(preprocessed_data, features)
    #    - Extract close_norm and sma_ww20_norm
    #    - Stack (year, month, day, hour, asset) → (samples, features)
    #    - Shape: (2260928, 2) for 253 assets × 8928 timesteps
    # 2. Stack target:
    #    y = adapter_handler.stack_features(preprocessed_data, ['close'])
    #    - Shape: (2260928, 1)
    # 3. Remove NaNs:
    #    valid = ~(np.isnan(X).any(1) | np.isnan(y).any(1))
    #    X_clean = X[valid]  # Shape: (~2000000, 2)
    #    y_clean = y[valid].ravel()
    # 4. Fit:
    #    model.fit(X_clean, y_clean)
    #    - LinearRegression learns coefficients
    #    - coef_ = [w1, w2], intercept_ = b
    
    # Step 6: Predict
    predictions = adapter.predict(
        ds=preprocessed_data,
        features=['close_norm', 'sma_ww20_norm']
    )
    # Internal operations:
    # 1. Stack features (same as fit)
    # 2. Predict:
    #    y_pred = model.predict(X)  # Shape: (2260928,)
    # 3. Unstack:
    #    pred_da = adapter_handler.unstack_predictions(
    #        y_pred,
    #        template=preprocessed_data['close']
    #    )
    #    - Reshape (2260928,) → (1, 12, 31, 24, 253)
    #    - Create xr.DataArray with same coords as 'close'
    #    - Name it 'close_pred'
    
    # Step 7: Add to dataset
    output_ds = preprocessed_data.copy()
    output_ds['close_pred'] = predictions
    
    # Save to cache (predictions + adapter)
    cache_manager.save_with_state(L5_key, output_ds, adapter)

# Load from cache
predictions, runner_result = cache_manager.load_with_state(L5_key)

# 4. RETURN RESULTS
# -----------------
result = ExecutionResult(
    spec=spec,
    raw_data=dataset,
    features_data=features_data,
    preprocessed_data=preprocessed_data,
    model_predictions=predictions,
    fitted_model=runner_result,      # Legacy alias
    model_runner_result=runner_result,
    learned_states=handler,
    cache_keys={
        'data': L2_key,
        'features': L3_key,
        'preprocessing': L4_key,
        'model': L5_key
    },
    execution_time=elapsed_time
)

return result
```

---

## Future Extensions

### 1. Walk-Forward Validation

Integration with `ModelRunner` for time-series cross-validation:

```yaml
model:
  adapter: "sklearn"
  type: "RandomForestRegressor"
  walk_forward:
    train_span_hours: 720   # 30 days
    infer_span_hours: 24    # 1 day
    step_hours: 24          # 1 day step
    min_train_hours: 168   # 7 days minimum
```

```python
# Future implementation in executor
from src.pipeline.model.runner import ModelRunner, make_plan

# Create segment plan
plan = make_plan(
    train_span_hours=720,
    infer_span_hours=24,
    step_hours=24,
    ds_for_bounds=preprocessed_data
)

# Model factory
def create_adapter():
    model = create_model_from_spec(spec.model)
    return SklearnAdapter(model=model, handler=handler, ...)

# Run walk-forward
runner = ModelRunner(
    handler=preprocessing_handler,
    plan=plan,
    model_factory=create_adapter,
    feature_cols=spec.model.features,
    label_col=spec.model.target
)

results = runner.run()
# Returns: List of (train_segment, infer_segment, predictions, metrics)
```

### 2. Additional Model Adapters

#### LightGBM Adapter

```python
# src/pipeline/model/adapter.py

class LightGBMAdapter(ModelAdapter):
    """Adapter for LightGBM models."""
    
    def fit(self, ds, features, label):
        import lightgbm as lgb
        
        X, y = self.stack_data(ds, features, label)
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X, label=y)
        
        # Train
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.params.get('num_boost_round', 100)
        )
    
    def predict(self, ds, features):
        X = self.stack_features(ds, features)
        predictions = self.model.predict(X)
        return self.unstack_predictions(predictions, ds, features[0])
```

```yaml
# YAML usage
model:
  adapter: "lightgbm"
  type: "LGBMRegressor"
  params:
    num_leaves: 31
    learning_rate: 0.05
    num_boost_round: 100
```

#### PyTorch Adapter

```python
# src/pipeline/model/adapter.py

class PyTorchAdapter(ModelAdapter):
    """Adapter for PyTorch models."""
    
    def __init__(self, model_class, model_params, training_params, **kwargs):
        super().__init__(**kwargs)
        self.model_class = model_class
        self.model_params = model_params
        self.training_params = training_params
    
    def fit(self, ds, features, label):
        import torch
        import torch.nn as nn
        
        X, y = self.stack_data(ds, features, label)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create model
        self.model = self.model_class(**self.model_params)
        
        # Training loop
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_params['learning_rate']
        )
        criterion = nn.MSELoss()
        
        for epoch in range(self.training_params['epochs']):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
    
    def predict(self, ds, features):
        import torch
        
        X = self.stack_features(ds, features)
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy()
        
        return self.unstack_predictions(predictions, ds, features[0])
```

```yaml
# YAML usage
model:
  adapter: "pytorch"
  type: "LSTM"
  params:
    input_size: 6
    hidden_size: 128
    num_layers: 2
    dropout: 0.2
  training_params:
    learning_rate: 0.001
    epochs: 100
    batch_size: 32
```

### 3. Ensemble Models

```yaml
model:
  adapter: "ensemble"
  type: "VotingRegressor"
  models:
    - adapter: "sklearn"
      type: "RandomForestRegressor"
      params: {n_estimators: 50}
      weight: 0.4
    - adapter: "lightgbm"
      type: "LGBMRegressor"
      params: {num_leaves: 31}
      weight: 0.3
    - adapter: "sklearn"
      type: "LinearRegression"
      params: {}
      weight: 0.3
  features: ["close_norm", "volume_norm"]
  target: "close"
```

### 4. Hyperparameter Optimization

```yaml
model:
  adapter: "sklearn"
  type: "RandomForestRegressor"
  hyperopt:
    method: "optuna"
    n_trials: 100
    params:
      n_estimators: [50, 100, 200, 500]
      max_depth: [5, 10, 15, 20]
      min_samples_split: [2, 5, 10]
    metric: "mse"
    cv_folds: 5
```

### 5. Feature Selection

```yaml
preprocessing:
  mode: "independent"
  learn:
    - type: "feature_selector"
      method: "mutual_info"
      n_features: 10
      vars: ["close_norm", "volume_norm", "sma_*", "ema_*"]
```

---

## Summary

The pipeline system is a **thin orchestration layer** that:

✅ **Wraps existing infrastructure**
- DataHandler for preprocessing
- ModelAdapter for models
- FormulaManager for features

✅ **Adds hierarchical caching**
- L2: Data loading
- L3: Feature engineering
- L4: Preprocessing
- L5: Model training

✅ **Provides declarative interface**
- YAML-based configuration
- Content-addressable caching
- Automatic dependency tracking

✅ **Maintains compatibility**
- Uses existing processor interface
- Uses existing adapter interface
- Integrates with walk-forward validation

✅ **Enables extensibility**
- Easy to add new adapters
- Easy to add new processors
- Easy to add new features

The system does **not** replace or duplicate existing infrastructure - it orchestrates and enhances it with caching and declarative configuration.
