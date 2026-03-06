# Hindsight Pipeline System

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Pipeline Stages](#pipeline-stages)
4. [Caching System](#caching-system)
5. [YAML Specification](#yaml-specification)
6. [Usage Guide](#usage-guide)
7. [Under the Hood](#under-the-hood)
8. [Examples](#examples)

---

## Overview

The Hindsight Pipeline System is an end-to-end framework for building, training, and deploying quantitative trading strategies. It provides:

- **Declarative YAML specifications** for defining complete pipelines
- **Hierarchical caching (L1-L5)** with content-addressable keys
- **Automatic cache reuse** across different pipeline configurations
- **JIT-compatible operations** using JAX for performance
- **Learned state persistence** for preprocessing and models

### Key Features

- 🚀 **6-18x speedup** from intelligent caching
- 📊 **End-to-end workflow**: Data → Features → Preprocessing → Model → Predictions
- 🔄 **Automatic cache invalidation** when dependencies change
- 🎯 **Content-addressable caching** ensures correctness
- 🧩 **Extensible design** for custom processors and models

---

## Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Pipeline Specification (YAML)               │
│  - Data sources                                                  │
│  - Feature formulas                                              │
│  - Preprocessing steps                                           │
│  - Model configuration                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline Executor                         │
│  Orchestrates execution with caching at each stage              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┴─────────────────────┐
        ↓                     ↓                      ↓
   ┌─────────┐          ┌──────────┐          ┌──────────┐
   │ L2: Data│          │L3:Feature│          │L4: Preproc│
   │ Loading │    →     │Engineer. │    →     │& Normalize│
   └─────────┘          └──────────┘          └──────────┘
        ↓
   ┌──────────┐
   │ L5: Model│
   │ Training │
   └──────────┘
        ↓
   ┌──────────────┐
   │ Predictions  │
   └──────────────┘
```

### Component Hierarchy

```
src/pipeline/
├── cache/                    # Caching infrastructure
│   ├── manager.py           # GlobalCacheManager
│   ├── metadata.py          # MetadataManager
│   └── stages.py            # CacheStage enum (L1-L5)
├── spec/                     # Pipeline specifications
│   ├── schema.py            # PipelineSpec, ModelSpec, etc.
│   ├── parser.py            # YAML → Python objects
│   ├── executor.py          # PipelineExecutor
│   ├── result.py            # ExecutionResult
│   ├── processor_registry.py # Processor instantiation
│   └── model_registry.py    # Model instantiation
└── data_handler/            # Data processing
    ├── core.py              # DataHandler
    └── processors.py        # CSZScore, PerAssetFFill, etc.
```

---

## Pipeline Stages

### L1: Raw Data (Internal)
- Raw xarray Datasets directly from data loaders
- Not directly cached (used internally by loaders)

### L2: Post-Processed Data
**What it does:**
- Loads data from providers (crypto, WRDS, OpenBB, etc.)
- Applies data-level processors (filters, merges, etc.)
- Creates xarray Datasets with proper dimensions

**Cache key based on:**
- Provider and dataset name
- Time range
- Filters and processors
- Frequency

**Example output:**
```python
<xarray.Dataset>
Dimensions:  (year: 1, month: 12, day: 31, hour: 24, asset: 253)
Variables:
  - open, high, low, close, volume
```

### L3: Features
**What it does:**
- Evaluates formulas using the AST system
- Computes technical indicators (SMA, EMA, RSI, etc.)
- Per-operation caching for fine-grained reuse

**Cache key based on:**
- Formula definitions and parameters
- L2 cache key (parent data)

**Example output:**
```python
<xarray.Dataset>
Dimensions:  (year: 1, month: 12, day: 31, hour: 24, asset: 253)
Variables:
  - open, high, low, close, volume  # Original
  - sma_ww20, sma_ww50              # Computed features
  - ema_ww12, ema_ww26
  - rsi
```

### L4: Preprocessed
**What it does:**
- Applies shared processors (forward fill, etc.)
- Fits learn processors (normalization, scaling)
- Stores learned states for inference
- Applies infer processors (transform-only)

**Cache key based on:**
- Preprocessing configuration
- L3 cache key (parent features)

**Learned states:**
- Normalization parameters (mean, std)
- Scaling factors
- Any stateful transformations

**Example output:**
```python
<xarray.Dataset>
Dimensions:  (year: 1, month: 12, day: 31, hour: 24, asset: 253)
Variables:
  - open, high, low, close, volume     # Original
  - sma_ww20, sma_ww50, ema_ww12, ...  # Features
  - close_norm, volume_norm, ...       # Normalized
  
Learned States:
  normalizer: {mu__close, sd__close, mu__volume, sd__volume, ...}
```

### L5: Model Predictions
**What it does:**
- Creates sklearn model instance from spec
- Wraps model in `SklearnAdapter` for consistent interface
- Trains model on preprocessed data using adapter
- Generates predictions through adapter
- Caches both predictions and fitted adapter

**Cache key based on:**
- Model type and hyperparameters
- Feature list and target variable
- Adapter parameters
- L4 cache key (parent preprocessed data)

**Integration with existing infrastructure:**
- Uses `SklearnAdapter` from `src.pipeline.model.adapter`
- Uses `DataHandler` for consistent data stacking
- Compatible with `ModelRunner` for walk-forward validation

**Example output:**
```python
<xarray.Dataset>
Dimensions:  (year: 1, month: 12, day: 31, hour: 24, asset: 253)
Variables:
  - [all previous variables]
  - close_pred                        # Model predictions

Fitted Adapter:
  SklearnAdapter(model=LinearRegression(...), ...)
```

---

## Caching System

### Content-Addressable Keys

Each cache key is a hash of:
1. **Stage configuration**: All parameters for that stage
2. **Parent cache keys**: Dependencies from previous stages

This ensures:
- ✅ Same config + same inputs = same cache key (deterministic)
- ✅ Changed config = different cache key (automatic invalidation)
- ✅ Changed inputs = different cache key (propagates changes)

### Cache Key Computation

```python
def _compute_key(stage, config, parent_keys):
    """
    Compute content-addressable cache key.
    
    Args:
        stage: CacheStage enum (L2, L3, L4, L5)
        config: Configuration dict for this stage
        parent_keys: List of cache keys this depends on
    
    Returns:
        16-character hex hash
    """
    # Normalize config (sort keys, handle special types)
    normalized = _normalize_config(config)
    
    # Create deterministic representation
    content = {
        'stage': stage.value,
        'config': normalized,
        'parents': sorted(parent_keys)
    }
    
    # Hash with SHA256
    content_str = json.dumps(content, sort_keys=True)
    hash_obj = hashlib.sha256(content_str.encode())
    
    return hash_obj.hexdigest()[:16]
```

### Cache Hierarchy Example

```
Pipeline A: crypto_momentum_baseline
  L2: a40d6ea463d5443f  (crypto/spot/binance, 2023-01-01 to 2023-12-31)
    ↓
  L3: 71e0190e64738beb  (sma[20,50], ema[12,26], rsi[14])
    ↓
  L4: 6154db95785f9a2c  (ffill + cs_zscore[close,volume,sma,ema])
    ↓
  L5: 3f8a2b1c4d5e6f7a  (linear model, 6 features)

Pipeline B: crypto_momentum_enhanced
  L2: a40d6ea463d5443f  ← REUSED (same data source)
    ↓
  L3: 9c3d7f2a8b4e1f5d  (additional features: sma[100], ema[50])
    ↓
  L4: 2e9f8c1b3a5d7f4e  (same preprocessing, different features)
    ↓
  L5: 8d4c2f1a9b3e7f5c  (lightgbm model, 8 features)
```

### Cache Storage

```
~/data/hindsight_cache/
├── l2_post/
│   ├── a40d6ea463d5443f.nc          # Dataset
│   ├── a40d6ea463d5443f.attrs.pkl   # Non-NetCDF attrs (TimeSeriesIndex, etc.)
│   └── a40d6ea463d5443f.meta.json   # Metadata
├── l3_features/
│   ├── 71e0190e64738beb.nc
│   ├── 71e0190e64738beb.attrs.pkl
│   └── 71e0190e64738beb.meta.json
├── l4_prep/
│   ├── 6154db95785f9a2c.pkl         # (dataset, DataHandler)
│   └── 6154db95785f9a2c.meta.json
└── l5_model/
    ├── 3f8a2b1c4d5e6f7a.pkl         # (predictions, ModelRunnerResult)
    └── 3f8a2b1c4d5e6f7a.meta.json
```

- Each `.nc` file may be accompanied by an `.attrs.pkl` that preserves attributes NetCDF cannot encode (notably the `TimeSeriesIndex` used by the `.dt` accessor). When loading from cache, these extras are rehydrated automatically.

---

## YAML Specification

### Complete Spec Structure

```yaml
spec_version: "1.0"
name: "pipeline_name"
version: "1.0"

# Global time range
time_range:
  start: "YYYY-MM-DD"
  end: "YYYY-MM-DD"

# Data sources (L2)
data:
  source_name:
    provider: "crypto" | "wrds" | "open_bb"
    dataset: "spot/binance" | "crsp_daily" | ...
    frequency: "H" | "D" | "M"
    filters: {}
    processors: []

# Feature engineering (L3)
features:
  operations:
    - name: "operation_name"
      formulas:
        formula_name:
          - {param1: value1, param2: value2}
          - {param1: value3}

# Preprocessing (L4)
preprocessing:
  mode: "independent" | "append"
  
  shared:
    - type: "processor_type"
      name: "processor_name"
      # processor-specific params
  
  learn:
    - type: "processor_type"
      name: "processor_name"
      # processor-specific params
  
  infer:
    - type: "processor_type"
      # transform-only processors

# Model training (L5)
model:
  adapter: "sklearn" | "lightgbm" | "pytorch" | ...
  type: "LinearRegression" | "RandomForestRegressor" | any supported model
  params:
    # model-specific hyperparameters passed to the constructor
  features:
    - "feature1"
    - "feature2"
  target: "target_variable"
  adapter_params:
    output_var: "predictions"  # name of prediction variable
    use_proba: false           # use predict_proba instead of predict
  walk_forward:
    train_span_hours: 120
    infer_span_hours: 24
    step_hours: 24
    gap_hours: 0
    start: "YYYY-MM-DDTHH:MM:SS"  # optional, defaults to dataset bounds
    end:   "YYYY-MM-DDTHH:MM:SS"  # optional
  runner_params:
    overlap_policy: "last"  # or "first"

# Metadata (optional)
metadata:
  description: "Pipeline description"
  author: "Author name"
  tags: ["tag1", "tag2"]
```

### Available Processors

**per_asset_ffill**: Forward fill missing values per asset
```yaml
- type: "per_asset_ffill"
  name: "ffill"
  vars: ["close", "volume"]
```

**cs_zscore**: Cross-sectional z-score normalization
```yaml
- type: "cs_zscore"
  name: "normalizer"
  vars: ["close", "volume"]
  out_suffix: "_norm"
  eps: 1e-8
```

**formula_eval**: Evaluate formulas (for complex preprocessing)
```yaml
- type: "formula_eval"
  name: "custom_features"
  formulas:
    log_return:
      - {}
```

### Available Models

The model system integrates with sklearn through the existing `SklearnAdapter` infrastructure. Any sklearn model can be used by specifying its class name.

**LinearRegression**: Simple linear regression
```yaml
model:
  type: "LinearRegression"
  params:
    fit_intercept: true
  adapter_params:
    output_var: "predictions"
```

**RandomForestRegressor**: Random forest ensemble
```yaml
model:
  type: "RandomForestRegressor"
  params:
    n_estimators: 100
    max_depth: 10
    n_jobs: -1
  adapter_params:
    output_var: "predictions"
```

**Other sklearn models**: Any sklearn estimator with `.fit()` and `.predict()` methods
```yaml
model:
  type: "Ridge"  # or "Lasso", "ElasticNet", "GradientBoostingRegressor", etc.
  params:
    alpha: 1.0
  adapter_params:
    output_var: "predictions"
```

The system automatically searches common sklearn modules:
- `sklearn.linear_model`
- `sklearn.ensemble`
- `sklearn.tree`
- `sklearn.svm`
- `sklearn.neighbors`

---

## Usage Guide

### Basic Usage

```python
from src.pipeline.cache import GlobalCacheManager
from src.pipeline.spec import SpecParser, PipelineExecutor
from src.data.managers.data_manager import DataManager

# Setup
cache_manager = GlobalCacheManager(cache_root="~/data/hindsight_cache")
data_manager = DataManager()
executor = PipelineExecutor(cache_manager, data_manager)

# Load and execute pipeline
spec = SpecParser.load_from_yaml("my_pipeline.yaml")
result = executor.execute(spec)

# Access results
print(f"Data: {list(result.data.keys())}")
print(f"Features: {list(result.features_data.data_vars)}")
print(f"Predictions: {result.model_predictions['close_pred']}")

# Access learned states (for inference)
normalizer_state = result.learned_states['normalizer']

# Access fitted model (for inference)
runner = result.model_runner_result
print(runner.attrs)
```

### Running the Example

```bash
# Navigate to examples directory
cd /home/suchismit/projects/hindsight/examples

# Set PYTHONPATH
export PYTHONPATH=/home/suchismit/projects/hindsight:$PYTHONPATH

# Run example (requires JAX environment)
conda activate jax
python run_pipeline_example.py
```

### Expected Output

```
================================================================================
Part 1: Execute Baseline Pipeline
================================================================================

Loading spec: crypto_momentum_baseline.yaml

Pipeline: crypto_momentum_baseline v1.0
  Data: ['crypto_prices']
  Features: 1 operation(s)
  Preprocessing: 1 learn processor(s)
  Model: linear

Executing baseline pipeline (all cache misses expected)...

================================================================================
Executing Pipeline: crypto_momentum_baseline
================================================================================

Loading data sources...
  Source: crypto_prices
Cache miss: l2_post/a40d6ea463d5... - computing...
  Shape: {'year': 1, 'month': 12, 'day': 31, 'hour': 24, 'asset': 253}

================================================================================
Feature Engineering Stage
================================================================================

Operation: momentum_indicators
Cache miss: l3_features/71e0190e6473... - computing...
  Formulas computed: ['sma_ww20', 'sma_ww50', 'ema_ww12', 'ema_ww26', 'rsi']

================================================================================
Preprocessing Stage
================================================================================

Cache miss: l4_prep/6154db95785f... - computing...
  Applying shared processor: forward_fill
  Applying learn processor: normalizer

================================================================================
Model Training/Prediction Stage
================================================================================

Cache miss: l5_model/3f8a2b1c4d5e... - computing...
  Training sklearn/LinearRegression model via ModelRunner...
  Walk-forward segments: 100%|████████████████████████████████| 365/365 [00:05<00:00, 72.1segment/s]
    Segments processed: 365
    Model trained successfully

Execution Summary:
  Total time: 8.45s
  Pipeline: crypto_momentum_baseline v1.0
  Data sources: ['crypto_prices']
  Features computed: 10 variables
  Preprocessed variables: 16
  Model type: linear
  Predictions: close_pred

================================================================================
Part 2: Re-execute Baseline Pipeline
================================================================================

Executing same pipeline again (all cache hits expected)...

Cache hit: l2_post/a40d6ea463d5... - loading...
Cache hit: l3_features/71e0190e6473... - loading...
Cache hit: l4_prep/6154db95785f... - loading...
Cache hit: l5_model/3f8a2b1c4d5e... - loading...

Execution Summary:
  Total time: 0.52s

Performance comparison:
  First execution:  8.45s
  Second execution: 0.52s
  Speedup: 16.3x

Cache verification:
  data: ✓ MATCH
  features: ✓ MATCH
  preprocessing: ✓ MATCH
  model: ✓ MATCH

================================================================================
Part 3: Execute Enhanced Pipeline
================================================================================

Loading spec: crypto_momentum_enhanced.yaml

Pipeline: crypto_momentum_enhanced v2.0
  Data: ['crypto_prices'] (SAME as baseline)
  Features: 2 operation(s) (EXTENDED)
  Model: lightgbm (DIFFERENT)

Expected cache behavior:
  - Data (L2): HIT (same data source)
  - Features (L3): PARTIAL (reuse baseline features, compute new ones)
  - Preprocessing (L4): MISS (different feature set)
  - Model (L5): MISS (different model type)

Cache hit: l2_post/a40d6ea463d5... - loading...
Cache hit: l3_features/71e0190e6473... - loading... (momentum_indicators)
Cache miss: l3_features/9c3d7f2a8b4e... - computing... (volatility_indicators)
Cache miss: l4_prep/2e9f8c1b3a5d... - computing...
Cache miss: l5_model/8d4c2f1a9b3e... - computing...
  Walk-forward segments: 100%|████████████████████████████████| 365/365 [00:08<00:00, 45.1segment/s]

Execution Summary:
  Total time: 3.21s

================================================================================
Summary
================================================================================

Key Takeaways:
  1. First execution computes all stages (8.45s)
  2. Second execution hits all caches (0.52s, 16.3x speedup)
  3. Different specs reuse shared stages automatically
  4. Cache keys are content-addressable
  5. Learned states and ModelRunner results are cached
```

---

## Under the Hood

This section provides detailed explanations of the internal mechanics, data flows, and implementation details of the pipeline system.

### Pipeline Execution Flow

```python
def execute(spec: PipelineSpec) -> ExecutionResult:
    """
    Execute complete pipeline with caching.
    
    Flow:
    1. Data Loading (L2)
    2. Feature Engineering (L3)
    3. Preprocessing (L4)
    4. Model Training (L5)
    """
    
    # Stage 1: Data Loading (L2)
    data, data_key = _execute_data_stage(spec)
    # Computes: hash(data_config)
    
    # Stage 2: Feature Engineering (L3)
    features, features_key = _execute_features_stage(spec, data, data_key)
    # Computes: hash(features_config, data_key)
    
    # Stage 3: Preprocessing (L4)
    preprocessed, prep_key, handler = _execute_preprocessing_stage(
        spec, features, features_key
    )
    # Computes: hash(preprocessing_config, features_key)
    # Returns: (dataset, DataHandler with learned states)
    
    # Stage 4: Model Training (L5)
    predictions, model_key, runner_result = _execute_model_stage(
        spec, preprocessed, prep_key, handler
    )
    # Computes: hash(model_config + walk_forward, prep_key)
    # Returns: (predictions, ModelRunnerResult)
    
    return ExecutionResult(
        data=data,
        features_data=features,
        preprocessed_data=preprocessed,
        model_predictions=predictions,
        learned_states=handler,      # DataHandler
        fitted_model=runner_result,   # Legacy alias
        model_runner_result=runner_result,
        cache_keys={
            'data': data_key,
            'features': features_key,
            'preprocessing': prep_key,
            'model': model_key
        }
    )
```

### DataHandler: Deep Dive

#### What is DataHandler?

`DataHandler` is the core preprocessing orchestrator in Hindsight. It manages:
- **Processor pipelines**: Shared, learn, and infer stages
- **State management**: Learned parameters (normalization stats, etc.)
- **View management**: Different data perspectives (RAW, LEARN, INFER)
- **Caching**: Intermediate transformation results

#### DataHandler Internal Structure

```python
@dataclass
class DataHandler:
    """
    Central orchestrator for data processing pipelines.
    
    Attributes:
        base: xr.Dataset - Raw input data
        config: HandlerConfig - Pipeline configuration
        cache: Dict - Cached views and states
        learn_states: List - Fitted processor states
        infer_states: List - Inference processor states
    """
    base: xr.Dataset
    config: HandlerConfig
    cache: Dict = field(default_factory=dict)
    learn_states: List = field(default_factory=list)
    infer_states: List = field(default_factory=list)
    
    def build(self) -> None:
        """
        Execute the complete pipeline.
        
        Internal flow:
        1. Apply shared processors (transform-only)
        2. Branch into learn and infer paths
        3. Fit learn processors on data
        4. Transform data through both paths
        5. Cache all views and states
        """
        # Step 1: Apply shared processors
        shared_view = self.base
        for processor in self.config.shared:
            shared_view = processor.transform(shared_view)
        self.cache['shared_view'] = shared_view
        
        # Step 2: Apply learn processors (fit + transform)
        learn_view = shared_view
        self.learn_states = []
        for processor in self.config.learn:
            # Fit the processor
            processor.fit(learn_view)
            # Transform the data
            learn_view = processor.transform(learn_view)
            # Store learned state
            state = processor.get_state()
            self.learn_states.append(state)
        self.cache['learn_view'] = learn_view
        
        # Step 3: Apply infer processors (transform-only)
        infer_view = shared_view
        for processor in self.config.infer:
            infer_view = processor.transform(infer_view)
        self.cache['infer_view'] = infer_view
    
    def view(self, which: View) -> xr.Dataset:
        """
        Get a specific data view.
        
        Views:
        - RAW: Original base dataset
        - LEARN: Shared + Learn transformations
        - INFER: Shared + Infer transformations
        """
        if which == View.RAW:
            return self.base
        if which == View.LEARN:
            return self.cache['learn_view']
        if which == View.INFER:
            return self.cache['infer_view']
```

#### How Pipeline Uses DataHandler

**Step 1: Parse YAML Preprocessing Spec**

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
```

**Step 2: Instantiate Processors via ProcessorRegistry**

```python
from src.pipeline.spec.processor_registry import ProcessorRegistry

# ProcessorRegistry maps YAML types to processor classes
shared_processors = [
    ProcessorRegistry.create_processor({
        'type': 'per_asset_ffill',
        'name': 'forward_fill',
        'vars': ['close', 'volume']
    })
]
# Result: [PerAssetFFill(name='forward_fill', vars=['close', 'volume'])]

learn_processors = [
    ProcessorRegistry.create_processor({
        'type': 'cs_zscore',
        'name': 'normalizer',
        'vars': ['close', 'volume', 'sma_ww20'],
        'out_suffix': '_norm'
    })
]
# Result: [CSZScore(name='normalizer', vars=[...], out_suffix='_norm')]
```

**Step 3: Create HandlerConfig**

```python
from src.pipeline.data_handler import HandlerConfig, PipelineMode

handler_config = HandlerConfig(
    shared=shared_processors,     # [PerAssetFFill(...)]
    learn=learn_processors,       # [CSZScore(...)]
    infer=[],                     # []
    mode=PipelineMode.INDEPENDENT,
    feature_cols=[],              # Not used in preprocessing-only
    label_cols=[]                 # Not used in preprocessing-only
)
```

**Step 4: Create and Build DataHandler**

```python
from src.pipeline.data_handler import DataHandler

# Create handler
handler = DataHandler(base=features_data, config=handler_config)

# Build pipeline - this executes all transformations
handler.build()
```

**What happens inside `handler.build()`:**

```python
# 1. Apply shared processors (transform-only)
shared_view = features_data
for processor in [PerAssetFFill(...)]:
    shared_view = processor.transform(shared_view)
    # PerAssetFFill.transform():
    #   - For each asset in the dataset
    #   - Forward-fill 'close' and 'volume' variables
    #   - Fills NaN values with last valid observation
    #   - Result: close and volume have fewer NaNs

# 2. Apply learn processors (fit + transform)
learn_view = shared_view
learn_states = []

processor = CSZScore(name='normalizer', vars=['close', 'volume', 'sma_ww20'], out_suffix='_norm')

# Fit: Learn normalization parameters
processor.fit(learn_view)
# CSZScore.fit():
#   - For each variable (close, volume, sma_ww20)
#   - Compute cross-sectional mean and std at each time step
#   - mean[t] = mean across all assets at time t
#   - std[t] = std across all assets at time t
#   - Store in processor.state = {'mean': {...}, 'std': {...}}

# Transform: Apply normalization
learn_view = processor.transform(learn_view)
# CSZScore.transform():
#   - For each variable
#   - Create new variable: var_norm = (var - mean) / std
#   - close_norm = (close - mean_close) / std_close
#   - volume_norm = (volume - mean_volume) / std_volume
#   - sma_ww20_norm = (sma_ww20 - mean_sma) / std_sma
#   - Add new variables to dataset

# Store learned state
learn_states.append(processor.get_state())
# Returns: {'mean': xr.Dataset, 'std': xr.Dataset}

# 3. Cache results
handler.cache['shared_view'] = shared_view
handler.cache['learn_view'] = learn_view
handler.cache['learn_states'] = learn_states
```

**Step 5: Get Preprocessed Data**

```python
from src.pipeline.data_handler import View

# Get the LEARN view (all transformations applied)
preprocessed_data = handler.view(View.LEARN)

# preprocessed_data now contains:
# - Original variables: open, high, low, close, volume, ema_ww12, etc.
# - Forward-filled: close, volume (modified in-place)
# - Normalized: close_norm, volume_norm, sma_ww20_norm (new variables)
```

**Step 6: Cache Handler and Data**

```python
# The executor caches BOTH dataset and handler
cache_manager.save_with_state(
    stage=CacheStage.L4_PREPROCESSED,
    key=cache_key,
    data=preprocessed_data,      # xr.Dataset
    state=handler                # DataHandler with learned states
)

# On cache hit, both are restored:
preprocessed_data, handler = cache_manager.load_with_state(cache_key)

# The handler can be used for inference:
# - Apply same transformations to new data
# - Use cached learn_states (no refitting)
new_data_preprocessed = handler.view(View.INFER)
```

### ModelAdapter: Deep Dive

#### What is ModelAdapter?

`ModelAdapter` provides a uniform interface for machine learning models, handling:
- **Data conversion**: xarray ↔ numpy
- **Model training**: Fit on training data
- **Prediction**: Generate predictions on new data
- **State management**: Fitted model parameters

#### ModelAdapter Internal Structure

```python
class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    
    Attributes:
        model: The underlying model (sklearn, lightgbm, pytorch, etc.)
        handler: DataHandler for data stacking/unstacking
        output_var: Name for prediction variable
    """
    
    def __init__(self, model, handler, output_var='predictions', **kwargs):
        self.model = model
        self.handler = handler
        self.output_var = output_var
    
    @abstractmethod
    def fit(self, ds: xr.Dataset, features: List[str], label: str):
        """
        Train the model.
        
        Internal flow:
        1. Stack features from xarray to numpy
        2. Stack labels from xarray to numpy
        3. Remove NaN rows
        4. Fit model on clean data
        """
        pass
    
    @abstractmethod
    def predict(self, ds: xr.Dataset, features: List[str]) -> xr.DataArray:
        """
        Generate predictions.
        
        Internal flow:
        1. Stack features from xarray to numpy
        2. Predict using fitted model
        3. Unstack predictions back to xarray
        """
        pass
    
    def stack_features(self, ds: xr.Dataset, features: List[str]) -> np.ndarray:
        """
        Convert xarray features to numpy array.
        
        Process:
        1. Extract specified variables from dataset
        2. Stack all dimensions into (samples, features)
        3. Handle multi-dimensional time (year, month, day, hour)
        """
        # Extract feature arrays
        feature_arrays = [ds[var] for var in features]
        
        # Stack dimensions
        # From: (year, month, day, hour, asset) per feature
        # To: (samples, n_features) where samples = year*month*day*hour*asset
        stacked = []
        for arr in feature_arrays:
            # Flatten all dimensions
            flat = arr.values.ravel()
            stacked.append(flat)
        
        # Combine features
        X = np.column_stack(stacked)
        # Shape: (n_samples, n_features)
        
        return X
    
    def unstack_predictions(self, predictions: np.ndarray, 
                           template: xr.DataArray) -> xr.DataArray:
        """
        Convert numpy predictions back to xarray.
        
        Process:
        1. Reshape flat predictions to match template shape
        2. Create DataArray with same coordinates
        3. Name it according to output_var
        """
        # Reshape to match template
        pred_shaped = predictions.reshape(template.shape)
        
        # Create DataArray
        pred_da = xr.DataArray(
            pred_shaped,
            coords=template.coords,
            dims=template.dims,
            name=self.output_var
        )
        
        return pred_da


class SklearnAdapter(ModelAdapter):
    """
    Adapter for scikit-learn models.
    """
    
    def fit(self, ds: xr.Dataset, features: List[str], label: str):
        """
        Fit sklearn model.
        """
        # Stack features
        X = self.stack_features(ds, features)
        # Shape: (n_samples, n_features)
        
        # Stack labels
        y = self.stack_features(ds, [label])
        # Shape: (n_samples, 1)
        y = y.ravel()  # Flatten to 1D
        
        # Remove NaNs
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        # Fit model
        self.model.fit(X_clean, y_clean)
    
    def predict(self, ds: xr.Dataset, features: List[str]) -> xr.DataArray:
        """
        Generate predictions.
        """
        # Stack features
        X = self.stack_features(ds, features)
        
        # Predict
        predictions = self.model.predict(X)
        # Shape: (n_samples,)
        
        # Unstack to xarray
        template = ds[features[0]]  # Use first feature as shape template
        pred_da = self.unstack_predictions(predictions, template)
        
        return pred_da
```

#### How Pipeline Uses ModelAdapter

**Step 1: Parse YAML Model Spec**

```yaml
model:
  adapter: "sklearn"
  type: "RandomForestRegressor"
  params:
    n_estimators: 50
    max_depth: 14
  features: ["close_norm", "volume_norm", "sma_ww20_norm"]
  target: "close"
  adapter_params:
    output_var: "close_pred"
```

**Step 2: Dynamically Import and Instantiate Model**

```python
import importlib

# Import model class
module = importlib.import_module('sklearn.ensemble')
model_class = getattr(module, 'RandomForestRegressor')

# Instantiate with params
model = model_class(n_estimators=50, max_depth=14)
# Result: RandomForestRegressor(n_estimators=50, max_depth=14)
```

**Step 3: Create DataHandler for Adapter**

```python
# The adapter needs a handler for data stacking
adapter_handler = DataHandler(
    base=preprocessed_data,
    config=HandlerConfig(
        shared=[],
        learn=[],
        infer=[],
        mode=PipelineMode.INDEPENDENT,
        feature_cols=['close_norm', 'volume_norm', 'sma_ww20_norm'],
        label_cols=['close']
    )
)
adapter_handler.build()
```

**Step 4: Create SklearnAdapter**

```python
from src.pipeline.model.adapter import SklearnAdapter

adapter = SklearnAdapter(
    model=model,                      # RandomForestRegressor
    handler=adapter_handler,          # DataHandler for stacking
    output_var='close_pred'           # Prediction variable name
)
```

**Step 5: Fit Model**

```python
features = ['close_norm', 'volume_norm', 'sma_ww20_norm']
target = 'close'

adapter.fit(ds=preprocessed_data, features=features, label=target)
```

**What happens inside `adapter.fit()`:**

```python
# 1. Stack features
X = adapter.stack_features(preprocessed_data, features)
# Process:
#   - Extract close_norm, volume_norm, sma_ww20_norm from dataset
#   - Each has shape: (year=1, month=12, day=31, hour=24, asset=253)
#   - Flatten: 1 * 12 * 31 * 24 * 253 = 2,260,928 samples
#   - Stack: (2260928, 3) for 3 features
# Result: X.shape = (2260928, 3)

# 2. Stack target
y = adapter.stack_features(preprocessed_data, [target])
# Process:
#   - Extract 'close' variable
#   - Flatten same way as features
# Result: y.shape = (2260928, 1)
y = y.ravel()  # Flatten to 1D: (2260928,)

# 3. Remove NaN rows
valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
# Find rows where neither features nor target have NaN
X_clean = X[valid_mask]
y_clean = y[valid_mask]
# Result: X_clean.shape ≈ (2000000, 3) after removing ~260k NaN rows

# 4. Fit model
model.fit(X_clean, y_clean)
# RandomForestRegressor trains on clean data
# Learns: 50 decision trees with max_depth=14
# Stores: Tree structures, split points, leaf values
```

**Step 6: Generate Predictions**

```python
pred_da = adapter.predict(ds=preprocessed_data, features=features)
```

**What happens inside `adapter.predict()`:**

```python
# 1. Stack features (same as fit)
X = adapter.stack_features(preprocessed_data, features)
# Result: X.shape = (2260928, 3)

# 2. Predict
predictions = model.predict(X)
# RandomForestRegressor generates predictions
# For each sample, average predictions from 50 trees
# Result: predictions.shape = (2260928,)

# 3. Unstack to xarray
template = preprocessed_data['close']
# template.shape = (1, 12, 31, 24, 253)

pred_da = adapter.unstack_predictions(predictions, template)
# Process:
#   - Reshape: (2260928,) → (1, 12, 31, 24, 253)
#   - Create xr.DataArray with same coords as template
#   - Name it 'close_pred'
# Result: xr.DataArray with dims (year, month, day, hour, asset)
```

**Step 7: Add Predictions to Dataset**

```python
output_ds = preprocessed_data.copy()
output_ds['close_pred'] = pred_da

# output_ds now contains:
# - All original variables
# - All preprocessed variables (*_norm)
# - Predictions (close_pred)
```

**Step 8: Cache Adapter and Predictions**

```python
cache_manager.save_with_state(
    stage=CacheStage.L5_MODEL,
    key=cache_key,
    data=output_ds,      # xr.Dataset with predictions
    state=adapter        # Fitted SklearnAdapter
)

# On cache hit:
predictions, adapter = cache_manager.load_with_state(cache_key)

# The adapter can generate predictions on new data:
new_predictions = adapter.predict(new_data, features)
```

### ModelRunner Execution (Actual Pipeline)

In the executor, these adapter steps run inside `ModelRunner.run()`:

1. **Segment planning** – construct a `SegmentConfig` from `model.walk_forward` (train/infer/span/step/gap/start/end) and build a `SegmentPlan` via `make_plan`.
2. **Per-segment adapters** – define a factory that instantiates a fresh adapter/model for each segment while reusing the preprocessing `DataHandler` (with cached learn states).
3. **Progress reporting** – iterate the plan with a tqdm progress bar (`"Walk-forward segments"`), fit learn processors on the train slice, apply learned states to the inference slice, fit the segment adapter, and scatter predictions back using `runner_params.overlap_policy`.
4. **Cached artifact** – receive a `ModelRunnerResult` (`pred_ds`, `segment_states`, `attrs`) which is cached alongside the predictions dataset so cache hits restore both immediately.

### Cache Manager: get_or_compute

```python
def get_or_compute(stage, config, parent_keys, compute_fn):
    """
    Get cached result or compute and cache.
    
    This is the core caching primitive used by all stages.
    """
    # 1. Compute content-addressable key
    cache_key = _compute_key(stage, config, parent_keys)
    
    # 2. Try to load from cache
    cache_path = _get_cache_path(stage, cache_key)
    if cache_path.exists():
        result = _load(cache_path, stage)
        metadata_manager.update_access(cache_key, stage)
        print(f"Cache hit: {stage.value}/{cache_key[:12]}...")
        return result, cache_key
    
    # 3. Cache miss - compute
    print(f"Cache miss: {stage.value}/{cache_key[:12]}...")
    result = compute_fn()
    
    # 4. Save to cache
    cache_path = _save(result, stage, cache_key)
    
    # 5. Record metadata
    metadata = CacheMetadata(
        key=cache_key,
        stage=stage.value,
        config=config,
        parent_keys=parent_keys,
        size_bytes=cache_path.stat().st_size,
        created_at=datetime.now()
    )
    metadata_manager.record_cache(metadata)
    
    return result, cache_key
```

### Feature Engineering: Per-Operation Caching

```python
def _execute_features_stage(spec, data, data_key):
    """
    Execute feature engineering with per-operation caching.
    
    Each operation is cached independently, enabling fine-grained reuse.
    """
    merged_data = data
    
    for operation in spec.features.operations:
        # Build config for this operation
        config = {
            'operation_name': operation.name,
            'formulas': operation.formulas
        }
        
        # Compute with caching
        def compute_fn():
            return _compute_formulas(operation.formulas, merged_data)
        
        result, op_key = cache_manager.get_or_compute(
            stage=CacheStage.L3_FEATURES,
            config=config,
            parent_keys=[data_key],
            compute_fn=compute_fn
        )
        
        # Merge results
        merged_data = xr.merge([merged_data, result])
    
    # Compute final cache key for all features
    features_key = _compute_key(
        CacheStage.L3_FEATURES,
        {'operations': [op.to_dict() for op in spec.features.operations]},
        [data_key]
    )
    
    return merged_data, features_key
```

### Preprocessing: State Persistence

```python
def _execute_preprocessing_stage(spec, features, features_key):
    """
    Execute preprocessing with learned state caching.
    
    Returns both transformed dataset AND learned states.
    """
    config = {
        'mode': spec.preprocessing.mode,
        'shared': spec.preprocessing.shared,
        'learn': spec.preprocessing.learn,
        'infer': spec.preprocessing.infer
    }
    
    def compute_fn():
        # Apply shared processors (stateless)
        data = features
        for processor in shared_processors:
            data = processor.transform(data)
        
        # Apply learn processors (fit_transform)
        learned_states = {}
        for processor in learn_processors:
            data, state = processor.fit_transform(data)
            learned_states[processor.name] = state
        
        # Apply infer processors (transform-only)
        for processor in infer_processors:
            data = processor.transform(data)
        
        return data, learned_states
    
    # Use specialized caching for (dataset, states) tuple
    preprocessed, states, prep_key = cache_manager.get_or_compute_with_state(
        stage=CacheStage.L4_PREPROCESSED,
        config=config,
        parent_keys=[features_key],
        compute_fn=compute_fn
    )
    
    return preprocessed, prep_key, states
```

### Model Training: Model Persistence

```python
def _execute_model_stage(spec, preprocessed, prep_key):
    """
    Execute model training with model caching.
    
    Returns both predictions AND fitted model.
    """
    config = {
        'model_type': spec.model.type,
        'params': spec.model.params,
        'features': spec.model.features,
        'target': spec.model.target
    }
    
    def compute_fn():
        # Extract features and target
        X = extract_features(preprocessed, spec.model.features)
        y = preprocessed[spec.model.target].values
        
        # Remove NaN samples
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        # Train model
        model = ModelRegistry.create_model(spec.model)
        model.fit(X_clean, y_clean)
        
        # Generate predictions
        predictions = np.full(len(X), np.nan)
        predictions[valid_mask] = model.predict(X_clean)
        
        # Add predictions to dataset
        output_ds = preprocessed.copy()
        output_ds[f'{spec.model.target}_pred'] = create_dataarray(predictions)
        
        return output_ds, model
    
    # Use specialized caching for (predictions, model) tuple
    predictions, model, model_key = cache_manager.get_or_compute_with_state(
        stage=CacheStage.L5_MODEL,
        config=config,
        parent_keys=[prep_key],
        compute_fn=compute_fn
    )
    
    return predictions, model_key, model
```

---

## Examples

### Example 1: Baseline Pipeline

**File:** `examples/pipeline_specs/crypto_momentum_baseline.yaml`

```yaml
spec_version: "1.0"
name: "crypto_momentum_baseline"
version: "1.0"

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
    - name: "momentum_indicators"
      formulas:
        sma:
          - {window: 20}
          - {window: 50}
        ema:
          - {window: 12}
          - {window: 26}
        rsi:
          - {window: 14}

preprocessing:
  mode: "independent"
  shared:
    - type: "per_asset_ffill"
      name: "forward_fill"
      vars: ["close", "volume"]
  learn:
    - type: "cs_zscore"
      name: "normalizer"
      vars: ["close", "volume", "sma_ww20", "sma_ww50", "ema_ww12", "ema_ww26"]
      out_suffix: "_norm"

model:
  adapter: "sklearn"
  type: "LinearRegression"
  params:
    fit_intercept: true
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
  walk_forward:
    train_span_hours: 120
    infer_span_hours: 24
    step_hours: 24
    gap_hours: 0
  runner_params:
    overlap_policy: "last"
```

**What it does:**
1. Loads hourly crypto data for 2023
2. Computes momentum indicators (SMA, EMA, RSI)
3. Forward fills missing values
4. Normalizes features using cross-sectional z-score
5. Trains linear regression model to predict close price

### Example 2: Enhanced Pipeline

**File:** `examples/pipeline_specs/crypto_momentum_enhanced.yaml`

```yaml
spec_version: "1.0"
name: "crypto_momentum_enhanced"
version: "2.0"

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
    - name: "momentum_indicators"
      formulas:
        sma:
          - {window: 20}
          - {window: 50}
        ema:
          - {window: 12}
          - {window: 26}
        rsi:
          - {window: 14}
    
    - name: "volatility_indicators"
      formulas:
        sma:
          - {window: 100}
        ema:
          - {window: 50}

preprocessing:
  mode: "independent"
  shared:
    - type: "per_asset_ffill"
      name: "forward_fill"
      vars: ["close", "volume"]
  learn:
    - type: "cs_zscore"
      name: "normalizer"
      vars: ["close", "volume", "sma_ww20", "sma_ww50", "sma_ww100", "ema_ww12", "ema_ww26", "ema_ww50"]
      out_suffix: "_norm"

model:
  adapter: "sklearn"
  type: "RandomForestRegressor"
  params:
    n_estimators: 50
    max_depth: 14
    min_samples_leaf: 10
    n_jobs: -1
    random_state: 0
  features:
    - "close_norm"
    - "volume_norm"
    - "sma_ww20_norm"
    - "sma_ww50_norm"
    - "sma_ww100_norm"
    - "ema_ww12_norm"
    - "ema_ww26_norm"
    - "ema_ww50_norm"
  target: "close"
  adapter_params:
    output_var: "close_pred"
  walk_forward:
    train_span_hours: 120
    infer_span_hours: 24
    step_hours: 24
    gap_hours: 0
  runner_params:
    overlap_policy: "last"
```

**What it does:**
1. Loads same data as baseline (reuses L2 cache)
2. Computes baseline features + additional volatility indicators
3. Same preprocessing (reuses baseline normalization for shared features)
4. Trains RandomForest model with more features

**Cache behavior:**
- L2 (Data): ✅ HIT (same data source)
- L3 (Features): ⚡ PARTIAL (reuses baseline, computes new)
- L4 (Preprocessing): ❌ MISS (different feature set)
- L5 (Model): ❌ MISS (different model type)

---

## Performance Characteristics

### Execution Times (Typical)

| Stage | First Run | Cache Hit | Speedup |
|-------|-----------|-----------|---------|
| Data Loading (L2) | 1-2s | 0.1s | 10-20x |
| Features (L3) | 3-4s | 0.1s | 30-40x |
| Preprocessing (L4) | 0.5s | 0.05s | 10x |
| Model Training (L5) | 1-2s | 0.1s | 10-20x |
| **Total** | **6-9s** | **0.5-1s** | **6-18x** |

### Cache Storage (Typical)

| Stage | Size per Entry | Notes |
|-------|----------------|-------|
| L2 (Data) | 50-200 MB | Depends on time range and assets |
| L3 (Features) | 100-400 MB | Includes all computed features |
| L4 (Preprocessing) | 100-400 MB | Dataset + learned states |
| L5 (Model) | 100-400 MB | Predictions + fitted model |

### Memory Usage

- **Peak memory**: ~2-4 GB for typical crypto dataset (1 year, 250 assets, hourly)
- **Streaming**: Not yet implemented (loads full dataset into memory)
- **JAX**: Uses GPU if available, falls back to CPU

---

## Best Practices

### 1. Organize Specs by Strategy

```
pipeline_specs/
├── momentum/
│   ├── baseline_v1.yaml
│   ├── baseline_v2.yaml
│   └── enhanced_v1.yaml
├── mean_reversion/
│   └── ...
└── ml_ensemble/
    └── ...
```

### 2. Use Meaningful Names

```yaml
name: "crypto_momentum_baseline"  # Good
name: "test1"                      # Bad

preprocessing:
  learn:
    - type: "cs_zscore"
      name: "normalizer"           # Good
      name: "proc1"                # Bad
```

### 3. Version Your Specs

```yaml
name: "crypto_momentum"
version: "1.0"  # Baseline
version: "2.0"  # Enhanced features
version: "3.0"  # Different model
```

### 4. Document with Metadata

```yaml
metadata:
  description: "Baseline momentum strategy using linear regression"
  author: "Your Name"
  tags: ["crypto", "momentum", "baseline"]
  notes: "Uses 6 normalized features for prediction"
```

### 5. Clear Cache Periodically

```bash
# Clear all cache
rm -rf ~/data/hindsight_cache

# Clear specific stage
rm -rf ~/data/hindsight_cache/l5_model
```

### 6. Monitor Cache Size

```python
stats = cache_manager.get_stats()
print(f"Total cache size: {stats['total_size_mb']:.2f} MB")
print(f"Total entries: {stats['total_entries']}")
```

---

## Troubleshooting

### Issue: Cache not hitting when it should

**Symptom:** Same spec runs twice, but second run doesn't hit cache

**Causes:**
1. Non-deterministic config (e.g., random seeds, timestamps)
2. Data changed between runs
3. Cache was cleared

**Solution:**
- Ensure all config is deterministic
- Check cache directory exists
- Verify cache keys match between runs

### Issue: Out of memory

**Symptom:** Process killed or OOM error

**Causes:**
1. Dataset too large for available RAM
2. Too many features computed at once
3. Model too large

**Solutions:**
- Reduce time range
- Reduce number of assets
- Use smaller model
- Increase system RAM

### Issue: Slow feature computation

**Symptom:** Feature stage takes very long

**Causes:**
1. Complex formulas (e.g., nested operations)
2. Large window sizes
3. Many assets

**Solutions:**
- Simplify formulas
- Reduce window sizes
- Use JAX JIT compilation (already enabled)
- Check for inefficient formula definitions

### Issue: Model training fails

**Symptom:** Error during model stage

**Causes:**
1. Missing features in dataset
2. All NaN values after preprocessing
3. Model library not installed (LightGBM, XGBoost)

**Solutions:**
- Verify feature names match between preprocessing and model
- Check for sufficient valid data after NaN removal
- Install required model libraries: `pip install lightgbm xgboost`

---

## Extension Points

### Adding Custom Processors

```python
# 1. Create processor class
from src.pipeline.data_handler.processors import Processor

class MyCustomProcessor(Processor):
    def __init__(self, name: str, my_param: float):
        super().__init__(name)
        self.my_param = my_param
    
    def fit(self, ds: xr.Dataset) -> xr.Dataset:
        # Learn parameters from data
        state = compute_state(ds, self.my_param)
        return state
    
    def transform(self, ds: xr.Dataset, state=None) -> xr.Dataset:
        # Apply transformation
        return apply_transform(ds, state, self.my_param)

# 2. Register processor
from src.pipeline.spec import ProcessorRegistry

ProcessorRegistry.register('my_custom', MyCustomProcessor)

# 3. Use in YAML
preprocessing:
  learn:
    - type: "my_custom"
      name: "custom_proc"
      my_param: 0.5
```

### Adding Custom Models

```python
# 1. Create model class
from src.pipeline.spec.model_registry import BaseModel

class MyCustomModel(BaseModel):
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Train model
        self.params_ = train_model(X, y, **self.params)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Make predictions
        return make_predictions(X, self.params_)

# 2. Register model
from src.pipeline.spec import ModelRegistry

ModelRegistry.register('my_custom_model', MyCustomModel)

# 3. Use in YAML
model:
  type: "my_custom_model"
  params:
    param1: value1
```

---

## Conclusion

The Hindsight Pipeline System provides a complete, production-ready framework for quantitative trading strategy development. Key benefits:

- ✅ **Declarative**: Define pipelines in YAML, not code
- ✅ **Fast**: 6-18x speedup from intelligent caching
- ✅ **Correct**: Content-addressable keys ensure cache correctness
- ✅ **Flexible**: Extensible processors and models
- ✅ **Complete**: End-to-end from data to predictions

For questions or issues, refer to the source code in `src/pipeline/` or run the example in `examples/run_pipeline_example.py`.

