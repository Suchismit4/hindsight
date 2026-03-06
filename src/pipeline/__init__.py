"""
Pipeline Module for Hindsight Financial Analysis Framework.

This module provides a comprehensive pipeline framework for financial data
processing, walk-forward analysis, and model orchestration. It follows qlib's
architectural principles of separating temporal concerns ("when") from data
processing concerns ("how") while providing high-performance implementations
optimized for financial time series analysis.

The pipeline framework consists of three main components:

Data Handler Pipeline:
    Manages data processing workflows with support for:
    - Multi-stage processing (shared, learn, infer)
    - Stateful processors with consistent state management
    - Flexible pipeline modes (independent, append)
    - Efficient caching and lazy evaluation
    - Integration with AST formula evaluation system

Walk-Forward Analysis:
    Provides temporal segmentation and execution for robust backtesting:
    - Configurable segment generation with gap handling
    - Automatic data boundary clipping and validation
    - Efficient segment-by-segment processing
    - Comprehensive result collection and analysis
    - Protection against lookahead bias

Model Integration:
    Orchestrates model training and inference across time segments:
    - Model adapter pattern for consistent interfaces
    - Segment-aware model runner with overlap handling
    - Support for various model types and frameworks
    - Performance optimization with JAX/threading controls

Key Features:

1. Separation of Concerns: Clear separation between data processing logic
   and temporal segmentation, enabling flexible and maintainable workflows.

2. Stateful Processing: Processors learn parameters on training data and
   apply them consistently to inference data, ensuring statistical validity.

3. Efficient Execution: Optimized implementations with pre-computation,
   caching, and parallel processing for high-performance backtesting.

4. Robust Validation: Comprehensive validation of temporal consistency,
   data availability, and pipeline configuration.

5. Extensible Design: Pluggable processor architecture and model adapter
   pattern support custom extensions and domain-specific logic.

Examples
--------
Complete pipeline setup with data processing and walk-forward analysis:

>>> import numpy as np
>>> from src.pipeline import (
...     DataHandler, HandlerConfig, View, PipelineMode,
...     make_plan, SegmentConfig, WalkForwardRunner
... )
>>> from src.pipeline.data_handler.processors import CSZScore, PerAssetFFill
>>> 
>>> # Configure data processing pipeline
>>> handler_config = HandlerConfig(
...     shared=[PerAssetFFill(name="ffill")],
...     learn=[CSZScore(name="norm", vars=["close", "volume"])],
...     mode=PipelineMode.INDEPENDENT,
...     feature_cols=["close_csz", "volume_csz"]
... )
>>> 
>>> # Create data handler
>>> handler = DataHandler(base=raw_dataset, config=handler_config)
>>> 
>>> # Configure walk-forward segments
>>> segment_config = SegmentConfig(
...     start=np.datetime64('2020-01-01'),
...     end=np.datetime64('2023-12-31'),
...     train_span=np.timedelta64(365, 'D'),
...     infer_span=np.timedelta64(30, 'D'),
...     step=np.timedelta64(30, 'D'),
...     gap=np.timedelta64(1, 'D')
... )
>>> 
>>> # Generate segment plan
>>> plan = make_plan(segment_config, ds_for_bounds=raw_dataset)
>>> 
>>> # Execute walk-forward analysis
>>> runner = WalkForwardRunner(handler=handler, plan=plan)
>>> results = runner.run()
>>> 
>>> print(f"Processed {len(results)} segments")
>>> print(f"Generated features: {handler.config.feature_cols}")

Advanced usage with formula evaluation and model integration:

>>> from src.pipeline.data_handler.processors import FormulaEval
>>> from src.pipeline.model import ModelRunner, SklearnAdapter
>>> from sklearn.ensemble import RandomForestRegressor
>>> 
>>> # Configure formula-based feature engineering
>>> formula_config = {
...     "rsi": [{"window": 14}, {"window": 21}],
...     "sma": [{"window": 20}, {"window": 50}],
...     "ema": [{"window": 12}, {"window": 26}]
... }
>>> 
>>> handler_config = HandlerConfig(
...     shared=[
...         PerAssetFFill(name="ffill"),
...         FormulaEval(name="formulas", formula_configs=formula_config)
...     ],
...     learn=[CSZScore(name="norm")],
...     feature_cols=["rsi_w14_csz", "rsi_w21_csz", "sma_w20_csz", "sma_w50_csz"],
...     label_cols=["target_return"]
... )
>>> 
>>> # Create model adapter
>>> model = SklearnAdapter(
...     model=RandomForestRegressor(n_estimators=100, random_state=42),
...     handler=handler,
...     output_var="predicted_return"
... )
>>> 
>>> # Create model runner
>>> model_runner = ModelRunner(
...     handler=handler,
...     plan=plan,
...     model_factory=model,
...     feature_cols=handler_config.feature_cols,
...     label_col="target_return"
... )
>>> 
>>> # Execute complete modeling pipeline
>>> model_results = model_runner.run()
>>> predictions = model_results.pred_ds
>>> 
>>> print(f"Generated predictions: {list(predictions.data_vars)}")

Architecture Overview:

The pipeline framework follows a layered architecture:

1. Data Layer (data_handler/):
   - Core types and interfaces (core.py)
   - Concrete processor implementations (processors.py)
   - Configuration management (config.py)
   - Main orchestration logic (handler.py)

2. Temporal Layer (walk_forward/):
   - Segment definitions and validation (segments.py)
   - Planning and generation utilities (planning.py)
   - Execution engine and results (execution.py)

3. Model Layer (model/):
   - Model adapter interfaces (adapter.py)
   - Model orchestration and prediction aggregation (runner.py)

This separation enables:
- Independent testing and development of each layer
- Flexible composition of processing workflows
- Clear interfaces between temporal and processing logic
- Efficient optimization at each layer

Performance Considerations:

The pipeline framework is optimized for high-performance financial analysis:

1. Lazy Evaluation: Views and transformations computed on-demand with caching
2. Pre-computation: Segment bounds and slice indices computed upfront
3. Vectorized Operations: JAX-accelerated transformations where possible
4. Memory Efficiency: Minimal data copying with in-place operations
5. Parallel Processing: Thread-safe operations with configurable parallelism

Best Practices:

1. Configure processors with appropriate caching and state management
2. Use data-aware planning to optimize segment schedules
3. Validate plans before execution for temporal consistency
4. Monitor memory usage with large datasets and long time periods
5. Leverage formula evaluation for complex feature engineering
6. Use appropriate processor sequencing for statistical validity
"""

# Import main components for convenient access
from .data_handler import (
    # Core types and enums
    View, PipelineMode, ProcessorContract,
    # Processor implementations  
    Processor, CSZScore, PerAssetFFill, FormulaEval,
    # Configuration and main handler
    HandlerConfig, DataHandler
)

from .walk_forward import (
    # Segment management
    Segment, SegmentPlan, SegmentConfig,
    # Planning functions
    make_plan, expand_plan_coverage, optimize_plan_for_dataset,
    # Execution components
    WalkForwardRunner, SegmentResult, WalkForwardResult
)

# Model components are imported separately to avoid circular dependencies
# Users should import from src.pipeline.model directly for model-related functionality

# Cache and specification system 
# These provide content-addressable caching and YAML-based pipeline specifications
try:
    from .cache import GlobalCacheManager, CacheStage, CacheMetadata, MetadataManager
    from .spec import (
        PipelineSpec, DataSourceSpec, FeaturesSpec, FormulaOperationSpec,
        PreprocessingSpec, SpecParser, PipelineExecutor, ExecutionResult, OperationResult
    )
    _CACHE_SPEC_AVAILABLE = True
except ImportError:
    _CACHE_SPEC_AVAILABLE = False

__all__ = [
    # Data handler components
    "View", "PipelineMode", "ProcessorContract",
    "Processor", "CSZScore", "PerAssetFFill", "FormulaEval", 
    "HandlerConfig", "DataHandler",
    
    # Walk-forward components
    "Segment", "SegmentPlan", "SegmentConfig",
    "make_plan", "expand_plan_coverage", "optimize_plan_for_dataset",
    "WalkForwardRunner", "SegmentResult", "WalkForwardResult",
]

# Add cache and spec components to __all__ if available
if _CACHE_SPEC_AVAILABLE:
    __all__.extend([
        # Cache system
        "GlobalCacheManager", "CacheStage", "CacheMetadata", "MetadataManager",
        # Specification system
        "PipelineSpec", "DataSourceSpec", "FeaturesSpec", "FormulaOperationSpec",
        "PreprocessingSpec", "SpecParser",
        # Executor and results
        "PipelineExecutor", "ExecutionResult", "OperationResult",
    ])
