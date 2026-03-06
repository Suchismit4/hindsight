"""
Model Integration Module.

This module provides model integration capabilities for the Hindsight pipeline
framework, enabling seamless orchestration of machine learning models within
walk-forward analysis workflows.

The module follows qlib's architectural principles by providing clean interfaces
between data processing pipelines and model training/inference, with support
for various model types and efficient execution patterns.

The module is organized into these components:

Model Interfaces:
- ModelAdapter: Abstract interface for model wrappers
- SklearnAdapter: Concrete implementation for scikit-learn models

Model Orchestration:
- ModelRunner: Main orchestrator that combines data processing with model execution
- ModelRunnerResult: Result type containing predictions and metadata

The design principles include:

1. Model Agnostic Interface: Adapters provide consistent interfaces regardless
   of underlying model implementation (sklearn, tensorflow, pytorch, etc.)

2. Gather>Scatter Pattern: Efficient execution strategy that pre-allocates
   global buffers and scatters segment predictions for optimal performance

3. Temporal Isolation: Strict separation between training and inference data
   with proper state management to prevent lookahead bias

4. Stateful Processing: Models are fitted per-segment with consistent
   feature transformations applied across time periods

5. Performance Optimization: Pre-computation, efficient indexing, and
   minimal data copying for large-scale backtesting

Examples
--------
Basic model integration with scikit-learn:

>>> from src.pipeline.model import ModelRunner, SklearnAdapter
>>> from sklearn.ensemble import RandomForestRegressor
>>> 
>>> # Create model adapter factory
>>> def make_adapter():
...     model = RandomForestRegressor(n_estimators=50, random_state=42)
...     return SklearnAdapter(
...         model=model,
...         handler=data_handler,
...         output_var="prediction"
...     )
>>> 
>>> # Configure model runner
>>> runner = ModelRunner(
...     handler=data_handler,
...     plan=segment_plan,
...     model_factory=make_adapter,
...     feature_cols=["feature1", "feature2"],
...     label_col="target",
...     overlap_policy="last"
... )
>>> 
>>> # Execute complete modeling pipeline
>>> results = runner.run()
>>> predictions = results.pred_ds

Advanced usage with custom models and debugging:

>>> # Custom model adapter
>>> class CustomAdapter(ModelAdapter):
...     def fit(self, ds, features, label=None, sample_weight=None):
...         # Custom training logic
...         return self
...     
...     def predict(self, ds, features):
...         # Custom prediction logic
...         return predictions
>>> 
>>> # Debug specific time periods and assets
>>> runner = ModelRunner(
...     handler=data_handler,
...     plan=segment_plan,
...     model_factory=lambda: CustomAdapter(),
...     feature_cols=feature_cols,
...     label_col=label_col,
...     debug_asset="AAPL",
...     debug_start=np.datetime64("2021-01-01"),
...     debug_end=np.datetime64("2021-02-01")
... )
>>> 
>>> results = runner.run()
>>> segment_states = results.segment_states

Architecture Overview:

The model module integrates with the broader pipeline architecture:

1. Data Layer: Receives processed datasets from DataHandler
2. Temporal Layer: Operates on segments from walk-forward planning
3. Model Layer: Applies ML models with consistent feature engineering
4. Result Layer: Aggregates predictions using efficient scatter operations

This separation enables:
- Independent model development and testing
- Flexible composition with different data processing workflows
- Consistent interfaces across various model types
- Efficient optimization at each layer
- Clear debugging and analysis capabilities

Performance Considerations:

The model integration is optimized for high-performance backtesting:

1. Pre-allocation: Global prediction buffers allocated once upfront
2. Efficient Indexing: Integer-based slicing and scatter operations
3. Minimal Copying: In-place operations where possible
4. Thread Safety: Configurable thread limits for BLAS operations
5. Memory Management: Efficient handling of large datasets

Best Practices:

1. Use model factories rather than pre-instantiated models for segment isolation
2. Configure appropriate overlap policies for time series with gaps
3. Leverage debug controls for development and validation
4. Monitor memory usage with large models and long time periods
5. Use appropriate thread limits to avoid oversubscription
"""

from .adapter import ModelAdapter, SklearnAdapter
from .runner import ModelRunner, ModelRunnerResult

__all__ = [
    # Core model interfaces
    "ModelAdapter",
    
    # Concrete model implementations  
    "SklearnAdapter",
    
    # Model orchestration
    "ModelRunner",
    "ModelRunnerResult",
]
