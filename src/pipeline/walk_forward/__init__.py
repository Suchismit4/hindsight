"""
Walk-Forward Analysis Module.

This module provides a comprehensive framework for walk-forward analysis
in financial backtesting, with architectural patterns of
separating temporal logic ("when") from data processing logic ("how").

The module is organized into three main components:

Segment Management:
- Segment: Individual time periods with training and inference windows
- SegmentPlan: Collections of segments with validation and utilities
- SegmentConfig: Configuration for automated segment generation

Planning and Generation:
- make_plan(): Generate segment plans from configuration parameters
- expand_plan_coverage(): Extend existing plans to cover more time periods
- optimize_plan_for_dataset(): Optimize plans based on data availability

Execution and Results:
- WalkForwardRunner: Execute data processing across temporal segments
- SegmentResult: Results from processing individual segments

The design principles include:

1. Temporal Isolation: Strict separation between training and inference periods
   with configurable gaps to prevent lookahead bias

2. Efficient Processing: Shared preprocessing with segment-specific fitting
   to minimize computation while maintaining statistical validity

3. Flexible Configuration: Support for various walk-forward patterns including
   rolling windows, expanding windows, and custom segment schedules

4. Data-Aware Planning: Automatic adjustment for data availability, business
   calendars, and missing data periods

5. Consistent State Management: Proper handling of learned processor states
   to ensure consistent transformations across time periods

[IMPORTANT] The walk-forward analysis is designed to be model-agnostic,
focusing on data processing and temporal segmentation. Model-specific logic
should be implemented in a separate module.

Examples
--------
Basic monthly walk-forward analysis:

>>> from src.pipeline.walk_forward import (
...     make_plan, SegmentConfig, WalkForwardRunner
... )
>>> from src.pipeline.data_handler import DataHandler, HandlerConfig
>>> 
>>> # Configure temporal segments
>>> config = SegmentConfig(
...     start=np.datetime64('2020-01-01'),
...     end=np.datetime64('2023-12-31'),
...     train_span=np.timedelta64(365, 'D'),  # 1 year training
...     infer_span=np.timedelta64(30, 'D'),   # 1 month inference
...     step=np.timedelta64(30, 'D'),         # Monthly steps
...     gap=np.timedelta64(1, 'D')            # 1 day gap
... )
>>> plan = make_plan(config, ds_for_bounds=dataset)
>>> 
>>> # Configure data processing
>>> handler_config = HandlerConfig(
...     shared=[PerAssetFFill(name="ffill")],
...     learn=[CSZScore(name="norm")],
...     feature_cols=["close_csz", "volume_csz"]
... )
>>> handler = DataHandler(base=dataset, config=handler_config)
>>> 
>>> # Execute walk-forward analysis
>>> runner = WalkForwardRunner(handler=handler, plan=plan)
>>> results = runner.run()
>>> 
>>> print(f"Processed {len(results)} segments")

Advanced usage with custom processors and validation:

>>> # Custom processing pipeline
>>> config = HandlerConfig(
...     shared=[FormulaEval(
...         name="formulas", 
...         formula_configs={"rsi": [{"window": 14}], "sma": [{"window": 20}]}
...     )],
...     learn=[CSZScore(name="norm", vars=["rsi", "sma"])],
...     infer=[],
...     feature_cols=["rsi_csz", "sma_csz"]
... )
>>> 
>>> # Validate plan before execution
>>> validation_issues = plan.validate()
>>> if validation_issues:
...     print(f"Plan validation issues: {validation_issues}")
>>> 
>>> # Optimize plan for data availability
>>> optimized_plan = optimize_plan_for_dataset(
...     plan, dataset, min_train_samples=500, min_infer_samples=20
... )
>>> 
>>> # Execute with optimized plan
>>> runner = WalkForwardRunner(handler=handler, plan=optimized_plan)
>>> results = runner.run()
>>> 
>>> # Analyze results
>>> for i, result in enumerate(results):
...     summary = result.get_state_summary()
...     print(f"Segment {i}: {summary['num_states']} learned states")
"""

from .segments import Segment, SegmentPlan, SegmentConfig
from .planning import make_plan, expand_plan_coverage, optimize_plan_for_dataset
from .execution import WalkForwardRunner, SegmentResult, WalkForwardResult

__all__ = [
    # Core segment types
    "Segment",
    "SegmentPlan", 
    "SegmentConfig",
    
    # Planning functions
    "make_plan",
    "expand_plan_coverage",
    "optimize_plan_for_dataset",
    
    # Execution components
    "WalkForwardRunner",
    "SegmentResult",
    "WalkForwardResult",
]
