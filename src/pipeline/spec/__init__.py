"""
Pipeline specification system for declarative workflow definitions.

This module provides a YAML-based specification format for defining end-to-end
data processing pipelines, from data loading through feature engineering and
preprocessing.

The specification system enables:
- Declarative pipeline definitions (version-controlled YAML files)
- Automatic cache reuse when specs share common stages
- Clear dependency tracking between pipeline stages
- Validation and error checking before execution

Main Components:
- PipelineSpec: Complete pipeline specification
- DataSourceSpec: Data source configuration
- FeaturesSpec: Feature engineering configuration
- PreprocessingSpec: Preprocessing configuration
- SpecParser: YAML file parser

Example:
    >>> from src.pipeline.spec import SpecParser
    >>> 
    >>> # Load a pipeline specification
    >>> spec = SpecParser.load_from_yaml("configs/momentum_model.yaml")
    >>> 
    >>> # Validate the specification
    >>> spec.validate()
    >>> 
    >>> # Access specification components
    >>> print(spec.name, spec.version)
    >>> print(spec.data.keys())  # Data sources
    >>> print(spec.features.operations)  # Feature operations

YAML Format Example:
    ```yaml
    spec_version: "1.0"
    name: "momentum_strategy_v1"
    
    time_range:
      start: "2010-01-01"
      end: "2023-12-31"
    
    data:
      equity_prices:
        provider: "wrds"
        dataset: "crsp"
        frequency: "daily"
        filters:
          share_classes: [10, 11]
        processors:
          - type: "set_permno_coord"
    
    features:
      operations:
        - name: "momentum_indicators"
          formulas:
            rsi: [{window: 14}, {window: 21}]
            sma: [{window: 20}, {window: 50}]
    
    preprocessing:
      mode: "independent"
      shared:
        - type: "per_asset_ffill"
          vars: ["close", "volume"]
      learn:
        - type: "cs_zscore"
          vars: ["ret_1m", "volume"]
    ```
"""

from .schema import (
    PipelineSpec,
    DataSourceSpec,
    FeaturesSpec,
    FormulaOperationSpec,
    PreprocessingSpec,
    ModelSpec,
)
from .parser import SpecParser
from .executor import PipelineExecutor
from .result import ExecutionResult, OperationResult
from .processor_registry import ProcessorRegistry

__all__ = [
    'PipelineSpec',
    'DataSourceSpec',
    'FeaturesSpec',
    'FormulaOperationSpec',
    'PreprocessingSpec',
    'ModelSpec',
    'SpecParser',
    'PipelineExecutor',
    'ExecutionResult',
    'OperationResult',
    'ProcessorRegistry',
]

