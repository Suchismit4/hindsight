"""
Data Handler Pipeline Module.

This module provides a comprehensive data processing pipeline framework following
qlib's architectural patterns. It separates "how data is processed" (processors)
from "when data is processed" (segments), enabling flexible and efficient
financial data transformations.

The module is organized into several components:

Core Components:
- View, PipelineMode: Fundamental enums for pipeline configuration
- ProcessorContract: Abstract interface for all data processors
- Processor: Base implementation class for processors

Concrete Processors:
- CSZScore: Cross-sectional z-score normalization
- PerAssetFFill: Per-asset forward filling
- FormulaEval: AST-based formula evaluation

Configuration:
- HandlerConfig: Complete pipeline configuration specification

Main Handler:
- DataHandler: Central orchestrator for the entire pipeline

The design follows these principles:
1. Separation of concerns between processing logic and temporal segmentation
2. Stateful processors that learn parameters and apply them consistently
3. Flexible pipeline modes supporting both independent and sequential execution
4. Efficient caching and lazy evaluation for performance
5. Type-safe interfaces with comprehensive documentation

Examples
--------
Basic usage with cross-sectional normalization:

>>> from src.pipeline.data_handler import (
...     DataHandler, HandlerConfig, View, PipelineMode
... )
>>> from src.pipeline.data_handler.processors import CSZScore, PerAssetFFill
>>> 
>>> # Configure processing pipeline
>>> config = HandlerConfig(
...     shared=[PerAssetFFill(name="ffill")],
...     learn=[CSZScore(name="norm", vars=["close", "volume"])],
...     mode=PipelineMode.INDEPENDENT,
...     feature_cols=["close_csz", "volume_csz"]
... )
>>> 
>>> # Create handler and process data
>>> handler = DataHandler(base=raw_dataset, config=config)
>>> processed = handler.view(View.LEARN)
>>> features = handler.fetch(View.LEARN, ["features"])

Advanced usage with formula evaluation:

>>> from src.pipeline.data_handler.processors import FormulaEval
>>> 
>>> # Configure formula-based features
>>> formula_config = {
...     "rsi": [{"window": 14}, {"window": 21}],
...     "sma": [{"window": 20}, {"window": 50}]
... }
>>> 
>>> config = HandlerConfig(
...     shared=[FormulaEval(
...         name="formulas",
...         formula_configs=formula_config,
...         use_jit=True
...     )],
...     feature_cols=["rsi_w14", "rsi_w21", "sma_w20", "sma_w50"]
... )
>>> 
>>> handler = DataHandler(base=raw_dataset, config=config)
>>> features = handler.fetch(View.RAW, ["features"])
"""

from .core import View, PipelineMode, ProcessorContract
from .processors import Processor, CSZScore, PerAssetFFill, FormulaEval, CrossSectionalSort, PortfolioReturns
from .config import HandlerConfig
from .handler import DataHandler
from .merge import DatasetMerger, MergeSpec, MergeMethod, TimeAlignment, merge_datasets

__all__ = [
    # Core types and interfaces
    "View",
    "PipelineMode", 
    "ProcessorContract",
    
    # Processor implementations
    "Processor",
    "CSZScore",
    "PerAssetFFill", 
    "FormulaEval",
    "CrossSectionalSort",
    "PortfolioReturns",
    
    # Configuration
    "HandlerConfig",
    
    # Main handler
    "DataHandler",
    
    # Merge utilities
    "DatasetMerger",
    "MergeSpec",
    "MergeMethod",
    "TimeAlignment",
    "merge_datasets",
]
