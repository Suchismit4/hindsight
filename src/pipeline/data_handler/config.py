"""
Configuration classes for the data handling pipeline.

This module defines configuration classes that control how data processing
pipelines are constructed and executed. These classes follow qlib's pattern
of separating configuration from implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .core import PipelineMode
from .processors import Processor


@dataclass
class HandlerConfig:
    """
    Configuration for DataHandler pipeline construction.
    
    This class defines the complete configuration for a data processing pipeline,
    including the processors for each stage and the execution mode. It follows
    qlib's pattern of separating "how data is processed" from "when".
    
    Parameters
    ----------
    shared : Sequence[Processor], default empty
        Processors that run once on the full dataset before segmentation.
        These are typically stateless transforms like data cleaning.
    learn : Sequence[Processor], default empty
        Processors that fit on training data and transform both train and inference.
        These learn parameters during training (e.g., normalization statistics).
    infer : Sequence[Processor], default empty
        Processors that run transform-only on inference data after learn processors.
        These are typically post-processing steps that don't require fitting.
    mode : PipelineMode, default INDEPENDENT
        Execution mode controlling how shared, learn, and infer pipelines combine.
    feature_cols : list of str, optional
        Names of columns to be treated as features for modeling.
    label_cols : list of str, optional
        Names of columns to be treated as labels/targets for modeling.
        
    Notes
    -----
    The pipeline execution follows this pattern:
    
    INDEPENDENT mode:
        shared -> learn (fit+transform on train, transform on infer)
        shared -> infer (transform-only)
        
    APPEND mode:
        shared -> infer -> learn (sequential, learn sees infer outputs)
        
    This design allows for flexible data processing workflows while maintaining
    clear separation between training and inference data flows.
    """
    shared: Sequence[Processor] = field(default_factory=list)
    learn: Sequence[Processor] = field(default_factory=list)
    infer: Sequence[Processor] = field(default_factory=list)
    mode: PipelineMode = PipelineMode.INDEPENDENT
    feature_cols: Optional[List[str]] = None
    label_cols: Optional[List[str]] = None
