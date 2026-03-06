"""
Core types and enums for the data handling pipeline.

This module defines the fundamental types, enums, and base classes used throughout
the data handling pipeline. It follows qlib's pattern of separating "how data is 
processed" (learn vs infer streams) from "when" (segments).

The design separates concerns between:
- Data views (RAW, LEARN, INFER) 
- Pipeline execution modes (INDEPENDENT, APPEND)
- Processing contracts and interfaces
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Sequence, Tuple
import xarray as xr

# Alias describing the opaque object a processor uses to persist learned
# information. Individual processors decide what they return (e.g., dicts,
# numpy arrays, dataclasses, xr objects, None, ...).
ProcessorState = Any

class View(Enum):
    """
    Enumeration of data processing views in the pipeline.
    
    This enum defines the different data processing contexts,
    separating training and inference data flows.
    
    Attributes:
        RAW: Raw data loaded as-is after feature graph, no train/infer specialization
        LEARN: Fit+transform path used for training with full processor fitting
        INFER: Transform-only path used for prediction with pre-fitted processors
    """
    RAW = "raw"       # loaded as-is (after feature graph), no train/infer specialization
    LEARN = "learn"   # fit+transform path used for training
    INFER = "infer"   # transform-only path used for prediction


class PipelineMode(Enum):
    """
    Enumeration of pipeline execution modes.
    
    This enum defines how the shared, learn, and infer processor pipelines
    are combined and executed.
    
    Attributes:
        INDEPENDENT: Two independent branches (shared -> learn, shared -> infer)
        APPEND: Sequential execution (shared -> infer -> learn, where learn sees extra steps)
    """
    INDEPENDENT = "independent"  # shared -> learn, shared -> infer (two branches)
    APPEND = "append"            # shared -> infer -> learn (learn sees extra steps)


class ProcessorContract:
    """
    Abstract base class defining the processor contract for xarray I/O.
    
    This class establishes the interface that all processors must implement,
    following the scikit-learn fit/transform pattern adapted for xarray datasets.
    
    The processor contract follows these conventions:
    - fit(ds) -> ProcessorState: learn parameters and return an opaque state object
    - transform(ds, state=None) -> xr.Dataset: apply the transformation using state
    - fit_transform(ds) -> (output, state): convenience method combining both
    
    State management conventions:
    - state_ds stores only parameters needed at inference time
    - Keep state compact and aligned by dimensions when possible
    - Use namespaced parameter names: f"{self.name}_param__{var}"
    
    Variable handling conventions:
    - Default to writing new variables with suffix/prefix for safety
    - Provide parameter to allow in-place replacement when desired
    """
    
    def fit(self, ds: xr.Dataset) -> ProcessorState:
        """
        Learn parameters from the input dataset.
        
        Parameters
        ----------
        ds : xr.Dataset
            Input dataset to learn parameters from
            
        Returns
        -------
        ProcessorState
            Opaque processor state containing only the parameters needed at
            inference time. Implementations decide the exact structure (e.g.,
            dicts, dataclasses, numpy arrays, xr objects, None, ...).
            
        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement fit method")

    def transform(
        self,
        ds: xr.Dataset,
        state: Optional[ProcessorState] = None,
    ) -> xr.Dataset:
        """
        Apply the transformation to the input dataset.
        
        Parameters
        ----------
        ds : xr.Dataset
            Input dataset to transform
        state : ProcessorState, optional
            State returned by ``fit``. If None, the processor must behave as stateless.
            
        Returns
        -------
        xr.Dataset
            Transformed dataset
            
        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement transform method")

    def fit_transform(self, ds: xr.Dataset) -> Tuple[xr.Dataset, ProcessorState]:
        """
        Convenience method that combines fit and transform operations.
        
        Default implementation calls fit() followed by transform(). Override if you 
        can compute both with a single pass for efficiency.
        
        Parameters
        ----------
        ds : xr.Dataset
            Input dataset to fit and transform
            
        Returns
        -------
        tuple[xr.Dataset, ProcessorState]
            Tuple of (transformed_dataset, state_object)
        """
        state = self.fit(ds)
        output = self.transform(ds, state)
        return output, state
