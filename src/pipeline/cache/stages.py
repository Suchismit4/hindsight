"""
Cache stage definitions for the hierarchical caching system.

This module defines the different cache levels used throughout the pipeline,
from raw data loading to preprocessed features ready for modeling.
"""

from enum import Enum


class CacheStage(Enum):
    """
    Enumeration of cache stages in the pipeline.
    
    Each stage represents a level in the hierarchical cache system, with
    dependencies flowing from L1 (raw data) through L5 (model predictions).
    
    Attributes:
        L1_RAW: Raw xarray Dataset directly from data loaders, before any processing
        L2_POSTPROCESSED: After xarray-level processors (merges, coordinates, transforms)
        L3_FEATURES: After formula evaluation and feature engineering
        L4_PREPROCESSED: After data handler preprocessing (ffill, normalization, etc.)
        L5_MODEL: After model training and predictions
    """
    
    L1_RAW = "l1_raw"
    L2_POSTPROCESSED = "l2_post"
    L3_FEATURES = "l3_features"
    L4_PREPROCESSED = "l4_prep"
    L5_MODEL = "l5_model"
    
    def __str__(self) -> str:
        """Return the string value of the stage."""
        return self.value
    
    @property
    def level(self) -> int:
        """Return the numeric level of this stage (1-5)."""
        level_map = {
            CacheStage.L1_RAW: 1,
            CacheStage.L2_POSTPROCESSED: 2,
            CacheStage.L3_FEATURES: 3,
            CacheStage.L4_PREPROCESSED: 4,
            CacheStage.L5_MODEL: 5,
        }
        return level_map[self]

