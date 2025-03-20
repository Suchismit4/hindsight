"""
Data loaders for various financial data providers in Hindsight.

This module provides specialized data loaders that retrieve financial data from
different sources and normalize it to the Hindsight data model.

Available data providers:
1. WRDS (Wharton Research Data Services) - Access to CRSP, Compustat, and other databases
2. OpenBB - Open-source financial data APIs

Each loader handles source-specific data retrieval, preprocessing, and conversion
to xarray Datasets with standardized dimensions and coordinates.
"""

from .wrds import *
from .open_bb import *

# Define public exports
__all__ = [
    # WRDS data loaders
    'WRDSDataLoader', 
    'GenericWRDSDataLoader',
    'CRSPDailyDataLoader',
    'CompustatDataLoader',
    
    # OpenBB data loaders
    'OpenBBDataLoader',
]