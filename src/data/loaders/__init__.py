"""
Data loaders for various financial data providers in Hindsight.

This module provides specialized data loaders that retrieve financial data from
different sources and normalize it to the Hindsight data model.

Available data providers:
1. WRDS (Wharton Research Data Services) - Access to CRSP, Compustat, and other databases
2. OpenBB - Open-source financial data APIs
3. Crypto - Local cryptocurrency data from CSV files

Each loader handles source-specific data retrieval, preprocessing, and conversion
to xarray Datasets with standardized dimensions and coordinates.
"""

# Import specific classes explicitly
from .wrds.generic import GenericWRDSDataLoader
from .wrds.crsp import CRSPDataFetcher
from .wrds.compustat import CompustatDataFetcher

# Import crypto loaders
from .crypto.local import LocalCryptoDataFetcher

# Try to import OpenBB loaders if they exist
try:
    from .open_bb import OpenBBDataLoader
except ImportError:
    # Create a placeholder if the module doesn't exist
    class OpenBBDataLoader:
        pass

# Define public exports
__all__ = [
    # WRDS data loaders
    'GenericWRDSDataLoader',
    'CRSPDataFetcher',
    'CompustatDataFetcher',
    
    # Crypto data loaders
    'LocalCryptoDataFetcher',
    
    # OpenBB data loaders
    'OpenBBDataLoader',
]