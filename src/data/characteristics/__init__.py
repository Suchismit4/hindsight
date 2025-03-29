"""
Financial characteristics computation module.

This module provides functionality for computing financial characteristics from raw
financial data. It uses a context-free grammar (CFG) approach for defining characteristic formulas in configuration files.

The module is organized into categories:
1. Accounting characteristics - Based on financial statement data
2. Market characteristics - Based on pricing and trading data
3. Combined characteristics - Based on both accounting and market data

Each characteristic is implemented as a function that takes datasets as input and
returns the computed characteristic values as an xarray DataArray or Dataset.
"""

from src.data.characteristics.accounting import *
from src.data.characteristics.market import *
from src.data.characteristics.combined import *
from src.data.characteristics.manager import CharacteristicsManager
from .formula import FormulaAST, FormulaCompiler
from .executor import FormulaExecutor
from .parser import FormulaParser, parse_formula

__all__ = [
    'CharacteristicsManager',
    'FormulaAST',
    'FormulaCompiler',
    'FormulaExecutor',
    'FormulaParser',
    'parse_formula',
    
    # Lists of available characteristics by category
    'ACCOUNTING_CHARACTERISTICS',
    'MARKET_CHARACTERISTICS',
    'COMBINED_CHARACTERISTICS',
    
    # Utility functions for characteristics computation
    'compute_characteristics'
] 