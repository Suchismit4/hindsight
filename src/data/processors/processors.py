# src/data/processors.py

from src.data.processors.registry import post_processor
import xarray
from typing import *

@post_processor
def processor_test(funcs: Dict[str, Any], ds: xarray.Dataset) -> xarray.Dataset:
    """
    Test function to test functionality of function decorators and registry
    """
    return ds