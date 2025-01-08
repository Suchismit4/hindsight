# src/data/processors.py

from processors_registry import post_processor
import xarray

@post_processor
def processor_test(funcs: Dict[str, Any], ds: xarray.Dataset) -> xarray.Dataset:
    """
    Test function to test functionality of function decorators and registry
    """
    return ds