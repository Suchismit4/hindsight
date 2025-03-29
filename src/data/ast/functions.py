"""
Function registry for financial formula execution.

This module provides a comprehensive function registry system for financial formulas.
It allows for registering custom functions that can be used in formula expressions
and manages the mapping between function names and their implementations.

The module operates primarily on xarray DataArrays and Datasets, preserving dimensions
and coordinates throughout calculations. All functions are designed to work with labeled
multi-dimensional data.

The module includes:
1. A registry for storing and retrieving registered functions
2. A decorator for easy function registration
3. Built-in financial functions commonly used in financial analysis
4. Utilities for creating function contexts for formula evaluation

Functions are designed to work with xarray objects while still being JAX-compatible
to enable JIT compilation for efficient computation.

Examples:
    Registering a custom function:
    >>> from src.data.ast.functions import register_function
    >>> @register_function
    ... def my_sum(a, b, c):
    ...     return a + b + c
    >>> from src.data.ast.parser import parse_formula
    >>> formula = parse_formula("my_sum(1, 2, 3)")
    >>> formula.evaluate(get_function_context())
    Array(6., dtype=float32)
"""

import functools
import inspect
from typing import Dict, Any, Callable, List, Union, Optional, TypeVar, cast, overload
import xarray as xr
import jax
import jax.numpy as jnp
import numpy as np
from src.data.core.operations import mean as core_mean
from src.data.core.operations import ema as core_ema
from src.data.core.operations import median as core_median
from src.data.core.operations import mode as core_mode

# Type variables for better type hinting
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')

# Dictionary to store registered functions
_FUNCTION_REGISTRY: Dict[str, Callable] = {}

# Category groupings for functions
_FUNCTION_CATEGORIES: Dict[str, List[str]] = {
    "arithmetic": [],
    "statistical": [],
    "financial": [],
    "temporal": [],
    "conditional": [],
    "miscellaneous": []
}

# Function metadata storage
_FUNCTION_METADATA: Dict[str, Dict[str, Any]] = {}

@overload
def register_function(func: F) -> F:
    """Decorator form without arguments."""
    ...

@overload
def register_function(
    name: Optional[str] = None,
    category: str = "miscellaneous",
    description: Optional[str] = None
) -> Callable[[F], F]:
    """Decorator form with arguments."""
    ...

def register_function(
    func_or_name: Union[Callable, str, None] = None,
    category: str = "miscellaneous",
    description: Optional[str] = None
) -> Union[Callable, Callable[[Callable], Callable]]:
    """
    Register a function for use in formulas.
    
    This can be used as a decorator with or without arguments:
    
    @register_function
    def my_func(a, b):
        return a + b
        
    @register_function(name="custom_name", category="arithmetic", description="Add two numbers")
    def my_func(a, b):
        return a + b
    
    Args:
        func_or_name: Either the function to register or a custom name for the function
        category: The category of the function (arithmetic, statistical, financial, etc.)
        description: A description of what the function does
    
    Returns:
        The registered function (when used as a simple decorator) or
        a decorator function (when used with arguments)
        
    Raises:
        ValueError: If the category is not valid or a function with the same name
                   is already registered
                   
    Examples:
        >>> @register_function
        ... def my_sum(a, b, c):
        ...     return a + b + c
        >>> my_sum(1, 2, 3)
        6
        
        >>> @register_function(name="custom_multiply", category="arithmetic")
        ... def multiply(a, b):
        ...     return a * b
        >>> custom_multiply(2, 3)
        6
    """
    # Check if the category is valid
    if category not in _FUNCTION_CATEGORIES and category != "miscellaneous":
        raise ValueError(f"Invalid category: {category}. Valid categories are: "
                         f"{', '.join(_FUNCTION_CATEGORIES.keys())}, or miscellaneous")
    
    # If func_or_name is a callable, this is the @register_function form
    if callable(func_or_name):
        func = func_or_name
        name = func.__name__
        
        # Register the function with its original name
        if name in _FUNCTION_REGISTRY:
            raise ValueError(f"Function '{name}' is already registered")
        
        _FUNCTION_REGISTRY[name] = func
        
        # Save metadata
        _FUNCTION_METADATA[name] = {
            "category": category,
            "description": description or func.__doc__ or f"Function '{name}'",
            "signature": str(inspect.signature(func))
        }
        
        # Add to category
        if category not in _FUNCTION_CATEGORIES:
            _FUNCTION_CATEGORIES[category] = []
        _FUNCTION_CATEGORIES[category].append(name)
        
        return func
    
    # Otherwise, this is the @register_function(name=...) form
    def decorator(func: Callable) -> Callable:
        nonlocal func_or_name
        
        # Use the provided name or fall back to the function name
        actual_name = func_or_name if isinstance(func_or_name, str) else func.__name__
        
        # Register the function
        if actual_name in _FUNCTION_REGISTRY:
            raise ValueError(f"Function '{actual_name}' is already registered")
        
        _FUNCTION_REGISTRY[actual_name] = func
        
        # Save metadata
        _FUNCTION_METADATA[actual_name] = {
            "category": category,
            "description": description or func.__doc__ or f"Function '{actual_name}'",
            "signature": str(inspect.signature(func))
        }
        
        # Add to category
        if category not in _FUNCTION_CATEGORIES:
            _FUNCTION_CATEGORIES[category] = []
        _FUNCTION_CATEGORIES[category].append(actual_name)
        
        # Return the original function
        return func
    
    return decorator

def unregister_function(name: str) -> None:
    """
    Unregister a previously registered function.
    
    Args:
        name: The name of the function to unregister
        
    Raises:
        ValueError: If the function is not registered
        
    Examples:
        >>> @register_function
        ... def temp_func(a, b):
        ...     return a + b
        >>> 'temp_func' in get_registered_functions()
        True
        >>> unregister_function('temp_func')
        >>> 'temp_func' in get_registered_functions()
        False
    """
    if name not in _FUNCTION_REGISTRY:
        raise ValueError(f"Function '{name}' is not registered")
    
    # Remove from registry
    del _FUNCTION_REGISTRY[name]
    
    # Remove from category
    for category, functions in _FUNCTION_CATEGORIES.items():
        if name in functions:
            functions.remove(name)
    
    # Remove metadata
    if name in _FUNCTION_METADATA:
        del _FUNCTION_METADATA[name]

def get_registered_functions() -> Dict[str, Callable]:
    """
    Get all registered functions.
    
    Returns:
        Dictionary mapping function names to function objects
        
    Examples:
        >>> funcs = get_registered_functions()
        >>> 'coalesce' in funcs
        True
    """
    return _FUNCTION_REGISTRY.copy()

def get_function_categories() -> Dict[str, List[str]]:
    """
    Get the categories of registered functions.
    
    Returns:
        Dictionary mapping category names to lists of function names
        
    Examples:
        >>> cats = get_function_categories()
        >>> 'arithmetic' in cats
        True
        >>> 'add' in cats['arithmetic']
        True
    """
    return {k: v.copy() for k, v in _FUNCTION_CATEGORIES.items() if v}

def get_function_metadata(name: str) -> Dict[str, Any]:
    """
    Get metadata for a registered function.
    
    Args:
        name: The name of the function
        
    Returns:
        Dictionary containing function metadata
        
    Raises:
        ValueError: If the function is not registered
        
    Examples:
        >>> metadata = get_function_metadata('coalesce')
        >>> 'category' in metadata
        True
        >>> 'description' in metadata
        True
    """
    if name not in _FUNCTION_METADATA:
        raise ValueError(f"Function '{name}' is not registered or has no metadata")
    
    return _FUNCTION_METADATA[name].copy()

def get_function_context() -> Dict[str, Callable]:
    """
    Get a context dictionary for all registered functions.
    
    This maps each function name to a context key of the form "_func_{name}".
    The resulting dictionary can be used as part of the context when evaluating
    formula AST nodes.
    
    Returns:
        Dictionary mapping function context keys to function objects
        
    Examples:
        >>> context = get_function_context()
        >>> '_func_coalesce' in context
        True
    """
    return {f"_func_{name}": func for name, func in _FUNCTION_REGISTRY.items()}

def clear_registry() -> None:
    """
    Clear all registered functions.
    
    This is mainly useful for testing or resetting the registry to a clean state.
    
    Examples:
        >>> register_function(lambda a, b: a + b, name='temp_add')
        >>> 'temp_add' in get_registered_functions()
        True
        >>> clear_registry()
        >>> 'temp_add' in get_registered_functions()
        False
    """
    _FUNCTION_REGISTRY.clear()
    for category in _FUNCTION_CATEGORIES:
        _FUNCTION_CATEGORIES[category] = []
    _FUNCTION_METADATA.clear()
    
    # Re-register built-in functions
    register_built_in_functions()

# ============================
# Built-in Financial Functions
# ============================

@register_function(category="conditional")
def coalesce(a, b):
    """
    Return the first non-null (non-NaN) value between a and b.
    
    Works with both xarray objects and JAX arrays.
    
    Args:
        a: The primary value to check
        b: The fallback value to use if a is NaN
        
    Returns:
        a if a is not NaN, otherwise b
    """
    if isinstance(a, (xr.DataArray, xr.Dataset)):
        return xr.where(xr.ufuncs.isnan(a), b, a)
    return jnp.where(jnp.isnan(a), b, a)

@register_function(category="statistical")
def std(data, dim=None):
    """
    Compute the standard deviation of data along the specified dimension.
    
    Works with both xarray objects and JAX arrays.
    
    Args:
        data: xarray DataArray or JAX array
        dim: The dimension to compute std along (for xarray objects)
        
    Returns:
        Standard deviation of the data, preserving other dimensions for xarray objects
    """
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        return data.std(dim=dim)
    return jnp.std(data)

@register_function(category="statistical")
def var(data, dim=None):
    """
    Compute the variance of data along the specified dimension.
    
    Works with both xarray objects and JAX arrays.
    
    Args:
        data: xarray DataArray or JAX array
        dim: The dimension to compute variance along (for xarray objects)
        
    Returns:
        Variance of the data, preserving other dimensions for xarray objects
    """
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        return data.var(dim=dim)
    return jnp.var(data)

@register_function(category="statistical")
def min(data, dim=None):
    """
    Compute the minimum of data along the specified dimension.
    
    Works with both xarray objects and JAX arrays.
    
    Args:
        data: xarray DataArray or JAX array
        dim: The dimension to compute minimum along (for xarray objects)
        
    Returns:
        Minimum of the data, preserving other dimensions for xarray objects
    """
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        return data.min(dim=dim)
    return jnp.min(data)

@register_function(category="statistical")
def max(data, dim=None):
    """
    Compute the maximum of data along the specified dimension.
    
    Works with both xarray objects and JAX arrays.
    
    Args:
        data: xarray DataArray or JAX array
        dim: The dimension to compute maximum along (for xarray objects)
        
    Returns:
        Maximum of the data, preserving other dimensions for xarray objects
    """
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        return data.max(dim=dim)
    return jnp.max(data)

@register_function(category="arithmetic")
def sum(data, dim=None):
    """
    Compute the sum of data along the specified dimension.
    
    Works with both xarray objects and JAX arrays.
    
    Args:
        data: xarray DataArray or JAX array
        dim: The dimension to sum along (for xarray objects)
        
    Returns:
        Sum of the data, preserving other dimensions for xarray objects
    """
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        return data.sum(dim=dim)
    return jnp.sum(data)

@register_function(category="arithmetic")
def sqrt(x):
    """
    Calculate square root.
    
    Works with both xarray DataArrays and JAX arrays.
    
    Args:
        x: Input data
        
    Returns:
        Square root with same structure as input
    """
    if isinstance(x, (xr.DataArray, xr.Dataset)):
        return xr.ufuncs.sqrt(x)
    return jnp.sqrt(x)

@register_function(category="arithmetic")
def abs(x):
    """
    Calculate absolute value.
    
    Works with both xarray DataArrays and JAX arrays.
    
    Args:
        x: Input data
        
    Returns:
        Absolute value with same structure as input
    """
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        return abs(x)  # xarray implements abs natively
    return jnp.abs(x)

@register_function(category="arithmetic")
def log(x):
    """
    Calculate natural logarithm.
    
    Works with both xarray DataArrays and JAX arrays.
    
    Args:
        x: Input data
        
    Returns:
        Natural logarithm with same structure as input
    """
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        return xr.ufuncs.log(x)
    return jnp.log(x)

@register_function(category="arithmetic")
def exp(x):
    """
    Calculate exponential (e^x).
    
    Works with both xarray DataArrays and JAX arrays.
    
    Args:
        x: Input data
        
    Returns:
        Exponential with same structure as input
    """
    if isinstance(x, (xr.DataArray, xr.Dataset)):
        return xr.ufuncs.exp(x)
    return jnp.exp(x)

@register_function(category="statistical")
def mean(data, dim=None):
    """
    Calculate the arithmetic mean along the specified dimension.
    
    Args:
        data: Input array or dataset
        dim: Dimension to reduce over. If None, calculate mean over all dimensions.
        
    Returns:
        Mean values along the specified dimension
        
    Examples:
        >>> mean(jnp.array([1, 2, 3, 4, 5]))
        Array(3., dtype=float32)
        >>> mean(jnp.array([[1, 2], [3, 4]]), dim=0)
        Array([2., 3.], dtype=float32)
    """
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        return data.mean(dim=dim)
    return jnp.mean(data, axis=dim)

@register_function(category="temporal")
def moving_mean(data, window):
    """
    Calculate the moving average along the time dimension.
    
    This function calculates the rolling mean of a dataset over a window
    of size 'window' along the time dimension. It's designed to work with
    the Hindsight data structure and rolling operations.
    
    When executed, this function will be processed using:
        dataset.dt.rolling(dim='time', window=window).reduce(core_mean)
        
    Args:
        data: Input array or dataset
        window: Size of the rolling window
        
    Returns:
        Dataset with rolling mean values
        
    Examples:
        >>> import xarray as xr
        >>> # Example with xarray dataset
        >>> ds = xr.Dataset(...)
        >>> moving_avg = moving_mean(ds, 30)  # 30-day moving average
        >>> # Equivalent to:
        >>> # moving_avg = ds.dt.rolling(dim='time', window=30).reduce(core_mean)
    """
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        return data.dt.rolling(dim='time', window=window).reduce(core_mean)
    raise ValueError("Input must be an xarray DataArray or Dataset")

@register_function(category="temporal")
def moving_ema(data, window):
    """
    Calculate the exponential moving average along the time dimension.
    
    This function calculates the rolling EMA of a dataset over a window
    of size 'window' along the time dimension. It uses an exponential
    weighting scheme.
    
    When executed, this function will be processed using:
        dataset.dt.rolling(dim='time', window=window).reduce(core_ema)
        
    Args:
        data: Input array or dataset
        window: Size of the rolling window
        
    Returns:
        Dataset with rolling EMA values
        
    Examples:
        >>> import xarray as xr
        >>> # Example with xarray dataset
        >>> ds = xr.Dataset(...)
        >>> ema_values = moving_ema(ds, 30)  # 30-day EMA
        >>> # Equivalent to:
        >>> # ema_values = ds.dt.rolling(dim='time', window=30).reduce(core_ema)
    """
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        return data.dt.rolling(dim='time', window=window).reduce(core_ema)
    raise ValueError("Input must be an xarray DataArray or Dataset")

@register_function(category="temporal")
def moving_median(data, window):
    """
    Calculate the moving median along the time dimension.
    
    This function calculates the rolling median of a dataset over a window
    of size 'window' along the time dimension.
    
    When executed, this function will be processed using:
        dataset.dt.rolling(dim='time', window=window).reduce(core_median)
        
    Args:
        data: Input array or dataset
        window: Size of the rolling window
        
    Returns:
        Dataset with rolling median values
        
    Examples:
        >>> import xarray as xr
        >>> # Example with xarray dataset
        >>> ds = xr.Dataset(...)
        >>> median_values = moving_median(ds, 30)  # 30-day moving median
        >>> # Equivalent to:
        >>> # median_values = ds.dt.rolling(dim='time', window=30).reduce(core_median)
    """
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        return data.dt.rolling(dim='time', window=window).reduce(core_median)
    raise ValueError("Input must be an xarray DataArray or Dataset")

def register_built_in_functions():
    """Register built-in functions for common operations."""
    # Most functions are registered via decorators above,
    # but we can also register existing functions directly
    pass

# Register built-in functions when the module is imported
register_built_in_functions() 