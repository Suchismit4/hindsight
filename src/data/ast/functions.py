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

import inspect
from typing import Dict, Any, Callable, List, Union, Optional, TypeVar, cast, overload
import xarray as xr
import jax.numpy as jnp
import numpy as np

from src.data.core.operations import mean as core_mean
from src.data.core.operations import ema as core_ema
from src.data.core.operations import median as core_median
from src.data.core.operations import mode as core_mode
from src.data.core.operations import gain as core_gain
from src.data.core.operations import loss as core_loss
from src.data.core.operations import rma as core_rma
from src.data.core.operations import wma as core_wma
from src.data.core.operations import triple_exponential_smoothing as core_triple_exponential_smoothing
from src.data.core.operations import adaptive_ema as core_adaptive_ema
from src.data.core.operations import sum_func as core_sum_func

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
    return xr.where(np.isnan(a), b, a)

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
    return data.std(dim=dim)

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
    return data.var(dim=dim)

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
    return data.min(dim=dim)

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
    return data.max(dim=dim)

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
    return data.sum(dim=dim)

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
    return np.sqrt(x)

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
    return np.abs(x)

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
    return np.log(x)

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
    return np.exp(x)

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
    return data.mean(dim=dim)

@register_function(category="temporal")
def sma(data, window):
    """
    Calculate the moving average along the time dimension.
    
    This function calculates the rolling mean of a dataset or DataArray over a window
    of size 'window' along the time dimension. It's designed to work with
    the Hindsight data structure and rolling operations.
    
    The function handles both Dataset and DataArray inputs:
    - For Dataset inputs, it uses the Dataset's built-in mask coordinates
    - For DataArray inputs, it uses the mask from the parent Dataset or from context
        
    Args:
        data: Input array or dataset
        window: Size of the rolling window
        
    Returns:
        Dataset or DataArray with rolling mean values
        
    Examples:
        >>> import xarray as xr
        >>> # Example with xarray dataset
        >>> ds = xr.Dataset(...)
        >>> # Whole dataset calculation
        >>> moving_avg = sma(ds, 30)  # 30-day moving average for all variables
        >>> # Single variable calculation
        >>> close_avg = sma($close, 30)  # 30-day moving average for just close prices
    """
    if isinstance(data, xr.Dataset):
        # Dataset case: use built-in rolling method with dataset's mask
        return data.dt.rolling(dim='time', window=window).reduce(core_mean)
    elif isinstance(data, xr.DataArray):
        # DataArray case: need to handle mask explicitly
        mask = None
        indices = None
        if hasattr(data, 'attrs') and '_parent_dataset' in data.attrs:
            parent_ds = data.attrs['_parent_dataset']
            if 'mask' in parent_ds.coords:
                mask = parent_ds.coords['mask'].values
            if 'mask_indices' in parent_ds.coords:
                indices = parent_ds.coords['mask_indices'].values
                
        if mask is not None and indices is not None:
            # Call dt.rolling directly, passing mask and indices
            return data.dt.rolling(dim='time', window=window, mask=mask, mask_indices=indices).reduce(core_mean)
        else:
            # If mask/indices cannot be found, raise an error
            raise ValueError(
                "Rolling operation on DataArray requires mask and mask_indices. "
                "Ensure the DataArray originated from a Dataset with these coordinates, "
                "or provide them explicitly if calling the function directly."
            )
    else:
        # Handle other types (e.g., numpy arrays)
        raise TypeError(f"Unsupported data type for sma: {type(data)}")

@register_function(category="temporal")
def ema(data, window):
    """
    Calculate the exponential moving average along the time dimension.
    
    This function calculates the rolling EMA of a dataset or DataArray over a window
    of size 'window' along the time dimension. It uses an exponential
    weighting scheme.
    
    The function handles both Dataset and DataArray inputs:
    - For Dataset inputs, it uses the Dataset's built-in mask coordinates
    - For DataArray inputs, it uses the mask from the parent Dataset or from context
        
    Args:
        data: Input array or dataset
        window: Size of the rolling window
        
    Returns:
        Dataset or DataArray with rolling EMA values
        
    Examples:
        >>> import xarray as xr
        >>> # Example with xarray dataset
        >>> ds = xr.Dataset(...)
        >>> # Whole dataset calculation
        >>> ema_values = ema(ds, 30)  # 30-day EMA for all variables
        >>> # Single variable calculation
        >>> close_ema = ema($close, 30)  # 30-day EMA for just close prices
    """    
    if isinstance(data, xr.Dataset):
        # Dataset case: use built-in rolling method with dataset's mask
        return data.dt.rolling(dim='time', window=window).reduce(core_ema)
    elif isinstance(data, xr.DataArray):
        # DataArray case: need to handle mask explicitly
        mask = None
        indices = None
        if hasattr(data, 'attrs') and '_parent_dataset' in data.attrs:
            parent_ds = data.attrs['_parent_dataset']
            if 'mask' in parent_ds.coords:
                mask = parent_ds.coords['mask'].values
            if 'mask_indices' in parent_ds.coords:
                indices = parent_ds.coords['mask_indices'].values
                
        if mask is not None and indices is not None:
            # Call dt.rolling directly, passing mask and indices
            return data.dt.rolling(dim='time', window=window, mask=mask, mask_indices=indices).reduce(core_ema)
        else:
            # If mask/indices cannot be found, raise an error
            raise ValueError(
                "Rolling operation on DataArray requires mask and mask_indices. "
                "Ensure the DataArray originated from a Dataset with these coordinates, "
                "or provide them explicitly if calling the function directly."
            )
    else:
        # Handle other types (e.g., numpy arrays)
        raise TypeError(f"Unsupported data type for ema: {type(data)}")

@register_function(category="temporal")
def rma(data, window):
    """
    Calculate the relative moving average along the time dimension.
    
    This function calculates the rolling RMA of a dataset or DataArray over a window
    of size 'window' along the time dimension. It uses a simple moving average
    weighting scheme.
    
    The function handles both Dataset and DataArray inputs:
    - For Dataset inputs, it uses the Dataset's built-in mask coordinates
    - For DataArray inputs, it uses the mask from the parent Dataset or from context
        
    Args:
        data: Input array or dataset
        window: Size of the rolling window
        
    Returns:
        Dataset or DataArray with rolling RMA values
        
    Examples:
        >>> import xarray as xr
        >>> # Example with xarray dataset
        >>> ds = xr.Dataset(...)
        >>> # Whole dataset calculation
        >>> rma_values = rma(ds, 30)  # 30-day RMA for all variables
        >>> # Single variable calculation
        >>> close_rma = rma($close, 30)  # 30-day RMA for just close prices
    """    
    if isinstance(data, xr.Dataset):
        # Dataset case: use built-in rolling method with dataset's mask
        return data.dt.rolling(dim='time', window=window).reduce(core_rma)
    elif isinstance(data, xr.DataArray):
        # DataArray case: need to handle mask explicitly
        mask = None
        indices = None
        if hasattr(data, 'attrs') and '_parent_dataset' in data.attrs:
            parent_ds = data.attrs['_parent_dataset']
            if 'mask' in parent_ds.coords:
                mask = parent_ds.coords['mask'].values
            if 'mask_indices' in parent_ds.coords:
                indices = parent_ds.coords['mask_indices'].values
                
        if mask is not None and indices is not None:
            # Call dt.rolling directly, passing mask and indices
            return data.dt.rolling(dim='time', window=window, mask=mask, mask_indices=indices).reduce(core_rma)
        else:
            # If mask/indices cannot be found, raise an error
            raise ValueError(
                "Rolling operation on DataArray requires mask and mask_indices. "
                "Ensure the DataArray originated from a Dataset with these coordinates, "
                "or provide them explicitly if calling the function directly."
            )
    else:
        # Handle other types (e.g., numpy arrays)
        raise TypeError(f"Unsupported data type for rma: {type(data)}")


@register_function(category="temporal") 
def gain(data, window):
    """
    Calculate the gain (positive change) along the time dimension.
    
    This function calculates the rolling gain of a dataset or DataArray over a window
    of size 'window' along the time dimension.
    
    The function handles both Dataset and DataArray inputs:
    - For Dataset inputs, it uses the Dataset's built-in mask coordinates
    - For DataArray inputs, it uses the mask from the parent Dataset or from context
        
    Args:
        data: Input array or dataset
        window: Size of the rolling window
        
    Returns:
        Dataset or DataArray with rolling gain values

    Examples:
        >>> import xarray as xr
        >>> # Example with xarray dataset
        >>> ds = xr.Dataset(...)
        >>> # Whole dataset calculation
        >>> gain_values = gain(ds, 30)  # 30-day gain for all variables
        >>> # Single variable calculation
        >>> close_gain = gain($close, 30)  # 30-day gain for just close prices
    """
    if isinstance(data, xr.Dataset):
        # Dataset case: use built-in rolling method with dataset's mask
        return data.dt.rolling(dim='time', window=window).reduce(core_gain)
    elif isinstance(data, xr.DataArray):
        # DataArray case: need to handle mask explicitly
        mask = None
        indices = None
        if hasattr(data, 'attrs') and '_parent_dataset' in data.attrs:
            parent_ds = data.attrs['_parent_dataset']
            if 'mask' in parent_ds.coords:
                mask = parent_ds.coords['mask'].values
            if 'mask_indices' in parent_ds.coords:
                indices = parent_ds.coords['mask_indices'].values
                
        if mask is not None and indices is not None:
            # Call dt.rolling directly, passing mask and indices
            return data.dt.rolling(dim='time', window=window, mask=mask, mask_indices=indices).reduce(core_gain)
        else:
            # If mask/indices cannot be found, raise an error
            raise ValueError(
                "Rolling operation on DataArray requires mask and mask_indices. "
                "Ensure the DataArray originated from a Dataset with these coordinates, "
                "or provide them explicitly if calling the function directly."
            )
    else:
        # Handle other types (e.g., numpy arrays)
        raise TypeError(f"Unsupported data type for gain: {type(data)}")

@register_function(category="temporal") 
def loss(data, window):
    """
    Calculate the loss (negative change) along the time dimension.
    
    This function calculates the rolling loss of a dataset or DataArray over a window
    of size 'window' along the time dimension.
    
    The function handles both Dataset and DataArray inputs:
    - For Dataset inputs, it uses the Dataset's built-in mask coordinates
    - For DataArray inputs, it uses the mask from the parent Dataset or from context
        
    Args:
        data: Input array or dataset
        window: Size of the rolling window
        
    Returns:
        Dataset or DataArray with rolling loss values
        
    Examples:
        >>> import xarray as xr
        >>> # Example with xarray dataset
        >>> ds = xr.Dataset(...)
        >>> # Whole dataset calculation
        >>> loss_values = loss(ds, 30)  # 30-day loss for all variables
        >>> # Single variable calculation
        >>> close_loss = loss($close, 30)  # 30-day loss for just close prices
    """
    if isinstance(data, xr.Dataset):
        # Dataset case: use built-in rolling method with dataset's mask
        return data.dt.rolling(dim='time', window=window).reduce(core_loss)
    elif isinstance(data, xr.DataArray):
        # DataArray case: need to handle mask explicitly
        mask = None
        indices = None
        if hasattr(data, 'attrs') and '_parent_dataset' in data.attrs:
            parent_ds = data.attrs['_parent_dataset']
            if 'mask' in parent_ds.coords:
                mask = parent_ds.coords['mask'].values
            if 'mask_indices' in parent_ds.coords:
                indices = parent_ds.coords['mask_indices'].values
                
        if mask is not None and indices is not None:
            # Call dt.rolling directly, passing mask and indices
            return data.dt.rolling(dim='time', window=window, mask=mask, mask_indices=indices).reduce(core_loss)
        else:
            # If mask/indices cannot be found, raise an error
            raise ValueError(
                "Rolling operation on DataArray requires mask and mask_indices. "
                "Ensure the DataArray originated from a Dataset with these coordinates, "
                "or provide them explicitly if calling the function directly."
            )
    else:
        # Handle other types (e.g., numpy arrays)
        raise TypeError(f"Unsupported data type for loss: {type(data)}")

@register_function(category="temporal")
def wma(data, window, weights=None):
    """
    Calculate the weighted moving average along the time dimension.
    
    This function calculates the weighted moving average of a dataset or DataArray over a window
    of size 'window' along the time dimension. If weights are not provided, it defaults to
    linearly increasing weights (1, 2, 3, ..., window).
    
    The function handles both Dataset and DataArray inputs:
    - For Dataset inputs, it uses the Dataset's built-in mask coordinates
    - For DataArray inputs, it uses the mask from the parent Dataset or from context
        
    Args:
        data: Input array or dataset
        window: Size of the rolling window
        weights: Optional array of weights. If None, defaults to linearly increasing weights.
                Must have length equal to window size.
        
    Returns:
        Dataset or DataArray with rolling weighted moving average values
        
    Examples:
        >>> import xarray as xr
        >>> import jax.numpy as jnp
        >>> # Example with xarray dataset
        >>> ds = xr.Dataset(...)
        >>> # Whole dataset calculation with default weights
        >>> wma_values = wma(ds, 30)  # 30-day WMA for all variables
        >>> # Single variable calculation with custom weights
        >>> custom_weights = jnp.array([0.1, 0.2, 0.3, 0.4])  # Must sum to 1.0 or will be normalized
        >>> close_wma = wma($close, 4, custom_weights)  # 4-day WMA for close prices
    """
    # Convert weights to JAX array if provided
    if weights is not None:
        if len(weights) != window:
            raise ValueError(f"Weights array length ({len(weights)}) must equal window size ({window})")
    
    if isinstance(data, xr.Dataset):
        # Dataset case: use built-in rolling method with dataset's mask
        return data.dt.rolling(dim='time', window=window).reduce(core_wma, weights=weights)
    elif isinstance(data, xr.DataArray):
        # DataArray case: need to handle mask explicitly
        mask = None
        indices = None
        if hasattr(data, 'attrs') and '_parent_dataset' in data.attrs:
            parent_ds = data.attrs['_parent_dataset']
            if 'mask' in parent_ds.coords:
                mask = parent_ds.coords['mask'].values
            if 'mask_indices' in parent_ds.coords:
                indices = parent_ds.coords['mask_indices'].values
                
        if mask is not None and indices is not None:
            # Call dt.rolling directly, passing mask and indices
            return data.dt.rolling(dim='time', window=window, mask=mask, mask_indices=indices).reduce(core_wma, weights=weights)
        else:
            # If mask/indices cannot be found, raise an error
            raise ValueError(
                "Rolling operation on DataArray requires mask and mask_indices. "
                "Ensure the DataArray originated from a Dataset with these coordinates, "
                "or provide them explicitly if calling the function directly."
            )
    else:
        # Handle other types (e.g., numpy arrays)
        raise TypeError(f"Unsupported data type for wma: {type(data)}")

@register_function(category="temporal")
def moving_median(data, window):
    """
    Calculate the moving median along the time dimension.
    
    This function calculates the rolling median of a dataset or DataArray over a window
    of size 'window' along the time dimension.
    
    The function handles both Dataset and DataArray inputs:
    - For Dataset inputs, it uses the Dataset's built-in mask coordinates
    - For DataArray inputs, it uses the mask from the parent Dataset or from context
        
    Args:
        data: Input array or dataset
        window: Size of the rolling window
        
    Returns:
        Dataset or DataArray with rolling median values
        
    Examples:
        >>> import xarray as xr
        >>> # Example with xarray dataset
        >>> ds = xr.Dataset(...)
        >>> # Whole dataset calculation
        >>> median_values = moving_median(ds, 30)  # 30-day moving median for all variables
        >>> # Single variable calculation
        >>> close_median = moving_median($close, 30)  # 30-day moving median for just close prices
    """
    if isinstance(data, xr.Dataset):
        # Dataset case: use built-in rolling method with dataset's mask
        return data.dt.rolling(dim='time', window=window).reduce(core_median)
    elif isinstance(data, xr.DataArray):
        # DataArray case: need to handle mask explicitly
        mask = None
        indices = None
        if hasattr(data, 'attrs') and '_parent_dataset' in data.attrs:
            parent_ds = data.attrs['_parent_dataset']
            if 'mask' in parent_ds.coords:
                mask = parent_ds.coords['mask'].values
            if 'mask_indices' in parent_ds.coords:
                indices = parent_ds.coords['mask_indices'].values
                
        if mask is not None and indices is not None:
            # Call dt.rolling directly, passing mask and indices
            return data.dt.rolling(dim='time', window=window, mask=mask, mask_indices=indices).reduce(core_median)
        else:
            # If mask/indices cannot be found, raise an error
            raise ValueError(
                "Rolling operation on DataArray requires mask and mask_indices. "
                "Ensure the DataArray originated from a Dataset with these coordinates, "
                "or provide them explicitly if calling the function directly."
            )
    else:
        # Handle other types (e.g., numpy arrays)
        raise TypeError(f"Unsupported data type for moving_median: {type(data)}")

@register_function(category="temporal")
def moving_mode(data, window):  
    """
    Calculate the moving mode along the time dimension.
    
    This function calculates the rolling mode of a dataset or DataArray over a window
    of size 'window' along the time dimension.
    
    The function handles both Dataset and DataArray inputs:
    - For Dataset inputs, it uses the Dataset's built-in mask coordinates
    - For DataArray inputs, it uses the mask from the parent Dataset or from context
        
    Args:
        data: Input array or dataset
        window: Size of the rolling window
        
    Returns:
        Dataset or DataArray with rolling mode values
        
    Examples:
        >>> import xarray as xr
        >>> # Example with xarray dataset
        >>> ds = xr.Dataset(...)
        >>> # Whole dataset calculation
        >>> mode_values = moving_mode(ds, 30)  # 30-day moving mode for all variables
        >>> # Single variable calculation
        >>> close_mode = moving_mode($close, 30)  # 30-day moving mode for just close prices
    """
    if isinstance(data, xr.Dataset):
        # Dataset case: use built-in rolling method with dataset's mask
        return data.dt.rolling(dim='time', window=window).reduce(core_mode)
    elif isinstance(data, xr.DataArray):
        # DataArray case: need to handle mask explicitly
        mask = None
        indices = None
        if hasattr(data, 'attrs') and '_parent_dataset' in data.attrs:
            parent_ds = data.attrs['_parent_dataset']
            if 'mask' in parent_ds.coords:
                mask = parent_ds.coords['mask'].values
            if 'mask_indices' in parent_ds.coords:
                indices = parent_ds.coords['mask_indices'].values
                
        if mask is not None and indices is not None:
            # Call dt.rolling directly, passing mask and indices
            return data.dt.rolling(dim='time', window=window, mask=mask, mask_indices=indices).reduce(core_mode)
        else:
            # If mask/indices cannot be found, raise an error
            raise ValueError(
                "Rolling operation on DataArray requires mask and mask_indices. "
                "Ensure the DataArray originated from a Dataset with these coordinates, "
                "or provide them explicitly if calling the function directly."
            )
    else:
        # Handle other types (e.g., numpy arrays)
        raise TypeError(f"Unsupported data type for moving_mode: {type(data)}")

# Add a utility function to handle shift operations with DataArrays
@register_function(category="temporal")
def shift(data, periods=1):
    """
    Shift data along the time dimension.
    
    This function shifts a dataset or DataArray by a specified number of periods
    along the time dimension. Positive values shift forward in time (introducing
    NaNs at the beginning), negative values shift backward in time (introducing
    NaNs at the end).
    
    Args:
        data: Input array or dataset
        periods: Number of periods to shift (default: 1)
        
    Returns:
        Dataset or DataArray with shifted values
        
    Examples:
        >>> import xarray as xr
        >>> # Example with xarray dataset
        >>> ds = xr.Dataset(...)
        >>> # Whole dataset shift
        >>> shifted_ds = shift(ds, 1)  # Shift all variables forward by 1 time period
        >>> # Single variable shift
        >>> shifted_close = shift($close, -1)  # Shift close prices backward by 1 time period
    """
    if isinstance(data, xr.Dataset):
        # Check if this dataset has the 3D time structure (year, month, day)
        if 'year' in data.coords and 'month' in data.coords and 'day' in data.coords:
            # Use business day aware shift
            return data.dt.shift(periods=periods)
        else:
            # Use simple xarray shift for regular time dimensions
            return data.shift(time=periods)
            
    elif isinstance(data, xr.DataArray):
        # Check if this DataArray has the 3D time structure
        if 'year' in data.coords and 'month' in data.coords and 'day' in data.coords:
            # DataArray case with 3D time: need to handle mask explicitly
            mask_indices = None
            if hasattr(data, 'attrs') and '_parent_dataset' in data.attrs:
                parent_ds = data.attrs['_parent_dataset']
                if 'mask_indices' in parent_ds.coords:
                    mask_indices = parent_ds.coords['mask_indices'].values
                    
            if mask_indices is not None:
                # Call dt.shift directly, passing mask_indices
                import jax.numpy as jnp
                return data.dt.shift(periods=periods, mask_indices=jnp.array(mask_indices))
            else:
                # If mask_indices cannot be found, raise an error
                raise ValueError(
                    "Shift operation on DataArray with 3D time structure requires mask_indices. "
                    "Ensure the DataArray originated from a Dataset with these coordinates, "
                    "or provide them explicitly if calling the function directly."
                )
        else:
            # Use simple xarray shift for regular time dimensions
            return data.shift(time=periods)
    else:
        # Handle numpy arrays or other types
        raise TypeError(f"Unsupported data type for shift: {type(data)}")

# Register a function to easily compute returns for financial time series
@register_function(category="financial")
def returns(price_data, periods=1):
    """
    Calculate simple returns for a price series.
    
    This function computes the simple returns (price_t / price_t-n - 1) for 
    a given price series over a specified number of periods.
    
    Args:
        price_data: Input price array or dataset 
        periods: Number of periods to use in return calculation (default: 1)
        
    Returns:
        Dataset or DataArray with returns values
        
    Examples:
        >>> import xarray as xr
        >>> # Example with xarray dataset
        >>> ds = xr.Dataset(...)
        >>> # Whole dataset returns
        >>> all_returns = returns(ds, 1)  # 1-period returns for all variables
        >>> # Single variable returns
        >>> close_returns = returns($close, 1)  # 1-period returns for just close prices
    """
    if isinstance(price_data, (xr.Dataset, xr.DataArray)):
        shifted_price = shift(price_data, periods)
        return price_data / shifted_price - 1
    else:
        # Handle numpy arrays or other types
        raise TypeError(f"Unsupported data type for returns: {type(price_data)}")

@register_function(category="temporal")
def triple_exponential_smoothing(data, alpha=0.2, beta=0.1, gamma=0.1):
    """
    Calculate triple exponential smoothing (Holt-Winters method) along the time dimension.
    
    This function implements the generic Holt-Winters triple exponential smoothing algorithm
    that maintains three state variables: level (F), trend (V), and acceleration (A).
    
    The computation follows the Holt-Winters equations:
    F[t] = (1-α) * (F[t-1] + V[t-1] + 0.5*A[t-1]) + α * X[t]
    V[t] = (1-β) * (V[t-1] + A[t-1]) + β * (F[t] - F[t-1])  
    A[t] = (1-γ) * A[t-1] + γ * (V[t] - V[t-1])
    
    Output = F[t] + V[t] + 0.5*A[t]
    
    This function can be used as a foundation for implementing HWMA and other
    triple exponential smoothing variants.
    
    Args:
        data: Input array or dataset
        alpha: Level smoothing parameter (0 < α < 1, default: 0.2)
        beta: Trend smoothing parameter (0 < β < 1, default: 0.1)  
        gamma: Acceleration smoothing parameter (0 < γ < 1, default: 0.1)
        
    Returns:
        Dataset or DataArray with triple exponential smoothing values
        
    Examples:
        >>> import xarray as xr
        >>> # Example with xarray dataset
        >>> ds = xr.Dataset(...)
        >>> # Apply triple exponential smoothing with default parameters
        >>> smoothed = triple_exponential_smoothing(ds)
        >>> # Apply with custom parameters
        >>> smoothed = triple_exponential_smoothing($close, alpha=0.3, beta=0.15, gamma=0.05)
    """
    # Validate parameters
    if not (0 < alpha < 1):
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
    if not (0 < beta < 1):
        raise ValueError(f"Beta must be between 0 and 1, got {beta}")
    if not (0 < gamma < 1):
        raise ValueError(f"Gamma must be between 0 and 1, got {gamma}")
    
    if isinstance(data, xr.Dataset):
        # Dataset case: use built-in rolling method with dataset's mask
        # For stateful operations, we use window=1 since each point depends on the previous state
        return data.dt.rolling(dim='time', window=1).reduce(
            core_triple_exponential_smoothing, 
            alpha=alpha, beta=beta, gamma=gamma
        )
    elif isinstance(data, xr.DataArray):
        # DataArray case: need to handle mask explicitly
        mask = None
        indices = None
        if hasattr(data, 'attrs') and '_parent_dataset' in data.attrs:
            parent_ds = data.attrs['_parent_dataset']
            if 'mask' in parent_ds.coords:
                mask = parent_ds.coords['mask'].values
            if 'mask_indices' in parent_ds.coords:
                indices = parent_ds.coords['mask_indices'].values
                
        if mask is not None and indices is not None:
            # Call dt.rolling directly, passing mask and indices
            return data.dt.rolling(dim='time', window=1, mask=mask, mask_indices=indices).reduce(
                core_triple_exponential_smoothing,
                alpha=alpha, beta=beta, gamma=gamma
            )
        else:
            # If mask/indices cannot be found, raise an error
            raise ValueError(
                "Rolling operation on DataArray requires mask and mask_indices. "
                "Ensure the DataArray originated from a Dataset with these coordinates, "
                "or provide them explicitly if calling the function directly."
            )
    else:
        # Handle other types (e.g., numpy arrays)
        raise TypeError(f"Unsupported data type for triple_exponential_smoothing: {type(data)}")

@register_function(category="temporal")
def adaptive_ema(data, smoothing_factors):
    """
    Calculate adaptive exponential moving average with varying smoothing factors.
    
    This function applies EMA-style smoothing where the smoothing parameter can vary
    at each time step. This enables implementation of indicators like KAMA and other
    adaptive smoothing methods.
    
    The computation follows: EMA[t] = smoothing[t] * value[t] + (1 - smoothing[t]) * EMA[t-1]
    
    Args:
        data: Input array or dataset (time series data)
        smoothing_factors: Array of smoothing factors for each time step (0 < factor < 1)
        
    Returns:
        Dataset or DataArray with adaptive EMA values
        
    Examples:
        >>> import xarray as xr
        >>> import jax.numpy as jnp
        >>> # Example with varying smoothing factors
        >>> smoothing = xr.DataArray(jnp.array([0.1, 0.2, 0.3, 0.2, 0.1]))
        >>> result = adaptive_ema($close, smoothing)
    """
    
    # Convert smoothing_factors to JAX array if it's an xarray object
    if isinstance(smoothing_factors, (xr.Dataset, xr.DataArray)):
        # Stack the time dimensions if it's a 3D structure to match how the rolling operations work
        if 'year' in smoothing_factors.coords and 'month' in smoothing_factors.coords and 'day' in smoothing_factors.coords:
            smoothing_stacked = smoothing_factors.stack(time_index=("year", "month", "day"))
            smoothing_stacked = smoothing_stacked.transpose("time_index", ...)
            smoothing_vals = smoothing_stacked.values
        else:
            smoothing_vals = smoothing_factors.values
            
        # Ensure we have the right shape for the core function: (T, N, 1)
        if smoothing_vals.ndim == 2:  # (T, N) - this is the expected case for dataset operations
            smoothing_vals = smoothing_vals[..., None]  # (T, N, 1)
        elif smoothing_vals.ndim == 1:  # (T,) - single asset case
            smoothing_vals = smoothing_vals[:, None, None]  # (T, 1, 1)
        elif smoothing_vals.ndim == 3:  # Already (T, N, 1)
            pass  # No change needed
        else:
            raise ValueError(f"Unexpected smoothing_factors shape: {smoothing_vals.shape}")
            
        smoothing_jax = jnp.asarray(smoothing_vals)
    else:
        # Already a JAX/numpy array
        smoothing_jax = jnp.asarray(smoothing_factors)
        if smoothing_jax.ndim == 1:
            smoothing_jax = smoothing_jax[:, None, None]
        elif smoothing_jax.ndim == 2:
            smoothing_jax = smoothing_jax[..., None]
    
    if isinstance(data, xr.Dataset):
        # Dataset case: use built-in rolling method with dataset's mask
        # For stateful operations, we use window=1 since each point depends on the previous state
        return data.dt.rolling(dim='time', window=1).reduce(
            core_adaptive_ema, 
            smoothing_factors=smoothing_jax
        )
    elif isinstance(data, xr.DataArray):
        # DataArray case: need to handle mask explicitly
        mask = None
        indices = None
        if hasattr(data, 'attrs') and '_parent_dataset' in data.attrs:
            parent_ds = data.attrs['_parent_dataset']
            if 'mask' in parent_ds.coords:
                mask = parent_ds.coords['mask'].values
            if 'mask_indices' in parent_ds.coords:
                indices = parent_ds.coords['mask_indices'].values
                
        if mask is not None and indices is not None:
            # Call dt.rolling directly, passing mask and indices
            return data.dt.rolling(dim='time', window=1, mask=mask, mask_indices=indices).reduce(
                core_adaptive_ema,
                smoothing_factors=smoothing_jax
            )
        else:
            # If mask/indices cannot be found, raise an error
            raise ValueError(
                "Rolling operation on DataArray requires mask and mask_indices. "
                "Ensure the DataArray originated from a Dataset with these coordinates, "
                "or provide them explicitly if calling the function directly."
            )
    else:
        # Handle other types (e.g., numpy arrays)
        raise TypeError(f"Unsupported data type for adaptive_ema: {type(data)}")

@register_function(category="temporal")
def rolling_sum(data, window):
    """
    Calculate the rolling sum along the time dimension.
    
    This function calculates the rolling sum of a dataset or DataArray over a window
    of size 'window' along the time dimension.
    
    Args:
        data: Input array or dataset
        window: Size of the rolling window
        
    Returns:
        Dataset or DataArray with rolling sum values
        
    Examples:
        >>> rolling_sum($close, 10)  # 10-period rolling sum
    """
    if isinstance(data, xr.Dataset):
        return data.dt.rolling(dim='time', window=window).reduce(core_sum_func)
    elif isinstance(data, xr.DataArray):
        # DataArray case: need to handle mask explicitly
        mask = None
        indices = None
        if hasattr(data, 'attrs') and '_parent_dataset' in data.attrs:
            parent_ds = data.attrs['_parent_dataset']
            if 'mask' in parent_ds.coords:
                mask = parent_ds.coords['mask'].values
            if 'mask_indices' in parent_ds.coords:
                indices = parent_ds.coords['mask_indices'].values
                
        if mask is not None and indices is not None:
            return data.dt.rolling(dim='time', window=window, mask=mask, mask_indices=indices).reduce(core_sum_func)
        else:
            raise ValueError(
                "Rolling operation on DataArray requires mask and mask_indices. "
                "Ensure the DataArray originated from a Dataset with these coordinates, "
                "or provide them explicitly if calling the function directly."
            )
    else:
        raise TypeError(f"Unsupported data type for rolling_sum: {type(data)}")

def register_built_in_functions():
    """Register built-in functions for common operations."""
    # Most functions are registered via decorators above,
    # but we can also register existing functions directly
    pass

# Register built-in functions when the module is imported
register_built_in_functions() 