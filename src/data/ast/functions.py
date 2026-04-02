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
    Array(6., dtype=float64)
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

def _dispatch_rolling(fn, data, *, window, func_name, **reduce_kwargs):
    """Dispatch a rolling reduction over Dataset or DataArray."""
    if isinstance(data, (xr.Dataset, xr.DataArray)):
        return data.dt.rolling(dim='time', window=window).reduce(fn, **reduce_kwargs)
    raise TypeError(f"Unsupported data type for {func_name}: {type(data)}")


def _dispatch_cross_sectional(fn, data, *, func_name, **kwargs):
    """Dispatch a cross-sectional operation over Dataset or DataArray.

    DataArray: calls fn(data, **kwargs) directly.
    Dataset: applies fn to each numeric variable that has an 'asset' dim.
    """
    if isinstance(data, xr.DataArray):
        return fn(data, **kwargs)
    if isinstance(data, xr.Dataset):
        result_vars = {}
        for var_name, da in data.data_vars.items():
            if np.issubdtype(da.dtype, np.number) and 'asset' in da.dims:
                result_vars[var_name] = fn(da, **kwargs)
        return xr.Dataset(result_vars, coords=data.coords)
    raise TypeError(f"Unsupported data type for {func_name}: {type(data)}")


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
    "cross_sectional": [],
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

# Built-in Financial Functions

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
        Array(3., dtype=float64)
        >>> mean(jnp.array([[1, 2], [3, 4]]), dim=0)
        Array([2., 3.], dtype=float64)
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
    return _dispatch_rolling(core_mean, data, window=window, func_name='sma')

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
    return _dispatch_rolling(core_ema, data, window=window, func_name='ema')

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
    return _dispatch_rolling(core_rma, data, window=window, func_name='rma')


@register_function(category="temporal") 
def gain(data, window):
    """
    Calculate the gain (positive change) along the time dimension.
    
    This function calculates the rolling gain of a dataset or DataArray over a window
    of size 'window' along the time dimension.
    
    The function handles both Dataset and DataArray inputs:
    - For Dataset inputs, it uses the Dataset's built-in rolling method
    - For DataArray inputs, it uses the DataArray's built-in rolling method
    - Masks are computed on-demand by the rolling operation itself
        
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
    return _dispatch_rolling(core_gain, data, window=window, func_name='gain')

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
    return _dispatch_rolling(core_loss, data, window=window, func_name='loss')

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
    
    return _dispatch_rolling(core_wma, data, window=window, func_name='wma', weights=weights)

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
    return _dispatch_rolling(core_median, data, window=window, func_name='moving_median')

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
    return _dispatch_rolling(core_mode, data, window=window, func_name='moving_mode')

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
        # Check if this dataset has the multi-dimensional time structure (year, month, day, optionally hour)
        if 'year' in data.coords and 'month' in data.coords and 'day' in data.coords:
            # Use business day aware shift
            return data.dt.shift(periods=periods)
        else:
            # Use simple xarray shift for regular time dimensions
            return data.shift(time=periods)
            
    elif isinstance(data, xr.DataArray):
        # Check if this DataArray has the multi-dimensional time structure
        if {'year', 'month', 'day'}.issubset(data.coords):
            # Delegate to the accessor which knows how to handle business-day-aware shifting
            print(f"Shifting with business-day-aware shifting...")
            return data.dt.shift(periods=periods)
        elif 'time' in data.dims:
            # Regular 1D time dimension – fall back to native xarray shift
            return data.shift(time=periods)
        else:
            # No recognizable time dimension; return a copy to avoid silent failures
            raise ValueError(
                "Cannot determine time dimension for shift; expected coords including "
                "'year','month','day' or a 'time' dimension."
            )
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
    
    return _dispatch_rolling(
        core_triple_exponential_smoothing, data,
        window=1, func_name='triple_exponential_smoothing',
        alpha=alpha, beta=beta, gamma=gamma
    )

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
        # Stack the time dimensions if it's a multi-dimensional structure to match how the rolling operations work
        if 'year' in smoothing_factors.coords and 'month' in smoothing_factors.coords and 'day' in smoothing_factors.coords:
            # Determine time dimensions to stack based on what's present
            time_dims = ["year", "month", "day"]
            if "hour" in smoothing_factors.coords:
                time_dims.append("hour")
            
            smoothing_stacked = smoothing_factors.stack(time_index=tuple(time_dims))
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
            
        smoothing_jax = jnp.asarray(smoothing_vals, dtype=jnp.float64)
    else:
        # Already a JAX/numpy array
        smoothing_jax = jnp.asarray(smoothing_factors, dtype=jnp.float64)
        if smoothing_jax.ndim == 1:
            smoothing_jax = smoothing_jax[:, None, None]
        elif smoothing_jax.ndim == 2:
            smoothing_jax = smoothing_jax[..., None]
    
    return _dispatch_rolling(
        core_adaptive_ema, data,
        window=1, func_name='adaptive_ema',
        smoothing_factors=smoothing_jax
    )

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
    return _dispatch_rolling(core_sum_func, data, window=window, func_name='rolling_sum')

# ---------------------------------------------------------------------------
# Cross-sectional operations
# These operate across the *asset* dimension at each time slice.
# None of them are JAX-compatible; they rely on numpy/xarray internally.
# ---------------------------------------------------------------------------

def _cs_rank_1d(values: np.ndarray) -> np.ndarray:
    """Percentile rank one time-slice (1-D numpy array) across assets.

    NaN assets receive NaN rank. Requires at least 2 valid values to rank;
    if fewer, all assets get NaN.
    """
    result = np.full_like(values, np.nan, dtype=np.float64)
    valid = ~np.isnan(values)
    n = int(valid.sum())
    if n < 2:
        return result
    # argsort of argsort gives rank (0-based); normalise to [0, 1]
    idx = np.where(valid)[0]
    sorted_ranks = np.argsort(np.argsort(values[idx]))
    result[idx] = sorted_ranks.astype(np.float64) / (n - 1)
    return result


@register_function(category="cross_sectional")
def cs_rank(data: xr.DataArray) -> xr.DataArray:
    """Cross-sectional percentile rank across the asset dimension.

    At each time slice all assets are ranked from 0.0 (lowest) to 1.0
    (highest). Assets with NaN signal receive NaN rank and are excluded
    from rank computation of the remaining assets.

    Args:
        data: DataArray with an 'asset' dimension.

    Returns:
        DataArray of same shape with percentile ranks in [0, 1].

    NaN: NaN inputs produce NaN outputs; no silent zeroing.
    JAX: No – uses numpy argsort internally.
    """
    if isinstance(data, xr.Dataset):
        return _dispatch_cross_sectional(cs_rank, data, func_name='cs_rank')
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"cs_rank expects xr.DataArray, got {type(data)}")
    return xr.apply_ufunc(
        _cs_rank_1d,
        data,
        input_core_dims=[['asset']],
        output_core_dims=[['asset']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float64],
    )


@register_function(category="cross_sectional")
def cs_quantile(data: xr.DataArray, q: float) -> xr.DataArray:
    """Compute the q-th quantile across the asset dimension at each time slice.

    NaN values are ignored (skipna). The result has no 'asset' dimension –
    it is a scalar per time step. Use with assign_bucket or where to bring
    it back to asset-level comparisons (xarray broadcasting handles this).

    Args:
        data: DataArray with an 'asset' dimension.
        q:    Quantile in [0, 1].

    Returns:
        DataArray without the 'asset' dimension.

    NaN: skipna=True; all-NaN slices return NaN.
    JAX: No.
    """
    if isinstance(data, xr.Dataset):
        raise TypeError("cs_quantile requires a DataArray, not a Dataset")
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"cs_quantile expects xr.DataArray, got {type(data)}")
    if not isinstance(q, (int, float)):
        raise TypeError(f"cs_quantile: q must be a scalar float, got {type(q)}")
    return data.quantile(float(q), dim='asset', skipna=True).drop_vars('quantile', errors='ignore')


@register_function(category="cross_sectional")
def cs_demean(data: xr.DataArray) -> xr.DataArray:
    """Subtract the cross-sectional mean from each asset at each time slice.

    Args:
        data: DataArray with an 'asset' dimension.

    Returns:
        DataArray of same shape, demeaned across assets (skipna mean).

    NaN: mean computed skipna; NaN assets remain NaN after demeaning.
    JAX: No.
    """
    def _demean(da: xr.DataArray) -> xr.DataArray:
        return da - da.mean(dim='asset', skipna=True)

    return _dispatch_cross_sectional(_demean, data, func_name='cs_demean')


# ---------------------------------------------------------------------------
# Comparison helpers & conditional
# These return boolean-like DataArrays for use inside `where`.
# ---------------------------------------------------------------------------

@register_function(category="conditional")
def gt(a, b) -> xr.DataArray:
    """Element-wise greater-than comparison (a > b).

    NaN compared to anything returns False (numpy/xarray default).
    """
    return a > b


@register_function(category="conditional")
def lt(a, b) -> xr.DataArray:
    """Element-wise less-than comparison (a < b)."""
    return a < b


@register_function(category="conditional")
def ge(a, b) -> xr.DataArray:
    """Element-wise greater-than-or-equal comparison (a >= b)."""
    return a >= b


@register_function(category="conditional")
def le(a, b) -> xr.DataArray:
    """Element-wise less-than-or-equal comparison (a <= b)."""
    return a <= b


@register_function(category="conditional")
def eq(a, b) -> xr.DataArray:
    """Element-wise equality comparison (a == b).

    Note: use for integer-like comparisons (e.g. exchcd == 1). Float equality
    is unreliable for computed values.
    """
    return a == b


@register_function(category="conditional")
def nan_const() -> float:
    """Return NaN as a scalar constant for use in formula expressions.

    Useful with `where`: ``where(gt($x, 0), $x, nan_const())``.
    """
    return np.nan


@register_function(category="conditional")
def where(condition, true_val, false_val):
    """Element-wise conditional selection.

    Returns ``true_val`` where ``condition`` is truthy, ``false_val``
    elsewhere. Wraps ``xr.where`` with full broadcasting support.

    Args:
        condition:  Boolean-like DataArray (e.g. result of gt/lt/eq).
        true_val:   DataArray or scalar used where condition is True.
        false_val:  DataArray or scalar used where condition is False.

    Returns:
        DataArray with selected values.

    NaN: NaN in condition treated as False by xr.where.
    JAX: No.
    """
    return xr.where(condition, true_val, false_val)


# ---------------------------------------------------------------------------
# Bucket assignment
# ---------------------------------------------------------------------------

@register_function(category="cross_sectional")
def assign_bucket(data: xr.DataArray, bp1, bp2=None, bp3=None) -> xr.DataArray:
    """Assign each asset to a bin based on pre-computed breakpoint values.

    Each breakpoint is a threshold (scalar or DataArray without 'asset' dim).
    Assets are assigned to bin 0 if below all breakpoints, bin 1 if above
    the first breakpoint, bin 2 if above the second, etc.

    Supports 1–3 breakpoints (2–4 bins). For Fama-French:
    - Size (2-bin):  ``assign_bucket($me, cs_quantile($me_nyse, 0.5))``
    - B/M  (3-bin):  ``assign_bucket($bm, cs_quantile($bm_nyse, 0.3),
                                         cs_quantile($bm_nyse, 0.7))``

    Breakpoints from ``cs_quantile`` have no 'asset' dim; xarray broadcasts
    them across assets automatically.

    Args:
        data: DataArray with an 'asset' dimension.
        bp1:  First breakpoint (scalar or DataArray without 'asset' dim).
        bp2:  Optional second breakpoint.
        bp3:  Optional third breakpoint.

    Returns:
        DataArray of same shape with integer bin labels (float64 dtype to
        accommodate NaN). NaN data → NaN bucket.

    NaN: NaN data produces NaN bucket; no silent zeroing.
    JAX: No.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"assign_bucket expects xr.DataArray, got {type(data)}")

    breakpoints = [bp1]
    if bp2 is not None:
        breakpoints.append(bp2)
    if bp3 is not None:
        breakpoints.append(bp3)

    # Start with all-zero bucket counter
    result = xr.zeros_like(data, dtype=np.float64)
    for bp in breakpoints:
        result = result + xr.where(data > bp, 1.0, 0.0)

    # Propagate NaN from input
    return xr.where(data.isnull(), np.nan, result)


@register_function(category="cross_sectional", description="Month coordinate as DataArray")
def month_coord(data):
    """Return the month coordinate broadcast to the shape of data.

    Useful for calendar masks inside formulas, for example:
        where(eq(month_coord($me), 6), $me, nan_const())
    """
    if isinstance(data, xr.Dataset):
        for var_name, da in data.data_vars.items():
            if np.issubdtype(da.dtype, np.number):
                data = da
                break
        else:
            raise ValueError("month_coord requires a dataset with at least one numeric variable")

    if not isinstance(data, xr.DataArray):
        raise TypeError(f"month_coord expects xr.Dataset or xr.DataArray, got {type(data)}")
    if "month" not in data.coords:
        raise ValueError("month_coord requires a 'month' coordinate")

    return data.coords["month"] * xr.ones_like(data)


def register_built_in_functions():
    """Register built-in functions for common operations."""
    # Most functions are registered via decorators above,
    # but we can also register existing functions directly
    pass

# Register built-in functions when the module is imported
register_built_in_functions()