"""
Weight generation functions for moving averages.

This module provides various weight generation functions that can be used
with weighted moving averages and other financial indicators.
"""

import numpy as np
import jax.numpy as jnp
from typing import Optional, Union


def linear_weights(window: int, **kwargs) -> jnp.ndarray:
    """
    Generate linearly increasing weights [1, 2, 3, ..., window].
    
    Args:
        window: Window size
        **kwargs: Additional parameters (ignored)
        
    Returns:
        JAX array of linearly increasing weights
    """
    return jnp.arange(1, window + 1, dtype=jnp.float32)


def exponential_weights(window: int, alpha: float = 0.1, **kwargs) -> jnp.ndarray:
    """
    Generate exponentially decaying weights.
    
    Args:
        window: Window size
        alpha: Decay parameter (higher = more recent weight)
        **kwargs: Additional parameters (ignored)
        
    Returns:
        JAX array of exponentially decaying weights
    """
    indices = jnp.arange(window, dtype=jnp.float32)
    weights = jnp.exp(-alpha * (window - 1 - indices))
    return weights


def alma_weights(window: int, offset: float = 0.85, sigma: int = 6, **kwargs) -> jnp.ndarray:
    """
    Generate Arnaud Legoux Moving Average (ALMA) weights.
    
    ALMA is a weighted moving average that uses a Gaussian kernel
    to assign weights, providing a good balance between responsiveness
    and smoothness.
    
    This implementation matches the pandas_ta ALMA algorithm where:
    - weights[0] corresponds to the most recent price
    - weights[1] corresponds to the second most recent price
    - etc.
    
    Args:
        window: Window size
        offset: Phase offset (0 to 1). 0.85 focuses on recent values (pandas_ta default),
                0.1 focuses on older values
        sigma: Smoothing parameter. Higher values make the weights
               more uniform across the window
        **kwargs: Additional parameters (ignored)
        
    Returns:
        JAX array of ALMA weights ordered from most recent to oldest
    """
    # Calculate the mean position based on offset
    m = offset * (window - 1)
    
    # Calculate the standard deviation
    s = window / sigma
    
    # Generate weights using Gaussian function
    # Note: indices go from 0 to window-1, where 0 represents most recent
    indices = jnp.arange(window, dtype=jnp.float32)
    weights = jnp.exp(-((indices - m) ** 2) / (2 * s ** 2))
    
    # Reverse the weights so that weights[0] corresponds to most recent price
    # This matches the pandas_ta convention where j=0 is most recent
    weights = weights[::-1]
    
    return weights

def fibonacci_weights(window: int,
                      asc: bool = True,
                      **kwargs) -> jnp.ndarray:
    """
    Generate a length-`window` vector of Fibonacci weights.

    - If asc=True, larger weights on more recent prices (weights increase towards end).
    - If asc=False, larger weights on older prices (weights decrease towards end).
    
    This matches pandas_ta behavior where weights are applied in chronological order
    (oldest to newest), so asc=True means the last elements get higher weights.
    """
    # Build Fibonacci sequence up to window terms:
    fib = [1, 1]
    for _ in range(2, window):
        fib.append(fib[-1] + fib[-2])
    w = jnp.array(fib[:window], dtype=jnp.float32)

    # Normalize:
    w = w / jnp.sum(w)

    # For asc=True: recent values (end of array) should get higher weights
    # For asc=False: older values (start of array) should get higher weights
    # pandas_ta applies weights in chronological order, so we don't reverse
    if not asc:
        w = w[::-1]

    return w

def triangular_weights(window: int, **kwargs) -> jnp.ndarray:
    """
    Generate triangular weights (ramp up to middle, then ramp down).
    
    Args:
        window: Window size
        **kwargs: Additional parameters (ignored)
        
    Returns:
        JAX array of triangular weights
    """
    if window == 1:
        return jnp.array([1.0])
    
    half = window // 2
    if window % 2 == 1:
        # Odd window size
        weights = jnp.concatenate([
            jnp.arange(1, half + 2, dtype=jnp.float32),
            jnp.arange(half, 0, -1, dtype=jnp.float32)
        ])
    else:
        # Even window size
        weights = jnp.concatenate([
            jnp.arange(1, half + 1, dtype=jnp.float32),
            jnp.arange(half, 0, -1, dtype=jnp.float32)
        ])
    
    return weights


def custom_weights(window: int, weights_array: Optional[Union[list, np.ndarray, jnp.ndarray]] = None, **kwargs) -> jnp.ndarray:
    """
    Use custom provided weights or fall back to linear weights.
    
    Args:
        window: Window size
        weights_array: Custom weight array. If None, uses linear weights
        **kwargs: Additional parameters (ignored)
        
    Returns:
        JAX array of weights
        
    Raises:
        ValueError: If weights_array length doesn't match window size
    """
    if weights_array is None:
        return linear_weights(window)
    
    weights_array = jnp.asarray(weights_array, dtype=jnp.float32)
    
    if len(weights_array) != window:
        raise ValueError(f"weights_array length ({len(weights_array)}) must match window size ({window})")
    
    return weights_array


# Registry of available weight functions
WEIGHT_GENERATORS = {
    'linear': linear_weights,
    'exponential': exponential_weights,
    'alma': alma_weights,
    'triangular': triangular_weights,
    'custom': custom_weights,
}


def get_weight_generator(name: str):
    """
    Get a weight generator function by name.
    
    Args:
        name: Name of the weight generator
        
    Returns:
        Weight generator function
        
    Raises:
        KeyError: If generator name is not found
    """
    if name not in WEIGHT_GENERATORS:
        available = ', '.join(WEIGHT_GENERATORS.keys())
        raise KeyError(f"Unknown weight generator '{name}'. Available generators: {available}")
    
    return WEIGHT_GENERATORS[name] 