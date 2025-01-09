import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=['window_size'])
def mean(i: int, carry, block: jnp.ndarray, window_size: int):
    """
    Compute the rolling mean over a window efficiently using an incremental sum.

    Assumes block shape (T, N, 1).

    Args:
        i (int): Current index in the time series.
        carry: Previous sum over the window, shape (N, 1).
        block (jnp.ndarray): The data block, shape (T, N, 1).
        window_size (int): Size of the moving window.

    Returns:
        tuple: (current_mean, updated_sum) where:
            - current_mean has shape (N, 1)
            - updated_sum (carry) has shape (N, 1)
    """
    if carry is None:
        # Sum over the initial window along time axis.
        initial_sum = jnp.sum(block[:window_size], axis=0)  # shape (N, 1)
        current_mean = initial_sum / window_size
        return current_mean, initial_sum

    # Update the rolling sum: subtract element leaving the window, add new element.
    new_sum = carry - block[i - window_size] + block[i]  # shape (N, 1)
    current_mean = new_sum / window_size
    return current_mean, new_sum

@partial(jax.jit, static_argnames=['window_size'])
def sum_func(i: int, carry, block: jnp.ndarray, window_size: int):
    """
    Compute the rolling sum over a window with shape (N, 1) output.

    Assumes block shape (T, N, 1).

    Args:
        i (int): Current index in the time series.
        carry: Previous sum over the window, shape (N, 1).
        block (jnp.ndarray): The data block, shape (T, N, 1).
        window_size (int): Size of the moving window.

    Returns:
        tuple: (current_sum, updated_sum) where both have shape (N, 1).
    """
    if carry is None:
        initial_sum = jnp.sum(block[:window_size], axis=0)  # shape (N, 1)
        return initial_sum, initial_sum

    new_sum = carry - block[i - window_size] + block[i]  # shape (N, 1)
    return new_sum, new_sum

@partial(jax.jit, static_argnames=['window_size'])
def median(i: int, carry, block: jnp.ndarray, window_size: int):
    """
    Compute the median over a rolling window for data of shape (T, N, 1).

    Args:
        i (int): Current index in the time series.
        carry: Not used for median calculation.
        block (jnp.ndarray): The data block, shape (T, N, 1).
        window_size (int): Size of the moving window.

    Returns:
        tuple: (current_median, None) where current_median has shape (N, 1).
    """
    # Extract the current window of data along time axis.
    window = jax.lax.dynamic_slice(
            block,
            start_indices=(i - window_size + 1, 0, 0),
            slice_sizes=(window_size, block.shape[1], 1)
        )    
    
    # Compute the median along the time dimension (axis 0).
    current_median = jnp.median(window, axis=0)  # shape (N, 1)
    return current_median, None

@partial(jax.jit, static_argnames=['window_size'])
def mode(i: int, carry, block: jnp.ndarray, window_size: int):
    """
    Compute the mode over a rolling window for data of shape (T, N, 1)
    using an approach inspired by SciPy, leveraging jnp.unique for efficiency.
    
    Args:
        i (int): Current index in the time series.
        carry: Not used for mode calculation.
        block (jnp.ndarray): The data block, shape (T, N, 1).
        window_size (int): Size of the moving window.
        
    Returns:
        tuple: (current_mode, None) where current_mode has shape (N, 1).
    """
    # Extract the current rolling window; shape: (window_size, N, 1)
    window = jax.lax.dynamic_slice(
        block,
        start_indices=(i - window_size + 1, 0, 0),
        slice_sizes=(window_size, block.shape[1], 1)
    )

    # Remove the singleton last dimension for ease of computation; shape becomes (window_size, N)
    window_2d = window[..., 0]

    def compute_mode_1d(series):
        """
        Compute the mode for a 1D array using jnp.unique.
        
        Args:
            series (jnp.ndarray): 1D array of shape (window_size,).
        
        Returns:
            The mode of the series.
        """
        # Get unique values and their counts.
        vals, cnts = jnp.unique(series, size=window_size, return_counts=True)
        # Find index of the most frequent element.
        idx = jnp.argmax(cnts)
        # Return the value corresponding to the highest count.
        return vals[idx]

    # Transpose to shape (N, window_size) to apply vmap over each stock/time series.
    # Each row corresponds to a series over the current window.
    series_data = window_2d.T  # shape (N, window_size)

    # Vectorize the mode computation across all series in parallel.
    modes = jax.vmap(compute_mode_1d)(series_data)  # shape (N,)

    # Expand last dimension to match output shape (N, 1)
    current_mode = modes[:, None]
    return current_mode, None

# Example for extended standard operations:

# Define a function to compute Exponential Moving Average (EMA)
# This function will be used with the u_roll method for efficient computation
@partial(jax.jit, static_argnames=['window_size'])
def ema(i: int, carry, block: jnp.ndarray, window_size: int):
    """
    Compute the Exponential Moving Average (EMA) for a given window.
    
    This function is designed to work with JAX's JIT compilation and
    the u_roll method defined in the Tensor class. It computes the EMA
    efficiently over a rolling window of data.
    
    Args:
    i (int): Current index in the time series
    state (tuple): Contains current values, carry (previous EMA), and data block
    window_size (int): Size of the moving window
    
    Returns:
    tuple: Updated state (new EMA value, carry, and data block)
    """
    
    # Initialize the first value
    if carry is None:
        # Compute the sum of the first window
        current_window_sum = block[:window_size].reshape(-1, 
                                                            block.shape[1], 
                                                            block.shape[2]).sum(axis=0)
    
        
        return (current_window_sum * (1/window_size), current_window_sum * (1/window_size))
    
    # Get the current price
    current_price = block[i]
    
    # Compute the new EMA
    # EMA = α * current_price + (1 - α) * previous_EMA
    # where α = 1 / (window_size)
    alpha = 1 / window_size
    
    new_ema = alpha * current_price + (1 - alpha) * carry
    
    return (new_ema, new_ema)