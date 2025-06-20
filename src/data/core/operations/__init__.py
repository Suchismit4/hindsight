# src/data/core/operations/__init__.py

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Union, Dict, List, Optional, Tuple, Any, Callable
import functools
from functools import partial

class TimeSeriesOps(eqx.Module):
    """
    Core operations for multi-dimensional panel data processing.
    Handles arrays with Time x Assets x Characteristics structure.
    """
        
    @staticmethod
    @partial(jax.jit, static_argnames=['periods'])
    def shift(
        data: jnp.ndarray, 
        indices: jnp.ndarray,
        periods: int
    ) -> jnp.ndarray:
        """
        Shift data along the time dimension respecting business days.
        
        Args:
            data: Input data with shape (T, ...)
            indices: Indices mapping mask -> valid business days with shape (T,)
            periods: Number of periods to shift (positive = forward, negative = backward)
            
        Returns:
            Shifted data with the same shape as input data
        """
        # Create result array filled with NaNs
        result = jnp.full_like(data, jnp.nan)
        
        # Return early if no shift is requested
        if periods == 0:
            return data.copy()
        
        # Find positions of valid business days
        valid_positions = indices
        
        # Return early if no valid positions or if shift is too large
        n_valid = len(valid_positions)
        if n_valid == 0 or abs(periods) >= n_valid:
            return result
        
        # Compute source and target indices for the shift
        if periods > 0:
            # For positive shift: take data[:-periods] and put it at positions[periods:]
            src_indices = jnp.arange(n_valid - periods)
            tgt_indices = jnp.arange(periods, n_valid)
        else:  # periods < 0
            # For negative shift: take data[-periods:] and put it at positions[:periods]
            src_indices = jnp.arange(-periods, n_valid)
            tgt_indices = jnp.arange(n_valid + periods)
        
        # Calculate actual positions to copy from/to
        src_positions = valid_positions[src_indices]
        tgt_positions = valid_positions[tgt_indices]
        
        # Get the data to be shifted
        src_data = jnp.take(data, src_positions, axis=0)
        
        # Copy data from source to target positions in one operation
        return result.at[tgt_positions].set(src_data)
    
    @staticmethod
    @partial(jax.jit, static_argnames=['window_size', 'func', 'overlap_factor'])
    def u_roll(
        data: jnp.ndarray,
        window_size: int,
        func: Callable[
            [int, Any, jnp.ndarray, int],
            Tuple[jnp.ndarray, Any]
        ],
        overlap_factor: float = None,
        **func_kwargs
    ) -> jnp.ndarray:
        """
        Applies a function over rolling windows along the 'time' dimension using block processing for parallelization.

        Args:
            window_size (int): Size of the rolling window.
            func (Callable[[int, Any, jnp.ndarray, int], Tuple[jnp.ndarray, Any]]):
                Function to apply over the rolling window. Should accept an index, the carry, the block, and window size,
                and return (value, new_carry).
            overlap_factor (float, optional): Factor determining the overlap between blocks.
            **func_kwargs: Additional keyword arguments to pass to the function.

        Returns:
            jnp.ndarray: Data array computed with the u_roll method.
        """
        
        # Set NaN values to zero to avoid issues with NaN in the data
        # data = jnp.nan_to_num(data)
        
        # Helper method to prepare blocks of data for u_roll
        def _prepare_blocks(
            data: jnp.ndarray,
            window_size: int,
            overlap_factor: float = None,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            Prepares overlapping blocks of data for efficient parallel processing.

            Args:
                window_size (int): Size of the rolling window.
                overlap_factor (float, optional): Factor determining the overlap between blocks.

            Returns:
                Tuple[jnp.ndarray, jnp.ndarray]: Padded data and block indices for slicing.
            """ 
            data = jnp.nan_to_num(data)

            num_time_steps = data.shape[0]
            other_dims = data.shape[1:]

            max_windows = num_time_steps - window_size + 1

            # If the overlap_factor (k) is not provided, default it to the ratio of max windows to window size
            if overlap_factor is None:
                overlap_factor = max_windows / window_size
                # Cap the overlap factor to maintain reasonable block sizes for parallelism
                # This prevents huge blocks when we have lots of data relative to window size
                max_reasonable_overlap = 20  # Allows blocks up to 20x window size
                overlap_factor = min(overlap_factor, max_reasonable_overlap)

            # Compute the effective block size (kw) based on the overlap factor and window size
            # This tells us how many time steps each block will span, including overlap
            # Since overlap_factor is now static, we can use regular Python operations
            import math
            block_size = math.ceil(overlap_factor * window_size)

            if block_size > max_windows:
                raise ValueError("Requested block size is larger than available data.")

            # Calculate the padding required to ensure that the data can be evenly divided into blocks
            padding_length = (block_size - max_windows % block_size) % block_size

            # Pad the data along the time dimension
            padding_shape = (padding_length,) + other_dims
            data_padded = jnp.concatenate(
                (data, jnp.zeros(padding_shape, dtype=data.dtype)), axis=0
            )

            # Total number of windows in the padded data
            total_windows = num_time_steps - window_size + 1

            # Starting indices for blocks
            block_starts = jnp.arange(0, total_windows, block_size)

            # Generate indices to slice the data into blocks
            block_indices = block_starts[:, None] + jnp.arange(window_size - 1 + block_size)[None, :]

            return data_padded, block_indices
        
        # Prepare blocks of data for processing
        blocks, block_indices = _prepare_blocks(data, window_size, overlap_factor)

        other_dims = data.shape[1:]
        num_time_steps = data.shape[0]

        # Function to apply over each block
        def process_block(
            block: jnp.ndarray,
            func: Callable[
                [int, Any, jnp.ndarray, int],
                Tuple[jnp.ndarray, Any]
            ],
        ) -> jnp.ndarray:
            t, n, j = block.shape

            values = jnp.zeros((t - window_size + 1, n, j), dtype=jnp.float32)

            # Initialize carry with the func (i == -1 case)
            initial_value, carry = func(-1, None, block, window_size, **func_kwargs)

            # Set the initial value in the values array
            values = values.at[0].set(initial_value)

            # Apply the step function iteratively
            def step_wrapper(i: int, state):
                values, carry = state
                new_value, new_carry = func(i, carry, block, window_size, **func_kwargs)
                idx = i - window_size + 1
                values = values.at[idx].set(new_value)
                return (values, new_carry)

            # Apply step_wrapper over the time dimension
            values, carry = jax.lax.fori_loop(
                window_size, t, step_wrapper, (values, carry)
            )

            return values

        # Vectorize over blocks
        blocks_results = jax.vmap(process_block, in_axes=(0, None))(blocks[block_indices], func)

        # Reshape the results to match the time dimension
        blocks_results = blocks_results.reshape(-1, *other_dims)

        # Concatenate the results along the time dimension
        # Back-pad the results to the original time dimension
        final = jnp.concatenate(
            (
                jnp.repeat(blocks_results[:1], window_size - 1, axis=0),
                blocks_results[: num_time_steps - window_size + 1],
            ),
            axis=0,
        )

        # Return a data array computed with the u_roll method
        return final
    
    
# Standard rolling functions

# TODO: Create a factory possibly?
from .standard import mean, mode, ema, median, gain, loss, sum_func, wma, triple_exponential_smoothing, adaptive_ema, rma