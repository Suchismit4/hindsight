# src/data/core/computations.py

import numpy as np
import xarray as xr
import xarray_jax as xj
import jax.numpy as jnp
import equinox as eqx
from typing import Union, Dict, List, Optional, Tuple, Any, Callable
import functools

from ..core.operations import TimeSeriesOps

class Rolling:
    """
    Custom Rolling class to apply rolling window operations using JAX.
    """
    
    def __init__(self, obj: Union[xr.DataArray, xr.Dataset], dim: str, window: int):
        """
        Initializes the Rolling object.

        Args:
            obj (Union[xr.DataArray, xr.Dataset]): The xarray object to apply rolling on.
            dim (str): The dimension over which to apply the rolling window.
            window (int): The size of the rolling window.
        """
        self.obj = obj
        self.dim = dim
        self.window = window

    def reduce(
        self, 
        func: Callable[[int, Any, jnp.ndarray, int], Tuple[jnp.ndarray, Any]], 
        **kwargs
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Applies the rolling function using the provided callable.

        Args:
            func (Callable[[int, Any, jnp.ndarray, int], Tuple[jnp.ndarray, Any]]): 
                The function to apply over each rolling window.
            **kwargs: Additional keyword arguments for the function.

        Returns:
            Union[xr.DataArray, xr.Dataset]: The resulting xarray object after applying the rolling function.
        """
        # Ensure that the specified dimension exists
        if self.dim not in self.obj.dims:
            raise ValueError(f"Dimension '{self.dim}' not found in the xarray object.")
        
        # Extract the axis corresponding to the dimension
        axis = self.obj.get_axis_num(self.dim)
        
        # Extract the data as a JAX array
        if isinstance(self.obj, xr.DataArray):
            data = self.obj.data 
        elif isinstance(self.obj, xr.Dataset):
            # Handle Dataset by applying rolling to each DataArray within
            rolled_data = {}
            for var in self.obj.data_vars:
                rolled = self.reduce(
                    self.obj[var], 
                    dim=self.dim, 
                    window=self.window, 
                    func=func, 
                    **kwargs
                )
                rolled_data[var] = rolled
            return xr.Dataset(rolled_data, coords=self.obj.coords)
        else:
            raise TypeError("Unsupported xarray object type.")

        # Apply the u_roll method from TimeSeriesOps
        rolled_array = TimeSeriesOps.u_roll(
            data=data,
            window_size=self.window,
            func=func,
            overlap_factor=kwargs.get('overlap_factor', None)
        )
        
        # Create a new xarray object with the rolled data
        rolled_obj = self.obj.copy(data=rolled_array)
        
        return rolled_obj
