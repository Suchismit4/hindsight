# data_layer/tensor_ops.py

from .tensor import Tensor
import jax.numpy as jnp
import jax

def _register_tensor_method(func):
    """
    Decorator to register a function as a method of the Tensor class.

    Args:
        func (Callable): Function to register as a method.

    Returns:
        Callable: The original function.
    """
    setattr(Tensor, func.__name__, func)
    return func

@_register_tensor_method
def mean(self, dimension_name: str) -> 'Tensor':
    """
    Computes the mean over the specified dimension.

    Args:
        dimension_name (str): Name of the dimension to compute the mean over.

    Returns:
        Tensor: New Tensor instance with mean computed over the specified dimension.
    """
    # Get the index of the specified dimension
    dim_idx = self.get_dimension_index(dimension_name)
    # Compute the mean over the specified dimension
    mean_data_jnp = jnp.mean(self.data, axis=dim_idx, keepdims=True)
    # Convert the result back to a NumPy array
    mean_data = jax.device_get(mean_data_jnp)
    # Return a new Tensor with the mean data
    return self._create_new_instance(data=mean_data)

@_register_tensor_method
def std(self, dimension_name: str) -> 'Tensor':
    """
    Computes the standard deviation over the specified dimension.

    Args:
        dimension_name (str): Name of the dimension to compute the std over.

    Returns:
        Tensor: New Tensor instance with std computed over the specified dimension.
    """
    # Get the index of the specified dimension
    dim_idx = self.get_dimension_index(dimension_name)
    # Compute the standard deviation over the specified dimension
    std_data_jnp = jnp.std(self.data, axis=dim_idx, keepdims=True)
    # Convert the result back to a NumPy array
    std_data = jax.device_get(std_data_jnp)
    # Return a new Tensor with the std data
    return self._create_new_instance(data=std_data)

@_register_tensor_method
def sum(self, dimension_name: str) -> 'Tensor':
    """
    Computes the sum over the specified dimension.

    Args:
        dimension_name (str): Name of the dimension to compute the sum over.

    Returns:
        Tensor: New Tensor instance with sum computed over the specified dimension.
    """
    # Get the index of the specified dimension
    dim_idx = self.get_dimension_index(dimension_name)
    # Compute the sum over the specified dimension
    sum_data_jnp = jnp.sum(self.data, axis=dim_idx, keepdims=True)
    # Convert the result back to a NumPy array
    sum_data = jax.device_get(sum_data_jnp)
    # Return a new Tensor with the sum data
    return self._create_new_instance(data=sum_data)

@_register_tensor_method
def max(self, dimension_name: str) -> 'Tensor':
    """
    Computes the maximum over the specified dimension.

    Args:
        dimension_name (str): Name of the dimension to compute the max over.

    Returns:
        Tensor: New Tensor instance with max computed over the specified dimension.
    """
    # Get the index of the specified dimension
    dim_idx = self.get_dimension_index(dimension_name)
    # Compute the maximum over the specified dimension
    max_data_jnp = jnp.max(self.data, axis=dim_idx, keepdims=True)
    # Convert the result back to a NumPy array
    max_data = jax.device_get(max_data_jnp)
    # Return a new Tensor with the max data
    return self._create_new_instance(data=max_data)

@_register_tensor_method
def min(self, dimension_name: str) -> 'Tensor':
    """
    Computes the minimum over the specified dimension.

    Args:
        dimension_name (str): Name of the dimension to compute the min over.

    Returns:
        Tensor: New Tensor instance with min computed over the specified dimension.
    """
    # Get the index of the specified dimension
    dim_idx = self.get_dimension_index(dimension_name)
    # Compute the minimum over the specified dimension
    min_data_jnp = jnp.min(self.data, axis=dim_idx, keepdims=True)
    # Convert the result back to a NumPy array
    min_data = jax.device_get(min_data_jnp)
    # Return a new Tensor with the min data
    return self._create_new_instance(data=min_data)
        
@_register_tensor_method
def variance(self, dimension_name: str) -> 'Tensor':
    """
    Computes the variance over the specified dimension.

    Args:
        dimension_name (str): Name of the dimension to compute the variance over.

    Returns:
        Tensor: New Tensor instance with variance computed over the specified dimension.
    """
    # Get the index of the specified dimension
    dim_idx = self.get_dimension_index(dimension_name)
    # Compute the variance over the specified dimension
    var_data_jnp = jnp.var(self.data, axis=dim_idx, keepdims=True)
    # Convert the result back to a NumPy array
    var_data = jax.device_get(var_data_jnp)
    # Return a new Tensor with the variance data
    return self._create_new_instance(data=var_data)

@_register_tensor_method
def median(self, dimension_name: str) -> 'Tensor':
    """
    Computes the median over the specified dimension.

    Args:
        dimension_name (str): Name of the dimension to compute the median over.

    Returns:
        Tensor: New Tensor instance with median computed over the specified dimension.
    """
    # Get the index of the specified dimension
    dim_idx = self.get_dimension_index(dimension_name)
    # Compute the median over the specified dimension
    median_data_jnp = jnp.median(self.data, axis=dim_idx, keepdims=True)
    # Convert the result back to a NumPy array
    median_data = jax.device_get(median_data_jnp)
    # Return a new Tensor with the median data
    return self._create_new_instance(data=median_data)

@_register_tensor_method
def prod(self, dimension_name: str) -> 'Tensor':
    """
    Computes the product over the specified dimension.

    Args:
        dimension_name (str): Name of the dimension to compute the product over.

    Returns:
        Tensor: New Tensor instance with product computed over the specified dimension.
    """
    # Get the index of the specified dimension
    dim_idx = self.get_dimension_index(dimension_name)
    # Compute the product over the specified dimension
    prod_data_jnp = jnp.prod(self.data, axis=dim_idx, keepdims=True)
    # Convert the result back to a NumPy array
    prod_data = jax.device_get(prod_data_jnp)
    # Return a new Tensor with the product data
    return self._create_new_instance(data=prod_data)
