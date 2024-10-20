# data_layer/tensor.py

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, List, Union, Dict, Callable, Any
from abc import ABC, abstractmethod
import math
from .coords import Coordinates

class Tensor(eqx.Module, ABC):
    """
    Abstract base class representing a generalized tensor structure.
    Inherits from Equinox's Module to ensure compatibility with JAX's pytree system.

    Attributes:
        data (jnp.ndarray):
            The underlying data stored as a JAX array.
            This array holds the numerical values of the tensor.
        dimensions (Tuple[str, ...]):
            Names of the dimensions of the tensor (e.g., 'time', 'asset', 'feature').
        feature_names (Tuple[str, ...]):
            Names of the features along the feature dimension.
            These names help in identifying and accessing specific features.
        Coordinates (Coordinates):
            Coordinate variables associated with the tensor dimensions.
            Ensures that each dimension has corresponding coordinate data.
    """

    data: jnp.ndarray
    dimensions: Tuple[str, ...]
    feature_names: Tuple[str, ...]
    Coordinates: Coordinates

    # Internal mappings for quick index retrieval by name
    # Not exposed in representations to maintain encapsulation
    _dimension_map: Dict[str, int] = eqx.field(init=False, repr=False)
    _feature_map: Dict[str, int] = eqx.field(init=False, repr=False)

    def __post_init__(self):
        """
        Validates the tensor's structure and initializes internal mappings.

        Raises:
            TypeError: If 'data' is not a JAX array.
            ValueError: If the data dimensions do not match the number of dimension names.
            ValueError: If tensor dimensions and coordinate keys do not match.
        """
        # Ensure 'data' is a JAX array for compatibility
        if not isinstance(self.data, jnp.ndarray):
            raise TypeError("Data must be a JAX array (jnp.ndarray).")

        # Ensure data dimensions match the number of dimension names
        if self.data.ndim != len(self.dimensions):
            raise ValueError(f"Data array has {self.data.ndim} dimensions, but {len(self.dimensions)} dimension names were provided.")

        # Ensure that all dimensions have corresponding coordinates
        dimensions_set = set(self.dimensions)
        coordinates_set = set(self.Coordinates.variables.keys())

        if dimensions_set != coordinates_set:
            missing_in_coords = dimensions_set - coordinates_set
            missing_in_dims = coordinates_set - dimensions_set
            error_message = "Tensor dimensions and coordinate keys do not match.\n"
            if missing_in_coords:
                error_message += f"Dimensions missing in Coordinates: {missing_in_coords}\n"
            if missing_in_dims:
                error_message += f"Coordinates missing in dimensions: {missing_in_dims}\n"
            error_message += f"Tensor dimensions: {self.dimensions}\n"
            error_message += f"Coordinate keys: {list(self.Coordinates.variables.keys())}"
            raise ValueError(error_message)

        # Create a mapping from dimension names to their indices for quick access
        object.__setattr__(self, '_dimension_map', {dim: idx for idx, dim in enumerate(self.dimensions)})

        # Create a mapping from feature names to their indices
        object.__setattr__(self, '_feature_map', {feature: idx for idx, feature in enumerate(self.feature_names)})

    @abstractmethod
    def select(self, dimension_name: str, index: Union[int, slice, List[int]]) -> 'Tensor':
        """
        Abstract method to select a subset of the tensor along a specified dimension.
        Must be implemented by subclasses to define specific selection behavior.

        Args:
            dimension_name (str):
                Name of the dimension to slice (e.g., 'time', 'asset', 'feature').
            index (Union[int, slice, List[int]]):
                Index or slice to select along the specified dimension.

        Returns:
            Tensor:
                A new instance of the Tensor subclass with the selected data.
        """
        pass

    def to_device_jax_array(self, device: Union[str, jax.Device] = None) -> jnp.ndarray:
        """
        Transfers the tensor's data array to the specified device (e.g., CPU, GPU).

        Args:
            device (Union[str, jax.Device], optional):
                Device identifier (e.g., 'cpu', 'gpu').
                If None, uses the default device.

        Returns:
            jnp.ndarray: JAX array on the specified device.
        """
        if device:
            return jax.device_put(self.data, device=device)
        else:
            return self.data  # Default device (usually CPU)

    def get_dimension_index(self, dimension_name: str) -> int:
        """
        Retrieves the index of the specified dimension within the tensor's data array.

        Args:
            dimension_name (str): Name of the dimension.

        Returns:
            int: Index of the dimension in the data array's shape.

        Raises:
            ValueError: If the dimension name is not found.
        """
        try:
            return self._dimension_map[dimension_name]
        except KeyError:
            raise ValueError(f"Dimension '{dimension_name}' not found in tensor dimensions.")

    def get_feature_index(self, feature_name: str) -> int:
        """
        Retrieves the index of the specified feature within the tensor's data array.

        Args:
            feature_name (str): Name of the feature.

        Returns:
            int: Index of the feature in the data array's feature dimension.

        Raises:
            ValueError: If the feature name is not found.
        """
        try:
            return self._feature_map[feature_name]
        except KeyError:
            raise ValueError(f"Feature '{feature_name}' not found in tensor features.")

    def __repr__(self):
        """
        Returns a string representation of the Tensor, including its shape, dimensions, feature names, and coordinates.

        Returns:
            str: String representation of the Tensor.
        """
        return (f"{self.__class__.__name__}(shape={self.data.shape}, "
                f"dimensions={self.dimensions}, "
                f"features={self.feature_names}, "
                f"Coordinates={self.Coordinates})")
        
    def _create_new_instance(self, data):
        """
        Creates a new instance of the same Tensor subclass with updated data.

        Args:
            data (jnp.ndarray): The data array for the new instance.

        Returns:
            Tensor: New instance of the Tensor subclass.
        """
        return self.__class__(data=data, dimensions=self.dimensions,
                              feature_names=self.feature_names, Coordinates=self.Coordinates)

    # Arithmetic operations
    def __sub__(self, other):
        """
        Subtracts another tensor or scalar from this tensor.

        Args:
            other (Union[Tensor, float, int]): The tensor or scalar to subtract.

        Returns:
            Tensor: A new tensor with the result of the subtraction.
        """
        if isinstance(other, Tensor):
            new_data = self.data - other.data
        else:
            new_data = self.data - other
        return self._create_new_instance(data=new_data)

    def __truediv__(self, other):
        """
        Divides this tensor by another tensor or scalar.

        Args:
            other (Union[Tensor, float, int]): The tensor or scalar to divide by.

        Returns:
            Tensor: A new tensor with the result of the division.
        """
        if isinstance(other, Tensor):
            new_data = self.data / other.data
        else:
            new_data = self.data / other
        return self._create_new_instance(data=new_data)

    def _set_nan_to_zero(self):
        """
        Sets NaN values in the data array to zero.
        """
        self.data = jnp.nan_to_num(self.data)

    # u_roll method
    def u_roll(
        self,
        window_size: int,
        func: Callable[[int, Tuple[jnp.ndarray, Any, jnp.ndarray]], Tuple[jnp.ndarray, Any, jnp.ndarray]],
        overlap_factor: float = None
    ) -> 'Tensor':
        """
        Applies a function over rolling windows along the 'time' dimension using block processing for parallelization.

        Args:
            window_size (int): Size of the rolling window.
            func (Callable[[int, Tuple[jnp.ndarray, Any, jnp.ndarray]], Tuple[jnp.ndarray, Any, jnp.ndarray]]):
                Function to apply over the rolling window. Should accept an index and a state tuple,
                and return (values, carry, block). This function is responsible for initialization when i == 0.
            overlap_factor (float, optional): Factor determining the overlap between blocks.

        Returns:
            Tensor: New tensor with the function applied over the time dimension.
        """
        
        # Set NaN values to zero to avoid issues with NaN in the data
        self._set_nan_to_zero()
        
        # Prepare blocks of data for processing
        blocks, block_indices = self._prepare_blocks(window_size, overlap_factor)
        
        other_dims = self.data.shape[1:]
        num_time_steps = self.data.shape[0]

        # Function to apply over each block
        @eqx.filter_jit
        def process_block(block: jnp.ndarray, func: Callable):
            t, n = block.shape
            
            values = jnp.zeros((t + window_size - 1, n), dtype=block.data.dtype)
            
            # Initialize state with the func (i == -1 case)
            state = func(-1, (values, None, block))
            
            # Apply the step function iteratively
            def step_wrapper(i, state):
                values, carry, block = state
                new_values, new_carry, _ = func(i, state)
                values = values.at[i - window_size + 1].set(new_values)
                return (values, new_carry, block)
            
            values, *_ = jax.lax.fori_loop(window_size, t, step_wrapper, state)
            
            return values
        
        # Vectorize over blocks
        blocks_results = jax.vmap(process_block, in_axes=(0, None))(blocks[block_indices], func)
        
        # Reshape the results to match the time dimension
        blocks_results = blocks_results.reshape(-1, *other_dims)
    
        # Concatenate the results along the time dimension
        # Back-pad the results to the original time dimension
        final = jnp.concatenate((jnp.repeat(blocks_results[:1], window_size - 1, axis=0),
                                 blocks_results[:num_time_steps - window_size + 1]), axis=0)

        # Return a new Tensor with the computed data
        return self._create_new_instance(data=final)

    # Helper method to prepare blocks of data for u_roll
    def _prepare_blocks(
        self,
        window_size: int,
        overlap_factor: float = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Prepares overlapping blocks of data for efficient parallel processing.

        Args:
            window_size (int): Size of the rolling window.
            overlap_factor (float, optional): Factor determining the overlap between blocks.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Padded data and block indices for slicing.
        """
        data = self.data
        num_time_steps = data.shape[0]
        other_dims = data.shape[1:]

        max_windows = num_time_steps - window_size + 1

        # If the overlap_factor (k) is not provided, we default it to the ratio of max windows to window size
        if overlap_factor is None:
            overlap_factor = max_windows / window_size

        # Compute the effective block size (kw) based on the overlap factor and window size
        # This tells us how many time steps each block will span, including overlap
        block_size = math.ceil(overlap_factor * window_size)

        if block_size > max_windows:
            raise ValueError('Requested block size is larger than available data.')

        # Calculate the padding required to ensure that the data can be evenly divided into blocks
        padding_length = (block_size - max_windows % block_size) % block_size

        # Pad the data along the time dimension
        padding_shape = (padding_length,) + other_dims
        data_padded = jnp.concatenate(
            (data, jnp.zeros(padding_shape, dtype=data.dtype)),
            axis=0
        )

        # Total number of time steps after padding
        num_padded_time_steps = data_padded.shape[0]
        # Total number of windows in the padded data
        total_windows = num_padded_time_steps - window_size + 1

        # Starting indices for blocks
        block_starts = jnp.arange(0, total_windows, block_size)

        # Generate indices to slice the data into blocks
        block_indices = block_starts[:, None] + jnp.arange(block_size + window_size - 1)[None, :]
        
        return data_padded, block_indices

class ReturnsTensor(Tensor):
    """
    Specialized Tensor for handling return-related data.
    Ensures consistency by enforcing fixed dimensions and feature names.

    Attributes:
        data (jnp.ndarray):
            Data array with shape corresponding to (time, asset, feature).
            Specifically tailored for return data with a single feature.
        Coordinates (Coordinates):
            Coordinate variables associated with the tensor dimensions.
    """
    data: jnp.ndarray
    Coordinates: Coordinates

    # Fixed dimensions and feature names are set internally and are not exposed to the user
    dimensions: Tuple[str, ...] = eqx.field(default=('time', 'asset', 'feature'), init=False)
    feature_names: Tuple[str, ...] = eqx.field(default=('return',), init=False)

    def __post_init__(self):
        """
        Validates that the tensor conforms to the expected structure for return data.
        Ensures that only one feature named 'return' exists.

        Raises:
            ValueError:
                - If 'feature_names' does not contain exactly one feature named 'return'.
                - If the data array does not have exactly one feature dimension.
        """
        # Enforce that there is exactly one feature named 'return'
        if self.feature_names != ('return',):
            raise ValueError("ReturnsTensor must have exactly one feature named 'return'.")

        # Ensure that the data array has one feature dimension
        if self.data.shape[-1] != 1:
            raise ValueError("ReturnsTensor data must have exactly one feature dimension.")

        # Initialize parent class mappings for dimensions and features
        super().__post_init__()

    def select(self, dimension_name: str, index: Union[int, slice, List[int]]) -> 'ReturnsTensor':
        """
        Selects a slice of the ReturnsTensor along the specified dimension.
        Returns a new ReturnsTensor instance with the sliced data.

        Args:
            dimension_name (str):
                Name of the dimension to slice (e.g., 'time', 'asset', 'feature').
            index (Union[int, slice, List[int]]):
                Index or slice to select along the specified dimension.

        Returns:
            ReturnsTensor:
                A new ReturnsTensor instance with the selected data.

        Raises:
            IndexError: If attempting to slice the 'feature' dimension in an unsupported way.
            ValueError: If slicing the 'feature' dimension improperly.
        """
        # Create a slicer list initialized to select all elements in each dimension
        slicer = [slice(None)] * self.data.ndim

        # Retrieve the index of the dimension to slice
        dim_idx = self.get_dimension_index(dimension_name)

        # Ensure that we use slicing that retains dimensions
        if isinstance(index, int):
            # Convert integer index to a slice that retains the dimension
            index = slice(index, index+1)
        elif isinstance(index, list):
            if len(index) == 1:
                # Convert single-element list to slice
                index = slice(index[0], index[0]+1)
            else:
                # Convert list to an array for advanced indexing
                index = jnp.array(index)

        # Update the slicer for the specified dimension with the provided index
        slicer[dim_idx] = index

        # Perform the slicing operation on the data array
        sliced_data = self.data[tuple(slicer)]

        # Dimensions remain the same
        new_dimensions = self.dimensions

        # Coordinates may need to be updated to reflect the selected indices
        sliced_Coordinates = self.Coordinates  # For simplicity, we're keeping Coordinates unchanged

        # Return a new ReturnsTensor instance with the sliced data and existing Coordinates
        return ReturnsTensor(
            data=sliced_data,
            Coordinates=sliced_Coordinates
        )

    # Override _create_new_instance
    def _create_new_instance(self, data):
        """
        Creates a new ReturnsTensor instance with the updated data.

        Args:
            data (jnp.ndarray): The data array for the new instance.

        Returns:
            ReturnsTensor: New instance of ReturnsTensor.
        """
        return self.__class__(data=data, Coordinates=self.Coordinates)

class CharacteristicsTensor(Tensor):
    """
    Specialized Tensor for handling characteristic-related data.
    Allows flexible dimensions and feature names as needed.

    Attributes:
        data (jnp.ndarray):
            Data array with dimensions corresponding to (time, asset, feature).
            Can hold multiple characteristics as features.
        dimensions (Tuple[str, ...]):
            Names of the dimensions (e.g., 'time', 'asset', 'feature').
        feature_names (Tuple[str, ...]):
            Names of the characteristics, corresponding to features.
        Coordinates (Coordinates):
            Coordinate variables associated with the tensor dimensions.
    """
    data: jnp.ndarray
    dimensions: Tuple[str, ...]
    feature_names: Tuple[str, ...]
    Coordinates: Coordinates

    def __post_init__(self):
        """
        Initializes the CharacteristicsTensor by calling the parent __post_init__ method.
        """
        super().__post_init__()

    def select(self, dimension_name: str, index: Union[int, slice, List[int]]) -> 'CharacteristicsTensor':
        """
        Selects a slice of the CharacteristicsTensor along the specified dimension.
        Returns a new CharacteristicsTensor instance with the sliced data and updated feature names if necessary.

        Args:
            dimension_name (str):
                Name of the dimension to slice (e.g., 'time', 'asset', 'feature').
            index (Union[int, slice, List[int]]):
                Index or slice to select along the specified dimension.

        Returns:
            CharacteristicsTensor:
                A new CharacteristicsTensor instance with the selected data and updated feature names.

        Raises:
            TypeError: If an unsupported index type is used for the 'feature' dimension.
        """
        # Create a slicer list initialized to select all elements in each dimension
        slicer = [slice(None)] * self.data.ndim

        # Retrieve the index of the dimension to slice
        dim_idx = self.get_dimension_index(dimension_name)

        # Update the slicer for the specified dimension with the provided index
        slicer[dim_idx] = index

        # Perform the slicing operation on the data array
        sliced_data = self.data[tuple(slicer)]

        # Initialize new feature names as the current feature names
        new_feature_names = self.feature_names

        # If slicing the 'feature' dimension, update the feature names accordingly
        if dimension_name == 'feature':
            if isinstance(index, int):
                # Selecting a single feature by index
                new_feature_names = (self.feature_names[index],)
            elif isinstance(index, slice):
                # Selecting a range of features
                new_feature_names = self.feature_names[index]
            elif isinstance(index, list):
                # Selecting multiple specific features by index
                new_feature_names = tuple(self.feature_names[i] for i in index)
            else:
                # Unsupported index type for 'feature' dimension
                raise TypeError("Unsupported index type for feature dimension.")

        # Coordinates remain unchanged when slicing dimensions other than 'feature'
        sliced_Coordinates = self.Coordinates  # For simplicity, keeping Coordinates unchanged

        # Return a new CharacteristicsTensor instance with sliced data and updated feature names
        return CharacteristicsTensor(
            data=sliced_data,
            dimensions=self.dimensions,
            feature_names=new_feature_names,
            Coordinates=sliced_Coordinates
        )


from . import tensor_ops        

