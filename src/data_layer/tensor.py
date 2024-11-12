# data_layer/tensor.py

import jax

from jax import numpy as jnp
import equinox as eqx
from typing import (
    Tuple,
    List,
    Union,
    Dict,
    Callable,
    Any,
    TypeVar,
    Generic,
    Type,
)
from abc import ABC, abstractmethod
import math
from .coords import Coordinates
import numpy as np

# Define a TypeVar bound to Tensor for type-safe method returns
T = TypeVar('T', bound='Tensor')


class Tensor(eqx.Module, ABC, Generic[T]):
    """
    Abstract base class representing a generalized tensor structure.
    
    Core intuition:
        A Tensor is a multi-dimensional array of data, with a set of coordinates that define the
        indices of the array. The coordinates are stored in a Coordinates object, which is a
        subclass of Equinox's Module.

    Inherits from Equinox's Module to ensure compatibility with JAX's pytree system.

    Attributes:
        data (np.ndarray):
            The underlying data stored as a NumPy array.
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

    data: np.ndarray
    dimensions: Tuple[str, ...] = eqx.static_field()
    feature_names: Tuple[str, ...] = eqx.static_field()
    Coordinates: Coordinates

    # Internal mappings for quick index retrieval by name
    _dimension_map: Dict[str, int] = eqx.static_field()
    _feature_map: Dict[str, int] = eqx.static_field()

    def __post_init__(self):
        """
        Validates the tensor's structure and initializes internal mappings.

        Raises:
            TypeError: If 'data' is not a NumPy array.
            ValueError: If the data dimensions do not match the number of dimension names.
            ValueError: If tensor dimensions and coordinate keys do not match.
        """
        # Ensure 'data' is a NumPy array for compatibility
        if not isinstance(self.data, np.ndarray):
            raise TypeError("Data must be a NumPy array (np.ndarray).")

        # Ensure data dimensions match the number of dimension names
        if self.data.ndim != len(self.dimensions):
            raise ValueError(
                f"Data array has {self.data.ndim} dimensions, but {len(self.dimensions)} dimension names were provided."
            )

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
        object.__setattr__(
            self, "_dimension_map", {dim: idx for idx, dim in enumerate(self.dimensions)}
        )

        # Create a mapping from feature names to their indices
        object.__setattr__(
            self, "_feature_map", {feature: idx for idx, feature in enumerate(self.feature_names)}
        )

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
        data_jnp = jnp.array(self.data)
        if device:
            return jax.device_put(data_jnp, device=device)
        else:
            return data_jnp  # Default device (usually CPU)

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

    def __repr__(self) -> str:
        """
        Returns a string representation of the Tensor, including its shape, dimensions, feature names, and coordinates.

        Returns:
            str: String representation of the Tensor.
        """
        return (
            f"{self.__class__.__name__}(shape={self.data.shape}, "
            f"dimensions={self.dimensions}, "
            f"features={self.feature_names}, "
            f"Coordinates={self.Coordinates})"
        )

    def _create_new_instance(self: T, data: np.ndarray) -> T:
        """
        Creates a new instance of the same Tensor subclass with updated data.

        Args:
            data (np.ndarray): The data array for the new instance.

        Returns:
            T: New instance of the Tensor subclass.
        """
        return self.__class__(
            data=data,
            dimensions=self.dimensions,
            feature_names=self.feature_names,
            Coordinates=self.Coordinates,
        )

    # Arithmetic operations
    def __sub__(self: T, other: Union[T, float, int]) -> T:
        """
        Subtracts another tensor or scalar from this tensor.

        Args:
            other (Union[T, float, int]): The tensor or scalar to subtract.

        Returns:
            T: A new tensor with the result of the subtraction.
        """
        if isinstance(other, Tensor):
            new_data = self.data - other.data
        else:
            new_data = self.data - other
        return self._create_new_instance(data=new_data)

    def __truediv__(self: T, other: Union[T, float, int]) -> T:
        """
        Divides this tensor by another tensor or scalar.

        Args:
            other (Union[T, float, int]): The tensor or scalar to divide by.

        Returns:
            T: A new tensor with the result of the division.
        """
        if isinstance(other, Tensor):
            new_data = self.data / other.data
        else:
            new_data = self.data / other
        return self._create_new_instance(data=new_data)
    
    def __mul__(self: T, other: Union[T, float, int]) -> T:
        """
        Multiplies this tensor by another tensor or scalar.

        Args:
            other (Union[T, float, int]): The tensor or scalar to multiply by.

        Returns:
            T: A new tensor with the result of the multiplication.
        """
        if isinstance(other, Tensor):
            if self.data.shape != other.data.shape:
                raise ValueError("Tensors must have the same shape to multiply.")
            new_data = self.data * other.data
        else:
            new_data = self.data * other
        return self._create_new_instance(data=new_data)

    @eqx.filter_jit
    def u_roll(
        self: T,
        window_size: int,
        func: Callable[
            [int, Any, jnp.ndarray, int],
            Tuple[jnp.ndarray, Any]
        ],
        overlap_factor: float = None,
    ) -> jnp.ndarray:
        """
        Applies a function over rolling windows along the 'time' dimension using block processing for parallelization.

        Args:
            window_size (int): Size of the rolling window.
            func (Callable[[int, Any, jnp.ndarray, int], Tuple[jnp.ndarray, Any]]):
                Function to apply over the rolling window. Should accept an index, the carry, the block, and window size,
                and return (value, new_carry).
            overlap_factor (float, optional): Factor determining the overlap between blocks.

        Returns:
            jnp.ndarray: Data array computed with the u_roll method.
        """

        # Set NaN values to zero to avoid issues with NaN in the data
        data = jnp.nan_to_num(self.data)

        # Prepare blocks of data for processing
        blocks, block_indices = self._prepare_blocks(window_size, overlap_factor)

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

            values = jnp.zeros((t - window_size + 1, n, j), dtype=jnp.float64)

            # Initialize carry with the func (i == -1 case)
            initial_value, carry = func(-1, None, block, window_size)

            # Set the initial value in the values array
            values = values.at[0].set(initial_value)

            # Apply the step function iteratively
            def step_wrapper(i: int, state):
                values, carry = state
                new_value, new_carry = func(i, carry, block, window_size)
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

    
    # Helper method to prepare blocks of data for u_roll
    @eqx.filter_jit
    def _prepare_blocks(
        self,
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
        data = jnp.nan_to_num(self.data)

        num_time_steps = data.shape[0]
        other_dims = data.shape[1:]

        max_windows = num_time_steps - window_size + 1

        # If the overlap_factor (k) is not provided, default it to the ratio of max windows to window size
        if overlap_factor is None:
            overlap_factor = max_windows / window_size

        # Compute the effective block size (kw) based on the overlap factor and window size
        # This tells us how many time steps each block will span, including overlap
        block_size = math.ceil(overlap_factor * window_size)

        if block_size > max_windows:
            raise ValueError("Requested block size is larger than available data.")

        # Calculate the padding required to ensure that the data can be evenly divided into blocks
        padding_length = (block_size - max_windows % block_size) % block_size

        # Pad the data along the time dimension
        padding_shape = (padding_length,) + other_dims
        data_padded = jnp.concatenate(
            (data, jnp.zeros(padding_shape, dtype=data.float64)), axis=0
        )

        # Total number of windows in the padded data
        total_windows = num_time_steps - window_size + 1

        # Starting indices for blocks
        block_starts = jnp.arange(0, total_windows, block_size)

        # Generate indices to slice the data into blocks
        block_indices = block_starts[:, None] + jnp.arange(window_size - 1 + block_size)[None, :]

        return data_padded, block_indices

    def select(
        self, 
        dimension_name: str, 
        index: Union[int, slice, List[int], str, List[str]]
    ) -> T:
        """
        Selects a slice of the Tensor along the specified dimension.
        Returns a new Tensor instance with the sliced data.

        Args:
            dimension_name (str):
                Name of the dimension to slice (e.g., 'time', 'asset', 'feature').
            index (Union[int, slice, List[int], str, List[str]]):
                Index or slice to select along the specified dimension.
                Can be an integer, slice, list of integers, string, or list of strings.

        Returns:
            T:
                A new Tensor instance with the selected data.
        """
        dim_idx = self.get_dimension_index(dimension_name)

        # Create a list of slicers for each dimension
        slicer = [slice(None)] * self.data.ndim

        # Initialize new_feature_names in case of feature dimension
        new_feature_names = self.feature_names

        # Handle the 'feature' dimension separately if index is str or list of str
        if dimension_name == "feature" and isinstance(index, (str, list)):
            if isinstance(index, str):
                # Get the index of the feature name
                index_num = self.get_feature_index(index)
                index = slice(index_num, index_num + 1)
                slicer[dim_idx] = index
                sliced_data = self.data[tuple(slicer)]
                new_feature_names = (index,)  # This will be updated below
            elif isinstance(index, list) and all(isinstance(i, str) for i in index):
                # Get indices for each feature name
                indices = [self.get_feature_index(name) for name in index]
                index_array = np.array(indices)
                sliced_data = np.take(self.data, index_array, axis=dim_idx)
                new_feature_names = tuple(index)
            else:
                raise TypeError("Unsupported index type in list for 'feature' dimension.")
        else:
            # Handle numeric indices or slices
            if isinstance(index, int):
                # Convert integer index to a slice that retains the dimension
                slicer[dim_idx] = slice(index, index + 1)
                sliced_data = self.data[tuple(slicer)]
            elif isinstance(index, slice):
                slicer[dim_idx] = index
                sliced_data = self.data[tuple(slicer)]
            elif isinstance(index, list):
                if all(isinstance(i, int) for i in index):
                    index_array = np.array(index)
                    sliced_data = np.take(self.data, index_array, axis=dim_idx)
                else:
                    raise TypeError("Unsupported index type in list for dimension.")
            else:
                raise TypeError("Unsupported index type for dimension.")

        # Update Coordinates
        new_variables = self.Coordinates.variables.copy()
        if dimension_name in new_variables:
            coord_var = self.Coordinates.variables[dimension_name]
            if isinstance(index, int):
                new_variables[dimension_name] = coord_var[index : index + 1]
            elif isinstance(index, slice):
                new_variables[dimension_name] = coord_var[index]
            elif isinstance(index, list):
                if all(isinstance(i, int) for i in index):
                    new_variables[dimension_name] = np.take(coord_var, index)
            elif dimension_name == "feature" and isinstance(index, (str, list)):
                if isinstance(index, str):
                    new_variables[dimension_name] = np.array([index])
                elif isinstance(index, list):
                    new_variables[dimension_name] = np.array(index)
        new_Coordinates = Coordinates(variables=new_variables)

        # Update feature names if slicing along 'feature' dimension
        if dimension_name == "feature":
            if isinstance(index, int):
                new_feature_names = (self.feature_names[index],)
            elif isinstance(index, slice):
                new_feature_names = self.feature_names[index]
            elif isinstance(index, list) and all(isinstance(i, str) for i in index):
                # Already handled above
                pass
            elif isinstance(index, list) and all(isinstance(i, int) for i in index):
                new_feature_names = tuple(self.feature_names[i] for i in index)

        # Return new Tensor instance with the sliced data
        return self.__class__(
            data=sliced_data,
            dimensions=self.dimensions,
            feature_names=new_feature_names,
            Coordinates=new_Coordinates,
        )


class ReturnsTensor(Tensor):
    """
    Specialized Tensor for handling return-related data.
    Ensures consistency by enforcing fixed dimensions and feature names.

    Attributes:
        data (np.ndarray):
            Data array with shape corresponding to (time, asset, feature).
            Specifically tailored for return data with a single feature.
        Coordinates (Coordinates):
            Coordinate variables associated with the tensor dimensions.
    """

    data: np.ndarray
    Coordinates: Coordinates = eqx.field(static=True)

    # Fixed dimensions and feature names are set internally and are not exposed to the user
    dimensions: Tuple[str, ...] = eqx.field(default=("time", "asset", "feature"), init=False)
    feature_names: Tuple[str, ...] = eqx.field(default=("return",), init=False)
    
    _dimension_map: Dict[str, int] = eqx.field(init=False, static=True)
    _feature_map: Dict[str, int] = eqx.field(init=False, static=True)

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
        if self.feature_names != ("return",):
            raise ValueError("ReturnsTensor must have exactly one feature named 'return'.")

        # Ensure that the data array has one feature dimension
        if self.data.shape[-1] != 1:
            raise ValueError("ReturnsTensor data must have exactly one feature dimension.")

        object.__setattr__(
            self, "_dimension_map", {dim: idx for idx, dim in enumerate(self.dimensions)}
        )
        object.__setattr__(
            self, "_feature_map", {feature: idx for idx, feature in enumerate(self.feature_names)}
        )


    # No need to implement select; it's handled by the base Tensor class

    # Override _create_new_instance
    def _create_new_instance(self: "ReturnsTensor", data: np.ndarray) -> "ReturnsTensor":
        """
        Creates a new ReturnsTensor instance with the updated data.

        Args:
            data (np.ndarray): The data array for the new instance.

        Returns:
            ReturnsTensor: New instance of ReturnsTensor.
        """
        return self.__class__(
            data=data,
            Coordinates=self.Coordinates,
        )


class CharacteristicsTensor(Tensor):
    """
    Specialized Tensor for handling characteristic-related data.
    Allows flexible dimensions and feature names as needed.

    Attributes:
        data (np.ndarray):
            Data array with dimensions corresponding to (time, asset, feature).
            Can hold multiple characteristics as features.
        dimensions (Tuple[str, ...]):
            Names of the dimensions (e.g., 'time', 'asset', 'feature').
        feature_names (Tuple[str, ...]):
            Names of the characteristics, corresponding to features.
        Coordinates (Coordinates):
            Coordinate variables associated with the tensor dimensions.
    """

    data: np.ndarray
    dimensions: Tuple[str, ...] = eqx.field(static=True)
    feature_names: Tuple[str, ...] = eqx.field(static=True)
    Coordinates: Coordinates = eqx.field(static=True)

    _dimension_map: Dict[str, int] = eqx.field(init=False, static=True)
    _feature_map: Dict[str, int] = eqx.field(init=False, static=True)

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            raise TypeError("Data must be a NumPy array (np.ndarray).")

        if self.data.ndim != len(self.dimensions):
            raise ValueError(
                f"Data array has {self.data.ndim} dimensions, but {len(self.dimensions)} dimension names were provided."
            )

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

        object.__setattr__(
            self, "_dimension_map", {dim: idx for idx, dim in enumerate(self.dimensions)}
        )
        object.__setattr__(
            self, "_feature_map", {feature: idx for idx, feature in enumerate(self.feature_names)}
        )

    # No need to implement select; it's handled by the base Tensor class

    # Override _create_new_instance
    def _create_new_instance(self: "CharacteristicsTensor", data: np.ndarray) -> "CharacteristicsTensor":
        """
        Creates a new CharacteristicsTensor instance with the updated data.

        Args:
            data (np.ndarray): The data array for the new instance.

        Returns:
            CharacteristicsTensor: New instance of CharacteristicsTensor.
        """
        return self.__class__(
            data=data,
            dimensions=self.dimensions,
            feature_names=self.feature_names,
            Coordinates=self.Coordinates,
        )


from . import tensor_ops
