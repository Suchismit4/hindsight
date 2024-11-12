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
    dimensions: Tuple[str, ...] = eqx.static_field()
    feature_names: Tuple[str, ...] = eqx.static_field()
    Coordinates: Coordinates

    # Internal mappings for quick index retrieval by name
    _dimension_map: Dict[str, int] = eqx.static_field()
    _feature_map: Dict[str, int] = eqx.static_field()

    def __post_init__(self):
        """
        Validates the tensor's structure, initializes internal mappings,
        and ensures data is a JAX array for efficient computations.

        Raises:
            TypeError: If 'data' is not a NumPy array or JAX array.
            ValueError: If the data dimensions do not match the number of dimension names.
            ValueError: If any tensor dimensions are missing in coordinate keys.
        """
        # Ensure 'data' is a NumPy array or JAX array for compatibility
        if not isinstance(self.data, (np.ndarray, jnp.ndarray)):
            raise TypeError("Data must be a NumPy array (np.ndarray) or JAX array (jnp.ndarray).")

        # Convert data to JAX array for efficient computations
        object.__setattr__(self, 'data', jnp.array(self.data))

        # Ensure data dimensions match the number of dimension names
        if self.data.ndim != len(self.dimensions):
            raise ValueError(
                f"Data array has {self.data.ndim} dimensions, but {len(self.dimensions)} dimension names were provided."
            )

        # Ensure that all dimensions have corresponding coordinates
        dimensions_set = set(self.dimensions)
        coordinates_set = set(self.Coordinates.variables.keys())

        missing_in_coords = dimensions_set - coordinates_set

        if missing_in_coords:
            error_message = "Tensor dimensions and coordinate keys do not match.\n"
            error_message += f"Dimensions missing in Coordinates: {missing_in_coords}\n"
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
        Ensures the tensor's data array is on the specified device (e.g., CPU, GPU).

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
            return self.data  # Data is already a JAX array

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

    def _create_new_instance(self: T,
                             data: Union[np.ndarray, jnp.ndarray],
                             Coordinates: Coordinates = None) -> T:
        """
        Creates a new instance of the same Tensor subclass with updated data and optional Coordinates.

        Args:
            data (Union[np.ndarray, jnp.ndarray]): The data array for the new instance.
            Coordinates (Coordinates, optional): The Coordinates object for the new instance.

        Returns:
            T: New instance of the Tensor subclass.
        """
        if Coordinates is None:
            Coordinates = self.Coordinates
        return self.__class__(
            data=data,
            dimensions=self.dimensions,
            feature_names=self.feature_names,
            Coordinates=Coordinates,
        )

    # Arithmetic operations
    def __sub__(self: T, other: Union[T, float, int]) -> T:
        """
        Subtracts another tensor or scalar from this tensor.
        Assumes tensors are already aligned.

        Args:
            other (Union[T, float, int]): The tensor or scalar to subtract.

        Returns:
            T: A new tensor with the result of the subtraction.
        """
        if isinstance(other, Tensor):
            if self.data.shape != other.data.shape:
                raise ValueError("Tensors must have the same shape to subtract.")
            new_data = self.data - other.data
        else:
            new_data = self.data - other
        return self._create_new_instance(data=new_data)

    def __truediv__(self: T, other: Union[T, float, int]) -> T:
        """
        Divides this tensor by another tensor or scalar.
        Assumes tensors are already aligned.

        Args:
            other (Union[T, float, int]): The tensor or scalar to divide by.

        Returns:
            T: A new tensor with the result of the division.
        """
        if isinstance(other, Tensor):
            if self.data.shape != other.data.shape:
                raise ValueError("Tensors must have the same shape to divide.")
            new_data = self.data / other.data
        else:
            new_data = self.data / other
        return self._create_new_instance(data=new_data)

    def __mul__(self: T, other: Union[T, float, int]) -> T:
        """
        Multiplies this tensor by another tensor or scalar.
        Assumes tensors are already aligned.

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

    # The rest of the methods remain as per your existing code

class TimeSeriesTensor(Tensor):
    """
    Tensor subclass for handling time series data with hierarchical time components.
    Provides methods to index using expanded time dimensions and aligns tensors based on time components.
    """
    # Declare additional attributes
    _dimension_map: Dict[str, int] = eqx.field(init=False)
    _feature_map: Dict[str, int] = eqx.field(init=False)

    def __post_init__(self):
        super().__post_init__()
        # Initialize the dimension and feature maps
        object.__setattr__(
            self, "_dimension_map", {dim: idx for idx, dim in enumerate(self.dimensions)}
        )
        object.__setattr__(
            self, "_feature_map", {feature: idx for idx, feature in enumerate(self.feature_names)}
        )
        # Ensure 'time' coordinate exists
        if 'time' not in self.Coordinates.variables:
            raise ValueError("TimeSeriesTensor requires a 'time' coordinate.")
        # Initialize time mappings in Coordinates
        self.Coordinates._initialize_time_mappings()


    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the apparent shape of the tensor, including expanded time dimensions.

        Returns:
            Tuple[int, ...]: The shape of the tensor with expanded time dimensions.
        """
        expanded_time_shape = self.Coordinates.get_expanded_time_shape()
        other_dims = self.data.shape[1:]  # Exclude the flat time dimension
        return expanded_time_shape + other_dims

    def __getitem__(self, indices: Union[int, slice, Tuple]) -> Any:
        """
        Allows indexing using expanded time dimensions.

        Args:
            indices (Union[int, slice, Tuple]):
                Indices corresponding to expanded time dimensions and other dimensions.

        Returns:
            Any: The data at the specified indices.
        """
        # Convert indices to a tuple
        if not isinstance(indices, tuple):
            indices = (indices,)

        time_component_names = self.Coordinates.get_time_component_names()
        num_time_components = len(time_component_names)
        if len(indices) < num_time_components:
            raise IndexError("Not enough indices provided for expanded time dimensions.")

        # Separate time indices and other dimension indices
        time_indices = indices[:num_time_components]
        other_indices = indices[num_time_components:]

        # Map expanded time indices to flat time indices
        # We'll handle integer indices
        if all(isinstance(idx, int) for idx in time_indices):
            # Single index into each time component
            expanded_time_index = tuple(
                self._get_time_component_value(name, idx)
                for name, idx in zip(time_component_names, time_indices)
            )
            flat_time_idx = self.Coordinates.get_flat_time_index(expanded_time_index)
            # Build full index for data array
            data_indices = (flat_time_idx,) + other_indices
            return self.data[data_indices]
        else:
            # Handle slices or arrays
            raise NotImplementedError("Indexing with slices or arrays is not yet implemented.")

    def _get_time_component_value(self, component_name: str, idx: int) -> Any:
        """
        Retrieves the value of a time component at a given index.

        Args:
            component_name (str): Name of the time component.
            idx (int): Index into the component.

        Returns:
            Any: The value at the specified index.
        """
        unique_values = self.Coordinates.get_time_component_unique_values(component_name)
        if idx < 0 or idx >= len(unique_values):
            raise IndexError(f"Index {idx} out of bounds for time component '{component_name}'.")
        return unique_values[idx]

    def __repr__(self) -> str:
        """
        Returns a string representation of the TimeSeriesTensor, including its apparent shape.

        Returns:
            str: String representation of the TimeSeriesTensor.
        """
        return (
            f"{self.__class__.__name__}(shape={self.shape}, "
            f"dimensions={self.Coordinates.get_time_component_names() + self.dimensions[1:]}, "
            f"features={self.feature_names}, "
            f"Coordinates={self.Coordinates})"
        )

    def _align_tensors(self: T, other: T) -> Tuple[jnp.ndarray, jnp.ndarray, Coordinates]:
        """
        Aligns self and other tensors based on their expanded time components.

        Args:
            other (T): Another TimeSeriesTensor to align with.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, Coordinates]:
                - Aligned data from self.
                - Aligned data from other.
                - New Coordinates object with common expanded time components.
        """
        # Get expanded time indices for both tensors
        self_expanded_time = self.Coordinates._expanded_time_index
        other_expanded_time = other.Coordinates._expanded_time_index

        # Use structured arrays to treat rows as single items
        dtype = [('comp' + str(i), self_expanded_time.dtype) for i in range(self_expanded_time.shape[1])]
        self_struct = self_expanded_time.view(dtype).reshape(-1)
        other_struct = other_expanded_time.view(dtype).reshape(-1)

        # Find common rows
        common_struct, self_idx, other_idx = jnp.intersect1d(
            self_struct, other_struct, assume_unique=False, return_indices=True
        )

        if common_struct.size == 0:
            raise ValueError("No overlapping time periods to align.")

        # Align data
        aligned_self_data = self.data[self_idx, ...]
        aligned_other_data = other.data[other_idx, ...]

        # Create new Coordinates with common expanded time components
        new_variables = self.Coordinates.variables.copy()
        for name in self.Coordinates._time_component_names:
            new_variables[name] = self.Coordinates.variables[name][self_idx]
        for name in self.Coordinates.variables:
            if name not in self.Coordinates._time_component_names:
                new_variables[name] = self.Coordinates.variables[name]
        new_Coordinates = Coordinates(variables=new_variables)

        return aligned_self_data, aligned_other_data, new_Coordinates

    def __mul__(self: T, other: Union[T, float, int]) -> T:
        """
        Multiplies this tensor by another tensor or scalar.
        Handles alignment based on expanded time dimensions when other is a TimeSeriesTensor.

        Args:
            other (Union[T, float, int]): The tensor or scalar to multiply by.

        Returns:
            T: A new TimeSeriesTensor with the result of the multiplication.
        """
        if isinstance(other, TimeSeriesTensor):
            # Align tensors based on expanded time components
            aligned_self_data, aligned_other_data, new_Coordinates = self._align_tensors(other)
            # Perform element-wise multiplication
            new_data = aligned_self_data * aligned_other_data
            return self._create_new_instance(data=new_data, Coordinates=new_Coordinates)
        else:
            # Scalar multiplication
            new_data = self.data * other
            return self._create_new_instance(data=new_data)

    def __sub__(self: T, other: Union[T, float, int]) -> T:
        """
        Subtracts another tensor or scalar from this tensor.
        Handles alignment based on expanded time dimensions when other is a TimeSeriesTensor.

        Args:
            other (Union[T, float, int]): The tensor or scalar to subtract.

        Returns:
            T: A new TimeSeriesTensor with the result of the subtraction.
        """
        if isinstance(other, TimeSeriesTensor):
            # Align tensors based on expanded time components
            aligned_self_data, aligned_other_data, new_Coordinates = self._align_tensors(other)
            # Perform element-wise subtraction
            new_data = aligned_self_data - aligned_other_data
            return self._create_new_instance(data=new_data, Coordinates=new_Coordinates)
        else:
            new_data = self.data - other
            return self._create_new_instance(data=new_data)

    def __truediv__(self: T, other: Union[T, float, int]) -> T:
        """
        Divides this tensor by another tensor or scalar.
        Handles alignment based on expanded time dimensions when other is a TimeSeriesTensor.

        Args:
            other (Union[T, float, int]): The tensor or scalar to divide by.

        Returns:
            T: A new TimeSeriesTensor with the result of the division.
        """
        if isinstance(other, TimeSeriesTensor):
            # Align tensors based on expanded time components
            aligned_self_data, aligned_other_data, new_Coordinates = self._align_tensors(other)
            # Perform element-wise division
            new_data = aligned_self_data / aligned_other_data
            return self._create_new_instance(data=new_data, Coordinates=new_Coordinates)
        else:
            new_data = self.data / other
            return self._create_new_instance(data=new_data)

    def select(
        self,
        dimension_name: str,
        index: Union[int, slice, List[int], str, List[str]]
    ) -> T:
        """
        Selects a slice of the TimeSeriesTensor along the specified dimension.
        Returns a new TimeSeriesTensor instance with the sliced data.

        Args:
            dimension_name (str):
                Name of the dimension to slice (e.g., 'time', 'asset', 'feature').
            index (Union[int, slice, List[int], str, List[str]]):
                Index or slice to select along the specified dimension.

        Returns:
            T:
                A new TimeSeriesTensor instance with the selected data.
        """
        # Use the select method from the base Tensor class
        new_tensor = super().select(dimension_name, index)

        # Ensure the returned instance is a TimeSeriesTensor
        return self.__class__(
            data=new_tensor.data,
            dimensions=new_tensor.dimensions,
            feature_names=new_tensor.feature_names,
            Coordinates=new_tensor.Coordinates,
        )

class ReturnsTensor(TimeSeriesTensor):
    """
    Specialized TimeSeriesTensor for handling return-related data.
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
    dimensions: Tuple[str, ...] = eqx.field(default=("time", "asset", "feature"), init=False)
    feature_names: Tuple[str, ...] = eqx.field(default=("return",), init=False)
    
    _dimension_map: Dict[str, int] = eqx.field(init=False)
    _feature_map: Dict[str, int] = eqx.field(init=False)

    def __post_init__(self):
        """
        Validates that the tensor conforms to the expected structure for return data.
        Ensures that only one feature named 'return' exists.
        """
        # Call the parent __post_init__ to handle data conversion and validation
        super().__post_init__()
        
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
    def _create_new_instance(self: "ReturnsTensor", data: jnp.ndarray, Coordinates: Coordinates = None) -> "ReturnsTensor":
        """
        Creates a new ReturnsTensor instance with the updated data.

        Args:
            data (jnp.ndarray): The data array for the new instance.
            Coordinates (Coordinates, optional): The Coordinates object for the new instance.

        Returns:
            ReturnsTensor: New instance of ReturnsTensor.
        """
        if Coordinates is None:
            Coordinates = self.Coordinates
        return self.__class__(
            data=data,
            Coordinates=Coordinates,
        )

class CharacteristicsTensor(TimeSeriesTensor):
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
    dimensions: Tuple[str, ...] = eqx.field()
    feature_names: Tuple[str, ...] = eqx.field()
    Coordinates: Coordinates = eqx.field()

    _dimension_map: Dict[str, int] = eqx.field(init=False)
    _feature_map: Dict[str, int] = eqx.field(init=False)

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            raise TypeError("Data must be a NumPy array (np.ndarray).")

        if self.data.ndim != len(self.dimensions):
            raise ValueError(
                f"Data array has {self.data.ndim} dimensions, but {len(self.dimensions)} dimension names were provided."
            )

        # Get the base dimensions that must match (excluding time components)
        dimensions_set = set(self.dimensions)
        coordinates_set = set(self.Coordinates.variables.keys())
        
        # Get time components
        time_components = {'year', 'quarter', 'month', 'day'}
        
        # Remove time components from the coordinates set for comparison
        base_coordinates = coordinates_set - time_components
        
        # Check if base dimensions match
        if not dimensions_set.issubset(base_coordinates):
            missing_dims = dimensions_set - base_coordinates
            error_message = "Required tensor dimensions are missing in coordinates.\n"
            error_message += f"Missing dimensions: {missing_dims}\n"
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
