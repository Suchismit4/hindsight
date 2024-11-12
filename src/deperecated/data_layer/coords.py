# data_layer/coords.py
import numpy as np
import equinox as eqx
from typing import Mapping, Any, Iterator, List, Tuple
import jax.numpy as jnp


class Coordinates(eqx.Module):
    """
    Manages coordinate variables associated with tensor dimensions.
    Supports numerical data types as JAX arrays and handles time component mappings.

    Attributes:
        variables (Mapping[str, Any]): 
            A mapping from dimension names to their coordinate data.
            - Numerical data types are converted to JAX arrays for compatibility with JAX transformations.
            - Non-numerical data types (e.g., strings) are stored as-is to maintain flexibility.
    """
    variables: Mapping[str, Any] = eqx.field()
    
    # Declare fields for attributes set after initialization
    _time_component_names: List[str] = eqx.field(init=False)
    _expanded_time_index: jnp.ndarray = eqx.field(init=False)
    _flat_time_index: jnp.ndarray = eqx.field(init=False)

    def __post_init__(self):
        """
        Processes the coordinate variables after initialization:
        - Converts numerical data types to JAX arrays for optimized computations with JAX.
        - Keeps non-numerical data types unchanged to preserve information such as categorical labels.
        """
        processed_vars = {}
        for name, var in self.variables.items():
            if isinstance(var, (jnp.ndarray, np.ndarray)):
                # Convert numpy arrays to JAX arrays
                processed_vars[name] = jnp.array(var)
            elif self._is_numerical(var):
                # Convert numerical lists or tuples to JAX arrays
                processed_vars[name] = jnp.array(var)
            else:
                # Keep non-numerical data types unchanged
                processed_vars[name] = var
        object.__setattr__(self, 'variables', processed_vars)
        # Initialize mappings for time components if 'time' is in variables
        if 'time' in self.variables:
            self._initialize_time_mappings()

    @staticmethod
    def _is_numerical(var: Any) -> bool:
        if isinstance(var, (list, tuple)):
            return all(isinstance(v, (int, float, complex)) for v in var)
        return isinstance(var, (int, float, complex, np.ndarray))
        
    def _initialize_time_mappings(self):
        """
        Initializes mappings between expanded time components and flat time indices.
        """
        # Assume time components are stored as variables (e.g., 'year', 'month', 'day')
        # Create a structured array for time components
        time_components = []
        time_component_names = []
        for name in ['year', 'quarter', 'month', 'day', 'hour', 'minute', 'second']:
            if name in self.variables:
                time_components.append(self.variables[name])
                time_component_names.append(name)
        if time_components:
            # Stack components to create a unique identifier for each time point
            object.__setattr__(self, '_time_component_names', time_component_names)
            expanded_time_index = jnp.stack(time_components, axis=-1)
            object.__setattr__(self, '_expanded_time_index', expanded_time_index)
            # Create a mapping from expanded indices to flat indices
            flat_time_index = jnp.arange(len(self.variables['time']))
            object.__setattr__(self, '_flat_time_index', flat_time_index)
        else:
            # No time components other than 'time' provided
            object.__setattr__(self, '_time_component_names', ['time'])
            expanded_time_index = self.variables['time'][:, None]
            object.__setattr__(self, '_expanded_time_index', expanded_time_index)
            flat_time_index = jnp.arange(len(self.variables['time']))
            object.__setattr__(self, '_flat_time_index', flat_time_index)

    def get_flat_time_index(self, expanded_index: Tuple) -> int:
        """
        Maps an expanded time index to the flat time index.

        Args:
            expanded_index (Tuple): A tuple of time component values (e.g., (2021, 1, 15))

        Returns:
            int: The flat time index corresponding to the expanded time index.

        Raises:
            ValueError: If the expanded index is not found.
        """
        # Convert expanded_index to JAX array
        expanded_index_array = jnp.array(expanded_index)
        # Compare with expanded_time_index to find the matching flat index
        matches = jnp.all(self._expanded_time_index == expanded_index_array, axis=-1)
        indices = jnp.nonzero(matches)[0]
        if indices.size == 0:
            raise ValueError(f"Expanded time index {expanded_index} not found.")
        return int(indices[0])  # Return the first matching index as int

    def get_expanded_time_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the expanded time dimensions.

        Returns:
            Tuple[int, ...]: A tuple representing the sizes of each time component dimension.
        """
        sizes = []
        for name in self._time_component_names:
            unique_values = jnp.unique(self.variables[name])
            sizes.append(len(unique_values))
        return tuple(sizes)

    def get_time_component_names(self) -> List[str]:
        """
        Returns the names of the time components.

        Returns:
            List[str]: A list of time component names.
        """
        return self._time_component_names

    def get_time_component_unique_values(self, component_name: str) -> jnp.ndarray:
        """
        Returns the unique values of a time component.

        Args:
            component_name (str): Name of the time component.

        Returns:
            jnp.ndarray: Array of unique values for the time component.
        """
        return jnp.unique(self.variables[component_name])

    def __getitem__(self, key: str) -> Any:
        return self.variables[key]

    def __repr__(self) -> str:
        return f"Coordinates(variables={self.variables})"
