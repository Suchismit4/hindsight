# src/data/processors/registry.py

from functools import wraps
from typing import Optional, List, Dict, Any, Callable, TypeVar, Generic
import xarray
import pandas as pd

T = TypeVar('T')  # Generic type for input
U = TypeVar('U')  # Generic type for output

class Registry(Generic[T, U]):
    """
    A generic registry for storing and retrieving processing functions.
    
    This registry can be used for different types of data processing functions,
    such as post-processors (xarray.Dataset -> xarray.Dataset) or 
    filters (pd.DataFrame -> pd.DataFrame).
    
    Args:
        T: Input type for registered functions
        U: Output type for registered functions
    """

    # Class attributes
    _instances = {}  # Dictionary to store instances by registry_name

    def __new__(cls, registry_name: str):
        """
        Override the __new__ method to control the creation of instances.
        Ensures that only one instance of Registry is created per name.
        
        Args:
            registry_name: A unique name for this registry instance
        """
        if registry_name not in cls._instances:            
            # Create instance and instantiate directory
            instance = super(Registry, cls).__new__(cls)
            instance._registry = {}
            instance._registry_name = registry_name
            cls._instances[registry_name] = instance
        return cls._instances[registry_name]

    def register(self, name: str, func: Callable[[T, Dict[str, Any]], U]) -> Callable[[T, Dict[str, Any]], U]:
        """
        Register a function in the registry.

        Parameters:
            name (str): The name under which to register the function.
            func (Callable): The function to register.

        Returns:
            Callable: The registered function.
        """
        self._registry[name] = func
        return func

    def get(self, name: str, default: Any = None) -> Callable[[T, Dict[str, Any]], U]:
        """
        Retrieves a registered function by its name with an option to return a default value if
        the function name is not found.

        Args:
            name (str): The name of the function to retrieve from the registry.
            default (Any, optional): The default value to return if the function name is not found.

        Returns:
            Callable: The function registered under the specified name, or the default value.
        """
        return self._registry.get(name, default)

    def __call__(self, func: Callable) -> Callable[[T, Dict[str, Any]], U]:
        """
        Allows the class instance to be used as a decorator.
        
        Args:
            func (Callable): The function to decorate, which will be registered in the registry.

        Returns:
            Callable: The wrapped function, which is now registered in the registry.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # register and return function
        return self.register(
            func.__name__,
            wrapper
        )

    def __getitem__(self, name: str) -> Callable[[T, Dict[str, Any]], U]:
        """
        Enables direct access to registered functions using dictionary-like subscript notation.
        
        Args:
            name (str): The name of the function to retrieve from the registry.

        Returns:
            Callable: The function registered under the specified name.

        Raises:
            KeyError: If no function is registered under the specified name.
        """
        if name in self._registry:
            return self._registry[name]
        # if function doesn't exist raise error
        raise KeyError(f"No function registered under the name {name} in registry {self._registry_name}")

# Create instances for different types of processors
post_processor = Registry[xarray.Dataset, xarray.Dataset]("post_processor")