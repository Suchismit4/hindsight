# src/data/processor_registry.py

from functools import wraps
from typing import Optional, List, Dict, Any, Callable
import xarray

class ProcessorRegistry:
    """
    A registry for storing and retrieving data processing functions.
    Each function must accept an xarray.Dataset and a dictionary of parameters (processors),
    and return a postprocessed xarray.Dataset.
    """

    # Class attributes
    Processor: Callable[[xarray.Dataset, Dict[str, Any]], xarray.Dataset] # Processor type to enforce function type
    _instance = None                                                      # Private class variable to store the singleton instance

    def __new__(cls): 
        """
        Override the __new__ method to control the creation of a new instance.
        Ensures that only one instance of ProcessorRegistry is created.
        """
        if not cls._instance:
            # Log creation 
            print("Instantiating ProcessorRegistry")
            
            # Create Singleton instance and instantiate directory (ONLY ONCE)
            cls._instance = super(ProcessorRegistry, cls).__new__(cls)
            cls._instance._registry: Dict[str, ProcessorRegistry.Processor] = {}
        return cls._instance

    def register(self, name: str, func: ProcessorRegistry.Processor) -> ProcessorRegistry.Processor: 
        """
        Register a function as a processor.

        Parameters:
            name (str): The name under which to register the processor.
            func (Processor): The processor function to register.

        Returns:
            Processor: The registered processor function.
        """
        self._registry[name] = func
        return func

    def get(self, name: str, default: Any = None) -> ProcessorRegistry.Processor:
        """
        Retrieves a registered function by its name with an option to return a default value if
        the function name is not found. This method does not raise a KeyError.

        Args:
            name (str): The name of the function to retrieve from the registry.
            default (Any, optional): The default value to return if the function name is not found.

        Returns:
            ProcessorRegistry.Processor: The function registered under the specified name, or
                                         the default value if the function is not found.

        Examples:
            function = registry.get('my_function', default=lambda x: x)
            This line retrieves 'my_function' if it exists, otherwise returns a lambda function
            that returns its input.
        """
        return self._registry.get(name, default)

    def __call__(self, func: Callable) -> ProcessorRegistry.Processor:
        """
        Allows the class instance to be used as a decorator. This method is called
        when the decorator is applied to a function. Function wraps the function and then
        registers it.
        
        The decorator modifies the function's name by removing a predefined prefix "processor_"
        before registering, which is useful for namespace management or simplifying
        function identifiers in the registry.

        Args:
            func (ProcessorRegistry.Processor): The function to decorate, which will be
                                                registered in the registry with its name
                                                possibly modified.

        Returns:
            ProcessorRegistry.Processor: The wrapped function, which is now registered in
                                         the registry under its potentially modified name.
        
        Examples:
            @processor
            def processor_function(x):
                return x * 2

            This usage decorates 'processor_function', removes "processor_" from its
            name, and registers it in the registry with binding "function": processor_function().
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # register and return function
        return self.register(
            func.__name__.replace('processor_', ''),
            wrapper
        )

    def __getitem__(self, name: str) -> ProcessorRegistry.Processor:
        """
        Enables direct access to registered functions using dictionary-like subscript notation.
        This method retrieves a function by its name, throwing a KeyError if the function does not exist.

        Args:
            name (str): The name of the function to retrieve from the registry.

        Returns:
            ProcessorRegistry.Processor: The function registered under the specified name.

        Raises:
            KeyError: If no function is registered under the specified name, a KeyError is raised.

        Examples:
            function = registry['my_function']
            This line retrieves 'my_function' from the registry if it exists.

        """

        if name in self._registry:
            return self._registry[name]
        # if function doesn't exist raise error
        raise KeyError(f"No function registered under the name {name}")


# Create instance of ProcessorRegistry for registering functions 
post_processor = ProcessorRegistry()