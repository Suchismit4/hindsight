# src/data/provider.py

"""
Provider management system for data sources in Hindsight.

This module implements a registry-based system for managing data providers and their
associated data loaders. Key components include:

1. Provider class: Represents a data source with associated fetchers
2. Provider registry: Maintains a global registry of available providers
3. Registration functions: For adding and retrieving providers

Providers serve as factories for data loaders, creating and managing instances
of specific data loader classes based on their configuration.
"""

from typing import Dict, Any, Type, Optional

class Provider:
    """
    Represents a data source provider with associated data loaders.
    
    A Provider maintains a collection of data loader instances for different
    data types available from a single source (e.g., WRDS, Yahoo Finance, etc.).
    It serves as a factory for creating and accessing these loaders.
    
    Attributes:
        name (str): Unique identifier for the provider
        website (str): URL of the provider's website
        description (str): Description of the provider and its data offerings
        fetcher_dict (Dict[str, Any]): Mapping from fetcher names to fetcher classes
        repr_name (str): Display name for the provider
        data_loaders (Dict[str, Any]): Instances of data loaders created from fetcher classes
    """
    def __init__(
        self,
        name: str,
        website: str,
        description: str,
        fetcher_dict: Dict[str, Type],
        repr_name: Optional[str] = None,
    ):
        """
        Initialize a Provider with metadata and data loader classes.
        
        Parameters:
            name (str): Unique identifier for the provider
            website (str): URL of the provider's website
            description (str): Description of the provider and its data offerings
            fetcher_dict (Dict[str, Type]): Mapping from fetcher names to fetcher classes
            repr_name (Optional[str]): Display name for the provider, defaults to name
        """
        self.name = name
        self.website = website
        self.description = description
        self.fetcher_dict = fetcher_dict
        self.repr_name = repr_name or name

        # Initialize a dictionary to hold data loaders
        self.data_loaders = {}
        # Register the fetchers
        for fetcher_name, fetcher_class in fetcher_dict.items():
            data_path = f"{self.name}/{fetcher_name}"
            instance = fetcher_class(data_path)
            self.data_loaders[data_path] = instance

    def get_fetcher(self, fetcher_key: str) -> Any:
        """
        Retrieve the fetcher class associated with the given key.
        
        Parameters:
            fetcher_key (str): The identifier for the fetcher class, typically in the 
                format 'category.subcategory.type' (e.g., 'equity.price.historical')
                
        Returns:
            Any: The fetcher class for the specified key, or None if not found
        """
        return self.fetcher_dict.get(fetcher_key)

    def __repr__(self) -> str:
        """
        Create a string representation of the Provider instance.
        
        Returns:
            str: A string representation of the Provider
        """
        return f"<Provider {self.repr_name}>"

# Global registry of available providers
_PROVIDER_REGISTRY = {}

def register_provider(provider: Provider) -> None:
    """
    Register a provider in the global provider registry.
    
    Parameters:
        provider (Provider): The provider instance to register
    """
    _PROVIDER_REGISTRY[provider.name] = provider

def get_provider(name: str) -> Provider:
    """
    Retrieve a provider from the global registry by name.
    
    Parameters:
        name (str): The unique identifier of the provider
        
    Returns:
        Provider: The provider instance
        
    Raises:
        KeyError: If no provider with the specified name exists
    """
    return _PROVIDER_REGISTRY[name]