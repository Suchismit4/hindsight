
# src/data/registry.py
"""
Registry module for managing data loader registration and discovery.

This module provides the central registry for data loaders and the decorator
used to register them. The registry maintains information about what data
types each loader can handle and where the source data is located.
"""

data_loader_registry = {}

def register_data_loader(data_path: str, data_types: list):
    """
    Decorator factory for registering data loaders with the framework.
    
    This decorator handles the registration of data loader classes, storing
    information about what data types they can handle and where their source
    data is located.
    
    Args:
        data_path: String identifying the location of the source data.
        data_types: List of string identifiers for the types of data this
                   loader can handle.
    
    Returns:
        callable: A decorator function that registers the decorated class.
    """
    def decorator(cls):
        data_loader_registry[data_path] = {
            'loader': cls(),
            'data_types': data_types
        }
        return cls
    return decorator
