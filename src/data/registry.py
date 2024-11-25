# src/data/registry.py

"""
Registry module for managing data loader registration and discovery.

This module provides the central registry for data loaders and the decorator
used to register them. The registry maintains information about what data
types each loader can handle and where the source data is located.
"""

data_loader_registry = {}

def register_data_loader(data_path: str):
    """
    Decorator factory for registering data loaders with the framework.
    
    This decorator handles the registration of data loader classes, storing
    them in a registry with their associated data paths.
    
    Args:
        data_path: String identifying the location of the source data.
    
    Returns:
        callable: A decorator function that registers the decorated class.
    """
    def decorator(cls):
        instance = cls()
        data_loader_registry[data_path] = instance
        return cls
    return decorator