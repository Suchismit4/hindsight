"""
src/__init__.py

This file serves as the initializer for the `src` package in the hindsight project. It is responsible for setting up
the package-level imports and configurations necessary for the modules within the `src` directory. Any global constants,
package-wide settings, or important metadata about the package can be defined here.

This file ensures that when the `src` package is imported, the necessary components are readily available.
"""

# Import specific modules or classes from the src package
from src.data_layer.coords import Coordinates
from src.data_layer.tensor import Tensor, ReturnsTensor, CharacteristicsTensor
from src.data_layer.data_manager import DataLoader

__version__ = "0.0.1"