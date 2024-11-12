# hindsight/example_working.py

# Importing necessary classes from the data_layer and coords modules
from src import Tensor, CharacteristicsTensor, Coordinates, DataLoader

# Importing JAX's NumPy module for numerical operations
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd  # Ensure pandas is imported for date handling


def main():
    """
    Main function to demonstrate the loading cached CRSP data using DataLoader.
    """
    
    data_loader = DataLoader()
    
    tensor = Tensor(data=jnp.array(),
                    Coordinates=Coordinates())
    
    print(apple)
    
if __name__ == "__main__":
    main()