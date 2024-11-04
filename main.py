# hindsight/main.py

# Importing necessary classes from the data_layer and coords modules
from data_layer.tensor import Tensor, ReturnsTensor, CharacteristicsTensor
from data_layer.coords import Coordinates
from data_layer.data_manager import DataLoader

# Importing JAX's NumPy module for numerical operations
import jax.numpy as jnp
import jax


def main():
    """
    Main function to demonstrate the initialization and usage of Coordinates.
    This function sets up coordinate variables, initializes a Coordinates instance,
    and prints out the coordinate details.
    """
    
    # Example of initializing Coordinates with numerical and categorical data
    coord_vars = {
        'time': jnp.arange(100),                  # Numerical coordinate: time indices from 0 to 99
        'asset': ['AAPL', 'GOOG', 'MSFT'],        # Categorical coordinate: asset names
        'feature': ['price', 'volume']            # Categorical coordinate: feature names
    }

    # Initialize the Coordinates object
    coords = Coordinates(coord_vars)

    # Accessing coordinate variables
    time_coords = coords['time']
    asset_coords = coords['asset']

    # Iterating over dimension names
    for dim in coords:
        print(f"Dimension: {dim}, Values: {coords[dim]}") 

    # Simulate some return data
    data = jax.random.normal(jax.random.PRNGKey(0), (100, 3, 1), dtype=jnp.float32)  # Shape: (time, asset, feature)
    coord_vars = {
        'time': jnp.arange(100),
        'asset': ['AAPL', 'GOOG', 'MSFT'],
        'feature': ['return']
    }
    coords = Coordinates(coord_vars)

    # Create a ReturnsTensor instance
    returns_tensor = ReturnsTensor(data=data, Coordinates=coords)

    # Select a subset of assets
    selected_returns = returns_tensor.select('asset', [0, 2])  # Select 'AAPL' and 'MSFT'

    # View the shape of the selected data
    print(selected_returns.data.shape)  # Should reflect the reduced asset dimension


    # Simulate characteristic data
    data = jax.random.normal(jax.random.PRNGKey(0), (100, 3, 5), dtype=jnp.float32)  # Shape: (time, asset, feature)
    feature_names = ('PE_ratio', 'momentum', 'volatility', 'liquidity', 'size')
    coord_vars = {
        'time': jnp.arange(100),
        'asset': ['AAPL', 'GOOG', 'MSFT'],
        'feature': feature_names
    }
    coords = Coordinates(coord_vars)

    # Create a CharacteristicsTensor instance
    characteristics_tensor = CharacteristicsTensor(
        data=data,
        dimensions=('time', 'asset', 'feature'),
        feature_names=feature_names,
        Coordinates=coords
    )

    # Select specific features
    selected_characteristics = characteristics_tensor.select('feature', [0, 2])  # Select 'PE_ratio' and 'volatility'

    # View the feature names after selection
    print(selected_characteristics.feature_names)  # Output: ('PE_ratio', 'volatility')

    
if __name__ == "__main__":
    main()