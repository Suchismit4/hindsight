# hindsight/example_crspastensors.py

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
    
    c_tensor, r_tensor = data_loader.load_cache(freq='D')
        
    print("Is Apple (14593) in the tensor?", 14593 in c_tensor.Coordinates.variables['asset'])
    
    # Get the assets
    assets = jnp.array(c_tensor.Coordinates.variables['asset'])
    
    # Find the index of the asset we are searching for
    asset_idx = jnp.where(assets == 14593)[0][0]
        
    # Get the asset 14593
    asset_14593 = c_tensor.select('asset', int(asset_idx))

    # Get the Coordinates of the asset
    asset_coords = asset_14593.Coordinates
    print("Asset Coordinates: \n", asset_coords)

    # Get the prices
    prices = asset_14593.select('feature', 'adj_prc')
    
    print("Asset 14593 Prices: \n", prices.data.squeeze())
    
    # Compute the moving average of the prices
    # Define the EMA function (similar to example_uroll.py)
    @partial(jax.jit, static_argnames=['window_size'])
    def ema(i: int, carry, block: jnp.ndarray, window_size: int):
        
        if carry is None:
            # Initialization step
            initial_window = block[:window_size]
            initial_ema = initial_window.mean(axis=0)
            
            return initial_ema, initial_ema
        else:
            
            current_price = block[i]
            alpha = 1 / window_size
            
            new_ema = alpha * current_price + (1 - alpha) * carry
            return new_ema, new_ema

    # Compute EMA with a window size of 20 days
    window_size = 20
    ema_prices = jax.device_get(prices.u_roll(window_size, ema))
    print("EMA Prices: \n", ema_prices)

    # Convert data to numpy for plotting
    time_coords = asset_coords.variables['time']  # Ensure accessing 'variables'
    # Convert int64 unix time to pd.DatetimeIndex
    time_coords = pd.to_datetime(time_coords, unit='s') 
    price_data = jnp.asarray(prices.data).squeeze()
    ema_data = jnp.asarray(ema_prices.data).squeeze()

    # Plot the prices and moving average
    plt.figure(figsize=(16, 9), dpi=300)  # Increased figure size and DPI for higher quality
    plt.plot(time_coords, price_data, label='Price', linewidth=2)
    plt.plot(time_coords, ema_data, label=f'EMA ({window_size} days)', linewidth=2)
    plt.title('Apple (14593) Stock Price and EMA', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping

    # Save the plot as a high-quality PNG file
    plt.savefig('example_crspastensors.png', dpi=300, bbox_inches='tight')
    print("Plot saved as example_crspastensors.png")

    # plt.show()  # Commented out to avoid displaying the plot

if __name__ == "__main__":
    main()
