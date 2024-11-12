import numpy as np
import pandas as pd
import time
from .tensor import Tensor, ReturnsTensor, CharacteristicsTensor
import jax
import jax.numpy as jnp
from typing import List, Tuple
from functools import partial

from .coords import Coordinates
from numba import jit

jax.config.update("jax_enable_x64", True)
import equinox as eqx

CACHE_PATH = "~/data/cache/crsp/"
EXCLUSION_FEATURE_LIST = ['date', 'permno', 'permco', 'hsiccd', 'hexcd', 'cusip', 'issuno', 'altprcdt']

class TensorCollection(eqx.Module):
    """
    Represents a collection of processed tensors, providing utilities for handling multiple tensors.
    This class is intended to manage data prepared for pre-processing or feature engineering.
    
    Moreover, it is intended to manage and prepare data for specific strategies. There is an
    upper-level abstraction to this which is more specific to Algorithms, and having a more 
    specific data manager. 
    
    This class serves as a computational hub where derived financial metrics are calculated
    and stored. For example, it can compute and store metrics such as book-to-market ratio,
    price-based characteristics, and other strategy-agnostic financial ratios. These computations
    are performed here to maintain a clear separation between raw data processing and 
    strategy-specific operations.

    The upper-level abstraction (e.g., a Algorithm layer) would then use this TensorCollection
    to create more specialized, strategy-specific DataSets.
    """
    def __init__(self):
        super().__init__()
        raise NotImplementedError("This class is not yet implemented.")

    # Additional methods for other financial computations

class DataLoader():
    """
    Responsible for loading data from various sources and creating tensor instances.
    Handles data retrieval, caching, and synchronization across different data sources.
    """
    def __init__(self):
        super().__init__()
        pass
        # Future initialization code (e.g., setting up data sources, initializing caches)

    def _generate_random_data(self, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Generates random data for simulation purposes.

        Args:
            shape (Tuple[int, ...]): Shape of the data array to generate.

        Returns:
            np.ndarray: Generated random data as a JAX array.
        """
        return jax.random.normal(jax.random.PRNGKey(0), shape, dtype=np.float32)

    def simulate_load_returns(self) -> ReturnsTensor:
        """
        Simulates loading return data by generating random data.

        Returns:
            ReturnsTensor: Simulated returns tensor.
        """
        data = self._generate_random_data((10_000, 3, 1))  # JAX array

        coord_vars = {
            'time': np.arange(10_000),
            'asset': ['AAPL', 'GOOG', 'MSFT'],
            'feature': ['return']
        }

        coords = Coordinates(variables=coord_vars)

        return ReturnsTensor(
            data=np.array(data),
            Coordinates=coords
        )
        
    def _populate_tensor(self, df: pd.DataFrame, features: List[str], tensor_shape: Tuple[int, int, int],
                         unique_permnos: np.ndarray, date_range) -> np.ndarray:
        """
        Populate the 3D tensor with feature data using vectorized operations.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            features (List[str]): List of feature columns to include.
            tensor_shape (Tuple[int, int, int]): Shape of the tensor (T, N, J).
            unique_permnos (np.ndarray): Array of unique permno identifiers.
            date_range (np.ndarray): Array of dates.

        Returns:
            np.ndarray: Populated tensor as a NumPy array.
        """
        
        @jit(nopython=True)
        def fast_populate(date_indices, permno_indices, feature_values, tensor):
            for i in range(len(date_indices)):
                tensor[date_indices[i], permno_indices[i], :] = feature_values[i]
            return tensor
        
        T, N, J = tensor_shape

        # Create index mappings for dates and permnos
        date_index = pd.Index(date_range)
        permno_index = pd.Index(unique_permnos)

        # Map dates and permnos to their respective indices
        df['date_idx'] = date_index.get_indexer(df['date'])
        df['permno_idx'] = permno_index.get_indexer(df['permno'])

        # Filter out invalid indices
        valid_mask = (df['date_idx'] >= 0) & (df['permno_idx'] >= 0)
        df_valid = df.loc[valid_mask, :]

        # Extract indices and feature values
        date_indices = df_valid['date_idx'].values
        permno_indices = df_valid['permno_idx'].values
        feature_values = df_valid[features].values.astype(np.float32)

        # Initialize tensor with NaNs
        tensor = np.full((T, N, J), np.nan, dtype=np.float32)

        # Assign feature values using advanced indexing
        # tensor[date_indices, permno_indices, :] = feature_values
        tensor = fast_populate(date_indices, permno_indices, feature_values, tensor)

        # Replace infinite values with NaN
        tensor = np.where(np.isinf(tensor), np.nan, tensor)

        return tensor
        
    def _load_crsp_from_parquet(self, cache_path: str, freq: str) -> Tuple[CharacteristicsTensor, ReturnsTensor]:
        """
        Loads CRSP data from a Parquet file, processes it, and populates characteristics and returns tensors.

        Args:
            cache_path (str): Path to the cached Parquet file.
            freq (str): Frequency for the date range (e.g., 'D' for daily).

        Returns:
            Tuple[CharacteristicsTensor, ReturnsTensor]: Populated characteristics and returns tensors.
        """
        df = pd.read_parquet(cache_path)
        
        # Convert 'date' to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Drop duplicates and rows with missing 'date' or 'permno'
        df = df.drop_duplicates(subset=['date', 'permno'], keep='last').reset_index(drop=True)
        df = df.dropna(subset=['date', 'permno']) 
        
        # Identify unique permnos and relevant features
        unique_permnos = df['permno'].unique()
        unique_permnos.sort()
        column_names = [col for col in df.columns if col not in EXCLUSION_FEATURE_LIST]
        
        # Retain only necessary columns to reduce memory usage
        df = df[['date', 'permno'] + column_names]
        
        # Use the unique dates from the DataFrame and convert to int64 Unix time
        date_range = np.sort(df['date'].unique())
        date_range_unix = date_range.astype(np.int64) // 10**9  # Convert to Unix timestamp in seconds
        
        # Define features for characteristics and returns
        c_features = [col for col in column_names if col != 'ret']
        r_features = ['ret']
        
        # Correct adjustment of price
        if 'prc' in df.columns and 'cfacpr' in df.columns:
            # https://wrds-www.wharton.upenn.edu/pages/support/support-articles/crsp/stock/adjusted-prices-stock-splits-and-dividends/
            df['adj_prc'] = (df['prc']) / df['cfacpr']
            c_features.append('adj_prc')

        # Create coordinates for characteristics
        c_coord_vars = {
            'time': date_range_unix,  # Use Unix timestamp
            'asset': unique_permnos,
            'feature': c_features
        }
        
        # Instantiate Coordinates for characteristics
        c_coords = Coordinates(variables=c_coord_vars)
        
        # Define tensor shape for characteristics
        T = len(date_range)
        N = len(unique_permnos)
        J_c = len(c_features)
        
        # Populate characteristics tensor
        c_tensor_data = self._populate_tensor(df, c_features, (T, N, J_c), unique_permnos, date_range)
        
        c_tensor = CharacteristicsTensor(
            data=c_tensor_data,
            dimensions=('time', 'asset', 'feature'), 
            feature_names=tuple(c_features),
            Coordinates=c_coords
        )
        
        # Create coordinates for returns
        r_coord_vars = {
            'time': date_range_unix,  # Use Unix timestamp
            'asset': unique_permnos,
            'feature': r_features
        }
        
        # Instantiate Coordinates for returns
        r_coords = Coordinates(variables=r_coord_vars)
        
        # Define tensor shape for returns
        J_r = len(r_features)  # Should be 1

        # Populate returns tensor
        r_tensor_data = self._populate_tensor(df, r_features, (T, N, J_r), unique_permnos, date_range)
        
        r_tensor = ReturnsTensor(
            data=r_tensor_data,
            Coordinates=r_coords
        )
        
        return c_tensor, r_tensor
        
    def load_cache(self, freq: str = 'M') -> Tuple[CharacteristicsTensor, ReturnsTensor]:
        """
        Loads all data sources from the cache. Currently, it only loads CRSP data.
        """
        # Currently only loads CRSP data 
        return self._load_crsp_from_parquet(CACHE_PATH + freq + '/data.parquet', freq=freq)
