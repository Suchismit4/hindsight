import os
import pandas as pd
import numpy as np
from typing import List, Tuple
from .data_cacher import DataCacher
from .tensor import CharacteristicsTensor, ReturnsTensor
from .coords import Coordinates

EXCLUSION_FEATURE_LIST = ['date', 'permno', 'permco', 'hsiccd', 'hexcd', 'cusip', 'issuno', 'altprcdt']
CACHE_PATH = "~/data/cache/crsp/"

class DataLoader:
    """
    Responsible for loading CRSP data from the cache and converting it into CharacteristicsTensor and ReturnsTensor.
    """
    def __init__(self, cache_dir: str = CACHE_PATH):
        self.cacher = DataCacher(cache_dir)

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic preprocessing to clean and format the data.
        """
        # Convert 'date' to datetime, drop duplicates and rows with missing 'date' or 'permno'
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.drop_duplicates(subset=['date', 'permno']).dropna(subset=['date', 'permno'])
        return df

    def load_crsp_data(self, freq: str) -> Tuple[CharacteristicsTensor, ReturnsTensor]:
        """
        Loads CRSP data from cache, preprocesses it, and converts it into CharacteristicsTensor and ReturnsTensor.
        
        Args:
            freq (str): Frequency for the date range (e.g., 'D' for daily, 'M' for monthly).
        
        Returns:
            Tuple[CharacteristicsTensor, ReturnsTensor]: Populated characteristics and returns tensors.
        """
        file_name = f"{freq}_data.parquet"
        df = self.cacher.load_from_cache(file_name)

        if df is None:
            #TODO Need to load data from external source into cache if not found in cache
            raise FileNotFoundError(f"CRSP data with frequency '{freq}' not found in cache.")

        # Preprocess DataFrame
        df = self._preprocess_dataframe(df)

        # Identify unique permnos and features
        unique_permnos = df['permno'].unique()
        unique_permnos.sort()
        column_names = [col for col in df.columns if col not in EXCLUSION_FEATURE_LIST]
        
        date_range = np.sort(df['date'].unique())
        
        # Define features for characteristics and returns
        c_features = [col for col in column_names if col != 'ret']
        r_features = ['ret']

        # Price adjustment
        if 'prc' in df.columns and 'cfacpr' in df.columns:
            df['adj_prc'] = df['prc'] / df['cfacpr']
            c_features.append('adj_prc')

        # Coordinates for characteristics
        c_coord_vars = {
            'time': date_range,
            'asset': unique_permnos,
            'feature': c_features
        }
        c_coords = Coordinates(variables=c_coord_vars)

        # Tensor shape for characteristics
        T = len(date_range)
        N = len(unique_permnos)
        J_c = len(c_features)

        # Populate characteristics tensor
        c_tensor_data = self._populate_tensor(df, c_features, (T, N, J_c), unique_permnos, date_range)
        c_tensor = CharacteristicsTensor(data=c_tensor_data, dimensions=('time', 'asset', 'feature'), 
                                         feature_names=tuple(c_features), Coordinates=c_coords)

        # Coordinates for returns
        r_coord_vars = {
            'time': date_range,
            'asset': unique_permnos,
            'feature': r_features
        }
        r_coords = Coordinates(variables=r_coord_vars)

        # Tensor shape for returns
        J_r = len(r_features)  # Usually 1 for returns

        # Populate returns tensor
        r_tensor_data = self._populate_tensor(df, r_features, (T, N, J_r), unique_permnos, date_range)
        r_tensor = ReturnsTensor(data=r_tensor_data, Coordinates=r_coords)

        return c_tensor, r_tensor

    def _populate_tensor(self, df: pd.DataFrame, features: List[str], tensor_shape: Tuple[int, int, int],
                         unique_permnos: np.ndarray, date_range: np.ndarray) -> np.ndarray:
        """
        Populate a 3D tensor with feature data using vectorized operations.

        Args:
            df (pd.DataFrame): DataFrame with data to populate the tensor.
            features (List[str]): List of feature columns to include.
            tensor_shape (Tuple[int, int, int]): Shape of the tensor (T, N, J).
            unique_permnos (np.ndarray): Array of unique permno identifiers.
            date_range (np.ndarray): Array of dates.
        
        Returns:
            np.ndarray: Populated tensor.
        """
        T, N, J = tensor_shape
        date_index = pd.Index(date_range)
        permno_index = pd.Index(unique_permnos)

        df['date_idx'] = date_index.get_indexer(df['date'])
        df['permno_idx'] = permno_index.get_indexer(df['permno'])

        valid_mask = (df['date_idx'] >= 0) & (df['permno_idx'] >= 0)
        df_valid = df.loc[valid_mask, :]

        date_indices = df_valid['date_idx'].values
        permno_indices = df_valid['permno_idx'].values
        feature_values = df_valid[features].values.astype(np.float32)

        tensor = np.full((T, N, J), np.nan, dtype=np.float32)
        tensor[date_indices, permno_indices, :] = feature_values
        tensor = np.where(np.isinf(tensor), np.nan, tensor)

        return tensor
