import os
import pandas as pd
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)

class DataCacher:
    """
    Manages data caching and loading for large datasets, reducing the need for repeated disk I/O.

    Attributes:
        cache_dir (str): Directory path for storing cached data.
    """
    def __init__(self, cache_dir: str = "~/data/cache/"):
        self.cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def save_to_cache(self, data: pd.DataFrame, filename: str) -> None:
        """
        Saves data to the cache directory as a Parquet (for DataFrames).

        Args:
            data (pd.DataFrame): Data to cache.
            filename (str): Filename for saving the cached data.
        """
        file_path = os.path.join(self.cache_dir, filename)
        if isinstance(data, pd.DataFrame):
            data.to_parquet(file_path, index=False)
            logging.info(f"DataFrame saved to {file_path}")
        else:
            raise TypeError("Unsupported data type for caching. Use DataFrame")

    def load_from_cache(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Loads data from cache if it exists, otherwise returns None.

        Args:
            filename (str): Filename of the cached data.

        Returns:
            Optional[pd.DataFrame]: Loaded data or None if file does not exist.
        """
        file_path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(file_path):
            logging.warning(f"Cache file {file_path} does not exist.")
            return None

        if filename.endswith(".parquet"):
            data = pd.read_parquet(file_path)
            logging.info(f"Loaded DataFrame from {file_path}")
            return data
        else:
            raise ValueError("Unsupported file format. Use .parquet for DataFrames")

    def clear_cache(self, filename: Optional[str] = None) -> None:
        """
        Clears a specific cached file or the entire cache directory.

        Args:
            filename (Optional[str]): Specific filename to delete; clears all if None.
        """
        if filename:
            file_path = os.path.join(self.cache_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Removed cached file: {file_path}")
            else:
                logging.warning(f"File {file_path} not found.")
        else:
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))
            logging.info("Cleared all cached files.")
