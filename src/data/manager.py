# src/data/manager.py

from abc import ABC, abstractmethod
import xarray as xr
from xarray import DataTree
from typing import Union, List, Dict, Any
import os
from .registry import data_loader_registry

class DataLoader(ABC):
    """
    Abstract base class for data loaders in the framework.

    This class defines the interface that all data loaders must implement.
    Each concrete data loader is responsible for loading specific types of data
    and converting them into xarray Datasets or DataTrees for standardized downstream processing.
    """

    @abstractmethod
    def load_data(self, **kwargs) -> Union[xr.Dataset, xr.DataTree]:
        """
        Load data and return it as an xarray Dataset or DataTree.

        This abstract method must be implemented by all concrete data loaders.
        The implementation should handle the specific logic required to load
        and process the data from its source format into an xarray Dataset or DataTree.

        Args:
            **kwargs: Additional arguments specific to the data loader implementation.

        Returns:
            Union[xr.Dataset, xr.DataTree]: The loaded data in a standardized format.
        """
        pass


class DataManager:
    """
    Central manager class for handling data loading and processing operations.

    This class serves as the main interface for clients to interact with the
    data framework. It coordinates between data loaders to provide a unified data access layer.
    """

    def __init__(self):
        """
        Initialize the DataManager.

        The manager loads the registry of data loaders upon initialization.
        """
        self.data_loaders = data_loader_registry

    def get_data(self, data_requests: List[Dict[str, Any]]) -> xr.DataTree:
        """
        Retrieve data for the specified data paths with their configurations.

        This method accepts a list of data requests, where each request is a
        dictionary containing the 'data_path' and its specific 'config'.

        Args:
            data_requests: A list of dictionaries, each containing:
                - 'data_path': The data path string.
                - 'config': A dictionary of configuration parameters for the data loader.

        Returns:
            xr.DataTree: The requested data merged into a single DataTree.

        Raises:
            ValueError: If no suitable loader is available for a data path.
        """
        data_dict = {}
        for request in data_requests:
            data_path = request.get('data_path')
            config = request.get('config', {})

            if data_path not in self.data_loaders:
                raise ValueError(f"No DataLoader available for data path '{data_path}'.")

            loader = self.data_loaders[data_path]
            data = loader.load_data(**config)

            if isinstance(data, xr.Dataset):
                data_dict[data_path] = data
            elif isinstance(data, xr.DataTree):
                # Convert the DataTree to a dictionary and merge paths
                tree_dict = self.data_tree_to_dict(data, prefix=data_path)
                data_dict.update(tree_dict)
            else:
                raise TypeError("DataLoader returned an unsupported data type.")

        # Create the DataTree from the data_dict
        merged_tree = DataTree.from_dict(data_dict)

        return merged_tree

    def data_tree_to_dict(self, tree: xr.DataTree, prefix: str = "") -> Dict[str, xr.Dataset]:
        """
        Convert a DataTree into a dictionary mapping paths to datasets.

        Args:
            tree: The DataTree to convert.
            prefix: A string to prefix to each path (used for data paths).

        Returns:
            A dictionary mapping paths to datasets.
        """
        data_dict = {}
        for node in tree.subtree:
            path = node.path
            if node.ds is not None:
                # Build the full path by combining the prefix and node path
                if path in ['', '/', '.', './']:
                    full_path = prefix
                else:
                    full_path = os.path.join(prefix, path.strip('/')).replace('\\', '/')
                data_dict[full_path] = node.ds
        return data_dict

    def list_available_data_paths(self) -> list:
        """
        Get a list of all available data paths in the registry.

        Returns:
            list: List of string identifiers for available data paths.
        """
        return list(self.data_loaders.keys())
