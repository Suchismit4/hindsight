# src/data/manager.py
from abc import ABC, abstractmethod
import xarray as xr
import yaml
import os
from .registry import data_loader_registry

class DataLoader(ABC):
    """
    Abstract base class for data loaders in the framework.
    
    This class defines the interface that all data loaders must implement.
    Each concrete data loader is responsible for loading specific types of data
    and converting them into xarray Datasets for standardized downstream processing.
    
    The class is part of a larger data management system that supports both base
    and derived data types, with extensibility for future compute modules.
    """
    
    @abstractmethod
    def load_data(self, **kwargs) -> xr.Dataset:
        """
        Load data and return it as an xarray Dataset.
        
        This abstract method must be implemented by all concrete data loaders.
        The implementation should handle the specific logic required to load
        and process the data from its source format into an xarray Dataset.
        
        Args:
            **kwargs: Additional arguments specific to the data loader implementation.
                     These might include date ranges, filters, or other parameters
                     needed to specify the exact data to be loaded.
        
        Returns:
            xr.Dataset: The loaded data in a standardized xarray Dataset format.
                       The structure should conform to the expectations defined
                       in the ontology configuration.
        """
        pass


class DataManager:
    """
    Central manager class for handling data loading and processing operations.
    
    This class serves as the main interface for clients to interact with the
    data framework. It coordinates between data loaders, the ontology configuration,
    and future compute modules to provide a unified data access layer.
    
    The manager maintains a mapping between data types and their corresponding
    loaders, manages the ontology configuration, and provides methods for
    data retrieval and discovery.
    """
    
    def __init__(self):
        """
        Initialize the DataManager with the built-in ontology configuration.
        
        The ontology configuration is loaded from a predefined path within the
        package, making it a hidden implementation detail rather than a
        user-configurable option.
        """
        # Define the path to the ontology file relative to this module
        ontology_path = os.path.join(
            os.path.dirname(__file__),
            'config',
            'ontology.yaml'
        )
        self.ontology = self.load_ontology(ontology_path)
    
    def load_ontology(self, ontology_path: str) -> dict:
        """
        Load and parse the ontology configuration file.
        
        Args:
            ontology_path: Path to the ontology YAML configuration file.
        
        Returns:
            dict: Parsed ontology configuration defining the relationships
                 between data types and their characteristics.
        
        Raises:
            FileNotFoundError: If the ontology configuration file cannot be found.
            yaml.YAMLError: If the ontology file contains invalid YAML syntax.
        """
        with open(ontology_path, 'r') as f:
            ontology = yaml.safe_load(f)
        return ontology
    
    def map_data_types_to_loaders(self) -> dict:
        """
        Create a mapping between data types and their corresponding loader classes.
        
        This method processes the registered data loaders and creates a lookup
        dictionary that allows quick access to the appropriate loader(s) for
        each data type.
        
        Returns:
            dict: Mapping from data type strings to lists of loader instances
                 capable of handling that data type.
        """
        mapping = {}
        for path, info in self.data_loaders.items():
            loader = info['loader']
            data_types = info['data_types']
            for data_type in data_types:
                if data_type in mapping:
                    mapping[data_type].append(loader)
                else:
                    mapping[data_type] = [loader]
        return mapping
    
    def get_data(self, data_type: str, provider: str = None, **kwargs) -> xr.Dataset:
        """
        Retrieve data of the specified type.
        
        This method serves as the main entry point for data retrieval. It handles
        both base and derived data types (though derived types are not yet
        implemented).
        
        Args:
            data_type: String identifier for the requested data type.
            **kwargs: Additional arguments to be passed to the data loader.
        
        Returns:
            xr.Dataset: The requested data.
        
        Raises:
            ValueError: If the data type is not found in the ontology or if no
                      suitable loader is available.
            NotImplementedError: If the requested data type is derived (currently
                               not supported).
        """
        
        self.data_loaders = data_loader_registry
        self.data_type_to_loader = self.map_data_types_to_loaders()
                
        if data_type not in self.ontology:
            raise ValueError(f"Data type '{data_type}' not found in ontology.")
        
        data_info = self.ontology[data_type]
        if data_info['type'] == 'base':
            return self.get_base_data(data_type, provider, **kwargs)
        elif data_info['type'] == 'derived':
            raise NotImplementedError(
                f"Derived data type '{data_type}' is not yet supported."
            )
        else:
            raise ValueError(f"Unknown data type '{data_type}'.")
    
    def get_base_data(self, data_type: str, provider: str = None, **kwargs) -> xr.Dataset:
        """
        Retrieve base (non-derived) data using the appropriate loader.
        
        Args:
            data_type: String identifier for the requested data type.
            **kwargs: Additional arguments to be passed to the data loader.
        
        Returns:
            xr.Dataset: The requested data.
        
        Raises:
            ValueError: If no loader is available for the requested data type
                      or if the loaded data doesn't contain the requested type.
        """
        
        if data_type not in self.data_type_to_loader:
            raise ValueError(
                f"No DataLoader available for data type '{data_type}'."
            )
        
        loaders = self.data_type_to_loader[data_type]

        if provider is None:
            loader = loaders[0]
            print("Warning: No provider was explicitly requested. Falling back to first available loaders.")
        else:
            loader = provider
        
        try:     
            return loader.load_data(**kwargs)
        except:
            raise FileNotFoundError("Failed to get a base dataset.")
        
    
    def list_available_data_types(self) -> list:
        """
        Get a list of all available data types defined in the ontology.
        
        Returns:
            list: List of string identifiers for available data types.
        """
        return list(self.ontology.keys())

