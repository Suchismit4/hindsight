# src/data/provider.py

from typing import Dict, Type
from .abstracts.base import BaseDataSource

class Provider:
    def __init__(
        self,
        name: str,
        website: str,
        description: str,
        fetcher_dict: Dict[str, Type[BaseDataSource]],
        repr_name: str = None,
    ):
        self.name = name
        self.website = website
        self.description = description
        self.fetcher_dict = fetcher_dict
        self.repr_name = repr_name or name

        # Initialize a dictionary to hold data loaders
        self.data_loaders = {}
        # Register the fetchers
        for fetcher_name, fetcher_class in fetcher_dict.items():
            data_path = f"{self.name}/{fetcher_name}"
            instance = fetcher_class(data_path)
            self.data_loaders[data_path] = instance

# Global provider registry
provider_registry = {}

def register_provider(provider: Provider):
    provider_registry[provider.name] = provider
