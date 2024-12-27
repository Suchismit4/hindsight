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

    def get_fetcher(self, fetcher_key: str):
        """
        Return the fetcher callable associated with the given fetcher_key,
        e.g. 'equity.price.historical' or 'crypto.price.historical'.
        """
        return self.fetcher_dict.get(fetcher_key)

    def __repr__(self):
        return f"<Provider {self.repr_name}>"

_PROVIDER_REGISTRY = {}

def register_provider(provider: Provider):
    _PROVIDER_REGISTRY[provider.name] = provider

def get_provider(name: str) -> Provider:
    return _PROVIDER_REGISTRY[name]