# data/loaders/wrds/__init__.py

"""
WRDS data loaders and provider registration.
"""

from src.data.core.provider import Provider, register_provider

# Fetchers
from .compustat import CompustatDataFetcher
from .crsp import CRSPDataFetcher

# Define the provider
wrds_provider = Provider(
    name="wrds",
    website="https://wrds-www.wharton.upenn.edu/",
    description="""WRDS is a powerful data platform that provides access to a wide range of financial, 
    statistical, and market data. This data loader facilitates the retrieval of data from WRDS, 
    enabling users to efficiently access and analyze financial information on both active 
    and inactive global companies.""",
    fetcher_dict={
        "equity/compustat": CompustatDataFetcher,
        "equity/crsp":      CRSPDataFetcher
    },
    repr_name="wrds",
)

# Register the provider
register_provider(wrds_provider)
