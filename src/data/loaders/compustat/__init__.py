# data/loaders/compustat/__init__.py

from ...provider import Provider, register_provider

# Fetchers
from .fundv import CompustatDataFetcher

# Define the provider
compustat_provider = Provider(
    name="compustat",
    website="https://wrds-www.wharton.upenn.edu/pages/get-data/compustat-capital-iq-standard-poors/compustat/",
    description="""Compustat is a comprehensive database of financial, statistical, 
    and market information on active and inactive global companies throughout the world. This data loader pulls data from
    a local source.""",
    fetcher_dict={
        "equities/fundamental": CompustatDataFetcher,
    },
    repr_name="Compustat",
)

# Register the provider
register_provider(compustat_provider)