# data/loaders/compustat/__init__.py

from ...provider import Provider, register_provider

# Fetchers
from .compustat import CompustatDataFetcher
from .crsp import CRSPDataFetcher

# Define the provider
compustat_provider = Provider(
    name="wrds",
    website="https://wrds-www.wharton.upenn.edu/",
    description="""WRDS is a powerful data platform that provides access to a wide range of financial, statistical, and market data.
    It serves as a comprehensive resource for researchers, analysts, and academics, offering datasets from various sources, including Compustat, CRSP, and others. 
    This data loader facilitates the retrieval of data from WRDS, enabling users to efficiently access and analyze financial information on both active and inactive global companies.""",
    fetcher_dict={
        "equities/compustat": CompustatDataFetcher,
        "equities/crsp": CRSPDataFetcher,
    },
    repr_name="wrds",
)

# Register the provider
register_provider(compustat_provider)