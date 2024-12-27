# data/loaders/openbb/__init__.py

from ...provider import Provider, register_provider
from .generic import GenericOpenBBDataFetcher
from openbb import obb

obb.user.preferences.output_type="dataframe"

def init_openbb():
    """
    Build a fetcher_dict from OpenBB's coverage providers.
    Then register this as the 'openbb' provider in your data system.
    """

    # obb.coverage.providers is typically a dict like:
    # {
    #   "yfinance": ["equities.price.historical", "crypto.price.historical", ...],
    #   "fmp":      ["equities.price.historical", "futures.historical", ...],
    #   ...
    # }
    
    coverage_dict = obb.coverage.providers

    # We only need the unique coverage paths (the user will pass 'provider' in config).
    # e.g. coverage_paths = {"equities.price.historical", "crypto.price.historical", "etf.info", ...}
    coverage_paths = set()
    for provider_name, path_list in coverage_dict.items():
        for coverage_path in path_list:
            coverage_paths.add(coverage_path)

    # Build the fetcher_dict: map slash-based paths -> GenericOpenBBDataFetcher
    # e.g. "equities.price.historical" => "equities/price/historical"
    fetcher_dict = {}
    for coverage_path in coverage_paths:
        slash_path = coverage_path.replace(".", "/")[1:] # remove the first slash
        fetcher_dict[slash_path] = GenericOpenBBDataFetcher
                
    # Define the provider
    openbb_provider = Provider(
        name="openbb",  # used in data_path as "openbb/..."
        website="https://openbb.co/",
        description="Integration with OpenBB's data providers",
        fetcher_dict=fetcher_dict,
        repr_name="openbb",
    )

    # Register the provider in your global registry
    register_provider(openbb_provider)

# Actually run the init
init_openbb()
