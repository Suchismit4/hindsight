"""
Crypto data loaders and provider registration.
"""

from src.data.core.provider import Provider, register_provider

# Fetchers
from .local import LocalCryptoDataFetcher

# Define the provider
crypto_provider = Provider(
    name="crypto",
    website="https://www.binance.com/",
    description="""Local crypto data provider that loads cryptocurrency price data from CSV files.
    Data includes OHLCV (Open, High, Low, Close, Volume) information for various cryptocurrency pairs,
    typically sourced from exchanges like Binance. The data is stored locally as CSV files with
    hourly frequency.""",
    fetcher_dict={
        "spot/binance": LocalCryptoDataFetcher,
    },
    repr_name="crypto",
)

# Register the provider
register_provider(crypto_provider) 