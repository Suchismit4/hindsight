# data/loaders/yfinance/__init__.py

from ...provider import Provider, register_provider

# Import fetchers (data loaders)
from .market.historical import YFinanceEquityHistoricalFetcher

# Define the provider
yfinance_provider = Provider(
    name="yfinance",
    website="https://finance.yahoo.com",
    description=""" Yahoo! Finance is a web-based platform that offers financial news,
                    data, and tools for investors and individuals interested in tracking and analyzing
                    financial markets and assets.""",
    fetcher_dict={
        "equities/market/historical": YFinanceEquityHistoricalFetcher,
    },
    repr_name="Yahoo Finance",
)

register_provider(yfinance_provider)
