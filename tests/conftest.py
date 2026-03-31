"""
Shared pytest fixtures for the dev/ test suite.
All fixtures use simulated data so tests run offline and deterministically.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.core.types import FrequencyType
from src.data.loaders.table import from_table, load_simulated_data


@pytest.fixture(scope="session")
def simulated_daily_ds():
    """3 assets, 60 daily timesteps, 3 numeric variables."""
    np.random.seed(42)
    return load_simulated_data(
        num_assets=3,
        num_timesteps=60,
        num_vars=3,
        freq=FrequencyType.DAILY,
        start_date="2023-01-02",
    )


@pytest.fixture(scope="session")
def simulated_monthly_ds():
    """2 assets, 24 monthly timesteps, 2 numeric variables."""
    np.random.seed(0)
    return load_simulated_data(
        num_assets=2,
        num_timesteps=24,
        num_vars=2,
        freq=FrequencyType.MONTHLY,
        start_date="2021-01-01",
    )


@pytest.fixture(scope="session")
def ohlcv_ds():
    """2 assets, 100 daily OHLCV bars with valid OHLCV constraints."""
    np.random.seed(7)
    assets = ["AAA", "BBB"]
    dates = pd.date_range("2022-01-03", periods=100, freq="D")

    rows = []
    for asset in assets:
        for dt in dates:
            close = float(np.random.uniform(90, 110))
            open_ = float(np.random.uniform(close * 0.98, close * 1.02))
            high = float(max(open_, close) * np.random.uniform(1.0, 1.03))
            low = float(min(open_, close) * np.random.uniform(0.97, 1.0))
            volume = float(np.random.randint(1000, 10000))
            rows.append(
                {
                    "time": dt,
                    "asset": asset,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

    df = pd.DataFrame(rows)
    return from_table(
        data=df,
        time_column="time",
        asset_column="asset",
        frequency=FrequencyType.DAILY,
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (crypto / large data)")
