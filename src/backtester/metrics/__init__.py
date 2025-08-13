from abc import ABC, abstractmethod
import xarray as xr
from typing import Any

from ..struct import BacktestState

class Metric(ABC):
    """
    Abstract base class for metrics.
    """
    _name: str = "Metric"
    
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, value: str):
        self._name = value

    @abstractmethod
    def calculate(self, 
                  all_positions: xr.DataArray, 
                  all_portfolio_values: xr.DataArray,
                  cumulative_returns: xr.DataArray, 
                  state: BacktestState) -> Any:
        pass