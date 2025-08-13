import xarray as xr
from abc import ABC, abstractmethod
from typing import Any
from enum import Enum

class OrderStatus(Enum):
    """
    Status of an order.
    """
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    
class OrderDirection(Enum):
    """
    Type of an order.
    """
    BUY = "buy"
    SELL = "sell"

class Order(ABC):
    """
    Abstract base class for orders.
    """
    
    def __init__(self, asset: Any, quantity: float, price: float, timestamp: int):
        self.asset = asset
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp
    
    def __str__(self):
        return f"Order(asset={self.asset}, quantity={self.quantity}, price={self.price}, timestamp={self.timestamp})"
    
    def __repr__(self):
        return self.__str__()
    
    def set_price(self, price: float):
        self.price = price
    
    def set_quantity(self, quantity: float):
        self.quantity = quantity
    
    def set_timestamp(self, timestamp: int):
        self.timestamp = timestamp
    
    
class MarketOrder(Order):
    """
    Market order.
    """
    
    timestamp_updated: int
    
    def __init__(self, asset: Any, quantity: float, direction: OrderDirection, timestamp: int):
        super().__init__(asset, quantity, -1, timestamp)
        self.direction = direction
        self.timestamp_updated = None
    
    def set_execution_timestamp(self, timestamp: int):
        """Set the timestamp when this order was executed."""
        self.timestamp_updated = timestamp
    
    def __str__(self):
        return f"MarketOrder(asset={self.asset}, quantity={self.quantity}, timestamp created={self.timestamp}, timestamp executed={self.timestamp_updated}, price_executed={self.price})"
    
    def __repr__(self):
        return self.__str__()
    
    
class LimitOrder(Order):
    
    def __init__(self, asset: Any, quantity: float, price: float, timestamp: int):
        raise NotImplementedError("LimitOrder is not implemented yet")
    
    
class BacktestState:
    """Lightweight container for backtest state."""
    
    def __init__(
        self,
        positions: xr.DataArray,            # (N,) - current positions in units
        cash: float,                        # scalar
        total_portfolio_value: float,       # scalar
        timestamp_idx: int                  # scalar int
    ):
        self.positions = positions
        self.cash = cash
        self.total_portfolio_value = total_portfolio_value
        self.timestamp_idx = timestamp_idx
        
        # Track portfolio value history for accurate return calculation
        self.portfolio_values = []  # List of actual portfolio values over time
        self.timestamps = []        # Corresponding timestamps
