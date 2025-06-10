import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, NamedTuple
from datetime import datetime, timedelta
from enum import Enum
import xarray as xr

class BacktestState(NamedTuple):
    """
        This is a named tuple for the backtest state.
    """
    positions: xr.DataArray               # Shape (N,) - Current Positions [-1, 1] or weights
    cash: np.float64                      # Available cash
    total_portfolio_value: np.float64     # Total portfolio value
    timestamp_idx: int                    # Index of the current timestamp

class EventBasedStrategy(ABC):
    """
        This is an abstract class for all event based strategies.
        
        Methods like next() must be implemented by subclasses, which
        are user defined strategies.
    """
    
    def __init__(self):
        """
            Initialize the strategy.
        """
        self.name = "A Event based strategy"
        self.description = "A description of the strategy"
        self.author = "Author"
        self.version = "1.0.0"
        
        pass
    
    def __str__(self):
        """
            Return a string representation of the strategy.
        """
        return f"{self.name} - {self.description}"
    
    def __repr__(self):
        """
            Return a string representation of the strategy.
        """
        # TODO: Add a more detailed representation of the strategy
        return f"{self.name} - {self.description}"
    
    @abstractmethod
    def next(self, 
             market_observations: xr.Dataset,
             character_observations: xr.Dataset,
             state: BacktestState
             ) -> xr.DataArray:
        """
            This method is called by the engine to get the execute the 
            a rules based strategy on next observation of data.
        """
        
        raise NotImplementedError("next() must be implemented by subclasses")

"""
This is the main class for the backtesting engine.

This accepts an EventBasedStrategy, a dataset of observations and a dataset of characteristics.
The engine will then execute the strategy on the observations and characteristics.

The engine will call the next() method of the strategy per observation indexed. It will then
pass the observations and characteristics to the next() method based on the available observations
till that index. 

The next method will return a DataArray of Positions. The positions are the weights of the assets in the portfolio.
COMPLETE DOC LATER.
"""
class BacktestEngine:
    """
        This is the main class for the backtest engine.
    """
    
    def __init__(self, 
                 strategy: EventBasedStrategy,
                 observations: xr.Dataset,
                 characteristics: xr.Dataset,
                 ) -> None:
        """
            Initialize the backtest engine.
        """
        self.strategy = strategy
        self.observations = observations
        self.characteristics = characteristics
        
        # Create an empty positions DataArray of (N,)
        positions = xr.DataArray(
            data=np.zeros(len(observations.asset)),
            dims=("asset",),
            coords={"asset": observations.asset},
            attrs={"description": "Positions for each asset at time t_i."}
        )
        
        # Initialize the state with the initial cash and positions
        self.state = BacktestState(
            positions=positions, 
            cash=0.0, 
            total_portfolio_value=0.0, 
            timestamp_idx=0)
        
        # Copy the market and characteristics data to avoid modifying the original data
        self.observations = observations.copy()
        self.characteristics = characteristics.copy()
    
    def execute(self): 
        """ Execute the backtest given a strategy."""
        
        # XArray Dataset of observations is expected of shape (T_composite, N_assets, J=5)
        # T_composite is (Y, M, D, H) and J is the number of features.
        
        # Flatten the T_composite to a single time-indexed using dt.
        
        
        # Iterate over the observations and call the next method of the strategy
        # for i in range(0, len(self.observations.index)):
        pass