from . import Metric
import xarray as xr
import numpy as np

from ..struct import BacktestState

class ProfitFactor(Metric):
    """
    Profit factor metric based on bar-by-bar portfolio value changes, 
    aligned with Timothy Masters' approach.
    """
    
    _name = "ProfitFactor"
    
    def calculate(self, all_positions: xr.DataArray, all_portfolio_values: xr.DataArray, 
                  cumulative_returns: xr.DataArray, state: BacktestState) -> float:
        """
        Calculate the profit factor as the ratio of total positive portfolio value 
        changes to total absolute negative portfolio value changes.

        Args:
            all_positions: DataArray of positions over time (time, asset).
            all_portfolio_values: DataArray of portfolio values over time (time).
            cumulative_returns: DataArray of cumulative log returns (time).
            state: Final backtest state.

        Returns:
            float: Profit factor value.
        """
        # Compute portfolio value differences between consecutive bars
        delta_pv = all_portfolio_values.diff(dim='time')
        
        # Sum positive changes (profits) and absolute negative changes (losses)
        profits = delta_pv.where(delta_pv > 0, 0).sum().item()
        losses = (-delta_pv.where(delta_pv < 0, 0)).sum().item()
        
        # Handle special cases
        if profits == 0 and losses == 0:
            return 1.0  # Neutral strategy (no change in portfolio value)
        elif losses == 0:
            return float('inf')  # Perfect strategy (no losses)
        else:
            return profits / losses  # Standard profit factor

class SharpeRatio(Metric):
    _name = "SharpeRatio"

    def __init__(self, riskfreerate=0.01, sessions_per_year=252):
        self.riskfreerate = riskfreerate
        self.sessions_per_year = sessions_per_year

    def calculate(self, all_positions: xr.DataArray, all_portfolio_values: xr.DataArray,
                  cumulative_returns: xr.DataArray, state: BacktestState) -> float:
        import pandas as pd

        # to pandas for easy daily grouping (assumes 'time' is datetime64)
        pv = pd.Series(all_portfolio_values.values,
                       index=pd.DatetimeIndex(all_portfolio_values.time.values))

        # Daily end-of-day portfolio value
        daily_pv = pv.groupby(pv.index.date).last()
        daily_ret = daily_pv.pct_change().dropna()

        if daily_ret.size < 2:
            return 0.0

        # Per-period rf (period = 1 trading day)
        rf_per_day = (1.0 + self.riskfreerate) ** (1.0 / self.sessions_per_year) - 1.0
        excess = daily_ret - rf_per_day

        mean_excess = excess.mean()
        std_excess = excess.std(ddof=1)  

        if std_excess == 0 or np.isnan(std_excess):
            return 0.0

        sharpe = mean_excess / std_excess
        return float(sharpe * np.sqrt(self.sessions_per_year))

class MaxDrawdown(Metric):
    """
    Maximum drawdown metric.
    """
    
    _name = "MaxDrawdown"
    
    def calculate(self, all_positions: xr.DataArray, all_portfolio_values: xr.DataArray, 
                  cumulative_returns: xr.DataArray, state: BacktestState) -> float:
        """
        Calculate the maximum drawdown as the largest percentage drop from a peak to a trough.

        Args:
            all_positions: DataArray of positions over time (time, asset).
            all_portfolio_values: DataArray of portfolio values over time (time).
            cumulative_returns: DataArray of cumulative log returns (time).
            state: Final backtest state.

        Returns:
            float: Maximum drawdown value (negative percentage).
        """
        # Compute running maximum of portfolio values
        running_max_values = np.maximum.accumulate(all_portfolio_values.values)
        running_max = xr.DataArray(
            running_max_values,
            dims=['time'],
            coords={'time': all_portfolio_values.time.values}
        )
        
        # Compute drawdown at each point
        drawdown = (all_portfolio_values - running_max) / running_max
        
        # Find the maximum drawdown (most negative value)
        max_drawdown = drawdown.min().item()
        
        return max_drawdown

class UlcerIndex(Metric):
    """
    Ulcer Index metric.
    """
    
    _name = "UlcerIndex"
    
    def calculate(self, all_positions: xr.DataArray, all_portfolio_values: xr.DataArray, 
                  cumulative_returns: xr.DataArray, state: BacktestState) -> float:
        """
        Calculate the Ulcer Index as the square root of the average squared percentage drawdowns.

        Args:
            all_positions: DataArray of positions over time (time, asset).
            all_portfolio_values: DataArray of portfolio values over time (time).
            cumulative_returns: DataArray of cumulative log returns (time).
            state: Final backtest state.

        Returns:
            float: Ulcer Index value.
        """
        # Compute running maximum of portfolio values
        running_max_values = np.maximum.accumulate(all_portfolio_values.values)
        running_max = xr.DataArray(
            running_max_values,
            dims=['time'],
            coords={'time': all_portfolio_values.time.values}
        )
        
        # Compute percentage drawdown at each point
        drawdown = (all_portfolio_values - running_max) / running_max * 100  # in percentage
        
        # Compute squared drawdowns
        squared_drawdown = drawdown ** 2
        
        # Compute average squared drawdown
        avg_squared_drawdown = squared_drawdown.mean().item()
        
        # Ulcer Index is the square root of the average squared drawdown
        ulcer_index = np.sqrt(avg_squared_drawdown)
        
        return ulcer_index

class MartinRatio(Metric):
    """
    Martin Ratio (Ulcer Performance Index) metric.
    """
    
    _name = "MartinRatio"
    
    def __init__(self, riskfreerate=0.01, periods_per_year=252):
        """
        Initialize the MartinRatio metric.

        Parameters:
        - riskfreerate (float): Annual risk-free rate (default is 0.01 or 1%).
        - periods_per_year (int): Number of periods in a year (default is 252 for daily data).
        """
        self.riskfreerate = riskfreerate
        self.periods_per_year = periods_per_year
    
    def calculate(self, all_positions: xr.DataArray, all_portfolio_values: xr.DataArray, 
                  cumulative_returns: xr.DataArray, state: BacktestState) -> float:
        """
        Calculate the Martin Ratio as the average excess return divided by the Ulcer Index.

        Args:
            all_positions: DataArray of positions over time (time_flat, asset).
            all_portfolio_values: DataArray of portfolio values over time (time_flat).
            cumulative_returns: DataArray of cumulative log returns (time_flat).
            state: Final backtest state.

        Returns:
            float: Martin Ratio value.
        """
        # Compute per-bar returns as percentage change
        prev_values = all_portfolio_values.shift(time=1)
        returns = (all_portfolio_values / prev_values) - 1
        
        # Convert annual risk-free rate to per-period rate
        period_rate = (1 + self.riskfreerate) ** (1 / self.periods_per_year) - 1
        
        # Compute excess returns
        excess_returns = returns - period_rate
        
        # Remove NaN values (e.g., first bar where prev_value is NaN)
        valid_excess_returns = excess_returns.dropna(dim='time')
        
        # Compute average excess return
        avg_excess_return = valid_excess_returns.mean().item()
        
        # Compute Ulcer Index
        running_max_values = np.maximum.accumulate(all_portfolio_values.values)
        running_max = xr.DataArray(
            running_max_values,
            dims=['time'],
            coords={'time': all_portfolio_values.time.values}
        )
        drawdown = (all_portfolio_values - running_max) / running_max * 100  # in percentage
        squared_drawdown = drawdown ** 2
        avg_squared_drawdown = squared_drawdown.mean().item()
        ulcer_index = np.sqrt(avg_squared_drawdown)
        
        # Handle division by zero
        if ulcer_index == 0:
            return float('inf') if avg_excess_return > 0 else 0.0
        
        # Compute Martin Ratio
        martin_ratio = avg_excess_return / ulcer_index
        
        return martin_ratio