import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, NamedTuple, Tuple
from datetime import datetime, timedelta
from enum import Enum
import xarray as xr
from tqdm import tqdm
import logging
import os
from pathlib import Path

from .struct import Order, MarketOrder, LimitOrder, OrderDirection, BacktestState
from .metrics import Metric


# Optional dependencies
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        """Fallback decorator when numba is not available."""
        def decorator(func):
            return func
        return decorator


class EventBasedStrategy(ABC):
    """
    Abstract base class for trading strategies.
    """
    
    def __init__(self, 
                 name: str = "Strategy",
                 description: str = "A trading strategy",
                 window_size: int = 1):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            description: Strategy description  
            window_size: Number of past observations to include (default=1 for current bar only)
        """
        self.name = name
        self.description = description
        self.window_size = window_size
        
    def asset_idx(self, ds: xr.Dataset, asset: str) -> int:
        if self._asset_to_idx is not None:
            return self._asset_to_idx[asset]
        # fallback once
        ai = int(np.where(ds.coords["asset"].values == asset)[0][0])
        if self._asset_to_idx is None: self._asset_to_idx = {asset: ai}
        return ai

    @staticmethod
    def series2(ds: xr.Dataset, var: str, asset_idx: int) -> Tuple[float, float]:
        v = ds[var].isel(asset=asset_idx).values  # shape (2,)
        return v[0], v[1]  # (prev, curr)

    @staticmethod
    def ffill2(prev: Optional[float], curr: Optional[float], last: Optional[float]):
        p = prev if np.isfinite(prev) else last
        c = curr if np.isfinite(curr) else last
        return p, c
    
    @abstractmethod
    def next(self, 
             market_data: xr.Dataset,           # Market data window
             characteristics: xr.Dataset,       # Characteristics data window
             state: BacktestState) -> Tuple[List[Order], List[str]]:
        """
        Strategy logic called for each bar.
        
        This method is called by the engine for each timestep. It receives
        xarray datasets for market data and characteristics.
        
        Args:
            market_data: Market data xarray dataset window
            characteristics: Characteristics data xarray dataset window
            state: Current backtest state
            
        Returns:
            List[Order]: List of orders to execute
        """
        raise NotImplementedError("Strategy must implement next() method")


class Broker:
    """
        Broker to handle the execution of orders 
        and track account state.
    """
    
    cash: float
    positions: xr.DataArray  # (N,) - current positions in units
    initial_cash: float
    commission_rate: float
    commission_rates: List[float]  # For testing multiple commission scenarios
    logger: Optional
    current_timestamp: int  # Current execution timestamp
    
    # Orders to be executed at the end of the day.
    pending_orders: List[Order]
    
    def __init__(self, initial_cash: float = 100000.0, commission_rate: float = 0.0005, 
                 commission_rates: Optional[List[float]] = None, logger=None):
        """
        Initialize broker.
        
        Args:
            initial_cash: Starting cash amount
            commission_rate: Default commission rate (0.0005 = 0.05% for Binance)
            commission_rates: List of commission rates to test (for sensitivity analysis)
            logger: Optional logger instance
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = None
        self.commission_rate = commission_rate
        self.commission_rates = commission_rates or [commission_rate]
        self.logger = logger
        self.current_timestamp = 0
        self.asset_to_idx = None  # injected by engine after precompute
        
        self.pending_orders = []
        
    def _set_positions(self, positions: xr.DataArray):
        """
        Set the positions of the broker.
        """
        self.positions = positions
        
    def _get_positions(self) -> xr.DataArray:
        """
        Get the positions of the broker.
        """
        return self.positions
    
    def _get_cash(self) -> float:   
        """
        Get the cash of the broker.
        """
        return self.cash
    
    def _get_total_portfolio_value(self, snapshot: xr.DataArray) -> float:  
        """
        Get the total portfolio value of the broker.
        
        Args:
            snapshot: Market data snapshot containing close prices
            
        Returns:
            Total portfolio value (cash + positions * prices)
        """
        # Extract close prices for portfolio valuation
        close_prices = snapshot['close']  # Shape: (N,)
        
        # self.positions has shape (N,), close_prices has shape (N,)
        # Both are 1D arrays, can multiply directly
        price_values = close_prices.values  # Shape: (N,)
        position_values = self.positions.values  # Shape: (N,)
        
        # Handle NaN prices by setting them to 0 for portfolio calculation
        valid_prices = np.where(np.isfinite(price_values), price_values, 0.0)
        
        # Calculate portfolio value: sum(positions * prices) + cash
        portfolio_value = float(np.sum(position_values * valid_prices)) + self.cash

        
        return portfolio_value
    
    def _get_nav(self, snapshot: xr.DataArray) -> float:
        """
        Get the net asset value of the broker.
        
        This is the same as the total portfolio value for now.
        """
        return self._get_total_portfolio_value(snapshot)
    
    def queue_orders(self, orders: List[Order]):
        """
        Queue the orders to be executed.
        """
        self.pending_orders.extend(orders)
        if self.logger and orders:
            self._log("debug", f"QUEUE | Added {len(orders)} orders (total pending: {len(self.pending_orders)})")
    
    def execute_orders_np(
        self,
        close_row_t1: np.ndarray,     # valuation closes at t+1 (NaN->0 already)
        open_row_t1:  np.ndarray,     # execution opens at t+1 (raw)
        date_t1,                      # e.g., numpy.datetime64 or str for logs
        asset_values: np.ndarray,     # engine's asset coord (N,)
    ) -> Tuple[xr.DataArray, float, float, float, float]:
        """
        NumPy-optimized version of execute_orders for day t+1.
        Uses arrays for speed; returns the same tuple as execute_orders().
        """
        if len(self.pending_orders) == 0:
            # No orders: just value the portfolio (NumPy dot)
            pos_arr = self.positions.values  # view; do not copy
            tpv = float(self.cash + pos_arr @ close_row_t1)
            return (self.positions, self.cash, tpv, 0.0, 0.0)

        # Timestamp/log setup
        self._log("info", f"EXEC | Processing {len(self.pending_orders)} orders [D={date_t1}]")

        total_commission = 0.0
        total_cash_spent = 0.0
        available_cash = self.cash
        trades_executed = 0
        orders_processed = 0

        # Work on a NumPy copy of positions; convert back to xarray at the end
        pos_arr = self.positions.values.copy()

        # Consume the queue in a drain pattern (no pop(0) churn)
        orders = self.pending_orders
        self.pending_orders = []
        requeue = []

        for order in orders:
            orders_processed += 1

            if not hasattr(order, 'direction'):
                self._log("warning", f"EXEC | Skipping non-market order: {order}")
                requeue.append(order)
                continue

            asset = order.asset
            try:
                asset_idx = self.asset_to_idx[asset] if self.asset_to_idx is not None else \
                            {a: i for i, a in enumerate(asset_values.tolist())}[asset]
            except KeyError:
                self._log("warning", f"EXEC | Order #{orders_processed}: Asset {asset} not in market data")
                requeue.append(order)
                continue

            _execution_price = float(open_row_t1[asset_idx])

            # Skip invalid execution prices
            if not np.isfinite(_execution_price) or _execution_price <= 0.0:
                self._log("warning", f"EXEC | Order #{orders_processed}: Invalid price for {asset}: {_execution_price}")
                requeue.append(order)
                continue

            current_position = float(pos_arr[asset_idx])
            requested_quantity = order.quantity
            direction = order.direction

            if direction.value == 'buy':
                notional_value = requested_quantity * _execution_price
                commission = self._calculate_commission(notional_value)
                total_cost = notional_value + commission

                if total_cost <= available_cash:
                    executed_quantity = requested_quantity
                    actual_notional = executed_quantity * _execution_price
                    actual_commission = self._calculate_commission(actual_notional)
                else:
                    if available_cash <= 0:
                        self._log("warning", f"EXEC | Order #{orders_processed}: No cash for BUY {asset}")
                        continue
                    max_affordable_quantity = available_cash / (_execution_price * (1 + self.commission_rate))
                    executed_quantity = max(0.0, max_affordable_quantity)
                    if executed_quantity < 1e-6:
                        self._log("warning", f"EXEC | Order #{orders_processed}: Insufficient cash for {asset} "
                                            f"(need ${total_cost:.2f}, have ${available_cash:.2f})")
                        continue
                    actual_notional = executed_quantity * _execution_price
                    actual_commission = self._calculate_commission(actual_notional)

                new_position = current_position + executed_quantity
                pos_arr[asset_idx] = new_position
                available_cash -= (actual_notional + actual_commission)
                total_cash_spent += actual_notional
                total_commission += actual_commission

                self._update_market_order(order, _execution_price, executed_quantity)
                fill_status = "FULL" if executed_quantity == requested_quantity else "PARTIAL"
                self._log("info", f"TRADE | {fill_status} BUY {executed_quantity:.6f}/{requested_quantity:.6f} {asset} @ {_execution_price:.6f} | "
                                f"Pos: {current_position:.6f}→{new_position:.6f} | Cash: ${available_cash:.2f}")

            elif direction.value == 'sell':
                executed_quantity = requested_quantity
                actual_notional = executed_quantity * _execution_price
                actual_commission = self._calculate_commission(actual_notional)

                new_position = current_position - executed_quantity
                pos_arr[asset_idx] = new_position
                available_cash += (actual_notional - actual_commission)
                total_cash_spent -= actual_notional
                total_commission += actual_commission

                self._update_market_order(order, _execution_price, executed_quantity)
                position_status = "SHORT" if new_position < 0 else "LONG" if new_position > 0 else "FLAT"
                self._log("info", f"TRADE | SELL {executed_quantity:.6f} {asset} @ {_execution_price:.6f} | "
                                f"Pos: {current_position:.6f}→{new_position:.6f} ({position_status}) | Cash: ${available_cash:.2f}")

            trades_executed += 1
            
        if requeue:
            self.pending_orders.extend(requeue)
            self._log("debug", "REQUEUE | Carried {len(requeue)} orders to next bar")

        # Persist new positions as xarray (coords preserved)
        self.positions = xr.DataArray(pos_arr, dims=['asset'], coords={'asset': asset_values})
        self.cash = available_cash

        # Fast portfolio valuation (dot with pre-cleaned close_row_t1)
        tpv = float(self.cash + pos_arr @ close_row_t1)

        if trades_executed > 0:
            avg_commission_rate = (total_commission / abs(total_cash_spent)) * 100 if total_cash_spent != 0 else 0.0
            self._log("info", f"SUMMARY | Executed {trades_executed}/{orders_processed} orders | "
                            f"Commission: ${total_commission:.2f} ({avg_commission_rate:.3f}%) | "
                            f"Cash Flow: ${total_cash_spent:.2f} | PV: ${tpv:.2f} | Date: {date_t1}")
        else:
            self._log("info", f"SUMMARY | No trades executed ({orders_processed} orders processed)")

        return (self.positions, self.cash, tpv, total_commission, total_cash_spent)

    
    def execute_orders(self, snapshot: xr.DataArray, execution_price: xr.DataArray) -> Tuple[xr.DataArray, float, float, float, float]:
        """
        Execute pending orders at market open prices.
        
        EXECUTION TIMING: Orders placed at day t's close are executed using day t+1's opening prices.
        The strategy that generated these orders had NO access to t+1's opening prices.
        
        Args:
            snapshot: Market data snapshot containing prices at t+1
            execution_price: Execution price for the orders at t+1 (N,)
        Returns:
            Tuple of (positions, cash, total_portfolio_value, total_commission, cash_spent_on_trades)
        """
        if len(self.pending_orders) == 0:
            # No orders to execute, return current state
            tpv = self._get_total_portfolio_value(snapshot)
            return (
                self.positions,
                self.cash,
                tpv,
                0.0,  # total_commission
                0.0   # cash_spent_on_trades
            )
        
        # Increment timestamp for this execution cycle
        date = snapshot.coords['time'].data
        self._log("info", f"EXEC | Processing {len(self.pending_orders)} orders [D={date}]")
        
        # Extract open prices for execution - these are t+1's opening prices
        # open_prices = snapshot['open']  # Shape: (N,)
        
        # Initialize tracking variables
        total_commission = 0.0
        total_cash_spent = 0.0
        available_cash = self.cash
        trades_executed = 0
        orders_processed = 0
        
        # Work on a copy of positions for updates
        new_positions = self.positions.copy()
        
        # Process orders serially - important for cash constraint handling
        for order in self.pending_orders:
            orders_processed += 1
            
            if not hasattr(order, 'direction'):  # Not a MarketOrder
                self._log("warning", f"EXEC | Skipping non-market order: {order}")
                continue
                
            asset = order.asset
            requested_quantity = order.quantity
            direction = order.direction
            
            # Get the asset index and execution price
            try:
                asset_idx = list(execution_price.asset.values).index(asset)
                _execution_price = float(execution_price.isel(asset=asset_idx).values)
                
                # Skip if price is invalid (NaN, zero, or negative)
                if not np.isfinite(_execution_price) or _execution_price <= 0:
                    self._log("warning", f"EXEC | Order #{orders_processed}: Invalid price for {asset}: {_execution_price}")
                    continue
                    
            except (ValueError, IndexError):
                self._log("warning", f"EXEC | Order #{orders_processed}: Asset {asset} not in market data")
                continue
            
            # Get current position for this asset
            current_position = float(new_positions.values[asset_idx])
            
            # Calculate trade details based on direction
            if direction.value == 'buy':
                # BUY ORDER LOGIC
                notional_value = requested_quantity * _execution_price
                commission = self._calculate_commission(notional_value)
                total_cost = notional_value + commission
                
                # Check if we have enough cash
                if total_cost <= available_cash:
                    # Execute full order
                    executed_quantity = requested_quantity
                    actual_notional = executed_quantity * _execution_price
                    actual_commission = self._calculate_commission(actual_notional)
                else:
                    # Partial execution - calculate max affordable quantity
                    if available_cash <= 0:
                        self._log("warning", f"EXEC | Order #{orders_processed}: No cash for BUY {asset}")
                        continue
                    
                    # Solve: quantity * price * (1 + commission_rate) = available_cash
                    max_affordable_quantity = available_cash / (_execution_price * (1 + self.commission_rate))
                    executed_quantity = max(0, max_affordable_quantity)
                    
                    if executed_quantity < 0.000001:  # Minimum meaningful quantity
                        self._log("warning", f"EXEC | Order #{orders_processed}: Insufficient cash for {asset} "
                                           f"(need ${total_cost:.2f}, have ${available_cash:.2f})")
                        continue
                    
                    actual_notional = executed_quantity * _execution_price
                    actual_commission = self._calculate_commission(actual_notional)
                
                # Execute the buy
                new_position = current_position + executed_quantity
                new_positions[asset_idx] = new_position
                available_cash -= (actual_notional + actual_commission)
                total_cash_spent += actual_notional
                total_commission += actual_commission
                
                # Update order with execution details
                self._update_market_order(order, _execution_price, executed_quantity, execution_timestamp)
                
                # Log trade execution
                fill_status = "FULL" if executed_quantity == requested_quantity else "PARTIAL"
                self._log("info", f"TRADE | {fill_status} BUY {executed_quantity:.6f}/{requested_quantity:.6f} {asset} @ {_execution_price:.6f} | "
                                f"Pos: {current_position:.6f}→{new_position:.6f} | Cash: ${available_cash:.2f}")
                
            elif direction.value == 'sell':
                # SELL ORDER LOGIC (supports shorting)
                executed_quantity = requested_quantity  # Always execute full quantity (allow shorting)
                actual_notional = executed_quantity * _execution_price
                actual_commission = self._calculate_commission(actual_notional)
                
                # Execute the sell
                new_position = current_position - executed_quantity
                new_positions[asset_idx] = new_position
                available_cash += (actual_notional - actual_commission)  # Receive cash minus commission
                total_cash_spent -= actual_notional  # Negative for sells
                total_commission += actual_commission
                
                # Update order with execution details
                self._update_market_order(order, _execution_price, executed_quantity, execution_timestamp)
                
                # Log trade execution
                position_status = "SHORT" if new_position < 0 else "LONG" if new_position > 0 else "FLAT"
                self._log("info", f"TRADE | SELL {executed_quantity:.6f} {asset} @ {_execution_price:.6f} | "
                                f"Pos: {current_position:.6f}→{new_position:.6f} ({position_status}) | Cash: ${available_cash:.2f}")
            
            trades_executed += 1
        
            # This order is processed, remove it from the pending orders
            self.pending_orders.pop(0)
        
        # Update broker state
        self.positions = new_positions
        self.cash = available_cash

        tpv = self._get_total_portfolio_value(snapshot)
                
        # Execution summary
        if trades_executed > 0:
            avg_commission_rate = (total_commission / abs(total_cash_spent)) * 100 if total_cash_spent != 0 else 0
            self._log("info", f"SUMMARY | Executed {trades_executed}/{orders_processed} orders | "
                            f"Commission: ${total_commission:.2f} ({avg_commission_rate:.3f}%) | "
                            f"Cash Flow: ${total_cash_spent:.2f} | PV: ${tpv:.2f} | Date: {date}")
        else:
            self._log("info", f"SUMMARY | No trades executed ({orders_processed} orders processed)")
        return (
            self.positions,      # (N,) format
            self.cash,           # Updated cash after trades
            tpv,                 # Total portfolio value
            total_commission,    # Total commission paid
            total_cash_spent     # Net cash spent (positive=net buys, negative=net sells)
        )
    
    def _log(self, level: str, message: str):
        """Efficient logging method that only logs if logger is available."""
        if self.logger:
            getattr(self.logger, level.lower())(message)

    def _update_market_order(self, order: 'MarketOrder', executed_price: float, 
                           executed_quantity: float):
        """
        Update MarketOrder with execution details.
        
        Args:
            order: MarketOrder to update
            executed_price: Price at which order was executed
            executed_quantity: Quantity actually executed (may be partial)
        """
        order.set_price(executed_price)
        order.set_quantity(executed_quantity)
        
        # Log the order update at debug level only
        self._log("debug", f"ORDER | {order.direction.value.upper()} {order.asset} updated: "
                          f"qty={executed_quantity:.6f} @ price={executed_price:.6f}")
    
    def _calculate_commission(self, notional_value: float, commission_rate: float = None) -> float:
        """
        Calculate commission based on notional value.
        
        Args:
            notional_value: Trade value (quantity * price)
            commission_rate: Commission rate to use (defaults to self.commission_rate)
            
        Returns:
            Commission amount
        """
        rate = commission_rate if commission_rate is not None else self.commission_rate
        return abs(notional_value) * rate  # Ensure positive commission


class BacktestEngine:
    """
    Backtesting engine
    """
    
    strategy: EventBasedStrategy
    market_data: xr.Dataset
    characteristics: xr.Dataset
    commission_rate: float
    verbose: int
    log_file: Optional[str]
    broker: Broker
    metrics: List[Metric]
    
    def __init__(self, 
                 strategy: Optional[EventBasedStrategy] = None,
                 market_data: Optional[xr.Dataset] = None,
                 characteristics: Optional[xr.Dataset] = None,
                 initial_cash: float = 100000.0,
                 commission_rate: float = 0.0005,  # 0.05% default commission rate
                 commission_rates: Optional[List[float]] = None,  # For sensitivity analysis
                 date: Optional[Tuple[str, str]] = None,
                 verbose: int = 0,
                 fpass: int = 0,
                 log_file: Optional[str] = None):
        """
        Initialize the backtesting engine.
        
        Args:
            strategy: Optional strategy instance
            market_data: Optional market data dataset
            characteristics: Optional characteristics dataset  
            initial_cash: Starting cash amount
            commission_rate: Default commission rate (0.0005 = 0.05% for Binance)
            commission_rates: List of commission rates for sensitivity testing
            verbose: Enable detailed logging
            log_file: Optional log file path. If None, creates timestamped log in current directory
        """
        self.initial_cash = initial_cash
        self.strategy = strategy
        self.market_data = market_data
        self.characteristics = characteristics
        self.commission_rate = commission_rate
        self.commission_rates = commission_rates
        self.verbose = verbose
        self.fpass = fpass
        self.start_date, self.end_date = date
        
        # Setup logging if verbose mode is enabled
        if self.verbose > 0:
            self._setup_logging(log_file, self.verbose)
        
        # Pre-computed windows for efficient backtesting
        self.market_windows = []
        self.char_windows = []
        self.metrics = []
        
        # Verify data if provided
        if self.market_data is not None and self.characteristics is not None:
            self._verify_data()
            if self.strategy is not None:
                self._precompute_windows()
                
        # Initialize broker with commission sensitivity support
        broker_logger = self.logger if self.verbose > 0 and hasattr(self, 'logger') else None
        self.broker = Broker(
            initial_cash=initial_cash, 
            commission_rate=commission_rate,
            commission_rates=commission_rates,
            logger=broker_logger
        )
        
    def add_metric(self, metric: Metric):
        """Add a metric to the engine."""
        self.metrics.append(metric)
        
    def add_strategy(self, strategy: EventBasedStrategy):
        """Add a strategy to the engine."""
        if not isinstance(strategy, EventBasedStrategy):
            raise ValueError("Strategy must inherit from EventBasedStrategy")
        self.strategy = strategy
        
        # Pre-compute windows if data is already available
        if self.market_data is not None and self.characteristics is not None:
            self._precompute_windows()
            
        if hasattr(self, "strategy") and self.strategy is not None:
            setattr(self.strategy, "_asset_to_idx", self._asset_to_idx)     
        
    def add_data(self, market_data: xr.Dataset, characteristics: xr.Dataset):
        """Add market data and characteristics to the engine."""
        self.market_data = market_data
        self.characteristics = characteristics
        self._verify_data()
        
        # Pre-compute windows if strategy is already available
        if self.strategy is not None:
            self._precompute_windows()
    
    def _verify_data(self):
        """Verify that market data and characteristics are compatible."""
        if self.market_data is None or self.characteristics is None:
            return
            
        # Check that both datasets have time_flat dimension
        if 'time_flat' not in self.market_data.dims:
            raise ValueError("Market data must have 'time_flat' dimension")
        if 'time_flat' not in self.characteristics.dims:
            raise ValueError("Characteristics data must have 'time_flat' dimension")
            
        # Check that both datasets have asset dimension
        if 'asset' not in self.market_data.dims:
            raise ValueError("Market data must have 'asset' dimension")
        if 'asset' not in self.characteristics.dims:
            raise ValueError("Characteristics data must have 'asset' dimension")
            
        # Check that assets match
        market_assets = set(self.market_data.asset.values)
        char_assets = set(self.characteristics.asset.values)
        
        if market_assets != char_assets:
            raise ValueError("Market data and characteristics must have same assets")
            
        # Check that time_flat dimensions match
        if len(self.market_data.time_flat) != len(self.characteristics.time_flat):
            raise ValueError("Market data and characteristics must have same time_flat length")
            
        # Check for required market data variables (OHLCV in that order)
        required_market_vars = ['open', 'high', 'low', 'close', 'volume']
        missing_vars = [var for var in required_market_vars if var not in self.market_data.data_vars]
        if missing_vars:
            raise ValueError(f"Market data missing required variables: {missing_vars}")
            
        # Check that mask and mask_indices exist (needed for business day operations)
        if 'mask' not in self.market_data.coords:
            raise ValueError("Market data must have 'mask' coordinate")
        if 'mask_indices' not in self.market_data.coords:
            raise ValueError("Market data must have 'mask_indices' coordinate")

    def _precompute_windows(self):
        """Pre-compute all data windows for efficient backtesting.
        
        This happens in O(T) time, where T is the number of timesteps, upfront.
        """
        if self.strategy is None:
            return
            
        self.T = len(self.market_data.time_flat)
        self.window_size = self.strategy.window_size
        
        # Ensure we have enough data for the window
        if self.T < self.strategy.window_size:
            raise ValueError(f"Not enough data: need at least {self.window_size} timesteps, got {self.T}")
        
        # Convert to time-indexed format (stack time dimensions)
        self.market_time_indexed = self.market_data.dt.to_time_indexed()
        self.char_time_indexed   = self.characteristics.dt.to_time_indexed()
        
        self.market_time_indexed_ffilled = self.market_data.dt.to_time_indexed().ffill(dim="time")
        
        mt    = self.market_time_indexed.transpose('time', 'asset', ...)
        mt_ff = self.market_time_indexed_ffilled.transpose('time', 'asset', ...)
        
        # Execution uses raw opens; valuation uses filled closes.
        self._open_np = mt['open'].data
        self._close_np = mt_ff['close'].data # Shape (T, N)
        
        self._close_np = np.where(np.isfinite(self._close_np), self._close_np, 0.0)
        
        # Cache
        self._asset_values = self.market_data.asset.values # (N,)
        self._time_values  = mt_ff['time'].values # (T,)
        self._T = self._open_np.shape[0]
        self._N = self._open_np.shape[1]
        
        # One-time, deterministic index map
        self._asset_to_idx = {str(a): i for i, a in enumerate(self._asset_values.tolist())}

        # Make it available to the strategy
        if hasattr(self, "strategy") and self.strategy is not None:
            setattr(self.strategy, "_asset_to_idx", self._asset_to_idx)

        # Make it available to the broker
        if hasattr(self, "broker") and self.broker is not None:
            self.broker.asset_to_idx = self._asset_to_idx
        
        self._log("info", f"Prepared data for {self.T - self.window_size + 1} windows")       
    
    def run(self) -> Tuple[BacktestState, xr.DataArray]:
        """Execute the backtest using pre-computed windows.
        
        Returns:
            Tuple of (final_state, cumulative_log_returns)
        """
        if self.strategy is None:
            raise ValueError("Must add strategy before running")
        if self.market_data is None or self.characteristics is None:
            raise ValueError("Must add data before running")
            
        N = len(self.market_data.asset)
        
        # Initialize state with proper xarray DataArray for positions in units.
        # Use (N,) format for current positions - no time dimension needed for point-in-time data
        initial_positions = xr.DataArray(
            np.zeros(N), # holding 0 unit of each asset.
            dims=['asset'],
            coords={
                'asset': self.market_data.asset.values
            }
        )
        
        state = BacktestState(
            positions=initial_positions,
            cash=self.initial_cash,
            total_portfolio_value=self.initial_cash,
            timestamp_idx=0  # Start at day 0
        )
        
        # Initialize portfolio value tracking with starting value
        
        portfolio_values = np.zeros(self.T)
        portfolio_values[0] = self.initial_cash # initial cash
        
        self.timestamps = []
        
        # Pre-allocate full positions DataArray for historical tracking (T, N)
        # all_positions = xr.DataArray(
        #     np.zeros((len(self.market_data.time_flat), N)),
        #     dims=['time_flat', 'asset'],
        #     coords={
        #         'time_flat': self.market_data.time_flat.values,
        #         'asset': self.market_data.asset.values
        #     }
        # )
        
        all_positions_np = np.zeros((len(self.market_data.time_flat), N), dtype=float)

        # Find the index of the start date
        start_idx = np.where(self.market_data.time.values.reshape(-1,) == np.datetime64(self.start_date))[0][0]
        self.fpass += start_idx
        
        # TODO: Add the end date to the backtest        
        
        self._log("info", "=" * 60)
        self._log("info", f"BACKTEST START | Days: {len(self.market_windows)} | Assets: {N}")
        self._log("info", f"BACKTEST START | Cash: ${self.initial_cash:,.0f} | Commission: {self.commission_rate*100:.3f}%")
        self._log("info", f"BACKTEST START | Strategy: {self.strategy.name}")
        self._log("info", "=" * 60)
        
        # Set initial positions in broker and reset timestamp
        self.broker._set_positions(initial_positions)
        # self.broker.current_timestamp = 0  # Reset broker timestamp
        
        num_steps = self.T - self.window_size + 1
        
        # Determine the forward-pass break-point (0 based)
        fpass_idx = max(self.fpass - 1, 0)
        
        for tv in range(fpass_idx):
            portfolio_values[tv] = state.total_portfolio_value
            all_positions_np[tv, :] = state.positions.values
            
        i_start = fpass_idx - (self.window_size - 1)
        if i_start < 0:
            i_start = 0            
                
        # Run through pre-computed windows
        for i in tqdm(range(i_start, num_steps), desc="Backtesting"):
        
            # Get pre-computed windows
            t = int(self.window_size - 1 + i)
            start_idx = t + 1 - self.window_size
            end_idx = t + 1
            
            market_window = self.market_time_indexed_ffilled.isel(time=slice(start_idx, end_idx), time_flat=slice(start_idx, end_idx))
            char_window = self.char_time_indexed.isel(time=slice(start_idx, end_idx), time_flat=slice(start_idx, end_idx))
            
            if i == num_steps - 1:
                # If we are on the last window of the data, we don't have a future to execute orders from.
                # TODO: Cash settle at last openning price.
                time_flat_idx = self.strategy.window_size - 1 + i
                portfolio_values[time_flat_idx] = portfolio_values[time_flat_idx - 1] 
                all_positions_np[time_flat_idx, :] = all_positions_np[time_flat_idx - 1, :]
                
                break
            
            # market_window_ffilled = self.market_time_indexed_ffilled.isel(time=t+1, time_flat=t+1)
            # execution_price = self.market_time_indexed.isel(time=t+1, time_flat=t+1)
            
            open_row_t1 = self._open_np[t+1] # Shape (N,)
            close_row_t1 = self._close_np[t+1] # Shape (N,)
            date_t1 = self._time_values[t+1] # numpy.datetime64
            
            # Execute step (end-of-day t -> open t+1)
            orders, logs = self.strategy.next(market_window, char_window, state)
            for log in logs:
                self._log("info", f"Strategy result at end-of-bar {market_window.time.values[-1]}: {log}")
            if orders:
                self._log("info", f"Strategy generated {len(orders)} orders next day.")
                
            # Queue orders for next execution cycle
            self.broker.queue_orders(orders)
            
            # Execute pending orders from previous day using next day's opening prices  
            (
                new_positions,
                new_cash,
                new_portfolio_value,
                tc,
                cf,
            ) = self.broker.execute_orders_np(close_row_t1, open_row_t1, date_t1, self._asset_values)
            
            # Update state
            state = BacktestState(
                positions=new_positions,
                cash=new_cash,
                total_portfolio_value=new_portfolio_value,
                timestamp_idx=state.timestamp_idx + 1
            )
            
            # day = market_window_ffilled.time.data.astype(str) # This should be a single day.
            
            # # Execute one step to collect orders.
            # state = self._step_backtest(day, state, 
            #                             market_window, char_window, 
            #                             market_window_ffilled,
            #                             execution_price['open'],
            #                             self.strategy)
            
            # Calculate the actual time_flat index for this backtest step
            time_flat_idx = t
            self.timestamps.append(time_flat_idx)

            # Track actual portfolio value after this step (includes all costs)
            portfolio_values[time_flat_idx] = state.total_portfolio_value
            
            # state.positions is (N,), all_positions[time_flat_idx, :] expects (N,)
            all_positions_np[time_flat_idx, :] = state.positions.values
            
            
        # Convert to xarray
        self.portfolio_values = xr.DataArray(
            portfolio_values,
            dims=['time'],
            coords={'time': self.market_data.time.values.reshape(-1,)}
        )
        
        all_positions = xr.DataArray(
            all_positions_np,
            dims=['time_flat', 'asset'],
            coords={
                'time_flat': self.market_data.time_flat.values,
                'asset': self.market_data.asset.values
            }
        )
        
        # Store final positions in state
        state.all_positions = all_positions
        
        # Compute cumulative log returns from actual portfolio values
        cumulative_returns = self._compute_actual_returns(self.portfolio_values)
        
        # Create a collector
        metrics_results = {k.name: None for k in self.metrics}

        
        # Calculate metrics
        for metric in self.metrics:
            metrics_results[metric.name] = metric.calculate(all_positions=all_positions, 
                                                            all_portfolio_values=self.portfolio_values,
                                                            cumulative_returns=cumulative_returns, 
                                                            state=state)        
        # Final statistics
        final_value = state.total_portfolio_value
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        final_log_return = float(cumulative_returns[-1]) if len(cumulative_returns) > 0 else 0.0
        
        # Log final summary
        self._log("info", "=" * 60)
        self._log("info", f"BACKTEST END | Portfolio: ${final_value:,.0f} | Return: {total_return:+.2f}%")
        self._log("info", f"BACKTEST END | Cash: ${state.cash:,.0f} | Log Return: {final_log_return:+.4f}")
        self._log("info", "=" * 60)
        
        # Close logging handlers to ensure all data is written
        if self.verbose and hasattr(self, 'logger'):
            for handler in self.logger.handlers:
                handler.flush()
                handler.close()
        
        return state, cumulative_returns, metrics_results
    
    def get_commission_sensitivity_config(self) -> Dict[str, Any]:
        """
        Get configuration for commission sensitivity analysis.
        
        Returns:
            Dictionary with commission testing configuration
        """
        return {
            'base_commission_rate': self.commission_rate,
            'test_commission_rates': self.commission_rates,
            'enabled': self.commission_rates is not None and len(self.commission_rates) > 1
        }
  
    def _step_backtest(
        self,
        day: np.datetime64,
        state: BacktestState,
        market_data: xr.Dataset,           # Market data window up to time t
        characteristics: xr.Dataset,       # Characteristics data window up to time t
        market_window_ffilled: xr.Dataset, # snapshot of t+1 open, forward filled
        execution_price: xr.Dataset,       # execution price for the orders at t+1 (N,)
        strategy: EventBasedStrategy,
    ) -> BacktestState:
        """
        Single step of the backtest loop.
        
        EXECUTION FLOW:
        1. At END of day t: Strategy sees data up to day t
        2. Strategy generates orders for execution at day t+1 open
        3. Orders executed using day t+1 opening prices (realistic timing)
        4. Positions updated and we move to day t+1
            
        Returns:
            Updated backtest state after executing at day t+1 open
        """
        
        # Strategy decision: Generate orders at the end of t
        orders, logs = strategy.next(market_data, characteristics, state)

        current_day = day
        
        # Log the logs
        for log in logs:
            self._log("info", f"Strategy result at end-of-bar {market_data.time.values[-1]}: {log}")
        
        if orders:
            self._log("info", f"Strategy generated {len(orders)} orders next day.")
        
        # Queue orders for next execution cycle
        self.broker.queue_orders(orders)
        
        # Extract day t+1's opening prices for order execution
        # Strategy has NOT seen these prices - they are in the future
        next_day_open = market_window_ffilled
        self._log("debug", f"Processing market open {current_day}@{float(next_day_open.sel(asset='BTCUSDT')['open'].values)}, executing orders from {market_data.time.values[-1]}")

        # Execute pending orders from previous day using next day's opening prices
        (
            new_positions,
            new_cash,
            new_portfolio_value,
            t,
            cf,
        ) = self.broker.execute_orders(next_day_open, execution_price)
        
        # Create updated state representing start of day t+1 (after fills)
        new_state = BacktestState(
            positions=new_positions,
            cash=new_cash,
            total_portfolio_value=new_portfolio_value,
            timestamp_idx=state.timestamp_idx + 1
        )        
        
        return new_state
    
    def _setup_logging(self, log_file: Optional[str] = None, verbose: int = 0):
        """Setup logging for verbose backtesting output."""
        
        # Create log file name if not provided
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"backtest_{timestamp}.log"
        
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create logger for this backtester instance
        self.logger = logging.getLogger(f"BacktestEngine_{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create file handler with buffering for efficiency
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Prevent propagation to root logger to avoid console output
        self.logger.propagate = False
        
        self.logger.info(f"Backtest logging initialized. Log file: {log_file}")
        
        # Update broker logger if it exists
        if hasattr(self, 'broker') and self.broker is not None:
            self.broker.logger = self.logger
        
    def _log(self, level: str, message: str):
        """Efficient logging method that only logs if verbose is enabled."""
        if self.verbose > 0 and hasattr(self, 'logger'):
            if level == "info" and self.verbose > 0:
                # 1 -> high priority debugs (strategy statements + trade info)
                getattr(self.logger, level.lower())(message)
            elif level == "debug" and self.verbose > 1:
                # 2 -> low priority debugs (full debugs)
                getattr(self.logger, level.lower())(message)
    
    def get_log_file_path(self) -> Optional[str]:
        """Get the path to the log file if logging is enabled."""
        if self.verbose > 0 and hasattr(self, 'logger'):
            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    return handler.baseFilename
        return None

    def _compute_actual_returns(self, portfolio_values: xr.DataArray) -> xr.DataArray:
        """
        Compute cumulative portfolio log returns from actual portfolio values.
        
        ACTUAL RETURNS METHODOLOGY:
        - Uses real portfolio values that include ALL costs:
          * Commission payments
          * Actual execution prices (vs theoretical)
          * Partial fill effects
          * Cash flows from real trades
        - This gives TRUE portfolio performance, not theoretical
        
        Args:
            portfolio_values: List of actual portfolio values over time
            timestamps: List of corresponding timestamps
            
        Returns:
            Cumulative log returns DataArray with dims (time_flat,)
        """
        if len(portfolio_values) < 2:
            self._log("warning", "RETURNS | Insufficient data for return calculation")
            return xr.DataArray([0.0], dims=['time_flat'], coords={'time_flat': [0]})
        
        self._log("info", f"RETURNS | Computing actual portfolio returns from {len(portfolio_values)} values")
        
        # Convert to numpy arrays for efficient computation
        pv_array = portfolio_values.data
                
        # Calculate period returns: (pv[t] - pv[t-1]) / pv[t-1]
        period_returns = np.diff(pv_array) / pv_array[:-1]
        
        # Add zero return for first period
        period_returns = np.insert(period_returns, 0, 0.0)
        
        # Handle invalid returns (inf, nan)
        period_returns = np.where(np.isfinite(period_returns), period_returns, 0.0)
        
        # Convert to log returns for numerical stability
        log_returns = np.log1p(period_returns)  # log(1 + return)
        
        # Calculate cumulative log returns
        cumulative_log_returns = np.cumsum(log_returns)
        
        # Create DataArray with proper time indexing
        cumulative_returns_da = xr.DataArray(
            cumulative_log_returns,
            dims=['time'],
            coords={'time': portfolio_values.time.values.reshape(-1,)}
        )
        
        # Log performance statistics
        total_return_pct = (pv_array[-1] / pv_array[0] - 1) * 100
        total_log_return = cumulative_log_returns[-1]
        
        self._log("info", f"RETURNS | Total period return: {total_return_pct:+.3f}%")
        self._log("info", f"RETURNS | Total log return: {total_log_return:+.6f}")
        self._log("info", f"RETURNS | Initial value: ${pv_array[0]:,.2f}")
        self._log("info", f"RETURNS | Final value: ${pv_array[-1]:,.2f}")
        
        return cumulative_returns_da 
