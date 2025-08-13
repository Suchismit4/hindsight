import os
import sys
import matplotlib.pyplot as plt
import mplfinance as mpf
import jax
import numpy as np

# Data Management imports
from src import DataManager
from src.data.ast.manager import FormulaManager
from src.data.ast.functions import register_built_in_functions, get_function_context
from src.data.core import prepare_for_jit

import xarray as xr

# Setup core
data_manager = DataManager()

# Load crypto USDT data
market_data = data_manager.load_builtin(
                    "crypto_standard", 
                    "2019-12-30", 
                    "2024-01-01")['crypto_prices'] # Load USDT Spot data from 2019 to 2024 at HOURLY timeframe

# market_data.dt.to_time_indexed().drop_vars(["time_flat", "mask", "mask_indices"]).sel(asset='BTCUSDT').drop_vars(["asset"]).to_dataframe().to_csv('BTCUSDT.csv')
# quit(1)

# Compute some characteristics 
# MA for different periods

# Create a closure that captures the static context
def create_jit_evaluator():
    function_context = get_function_context()
    manager = FormulaManager()
        
    # Capture the static context in the closure
    static_context = {
        "price": "close",
        **function_context
    }
    
    # JIT compile only the dataset processing part
    @jax.jit
    def evaluate_formulas_jit(dataset):
        # Reconstruct the full context inside the JIT function
        context = {
            "_dataset": dataset,
            **static_context
        }
        
        # Multi-configuration evaluation with lag examples
        formula_configs = {
            "sma": [
                {"window": 100},
                {"window": 200}
            ],
        }
        
        return manager.evaluate_bulk(formula_configs, context)
    
    return evaluate_formulas_jit

# Prepare data for JIT, since some vars are not JIT compatible. For example, strings.
market_data_jit, recover = prepare_for_jit(market_data)

evaluate_formulas_jit = create_jit_evaluator()
characteristics = evaluate_formulas_jit(market_data_jit)

# Backtester imports
from src.backtester.core import BacktestState, EventBasedStrategy, BacktestEngine
from src.backtester.struct import MarketOrder, OrderDirection
from src.backtester.metrics.standard import SharpeRatio, ProfitFactor, MaxDrawdown, UlcerIndex, MartinRatio

# Simple Buy and Hold Strategy test
class BUYHLDBitcoin(EventBasedStrategy):
    
    def __init__(self):
        super().__init__("Buy and Hold Bitcoin Strategy", window_size=10)

    def next(self, market_data, characteristics, state):
        
        # orders = []
        # if market_data.time.data[0] == np.datetime64('2020-01-08 17:00:00'):
        #     import math 
        #     orders.append(MarketOrder(asset='BTCUSDT', quantity=math.floor(state.cash / market_data.sel(asset='BTCUSDT')['close'][0].data), direction=OrderDirection.BUY, timestamp=state.timestamp_idx))
        
        return [], []
    
# MA-Crossover strategy (state based only BTCUSDT)
class MACrossoverStrategy(EventBasedStrategy):
    
    def __init__(self):
        super().__init__("MA Crossover Strategy", window_size=2)
        # Memory for last valid SMA values
        self.last_sma100 = None
        self.last_sma200 = None

    def next(self, market_data, characteristics, state):
        orders = []
        logs = []
        asset = 'BTCUSDT'

        current_position = float(state.positions.sel(asset=asset))

        sma100 = characteristics['sma_ww100'].sel(asset=asset).data
        sma200 = characteristics['sma_ww200'].sel(asset=asset).data

        # Forward-fill from memory if NaN
        prev100 = float(sma100[0]) if not np.isnan(sma100[0]) else self.last_sma100
        curr100 = float(sma100[1]) if not np.isnan(sma100[1]) else self.last_sma100
        prev200 = float(sma200[0]) if not np.isnan(sma200[0]) else self.last_sma200
        curr200 = float(sma200[1]) if not np.isnan(sma200[1]) else self.last_sma200

        # Update memory if we got fresh valid values
        if not np.isnan(sma100[1]):
            self.last_sma100 = float(sma100[1])
        if not np.isnan(sma200[1]):
            self.last_sma200 = float(sma200[1])

        # Skip if we still don't have enough data
        if None in (prev100, curr100, prev200, curr200):
            return orders, logs

        # Signal logic
        if (prev100 > prev200) and (curr100 <= curr200):
            target_position = 1
        elif (prev100 < prev200) and (curr100 >= curr200):
            target_position = -1
        else:
            return orders, logs

        delta = target_position - current_position
        if abs(delta) > 1e-12:
            if delta > 0:
                orders.append(MarketOrder(asset=asset, quantity=delta,
                                          direction=OrderDirection.BUY,
                                          timestamp=state.timestamp_idx))
                logs.append(
                    f"BUY {delta:.6f} {asset} | prev100={prev100:.3f}, curr100={curr100:.3f}, "
                    f"prev200={prev200:.3f}, curr200={curr200:.3f} "
                    f"D={market_data.coords['time'].data[1]}"
                )
            else:
                orders.append(MarketOrder(asset=asset, quantity=-delta,
                                          direction=OrderDirection.SELL,
                                          timestamp=state.timestamp_idx))
                logs.append(
                    f"SELL {-delta:.6f} {asset} | prev100={prev100:.3f}, curr100={curr100:.3f}, "
                    f"prev200={prev200:.3f}, curr200={curr200:.3f} "
                    f"D={market_data.coords['time'].data[1]}"
                )

        return orders, logs
        
class MACrossoverStrategyOptimized(EventBasedStrategy):
    def __init__(self):
        super().__init__("MA Crossover Strategy", window_size=2)
        self.asset = "BTCUSDT"
        # memory for last valid SMA values
        self.last_sma100 = None
        self.last_sma200 = None

    def next(self, market_data: xr.Dataset, characteristics: xr.Dataset, state: BacktestState):
        orders, logs = [], []

        # fast asset index (uses engine-injected map if present; otherwise falls back)
        ai = self.asset_idx(characteristics, self.asset)

        # pull the 2-length window for each SMA as (prev, curr)
        p100_raw, c100_raw = self.series2(characteristics, "sma_ww100", ai)
        p200_raw, c200_raw = self.series2(characteristics, "sma_ww200", ai)

        # forward-fill with our remembered last values (same semantics as before)
        p100, c100 = self.ffill2(p100_raw, c100_raw, self.last_sma100)
        p200, c200 = self.ffill2(p200_raw, c200_raw, self.last_sma200)

        # update memory if we saw fresh values
        if np.isfinite(c100_raw): self.last_sma100 = float(c100_raw)
        if np.isfinite(c200_raw): self.last_sma200 = float(c200_raw)

        # still insufficient data?
        if (p100 is None) or (c100 is None) or (p200 is None) or (c200 is None):
            return orders, logs

        # crossover logic (unchanged)
        if (p100 > p200) and (c100 <= c200):
            target = 1.0
        elif (p100 < p200) and (c100 >= c200):
            target = -1.0
        else:
            return orders, logs

        # current position (fast path)
        current = float(state.positions.isel(asset=ai).values)
        delta = target - current

        if abs(delta) > 1e-12:
            if delta > 0:
                orders.append(MarketOrder(
                    asset=self.asset, quantity=delta,
                    direction=OrderDirection.BUY, timestamp=state.timestamp_idx
                ))
                logs.append(
                    f"BUY {delta:.6f} {self.asset} | prev100={p100:.3f}, curr100={c100:.3f}, "
                    f"prev200={p200:.3f}, curr200={c200:.3f} D={market_data.time.values[1]}"
                )
            else:
                orders.append(MarketOrder(
                    asset=self.asset, quantity=-delta,
                    direction=OrderDirection.SELL, timestamp=state.timestamp_idx
                ))
                logs.append(
                    f"SELL {-delta:.6f} {self.asset} | prev100={p100:.3f}, curr100={c100:.3f}, "
                    f"prev200={p200:.3f}, curr200={c200:.3f} D={market_data.time.values[1]}"
                )

        return orders, logs

engine = BacktestEngine(
    strategy        = MACrossoverStrategyOptimized(),
    market_data     = market_data,
    characteristics = characteristics,
    verbose         = 1,
    fpass           = 0,
    commission_rate = 0,
    date            = ("2020-01-08 16:00:00", "2020-01-08 16:00:00")
    # date            = ("2020-01-08 16:00:00", "2020-01-08 16:00:00")
)

engine.add_metric(SharpeRatio(riskfreerate=0.01,))
engine.add_metric(ProfitFactor())
engine.add_metric(MaxDrawdown())
engine.add_metric(UlcerIndex())
engine.add_metric(MartinRatio(riskfreerate=0.01, periods_per_year=8760))

state, cumulative_returns, metrics_results = engine.run()

print(metrics_results)

# import pandas as pd
# # merge the the BTC open close and etc and sma 100 and sma 200

# btc_df = market_data.dt.to_time_indexed().drop_vars(["time_flat", "mask", "mask_indices"]).sel(asset='BTCUSDT').drop_vars(["asset"]).to_dataframe()
# ch_df = characteristics.dt.to_time_indexed().drop_vars(["time_flat", "mask", "mask_indices"]).sel(asset='BTCUSDT').drop_vars(["asset"]).to_dataframe()

# # merge the two dataframes on the index
# merged_df = pd.merge(btc_df, ch_df, left_index=True, right_index=True)
# merged_df.to_csv('merged_df.csv')
