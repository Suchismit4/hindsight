"""
Hindsight Pipeline Framework Example

This script demonstrates a complete end-to-end workflow using the Hindsight pipeline framework
for financial time series analysis. The pipeline architecture separates data processing ("how") 
from temporal segmentation ("when"), enabling flexible and robust financial research.

Key Components Demonstrated:
- Data loading with DataManager
- Feature engineering with formula evaluation 
- Walk-forward analysis configuration
- Model integration with scikit-learn
- Result visualization and analysis
"""
from __future__ import annotations
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from src.pipeline import (
    HandlerConfig, DataHandler, PipelineMode,
    FormulaEval, PerAssetFFill,
    SegmentConfig, make_plan
)
from src.pipeline.model.adapter import SklearnAdapter
from src.pipeline.model.runner import ModelRunner
from src import DataManager

# ===== Data Loading =====
# Load hourly cryptocurrency price data for backtesting
print("Loading cryptocurrency data...")
data_manager = DataManager()
ds = data_manager.load_builtin(
    "crypto_standard",
    "2019-12-30",
    "2024-01-01"
)['crypto_prices']
print(f"Loaded data with {ds.sizes['asset']} assets and {ds.sizes.get('hour', 0)} time periods")


# ===== Feature Engineering Configuration =====
# Define formulas for technical indicators and target variables
formulas = {
    "sma": [{"window": 100}, {"window": 200}],  # Simple moving averages
    "rsi": [{"window": 14}],                    # Relative strength index
    "fwd_return": [{"periods": 5}],             # Target: 5-period forward return
    "price_ret_var": [                          # Price return variance features
        {"periods": 1}, {"periods": 3}, {"periods": 6}, 
        {"periods": 12}, {"periods": 24}, {"periods": 48}
    ],
}

# Configure data processing pipeline
shared_processors = [
    FormulaEval(
        name="formulas_core",
        formula_configs=formulas,
        static_context={"price": "close", "prc": "close"},
        use_jit=False,
        assign_in_place=True
    )
]

# Create handler configuration with feature and target specifications
cfg_handler = HandlerConfig(
    shared=shared_processors,
    learn=[],  # No stateful processors in this example
    infer=[],  # No inference-only processors
    mode=PipelineMode.INDEPENDENT,
    feature_cols=[
        "sma_ww200", "rsi", 
        "price_ret_var_p1", "price_ret_var_p3", "price_ret_var_p6", 
        "price_ret_var_p12", "price_ret_var_p24", "price_ret_var_p48"
    ],
    label_cols=["fwd_return_p5"] # ["fwd_return_p5"] #sma_ww100
)


# ===== Data Handler Setup =====
# Instantiate and build the data processing pipeline
print("Configuring data handler...")
handler = DataHandler(base=ds, config=cfg_handler)
handler.build()
print("Data handler configured successfully")


# ===== Walk-Forward Analysis Configuration =====
# Define temporal structure for walk-forward validation
start = np.datetime64("2020-01-22 18:00:00")
end = np.datetime64("2023-12-01 00:00:00")

cfg_segment = SegmentConfig(
    start=start,
    end=end,
    train_span=np.timedelta64(24 * 5, "h"),   # 5 days training window
    infer_span=np.timedelta64(24 * 1, "h"),   # 1 day inference window
    step=np.timedelta64(24 * 1, "h"),         # 1 day step forward
    gap=np.timedelta64(0, "h"),               # No gap between train/infer
    clip_to_data=True                         # Clip to available data
)

# Generate segment plan
plan = make_plan(cfg_segment, ds_for_bounds=ds)
print(f"Created walk-forward plan with {len(plan)} segments")


# ===== Model Configuration =====
# Define model adapter factory for segment isolation
def make_adapter():
    """Create a fresh RandomForestRegressor adapter for each segment."""
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=14,
        min_samples_leaf=10,
        bootstrap=True,
        max_samples=0.5,
        n_jobs=-1,
        random_state=0,
    )
    return SklearnAdapter(
        model=model, 
        handler=handler, 
        output_var="score", 
        use_proba=False
    )

# Configure model runner for walk-forward execution
runner = ModelRunner(
    handler=handler,
    plan=plan,
    model_factory=make_adapter,
    feature_cols=cfg_handler.feature_cols,
    label_col=cfg_handler.label_cols[0],
    overlap_policy="last",
    output_var="score",
    # Debug parameters for focused analysis
    debug_start=np.datetime64("2021-01-30T01:00:00"),
    debug_end=np.datetime64("2021-01-30T07:00:00"),
    debug_asset="BTCUSDT"
)


# ===== Pipeline Execution (Commented for Demo) =====
# Uncomment to run the full pipeline

print("Starting model runner execution...")
results = runner.run()
print("Model execution completed")

# Save predictions
pred_ds = results.pred_ds
pred_ds.to_netcdf("dev/predictions.nc")
print("Predictions saved to dev/predictions.nc")


# ===== Results Analysis and Visualization =====
# Load pre-computed predictions for analysis
print("Loading predictions for visualization...")
pred = xr.open_dataset("dev/pred_debug.nc")

# Merge predictions with original data
print("Merging predictions with processed data...")
start_time = time.time()
processed_ds = handler.shared_view()
combined_ds = processed_ds.merge(pred)
print(f"Merge completed in {time.time() - start_time:.2f} seconds")

# Extract BTCUSDT data for visualization
btcusdt = combined_ds.sel(asset="BTCUSDT").dt.to_time_indexed()

# Create prediction vs actual comparison plot
print("Generating visualizations...")
btcusdt["score"].plot.line(x="time", label="Predicted Returns", alpha=0.7)
btcusdt["fwd_return_p5"].plot.line(x="time", label="Actual Returns", alpha=0.7)
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("BTCUSDT: Predicted vs Actual Returns")
plt.xlabel("Time")
plt.ylabel("5-Period Forward Returns")
plt.tight_layout()
plt.savefig("prediction_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# Create dual-axis plot with price and predictions
fig, ax1 = plt.subplots(figsize=(12, 6))

# Price on left axis
btcusdt["close"].plot.line(x="time", ax=ax1, label="Price", color="blue", linewidth=1)
ax1.set_xlabel("Time")
ax1.set_ylabel("Price (USD)", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

# Predictions on right axis
ax2 = ax1.twinx()
btcusdt["score"].plot.line(x="time", ax=ax2, label="Predicted Returns", color="red", alpha=0.7)
ax2.set_ylabel("Predicted 5-Period Returns", color="red")
ax2.tick_params(axis="y", labelcolor="red")

# Combined legend
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left")

plt.title("BTCUSDT: Price and Predicted Returns")
plt.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("price_and_predictions.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

print("Example completed! Check dev/ folder for generated plots.")
print("\nPipeline Summary:")
print(f"- Features: {len(cfg_handler.feature_cols)} technical indicators")
print(f"- Segments: {len(plan)} walk-forward windows")
print(f"- Assets: {ds.sizes['asset']} cryptocurrencies")
print(f"- Target: {cfg_handler.label_cols[0]} (5-period forward returns)")