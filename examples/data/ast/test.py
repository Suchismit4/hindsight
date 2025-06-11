import os
import sys
import jax
import xarray_jax
from functools import partial
# Add the project root to the path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src import DataManager
from src.data.ast.manager import FormulaManager
from src.data.core import prepare_for_jit

from src.data.ast.functions import (
    get_function_context,
)

# Create manager (automatically loads technical.yaml and other definitions)
manager = FormulaManager()

# Load CRSP data
dm = DataManager()
configs = dm.get_builtin_configs()
print("Available configs:", configs)
ds = dm.load_builtin("crypto_standard", "2019-01-01", "2024-01-01")['crypto_prices']

# create the closing prices
# ds["close"] = ds["prc"] / ds["cfacpr"]

# Prepare data for JIT, since some vars are not JIT compatible. For example, strings.
ds_jit, recover = prepare_for_jit(ds)

# Get function context (this contains the registered functions)
function_context = get_function_context()

# Create a closure that captures the static context
def create_jit_evaluator():
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
        # Using smaller windows to fit available data
        formula_configs = {
            "wma": [
                {"window": 5},
                {"window": 10}, 
                {"window": 15} 
            ],
            "rsi": [
                {"window": 7},     
                {"window": 14}  
            ],
            "dema": {"window": 10},  
            "alma": [
                {"window": 5, "offset": 0.85, "sigma": 6},
                {"window": 10, "offset": 0.9, "sigma": 8,}  
            ],
            "hma": [
                {"window": 100},
                {"window": 200}, 
            ]
        }
        
        return manager.evaluate_bulk(formula_configs, context, jit_compile=True)
    
    return evaluate_formulas_jit

evaluate_formulas_jit = create_jit_evaluator()
results = evaluate_formulas_jit(ds_jit)

print("Available formulas computed (JIT):", list(results.data_vars.keys()))
print("Results shape:", {name: var.shape for name, var in results.data_vars.items()})

# if 'divamt' in ds.data_vars:
#     print("Distribution data available!")
#     # Access dividend amounts for a specific asset
#     apple_permno = 14593
#     apple_dividends = ds['divamt'].sel(asset=apple_permno)
#     print(f"Apple dividends shape: {apple_dividends.shape}")
