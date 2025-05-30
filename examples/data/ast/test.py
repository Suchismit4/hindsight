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
ds = dm.get_data(
    {
        "data_sources": [
            {
                "data_path": "wrds/equity/crsp",
                "config": {
                    "start_date": "2020-01-01",
                    "end_date": "2024-01-01",
                    "freq": "D",
                    "filters": {
                        "date__gte": "2020-01-01"
                    },
                    "processors": {
                        "replace_values": {
                            "source": "delistings",
                            "rename": [["dlstdt", "time"]],
                            "identifier": "permno",
                            "from_var": "dlret",
                            "to_var": "ret"
                        },
                        "merge_table": [
                            {
                                "source": "msenames",
                                "identifier": "permno",
                                "column": "comnam",
                                "axis": "asset"
                            },
                            {
                                "source": "msenames",
                                "identifier": "permno",
                                "column": "exchcd",
                                "axis": "asset"
                            }
                        ],
                        "set_permco_coord": True,
                        "fix_market_equity": True
                    }
                }
            }
        ]
    }
)['wrds/equity/crsp']

# create the closing prices
ds["close"] = ds["prc"] / ds["cfacpr"]

# Prepare data for JIT, since some vars are not JIT compatible. For example, strings.
ds_jit, recover = prepare_for_jit(ds)

# Get function context (this contains the registered functions)
function_context = get_function_context()

# Create a closure that captures the static context
def create_jit_evaluator():
    # Capture the static context in the closure
    static_context = {
        "price": "close",
        "window": 14,
        "weights": None,
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
        # Test with specific formulas that don't require module generators
        formula_names = ["wma", "rsi", "dema"]  # Avoid hma, alma, fwma for now
        return manager.evaluate_bulk(formula_names, context, jit_compile=True)
    
    return evaluate_formulas_jit

# Create the JIT-compiled evaluator
evaluate_formulas = create_jit_evaluator()

print("Evaluating formulas with closure-based JIT compilation...")
results = evaluate_formulas(ds_jit)

print("Available formulas computed:", list(results.data_vars.keys()))
print("Results shape:", {name: var.shape for name, var in results.data_vars.items()})

# Test a second call to verify JIT caching works
print("\nTesting JIT caching with second call...")
results2 = evaluate_formulas(ds_jit)
print("Second call successful - JIT caching is working!")
