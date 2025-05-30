"""
Simple test for formula loading and evaluation.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import matplotlib.pyplot as plt
import jax

from src import DataManager
from src.data.ast.manager import FormulaManager
from src.data.ast.functions import register_built_in_functions, get_function_context
from src.data.core import prepare_for_jit

# Register built-in functions
register_built_in_functions()

# Initialize and load formulas
manager = FormulaManager()
manager.load_directory("src/data/ast/definitions")

# Load CRSP data
dm = DataManager()
ds = dm.get_data(
    {
        "data_sources": [
            {
                "data_path": "wrds/equity/crsp",
                "config": {
                    "start_date": "2000-01-01",
                    "end_date": "2024-01-01",
                    "freq": "D",
                    "filters": {
                        "date__gte": "2000-01-01"
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

# Create adjusted price variable
ds["close"] = ds["prc"] / ds["cfacpr"]

# Get RSI formula
formula = manager.get_formula("rsi")

# Prepare data for JIT
ds_jit = prepare_for_jit(ds)

print(ds_jit["close"])

# Create evaluation context with both data and functions
context = {
    "price": ds_jit["close"],
    "window": 14,
    "_dataset": ds_jit
}
context.update(get_function_context())  # Add all registered functions

# JIT compile the evaluation
@jax.jit
def evaluate_rsi(context):
    result, _ = manager.evaluate("rsi", context)
    return result

# Evaluate RSI
result = evaluate_rsi(context)

# Plot RSI for Apple
plt.figure(figsize=(15, 8))
stocks_to_plot = [14593]  # AAPL's PERMNO
for stock in stocks_to_plot:
    rsi_values = result.sel(asset=stock)
    plt.plot(ds.time, rsi_values, label=ds.comnam.sel(asset=stock).item())

plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
plt.title('RSI(14) for Apple')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.grid(True)
plt.show()
