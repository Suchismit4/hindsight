import os
import sys
import jax
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

# Setup context once
context = {
    "price": "close",
    "window": 14,  
    "weights": None,  # Default for WMA weights (will use linearly increasing weights)
    "_dataset": ds_jit
}
context.update(get_function_context())

@partial(jax.jit, static_argnames=['context'])
def func(context):
    results = manager.evaluate_all_loaded(context, jit_compile=True)
    return results

results = func(context)

print(results)
