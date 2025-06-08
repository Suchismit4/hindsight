import os
import sys
import jax
import xarray as xr
import xarray_jax
import numpy as np
from functools import partial

# Add the project root to the path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src import DataManager
from src.data.ast.manager import FormulaManager
from src.data.core import prepare_for_jit
from src.data.ast.functions import get_function_context

def test_marketchars():
    """Test the new market characteristics formulas"""
    
    print("=== Testing Market Characteristics Formulas ===")
    
    # Create manager (automatically loads marketchars.yaml and other definitions)
    manager = FormulaManager()
    
    # Load CRSP data
    dm = DataManager()
    ds = dm.load_builtin("equity_standard", "2020-01-01", "2024-01-01")['equity_prices']
    
    # Create synthetic data for testing if some fields are missing
    print(f"Available data fields: {list(ds.data_vars.keys())}")
    
    # Create basic data fields that might be missing
    if 'prc' not in ds.data_vars:
        ds['prc'] = ds['close']  # Use close price as raw price
        
    if 'adjfct' not in ds.data_vars:
        ds['adjfct'] = ds['cfacpr']  # Use cumulative adjustment factor
        
    if 'ret' not in ds.data_vars:
        # Create total returns from adjusted prices
        close_adj = ds['prc'] / ds['cfacpr']
        ds['ret'] = close_adj / close_adj.shift(time_index=1) - 1
        
    if 'shares' not in ds.data_vars:
        # Create synthetic shares data
        ds['shares'] = ds['vol'] * 1000  # Synthetic shares based on volume
        
    if 'me' not in ds.data_vars:
        # Market equity = price * shares
        ds['me'] = (ds['prc'] / ds['cfacpr']) * ds['shares']
    
    # Prepare data for JIT
    ds_jit, recover = prepare_for_jit(ds)
    
    # Get function context
    function_context = get_function_context()
    
    # Test basic market characteristics
    basic_formulas = [
        "market_equity",
        "prc_adj", 
        "price_ret",
        "shares_adj",
        "ret_1m"
    ]
    
    print("\n=== Testing Basic Formulas ===")
    for formula_name in basic_formulas:
        try:
            # Create evaluation context
            context = {
                "_dataset": ds_jit,
                "prc": "prc",
                "adjfct": "adjfct", 
                "ret": "ret",
                "shares": "shares",
                "me": "me",
                **function_context
            }
            
            result = manager.evaluate(formula_name, context)
            print(f"✓ {formula_name}: {result.shape} - {result.dtype}")
            
            # Show some statistics
            if hasattr(result, 'values'):
                values = result.values
                finite_mask = np.isfinite(values)
                if np.any(finite_mask):
                    print(f"  Range: [{np.min(values[finite_mask]):.4f}, {np.max(values[finite_mask]):.4f}]")
                    print(f"  Non-null values: {np.sum(finite_mask)} / {values.size}")
                else:
                    print(f"  All values are NaN/Inf")
            
        except Exception as e:
            print(f"✗ {formula_name}: Error - {str(e)}")
    
    # Test formulas with dependencies
    dependency_formulas = [
        "dividend",
        "dividend_times_shares", 
        "div1m_me",
        "chcsho_1m"
    ]
    
    print("\n=== Testing Dependency Formulas ===") 
    for formula_name in dependency_formulas:
        try:
            context = {
                "_dataset": ds_jit,
                "prc": "prc",
                "adjfct": "adjfct",
                "ret": "ret", 
                "shares": "shares",
                "me": "me",
                **function_context
            }
            
            result = manager.evaluate(formula_name, context)
            print(f"✓ {formula_name}: {result.shape} - {result.dtype}")
            
            # Show some statistics
            if hasattr(result, 'values'):
                values = result.values
                finite_mask = np.isfinite(values)
                if np.any(finite_mask):
                    print(f"  Range: [{np.min(values[finite_mask]):.4f}, {np.max(values[finite_mask]):.4f}]")
                    print(f"  Non-null values: {np.sum(finite_mask)} / {values.size}")
                else:
                    print(f"  All values are NaN/Inf")
                    
        except Exception as e:
            print(f"✗ {formula_name}: Error - {str(e)}")
    
    # Test bulk evaluation of multiple characteristics
    print("\n=== Testing Bulk Evaluation ===")
    try:
        bulk_formulas = {
            "market_equity": {},
            "prc_adj": {},
            "price_ret": {},
            "dividend": {},
            "div1m_me": {},
            "div3m_me": {},
            "chcsho_1m": {},
            "chcsho_3m": {},
            "ret_3m": {}
        }
        
        context = {
            "_dataset": ds_jit,
            "prc": "prc",
            "adjfct": "adjfct",
            "ret": "ret",
            "shares": "shares", 
            "me": "me",
            **function_context
        }
        
        results = manager.evaluate_bulk(bulk_formulas, context)
        print(f"✓ Bulk evaluation successful!")
        print(f"  Computed characteristics: {list(results.data_vars.keys())}")
        print(f"  Result dataset shape: {results.dims}")
        
        # Show summary statistics for each computed characteristic
        for var_name, var_data in results.data_vars.items():
            values = var_data.values
            finite_mask = np.isfinite(values)
            if np.any(finite_mask):
                mean_val = np.mean(values[finite_mask])
                std_val = np.std(values[finite_mask])
                print(f"  {var_name}: mean={mean_val:.4f}, std={std_val:.4f}, non-null={np.sum(finite_mask)}")
            else:
                print(f"  {var_name}: All values are NaN/Inf")
        
    except Exception as e:
        print(f"✗ Bulk evaluation failed: {str(e)}")
    
    # Test available formulas in manager
    print(f"\n=== Available Formulas ===")
    available_formulas = manager.list_formulas()
    marketchars_formulas = [f for f in available_formulas if any(f.startswith(prefix) for prefix in 
                           ['market_', 'prc_', 'price_', 'div', 'chcsho_', 'ret_', 'shares_', 'dividend'])]
    print(f"Market characteristics formulas loaded: {len(marketchars_formulas)}")
    for formula in sorted(marketchars_formulas):
        print(f"  - {formula}")

if __name__ == "__main__":
    test_marketchars() 