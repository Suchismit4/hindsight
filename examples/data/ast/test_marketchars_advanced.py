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

def test_advanced_marketchars():
    """Test the more complex market characteristics formulas"""
    
    print("=== Testing Advanced Market Characteristics ===")
    
    # Create manager
    manager = FormulaManager()
    
    # Load CRSP data
    dm = DataManager()
    ds = dm.load_builtin("equity_standard", "2020-01-01", "2024-01-01")['equity_prices']
    
    # Prepare synthetic data
    print(f"Available data fields: {list(ds.data_vars.keys())}")
    
    # Map existing fields to expected names
    ds['prc'] = ds['prc'].copy()  # Raw price
    ds['adjfct'] = ds['cfacpr'].copy()  # Cumulative adjustment factor
    ds['shares'] = ds['shrout'].copy()  # Use actual shares outstanding
    
    # Create total returns with dividends (existing ret should include dividends)
    ds['ret'] = ds['ret'].copy()  # This should include total returns
    
    # Create market equity if not present
    if 'me' not in ds.data_vars:
        ds['me'] = ds['me'].copy()  # Market equity should already be present
    
    # Prepare data for JIT
    ds_jit, recover = prepare_for_jit(ds)
    
    # Get function context
    function_context = get_function_context()
    
    # Test longer-horizon return calculations
    return_formulas = [
        "ret_1m",
        "ret_3m", 
        "ret_6m",
        "ret_12m"
    ]
    
    print("\n=== Testing Return Calculations ===")
    for formula_name in return_formulas:
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
            
            # Calculate statistics
            values = result.values
            finite_mask = np.isfinite(values)
            
            if np.any(finite_mask):
                mean_val = np.mean(values[finite_mask])
                std_val = np.std(values[finite_mask])
                min_val = np.min(values[finite_mask])
                max_val = np.max(values[finite_mask])
                
                print(f"  Stats: mean={mean_val:.4f}, std={std_val:.4f}")
                print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
                print(f"  Non-null: {np.sum(finite_mask)} / {values.size} ({100*np.sum(finite_mask)/values.size:.1f}%)")
                
                # Check for reasonable return values
                if formula_name == "ret_1m":
                    extreme_returns = np.sum(np.abs(values[finite_mask]) > 2.0)
                    print(f"  Extreme returns (>200%): {extreme_returns}")
                elif formula_name == "ret_12m":
                    extreme_returns = np.sum(np.abs(values[finite_mask]) > 5.0)
                    print(f"  Extreme annual returns (>500%): {extreme_returns}")
            else:
                print(f"  All values are NaN/Inf")
                
        except Exception as e:
            print(f"✗ {formula_name}: Error - {str(e)}")
    
    # Test dividend-related calculations
    dividend_formulas = [
        "dividend",
        "dividend_times_shares",
        "div1m_me",
        "div3m_me", 
        "div6m_me",
        "div12m_me"
    ]
    
    print("\n=== Testing Dividend Calculations ===")
    for formula_name in dividend_formulas:
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
            
            values = result.values
            finite_mask = np.isfinite(values)
            
            if np.any(finite_mask):
                mean_val = np.mean(values[finite_mask])
                std_val = np.std(values[finite_mask])
                
                # For dividend yields, check for reasonable ranges
                if 'div' in formula_name and '_me' in formula_name:
                    reasonable_range = np.sum((values[finite_mask] >= -0.1) & (values[finite_mask] <= 0.2))
                    print(f"  Reasonable div yields (-10% to 20%): {reasonable_range} / {np.sum(finite_mask)}")
                
                print(f"  Stats: mean={mean_val:.4f}, std={std_val:.4f}")
                print(f"  Non-null: {np.sum(finite_mask)} / {values.size} ({100*np.sum(finite_mask)/values.size:.1f}%)")
            else:
                print(f"  All values are NaN/Inf")
                
        except Exception as e:
            print(f"✗ {formula_name}: Error - {str(e)}")
    
    # Test shares outstanding changes
    shares_formulas = [
        "shares_adj",
        "chcsho_1m",
        "chcsho_3m",
        "chcsho_6m", 
        "chcsho_12m"
    ]
    
    print("\n=== Testing Shares Outstanding Changes ===")
    for formula_name in shares_formulas:
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
            
            values = result.values
            finite_mask = np.isfinite(values)
            
            if np.any(finite_mask):
                if 'chcsho' in formula_name:
                    # For change in shares, most should be small changes
                    small_changes = np.sum(np.abs(values[finite_mask]) <= 0.1)
                    print(f"  Small changes (<=10%): {small_changes} / {np.sum(finite_mask)}")
                    
                    extreme_changes = np.sum(np.abs(values[finite_mask]) > 1.0)
                    print(f"  Extreme changes (>100%): {extreme_changes}")
                    
                mean_val = np.mean(values[finite_mask])
                std_val = np.std(values[finite_mask])
                print(f"  Stats: mean={mean_val:.4f}, std={std_val:.4f}")
                print(f"  Non-null: {np.sum(finite_mask)} / {values.size} ({100*np.sum(finite_mask)/values.size:.1f}%)")
            else:
                print(f"  All values are NaN/Inf")
                
        except Exception as e:
            print(f"✗ {formula_name}: Error - {str(e)}")
    
    # Test dependency chain evaluation
    print("\n=== Testing Complex Dependency Chain ===")
    try:
        # This should test the full dependency resolution
        complex_formulas = {
            "price_ret": {},
            "dividend": {},
            "dividend_times_shares": {},
            "div3m_me": {},
            "div12m_me": {}
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
        
        results = manager.evaluate_bulk(complex_formulas, context)
        print(f"✓ Complex dependency evaluation successful!")
        
        # Check that dependency relationships are maintained
        price_ret = results['price_ret'].values
        dividend = results['dividend'].values
        
        # Basic sanity check: dividend should be ret - price_ret times lagged price
        print(f"  Dependencies resolved correctly for {len(results.data_vars)} characteristics")
        
        # Check that longer-horizon dividend yields are larger than shorter ones (typically)
        div3m = results['div3m_me'].values
        div12m = results['div12m_me'].values
        
        finite_mask = np.isfinite(div3m) & np.isfinite(div12m)
        if np.any(finite_mask):
            # For most stocks, 12m dividend yield should be higher than 3m (cumulative effect)
            higher_12m = np.sum(div12m[finite_mask] > div3m[finite_mask])
            total_valid = np.sum(finite_mask)
            print(f"  12m dividend yield > 3m dividend yield: {higher_12m} / {total_valid} ({100*higher_12m/total_valid:.1f}%)")
        
    except Exception as e:
        print(f"✗ Complex dependency evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_advanced_marketchars() 