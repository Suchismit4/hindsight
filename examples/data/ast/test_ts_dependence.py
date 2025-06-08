import os
import sys
import jax
import xarray_jax

# Add the project root to the path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src import DataManager
from src.data.ast.manager import FormulaManager
from src.data.core import prepare_for_jit
from src.data.ast.functions import get_function_context

def test_time_series_dependence():
    """Test time series dependence feature where formulas can reference other formulas as dataarray variables."""
    
    print("=== Testing Time Series Dependence ===")
    
    # Create empty formula manager to avoid conflicts
    manager = FormulaManager.__new__(FormulaManager)
    manager.formulas = {}
    manager._registered_functions = set()
    manager._module_cache = {}
    manager._formula_functions = {}
    manager._dependency_graph = {}
    
    # Load the schema
    import yaml
    schema_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'data', 'ast', 'definitions', 'schema.yaml')
    with open(schema_path) as f:
        manager._schema = yaml.safe_load(f)
    
    # Load the time series dependence definitions
    ts_def_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'data', 'ast', 'definitions', 'ts_dependence.yaml')
    manager.load_file(ts_def_path)
    
    # Load RSI formula manually since we use it in our test
    rsi_def_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'data', 'ast', 'definitions', 'technical.yaml')  
    if os.path.exists(rsi_def_path):
        with open(rsi_def_path) as f:
            import yaml
            tech_formulas = yaml.safe_load(f)
            if 'rsi' in tech_formulas:
                manager.add_formula('rsi', tech_formulas['rsi'])
    
    print(f"Loaded formulas: {manager.list_formulas()}")
    
    # Analyze dependencies
    print("\n=== Dependency Analysis ===")
    for formula_name in ['ts_momentum', 'ts_signal', 'ts_combined']:
        if formula_name in manager.formulas:
            print(f"\nFormula: {formula_name}")
            
            # Get different types of dependencies
            ts_deps = manager.get_time_series_dependencies(formula_name)
            func_deps = manager.get_functional_dependencies(formula_name)
            all_deps = manager.get_formula_dependencies(formula_name)
            
            print(f"  Time Series Dependencies: {ts_deps}")
            print(f"  Functional Dependencies: {func_deps}")
            print(f"  All Dependencies: {all_deps}")
            
            # Get dependency chain
            try:
                dep_chain = manager.get_dependency_chain(formula_name)
                print(f"  Dependency Chain: {dep_chain}")
            except Exception as e:
                print(f"  Dependency Chain Error: {e}")
    
    # Load CRSP data (same as test.py)
    print("\n=== Loading Data ===")
    dm = DataManager()
    ds = dm.load_builtin("equity_standard", "2020-01-01", "2024-01-01")['equity_prices']
    ds["close"] = ds["prc"] / ds["cfacpr"]
    ds_jit, recover = prepare_for_jit(ds)
    
    print(f"Dataset variables: {list(ds.data_vars.keys())}")
    print(f"Dataset shape: {ds.dims}")
    
    # Test evaluation with time series dependencies
    print("\n=== Evaluating Formulas with Time Series Dependencies ===")
    
    # Get function context
    function_context = get_function_context()
    
    # Create evaluation context
    context = {
        "_dataset": ds_jit,
        "price": "close",  # Map price variable to close prices
        **function_context
    }
    
    try:
        # Test individual formula evaluation
        print("\n1. Testing individual formula evaluation:")
        
        # Evaluate ts_momentum (base formula)
        result1 = manager.evaluate("ts_momentum", context)
        print(f"   ts_momentum result shape: {result1.shape}")
        print(f"   ts_momentum result type: {type(result1)}")
        
        # Evaluate ts_signal (depends on ts_momentum)
        result2 = manager.evaluate("ts_signal", context)
        print(f"   ts_signal result shape: {result2.shape}")
        print(f"   ts_signal result type: {type(result2)}")
        
        # Evaluate ts_combined (depends on ts_signal + functional dependency)
        result3 = manager.evaluate("ts_combined", context)
        print(f"   ts_combined result shape: {result3.shape}")
        print(f"   ts_combined result type: {type(result3)}")
        
    except Exception as e:
        print(f"   Individual evaluation error: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Test bulk evaluation (should handle dependencies automatically)
        print("\n2. Testing bulk evaluation with dependencies:")
        
        results = manager.evaluate_bulk(['ts_combined'], context)
        print(f"   Bulk evaluation results: {list(results.data_vars.keys())}")
        print(f"   Results shapes: {[(name, var.shape) for name, var in results.data_vars.items()]}")
        
        # Check if intermediate dependencies were computed
        if 'ts_momentum' in results.data_vars:
            print("   ✓ ts_momentum was computed as dependency")
        if 'ts_signal' in results.data_vars:
            print("   ✓ ts_signal was computed as dependency")
        if 'ts_combined' in results.data_vars:
            print("   ✓ ts_combined was computed as target")
            
    except Exception as e:
        print(f"   Bulk evaluation error: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Test evaluation of all time series dependence formulas
        print("\n3. Testing evaluation of all formulas:")
        
        all_results = manager.evaluate_bulk(['ts_momentum', 'ts_signal', 'ts_combined'], context)
        print(f"   All results: {list(all_results.data_vars.keys())}")
        
        # Show some sample values
        for var_name, var_data in all_results.data_vars.items():
            if hasattr(var_data, 'values'):
                # Use the actual dimension names from the dataset
                sample_val = var_data.isel(year=slice(-1, None), month=slice(-1, None), day=slice(-5, None), asset=0).values
                print(f"   {var_name} last 5 days for first asset: {sample_val.flatten()}")
        
    except Exception as e:
        print(f"   All formulas evaluation error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_time_series_dependence() 