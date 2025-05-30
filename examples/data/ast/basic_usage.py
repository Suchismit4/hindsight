#!/usr/bin/env python
"""
Basic usage examples for the AST-based formula evaluation system.

This example demonstrates the core functionality of the AST system:
- Parsing formulas into AST nodes
- Evaluating formulas with different types of inputs
- Visualizing formula ASTs
- Working with the function registry
- Optimizing formulas

Each example is clearly documented and designed to help new users
understand how to use the AST system effectively.
"""

import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
# Add the project root to the path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src import DataManager
from src.data.core import prepare_for_jit
from src.data.ast import (
    parse_formula,
    register_function,
    extract_variables,
    extract_functions,
    visualize_ast,
    visualize_parse_tree,
    get_grammar_description,
    optimize_formula,
)

from src.data.ast.functions import (
    get_function_context,
    get_function_categories
)

def example_simple_parsing():
    """
    Example 1: Parsing simple formulas into ASTs.
    
    Demonstrates how to parse different types of formulas and inspect their structure.
    """
    print("\n=== Example 1: Simple Formula Parsing ===")
    
    # Example formulas from simple to complex
    formulas = [
        "2 + 3",                      # Simple addition
        "x + y * z",                  # Variables with operator precedence
        "(a + b) / (c - d)",          # Parentheses grouping
        "sin(x) + log(y + 1)",        # Function calls
        "price * (1 + tax_rate)"      # Financial formula
    ]
    
    for formula in formulas:
        print(f"\nFormula: {formula}")
        
        # Parse the formula into an AST
        ast = parse_formula(formula)
        
        # Extract variables and functions
        variables = extract_variables(formula)
        functions = extract_functions(formula)
        
        # Print the AST and extracted information
        print(f"AST structure: {ast}")
        print(f"Variables used: {variables}")
        print(f"Functions used: {functions}")

def example_formula_evaluation():
    """
    Example 2: Evaluating formulas with different inputs.
    
    Shows how to evaluate formulas with scalar values, arrays, and mixed inputs.
    """
    print("\n=== Example 2: Formula Evaluation ===")
    
    # Parse a formula for evaluation
    formula = "x + y * z"
    ast = parse_formula(formula)
    print(f"Formula: {formula}")
    
    # Evaluate with scalar inputs
    scalar_context = {
        'x': jnp.array(1.0),
        'y': jnp.array(2.0),
        'z': jnp.array(3.0)
    }
    
    scalar_result = ast.evaluate(scalar_context)
    print(f"\nWith scalar inputs (x=1, y=2, z=3):")
    print(f"Result: {scalar_result}")
    print(f"Expected: 1 + 2*3 = 7")
    
    # Evaluate with array inputs (element-wise operations)
    array_context = {
        'x': jnp.array([1.0, 2.0, 3.0]),
        'y': jnp.array([4.0, 5.0, 6.0]),
        'z': jnp.array([7.0, 8.0, 9.0])
    }
    
    array_result = ast.evaluate(array_context)
    print(f"\nWith array inputs:")
    print(f"x = {array_context['x']}")
    print(f"y = {array_context['y']}")
    print(f"z = {array_context['z']}")
    print(f"Result: {array_result}")
    print(f"Expected: [1+4*7, 2+5*8, 3+6*9] = [29, 42, 57]")
    
    # Evaluate with mixed dimensions (JAX handles broadcasting)
    mixed_context = {
        'x': jnp.array(2.0),  # scalar
        'y': jnp.array([1.0, 2.0, 3.0]),  # vector
        'z': jnp.array(3.0)   # scalar
    }
    
    mixed_result = ast.evaluate(mixed_context)
    print(f"\nWith mixed inputs (broadcasting):")
    print(f"x = {mixed_context['x']} (scalar)")
    print(f"y = {mixed_context['y']} (vector)")
    print(f"z = {mixed_context['z']} (scalar)")
    print(f"Result: {mixed_result}")
    print(f"Expected: [2+1*3, 2+2*3, 2+3*3] = [5, 8, 11]")

def example_custom_functions():
    """
    Example 3: Working with custom functions.
    
    Demonstrates how to register and use custom functions in formulas.
    """
    print("\n=== Example 3: Custom Functions ===")
    
    # Register some custom functions
    @register_function(category="financial", description="Calculate compound interest")
    def compound_interest(principal, rate, periods):
        """Calculate compound interest: principal * (1 + rate)^periods."""
        return principal * jnp.power(1 + rate, periods)
    
    @register_function(func_or_name="discount", category="financial")
    def present_value(future_value, rate, periods):
        """Calculate present value with discount rate."""
        return future_value / jnp.power(1 + rate, periods)
    
    @register_function(category="statistical")
    def weighted_average(values, weights):
        """Calculate weighted average of values."""
        return jnp.sum(values * weights) / jnp.sum(weights)
    
    # Print registered function categories
    categories = get_function_categories()
    print("Registered function categories:")
    for category, functions in categories.items():
        if functions:  # Only show non-empty categories
            print(f"  {category}:")
            for func in functions:
                print(f"    - {func}")
    
    # Example formulas using custom functions
    formulas = [
        "compound_interest(1000, 0.05, 5)",
        "discount(1200, 0.05, 3)",
        "weighted_average(values, weights)"
    ]
    
    # Get the function context for evaluation
    context = get_function_context()
    
    # Add test values to the context
    context.update({
        'principal': jnp.array(1000.0),
        'rate': jnp.array(0.05),
        'periods': jnp.array(5.0),
        'future_value': jnp.array(1200.0),
        'values': jnp.array([10.0, 20.0, 30.0]),
        'weights': jnp.array([1.0, 2.0, 3.0])
    })
    
    # Evaluate each formula
    for formula in formulas:
        print(f"\nFormula: {formula}")
        ast = parse_formula(formula)
        result = ast.evaluate(context)
        print(f"Result: {result}")
        
        # Add explanations for each result
        if formula == "compound_interest(1000, 0.05, 5)":
            expected = 1000 * (1.05 ** 5)
            print(f"Expected: 1000 * (1.05 ^ 5) = {expected:.2f}")
        elif formula == "discount(1200, 0.05, 3)":
            expected = 1200 / (1.05 ** 3)
            print(f"Expected: 1200 / (1.05 ^ 3) = {expected:.2f}")
        elif formula == "weighted_average(values, weights)":
            values = context['values']
            weights = context['weights']
            expected = np.sum(values * weights) / np.sum(weights)
            print(f"Expected: (10*1 + 20*2 + 30*3) / (1+2+3) = {expected:.2f}")

def example_formula_visualization():
    """
    Example 4: Visualizing formula ASTs.
    
    Shows how to create visual representations of formula ASTs.
    """
    print("\n=== Example 4: AST Visualization ===")
    
    # Create directory for visualizations
    os.makedirs("examples/visualizations", exist_ok=True)
    
    # Example formula for visualization
    formula = "x * (y + z) / (w - 2)"
    print(f"Visualizing formula: {formula}")
    
    # Create a parse tree visualization
    output_file = visualize_parse_tree(
        formula,
        output_path="examples/visualizations/parse_tree",
        view=False
    )
    print(f"Parse tree visualization saved to: {output_file}")
    
    # Create an AST visualization
    ast = parse_formula(formula)
    ast_output_file = visualize_ast(
        ast,
        output_path="examples/visualizations/ast_tree",
        view=False
    )
    print(f"AST visualization saved to: {ast_output_file}")
    
    # Explain the difference between parse tree and AST
    print("\nExplanation of visualizations:")
    print("- Parse Tree: Shows the complete parsing process and grammar rules applied")
    print("- AST: Shows the simplified abstract syntax tree used for evaluation")

def example_optimization():
    """
    Example 5: Formula optimization.
    
    Demonstrates how formulas can be optimized through constant folding.
    """
    print("\n=== Example 5: Formula Optimization ===")
    
    # Create directory for visualizations
    os.makedirs("examples/visualizations", exist_ok=True)
    
    # Example formulas to optimize
    formulas = [
        "2 + 3 * 4",
        "(5 - 2) * (4 + 1)",
        "x + (2 * 3)",
        "(y * 2) + (z * 3)",
        "5 * x * 2"  # Shows associative property
    ]
    
    for formula in formulas:
        print(f"\nOriginal formula: {formula}")
        
        # Parse the original formula
        original_ast = parse_formula(formula)
        
        # Optimize the formula
        optimized_ast = optimize_formula(formula)
        
        # Show the before and after
        print(f"Original AST: {original_ast}")
        print(f"Optimized AST: {optimized_ast}")
        
        # Create a visualization comparing the two ASTs
        from src.data.ast import compare_asts
        
        output_file = compare_asts(
            [(original_ast, "Original"), (optimized_ast, "Optimized")],
            output_path=f"examples/visualizations/opt_{formula.replace(' ', '_')}",
            view=False
        )
        print(f"Comparison visualization saved to: {output_file}")

def example_jit_compilation():
    """
    Example 6: Using JAX's JIT compilation.
    
    Shows how to use JAX's JIT compilation for faster formula evaluation.
    """
    print("\n=== Example 6: JIT Compilation ===")
    
    # Create a moderately complex formula
    formula = "a * sin(x) + b * cos(y) + c * tan(z)"
    ast = parse_formula(formula)
    print(f"Formula: {formula}")
    
    # Register trig functions if they aren't already
    @register_function(category="arithmetic")
    def sin(x):
        """Sine function."""
        return jnp.sin(x)
    
    @register_function(category="arithmetic")
    def cos(x):
        """Cosine function."""
        return jnp.cos(x)
    
    @register_function(category="arithmetic")
    def tan(x):
        """Tangent function."""
        return jnp.tan(x)
    
    # Get the function context
    context = get_function_context()
    
    # Define regular and JIT-compiled evaluation functions
    def evaluate_normal(a, b, c, x, y, z):
        """Regular evaluation function."""
        eval_context = context.copy()
        eval_context.update({
            'a': a, 'b': b, 'c': c,
            'x': x, 'y': y, 'z': z
        })

        result = ast.evaluate(eval_context)
        return result
    
    @jax.jit
    def evaluate_jit(a, b, c, x, y, z):
        """JIT-compiled evaluation function."""
        eval_context = {
            'a': a, 'b': b, 'c': c,
            'x': x, 'y': y, 'z': z
        }
        # Need to include the function context in the JIT
        for k, v in context.items():
            eval_context[k] = v

        result = ast.evaluate(eval_context)
        return result
    
    # Create large arrays for benchmarking
    size = 1000
    a = jnp.ones(size)
    b = jnp.ones(size) * 2
    c = jnp.ones(size) * 3
    x = jnp.linspace(0, jnp.pi, size)
    y = jnp.linspace(0, jnp.pi/2, size)
    z = jnp.linspace(0, jnp.pi/4, size)
    
    # Compile the JIT function (first call)
    print("Compiling JIT function...")
    start = time.time()
    _ = evaluate_jit(a, b, c, x, y, z)
    compile_time = time.time() - start
    print(f"JIT compilation time: {compile_time:.6f} seconds")
    
    # Benchmark both versions
    runs = 5
    jit_times = []
    normal_times = []
    
    print("\nBenchmarking JIT vs. non-JIT evaluation...")
    jit_result_val = None # Store the actual JIT result value
    normal_result_val = None # Store the actual non-JIT result value
    for i in range(runs):
        # Time JIT version
        start = time.time()
        jit_result_val = evaluate_jit(a, b, c, x, y, z)
        jit_times.append(time.time() - start)
        
        # Time normal version
        start = time.time()
        normal_result_val = evaluate_normal(a, b, c, x, y, z)
        normal_times.append(time.time() - start)
    
    # Compute averages
    avg_jit = sum(jit_times) / len(jit_times)
    avg_normal = sum(normal_times) / len(normal_times)
    speedup = avg_normal / avg_jit
    
    # Print results
    print(f"\nResults summary:")
    print(f"Average JIT time: {avg_jit:.6f} seconds")
    print(f"Average non-JIT time: {avg_normal:.6f} seconds")
    print(f"Speedup factor: {speedup:.2f}x")
    
    # Create a simple bar chart for visualization
    plt.figure(figsize=(10, 6))
    plt.bar(['JIT', 'Non-JIT'], [avg_jit, avg_normal])
    plt.title('JIT vs. Non-JIT Performance')
    plt.ylabel('Average Execution Time (seconds)')
    plt.yscale('log')  # Log scale for better visibility
    plt.savefig('examples/visualizations/jit_benchmark.png')
    print("\nBenchmark plot saved to: examples/visualizations/jit_benchmark.png")
    
    # Verify results match
    # Compare the actual result values extracted from the tuples
    match = jnp.allclose(jit_result_val, normal_result_val)
    print(f"Results match: {match}")

def example_rolling_operations():
    """
    Example 7: Rolling Operations with DataArray References and Result Datasets.
    
    Demonstrates how to use the $variable syntax
    """
    print("\n=== Example 7: Rolling Operations with Result Datasets ===")
    
    # Load a small dataset for demonstration
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
                    "set_permco_coord":  True,
                    "fix_market_equity": True
                }
            }
                }
            ]
        }
    )['wrds/equity/crsp']
    
    # Create adjusted price variable
    ds["adj_prc"] = ds["prc"] / ds["cfacpr"]
    
    print("\n=== Using $variable Syntax for DataArray References ===")
    
    # Define a complex formula involving intermediate steps
    formula = "1 + sma($adj_prc, 50) / sma($adj_prc, 200)"
    print(f"Formula: {formula}")
    
    # Parse the formula
    ast = parse_formula(formula)
    
    # Setup execution environment
    context = get_function_context()
    context['_dataset'] = ds  # Dataset for $variable references and result structure
    
    print("\nEvaluating formula")
    try:
        final_ds = ast.evaluate(
            context
        )
        print(f"Final value type: {type(final_ds).__name__}")

        # --- Plotting Example ---
        # Let's plot the 50-day and 200-day SMA
        if "sma($adj_prc, 50)" in final_ds and "sma($adj_prc, 200)" in final_ds and '$adj_prc' in final_ds:
            print("\nPlotting SMAs for the first asset...")
            
            # Select data for the first asset using its identifier (e.g., PERMNO)
            asset_permno = 14593  # Example: Apple's PERMNO
            if asset_permno not in ds['asset'].values:
                 print(f"Warning: Asset {asset_permno} not found. Using first available asset.")
                 asset_permno = ds['asset'].values[0]
                 
            # Select the subset for the chosen asset
            plot_vars = ["$adj_prc", "sma($adj_prc, 50)", "sma($adj_prc, 200)"]
            asset_ds_subset = final_ds[plot_vars].sel(asset=asset_permno)
            
            try:
                asset_ts = asset_ds_subset.dt.to_time_indexed()
            except Exception as plot_prep_e:
                 print(f"Error preparing data for plotting: {plot_prep_e}")
                 asset_ts = None

            if asset_ts is not None:
                plt.figure(figsize=(12, 6))
                
                # Plot original price and SMAs
                asset_ts['$adj_prc'].plot.line(x="time", label='Original adj_prc', color='blue', alpha=0.5)
                asset_ts["sma($adj_prc, 50)"].plot.line(x="time", label='SMA(50)', color='orange')
                asset_ts["sma($adj_prc, 200)"].plot.line(x="time", label='SMA(200)', color='red')
            
                plt.title(f"Adjusted Price vs. SMAs (Asset: {asset_permno})")
                plt.xlabel("Time")
                plt.ylabel("Price")
                plt.legend()
                plt.grid(True)
                
                plot_path = "examples/visualizations/sma_comparison.png"
                plt.savefig(plot_path)
                plt.close()
                print(f"SMA comparison plot saved to: {plot_path}")
            else:
                 print("Skipping plot due to data preparation error.")
        # ------------------------

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

def main():
    """Run all examples."""
    print("=== AST-Based Formula Evaluation System Examples ===")
    print("This script demonstrates the core features of the AST formula system.")
    
    # Create visualizations directory
    os.makedirs("examples/visualizations", exist_ok=True)
    
    # Run examples
    example_simple_parsing()
    example_formula_evaluation()
    example_custom_functions()
    example_formula_visualization()
    example_optimization()
    example_jit_compilation()
    example_rolling_operations()
    
    print("\nAll examples completed. Visualizations saved to examples/visualizations/")

if __name__ == "__main__":
    main() 