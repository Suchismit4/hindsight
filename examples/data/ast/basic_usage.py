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

# Add the project root to the path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src import DataManager

from src.data.ast import (
    parse_formula,
    register_function,
    extract_variables,
    extract_functions,
    visualize_ast,
    visualize_parse_tree,
    get_grammar_description,
    optimize_formula
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
        return ast.evaluate(eval_context)
    
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
        return ast.evaluate(eval_context)
    
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
    for i in range(runs):
        # Time JIT version
        start = time.time()
        jit_result = evaluate_jit(a, b, c, x, y, z)
        jit_times.append(time.time() - start)
        
        # Time normal version
        start = time.time()
        normal_result = evaluate_normal(a, b, c, x, y, z)
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
    match = jnp.allclose(jit_result, normal_result)
    print(f"Results match: {match}")

def example_rolling_operations():
    """
    Example 6: Rolling Operations.
    
    Demonstrates how to use rolling statistical functions to perform
    calculations on time series data. Shows both the AST formula representation
    and the actual computation using the dataset.dt.rolling accessor.
    Also benchmarks JIT vs non-JIT performance for these operations.
    """
    print("\n=== Example 6: Rolling Operations ===")
    
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
    
    print(f"Created sample dataset: {ds}")
    
    ds["adj_prc"] = ds["prc"] / ds["cfacpr"]
    
    # Create formula AST for calculations
    print("\nParsing formulas for rolling calculations:")
    
    # Moving average with window of 30 days
    ma_formula = "moving_mean(dataset, 30)"
    ma_ast = parse_formula(ma_formula)
    print(f"Formula: {ma_formula}")
    
    # Exponential moving average with window of 30 days
    ema_formula = "moving_ema(dataset, 30)"
    ema_ast = parse_formula(ema_formula)
    print(f"Formula: {ema_formula}")
    
    # Example of more complex formula using rolling operations
    complex_formula = "moving_mean(dataset, 10) / moving_mean(dataset, 30)"
    complex_ast = parse_formula(complex_formula)
    print(f"Complex ratio formula: {complex_formula}")
    
    # Visualize the AST for a complex formula
    os.makedirs("examples/visualizations", exist_ok=True)
    vis_file = visualize_ast(
        complex_ast,
        output_path="examples/visualizations/rolling_operations_ast",
        view=False
    )
    print(f"AST visualization saved to: {vis_file}")
    
    # =======================================================
    # Benchmark JIT vs non-JIT performance for rolling operations
    # =======================================================
    print("\n=== Benchmarking Rolling Operations: JIT vs Non-JIT ===")
    
    # Get the function context
    context = get_function_context()
    
    # Create non-JIT and JIT evaluation functions
    def evaluate_normal(dataset, window1, window2):
        """Evaluate all formulas without JIT compilation."""
        eval_context = context.copy()
        eval_context.update({
            'dataset': dataset
        })
        
        # Evaluate moving mean with window1
        eval_context['window'] = window1
        ma_result1 = ma_ast.evaluate(eval_context)
        
        # Evaluate moving mean with window2
        eval_context['window'] = window2
        ma_result2 = ma_ast.evaluate(eval_context)
        
        # Evaluate moving EMA with window1
        ema_result = ema_ast.evaluate(eval_context)
        
        # Evaluate complex formula (ratio of moving means)
        complex_result = complex_ast.evaluate(eval_context)
        
        return ma_result1, ma_result2, ema_result, complex_result
    
    @jax.jit
    def evaluate_jit(dataset, window1, window2):
        """Evaluate all formulas with JIT compilation."""
        eval_context = {
            'dataset': dataset
        }
        
        # Include the function context
        for k, v in context.items():
            eval_context[k] = v
        
        # Evaluate moving mean with window1
        eval_context['window'] = window1
        ma_result1 = ma_ast.evaluate(eval_context)
        
        # Evaluate moving mean with window2
        eval_context['window'] = window2
        ma_result2 = ma_ast.evaluate(eval_context)
        
        # Evaluate moving EMA with window1
        ema_result = ema_ast.evaluate(eval_context)
        
        # Evaluate complex formula (ratio of moving means)
        complex_result = complex_ast.evaluate(eval_context)
        
        return ma_result1, ma_result2, ema_result, complex_result
    
    # Prepare data for benchmarking
    try:
        # Use a subset of assets for the benchmark
        if len(ds.asset) > 10:
            subset_ds = ds.isel(asset=slice(0, 10))
        else:
            subset_ds = ds
            
        # Use a subset of time for faster benchmarking if needed
        if len(subset_ds.time) > 500:
            subset_ds = subset_ds.isel(time=slice(0, 500))
        
        # Convert dataset to dictionary for the benchmark
        dataset_dict = {}
        for var_name in subset_ds.data_vars:
            dataset_dict[var_name] = subset_ds[var_name].values
            
        window1 = jnp.array(30)
        window2 = jnp.array(10)
        
        # Compile the JIT function (first call)
        print("Compiling JIT function...")
        start = time.time()
        _ = evaluate_jit(dataset_dict, window1, window2)
        compile_time = time.time() - start
        print(f"JIT compilation time: {compile_time:.6f} seconds")
        
        # Benchmark both versions
        runs = 10
        jit_times = []
        normal_times = []
        
        print("\nBenchmarking rolling operations with JIT vs. non-JIT evaluation...")
        for i in range(runs):
            # Time JIT version
            start = time.time()
            jit_results = evaluate_jit(dataset_dict, window1, window2)
            jit_times.append(time.time() - start)
            
            # Time normal version
            start = time.time()
            normal_results = evaluate_normal(dataset_dict, window1, window2)
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
        plt.title('Rolling Operations: JIT vs. Non-JIT Performance')
        plt.ylabel('Average Execution Time (seconds)')
        plt.yscale('log')  # Log scale for better visibility
        plt.savefig('examples/visualizations/rolling_operations_benchmark.png')
        print("\nBenchmark plot saved to: examples/visualizations/rolling_operations_benchmark.png")
        
        # Verify results match
        results_match = all([
            jnp.allclose(jit_results[i], normal_results[i], equal_nan=True)
            for i in range(len(jit_results))
        ])
        print(f"Results match: {results_match}")
        
        # Print detailed results for one variable in the dataset
        var_name = "adj_prc"
        if var_name in dataset_dict:
            print(f"\nDetailed results for {var_name} (first asset, last time point):")
            print(f"30-day MA: {normal_results[0][var_name][0, -1] if normal_results[0][var_name].size > 0 else 'N/A'}")
            print(f"10-day MA: {normal_results[1][var_name][0, -1] if normal_results[1][var_name].size > 0 else 'N/A'}")
            print(f"30-day EMA: {normal_results[2][var_name][0, -1] if normal_results[2][var_name].size > 0 else 'N/A'}")
            print(f"Ratio (10-day MA / 30-day MA): {normal_results[3][var_name][0, -1] if normal_results[3][var_name].size > 0 else 'N/A'}")
        
    except Exception as e:
        print(f"\nBenchmark could not be completed: {e}")
        print("Make sure the dataset contains the expected structure and variables.")

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