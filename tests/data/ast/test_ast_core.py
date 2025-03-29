#!/usr/bin/env python
"""
Unit tests for the AST-based formula evaluation system core functionality.

These tests verify the core functionality of the AST system:
- Parsing formulas into AST nodes
- Evaluating formulas with different types of input values
- Handling operator precedence and associativity
- Variable substitution and resolution
- Error handling and edge cases

Each test includes detailed assertions and documentation to make it clear
what is being tested and what the expected outcomes are.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os
import time

# Add the project to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.data.ast import (
    parse_formula,
    optimize_formula,
    extract_variables,
    extract_functions
)

from src.data.ast.functions import get_function_context

class TestASTCore(unittest.TestCase):
    """Test the core functionality of the AST-based formula evaluation system."""

    def setUp(self):
        """Set up common test variables."""
        # Common test contexts
        self.scalar_context = {
            'x': jnp.array(1.0),
            'y': jnp.array(2.0),
            'z': jnp.array(3.0),
            'w': jnp.array(4.0)
        }
        
        self.array_context = {
            'x': jnp.array([1.0, 2.0, 3.0]),
            'y': jnp.array([4.0, 5.0, 6.0]),
            'z': jnp.array([7.0, 8.0, 9.0]),
            'w': jnp.array([10.0, 11.0, 12.0])
        }
        
        self.mixed_context = {
            'scalar': jnp.array(5.0),
            'vector': jnp.array([1.0, 2.0, 3.0]),
            'matrix': jnp.array([[1.0, 2.0], [3.0, 4.0]])
        }

    def test_simple_arithmetic(self):
        """Test basic arithmetic operations with constants."""
        test_cases = [
            # (formula, expected_result)
            ("2 + 3", 5.0),
            ("2 - 3", -1.0),
            ("2 * 3", 6.0),
            ("6 / 3", 2.0),
            ("2 ^ 3", 8.0),  # Exponentiation
            ("-5", -5.0),    # Unary minus
            ("+5", 5.0)      # Unary plus
        ]
        
        for formula, expected in test_cases:
            with self.subTest(formula=formula):
                ast = parse_formula(formula)
                result = ast.evaluate({})
                self.assertAlmostEqual(float(result), expected, 
                    msg=f"Formula '{formula}' should evaluate to {expected}, got {result}")

    def test_operator_precedence(self):
        """Test that operator precedence follows mathematical conventions."""
        test_cases = [
            # (formula, expected_result)
            ("2 + 3 * 4", 14.0),       # Multiplication before addition
            ("2 * 3 + 4", 10.0),       # Multiplication before addition
            ("6 / 3 + 2", 4.0),        # Division before addition
            ("6 / (3 + 2)", 1.2),      # Parentheses override precedence
            ("2 ^ 3 * 4", 32.0),       # Exponentiation before multiplication
            ("2 * 3 ^ 2", 18.0),       # Exponentiation before multiplication
            ("(2 + 3) * 4", 20.0),     # Parentheses override precedence
            ("-2 ^ 2", -4.0),          # Unary minus before exponentiation
            ("-(2 ^ 2)", -4.0),        # Parentheses with unary minus
            ("2 * -3", -6.0),          # Unary minus with multiplication
            ("5 - 2 + 1", 4.0),        # Left-to-right for same precedence
            ("5 - (2 + 1)", 2.0)       # Parentheses override associativity
        ]
        
        for formula, expected in test_cases:
            with self.subTest(formula=formula):
                ast = parse_formula(formula)
                result = ast.evaluate({})
                self.assertAlmostEqual(float(result), expected, 
                    msg=f"Formula '{formula}' should evaluate to {expected}, got {result}")

    def test_variable_substitution(self):
        """Test that variables can be substituted with values in different contexts."""
        test_cases = [
            # (formula, expected_with_scalar_context)
            ("x", 1.0),
            ("y", 2.0),
            ("x + y", 3.0),
            ("x * y", 2.0),
            ("x + y * z", 7.0),        # Should be 1 + (2 * 3)
            ("(x + y) * z", 9.0),      # Should be (1 + 2) * 3
            ("w / (x + y)", 4.0/3.0),  # Should be 4 / (1 + 2)
            ("x ^ y", 1.0),            # 1^2 = 1
            ("y ^ x", 2.0),            # 2^1 = 2
            ("z ^ w", 81.0)            # 3^4 = 81
        ]
        
        for formula, expected in test_cases:
            with self.subTest(formula=formula):
                ast = parse_formula(formula)
                result = ast.evaluate(self.scalar_context)
                self.assertAlmostEqual(float(result), expected, 
                    msg=f"Formula '{formula}' with scalar context should evaluate to {expected}, got {result}")

    def test_array_operations(self):
        """Test operations with array inputs."""
        test_cases = [
            # (formula, expected_calc)
            ("x + y", lambda x, y, z, w: x + y),
            ("x * y", lambda x, y, z, w: x * y),
            ("x / y", lambda x, y, z, w: x / y),
            ("x + y * z", lambda x, y, z, w: x + y * z),
            ("(x + y) * z", lambda x, y, z, w: (x + y) * z),
            ("w - x", lambda x, y, z, w: w - x)
        ]
        
        # Use the array context for these tests
        x = self.array_context['x']
        y = self.array_context['y']
        z = self.array_context['z']
        w = self.array_context['w']
        
        for formula, expected_calc in test_cases:
            with self.subTest(formula=formula):
                ast = parse_formula(formula)
                result = ast.evaluate(self.array_context)
                expected = expected_calc(x, y, z, w)
                self.assertTrue(jnp.allclose(result, expected), 
                    msg=f"Formula '{formula}' with array context failed. Expected {expected}, got {result}")

    def test_mixed_dimensionality(self):
        """Test operations with mixed dimensionality inputs."""
        # These operations should work with JAX's broadcasting rules
        test_cases = [
            # (formula, shape_check)
            ("scalar + vector", lambda res: res.shape == (3,)),
            ("scalar * matrix", lambda res: res.shape == (2, 2)),
            ("vector + 5", lambda res: res.shape == (3,)),
            ("matrix * scalar", lambda res: res.shape == (2, 2)),
            ("matrix + 1", lambda res: res.shape == (2, 2))
        ]
        
        for formula, shape_check in test_cases:
            with self.subTest(formula=formula):
                ast = parse_formula(formula)
                result = ast.evaluate(self.mixed_context)
                self.assertTrue(shape_check(result), 
                    msg=f"Formula '{formula}' did not produce expected shape. Got shape {result.shape}")

    def test_error_handling(self):
        """Test that appropriate errors are raised for invalid formulas or operations."""
        error_cases = [
            # (formula, context, expected_error_type, error_msg_contains)
            ("x + y", {}, KeyError, "variable 'x' not found"),
            # Note: JAX doesn't raise ZeroDivisionError, it returns NaN silently
            # So we don't test for division by zero errors
        ]
        
        for formula, context, error_type, error_msg in error_cases:
            with self.subTest(formula=formula):
                ast = parse_formula(formula)
                with self.assertRaises(error_type) as cm:
                    ast.evaluate(context)
                if error_msg:
                    self.assertIn(error_msg.lower(), str(cm.exception).lower(), 
                        f"Expected error message containing '{error_msg}' for formula '{formula}'")
        
        # Test that division by zero returns NaN rather than raising an exception
        div_zero_formulas = ["1 / 0", "x / 0"]
        for formula in div_zero_formulas:
            with self.subTest(formula=formula):
                ast = parse_formula(formula)
                if formula == "x / 0":
                    result = ast.evaluate({'x': jnp.array(1.0)})
                else:
                    result = ast.evaluate({})
                self.assertTrue(jnp.isnan(result), f"Expected NaN for {formula}, got {result}")

    def test_formula_extraction(self):
        """Test extraction of variables and functions from formulas."""
        test_cases = [
            # (formula, expected_variables, expected_functions)
            ("x + y", {'x', 'y'}, set()),
            ("x + y * z", {'x', 'y', 'z'}, set()),
            ("add(x, y)", {'x', 'y'}, {'add'}),
            ("add(x, multiply(y, z))", {'x', 'y', 'z'}, {'add', 'multiply'}),
            ("5 + 3", set(), set()),
            ("sin(x) + cos(y)", {'x', 'y'}, {'sin', 'cos'}),
            ("x + f(y, g(z))", {'x', 'y', 'z'}, {'f', 'g'})
        ]
        
        for formula, expected_vars, expected_funcs in test_cases:
            with self.subTest(formula=formula):
                variables = extract_variables(formula)
                functions = extract_functions(formula)
                
                self.assertEqual(variables, expected_vars, 
                    f"Expected variables {expected_vars} for formula '{formula}', got {variables}")
                self.assertEqual(functions, expected_funcs, 
                    f"Expected functions {expected_funcs} for formula '{formula}', got {functions}")

    def test_optimization(self):
        """Test that formulas can be optimized by constant folding."""
        test_cases = [
            # (formula, expected_optimized_str)
            ("2 + 3", "5"),  # Integer literals are formatted without decimal points
            ("2 * 3 + 4", "10"),
            ("(2 + 3) * (4 - 1)", "15"),
            ("2 + x", "2 + x"),  # Partial optimization
            ("x + 2 + 3", "x + 2 + 3"),  # Our optimizer doesn't handle associative operations yet
            ("x * (2 + 3)", "x * 5"),  # Partial optimization
            ("(x + y) * (2 + 3)", "(x + y) * 5"),  # Partial optimization
        ]
        
        for formula, expected_str in test_cases:
            with self.subTest(formula=formula):
                optimized_ast = optimize_formula(formula)
                self.assertEqual(str(optimized_ast), expected_str, 
                    f"Formula '{formula}' should optimize to '{expected_str}', got '{str(optimized_ast)}'")

    def test_jit_performance(self):
        """Test that JAX JIT compilation improves performance."""
        # Choose a moderately complex formula for benchmarking
        formula = "x * (y + z) / (w - y + 1)"
        ast = parse_formula(formula)
        
        # Create large arrays for performance testing
        large_context = {
            'x': jnp.ones((500, 500)),
            'y': jnp.ones((500, 500)) * 2,
            'z': jnp.ones((500, 500)) * 3,
            'w': jnp.ones((500, 500)) * 6
        }
        
        # Define evaluation functions
        def evaluate_no_jit(x, y, z, w):
            context = {'x': x, 'y': y, 'z': z, 'w': w}
            return ast.evaluate(context)
        
        @jax.jit
        def evaluate_jit(x, y, z, w):
            context = {'x': x, 'y': y, 'z': z, 'w': w}
            return ast.evaluate(context)
        
        # JIT warmup - first call includes compilation
        _ = evaluate_jit(**{k: large_context[k] for k in ['x', 'y', 'z', 'w']})
        
        # Time JIT version
        jit_times = []
        for _ in range(3):
            start = time.time()
            _ = evaluate_jit(**{k: large_context[k] for k in ['x', 'y', 'z', 'w']})
            jit_times.append(time.time() - start)
        
        # Time non-JIT version
        no_jit_times = []
        for _ in range(3):
            start = time.time()
            _ = evaluate_no_jit(**{k: large_context[k] for k in ['x', 'y', 'z', 'w']})
            no_jit_times.append(time.time() - start)
        
        avg_jit_time = sum(jit_times) / len(jit_times)
        avg_no_jit_time = sum(no_jit_times) / len(no_jit_times)
        
        # JIT should be significantly faster (at least 10x)
        self.assertTrue(avg_no_jit_time > avg_jit_time * 10, 
            f"JIT compilation should provide at least 10x speedup. JIT: {avg_jit_time:.6f}s, Non-JIT: {avg_no_jit_time:.6f}s")
        
        print(f"\nPerformance test results for formula '{formula}':")
        print(f"JIT average time: {avg_jit_time:.6f} seconds")
        print(f"Non-JIT average time: {avg_no_jit_time:.6f} seconds")
        print(f"Speedup: {avg_no_jit_time / avg_jit_time:.2f}x")

if __name__ == "__main__":
    unittest.main() 