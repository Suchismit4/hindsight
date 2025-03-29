#!/usr/bin/env python
"""
Unit tests for the AST function registry system.

These tests verify the function registry system works correctly:
- Function registration and retrieval
- Function categorization
- Custom function evaluation
- Function metadata handling
- Error cases and edge behaviors

Each test includes detailed assertions and documentation to make it clear
what is being tested and what the expected outcomes are.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add the project to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.data.ast import (
    parse_formula,
    register_function,
    extract_functions
)

from src.data.ast.functions import (
    get_registered_functions,
    get_function_context,
    get_function_categories,
    get_function_metadata,
    unregister_function,
    clear_registry
)

class TestFunctionRegistry(unittest.TestCase):
    """Test the function registry system of the AST module."""
    
    def setUp(self):
        """Set up test environment and register test functions."""
        # Save the current functions so we can restore them later
        self.original_functions = get_registered_functions().copy()
        
        # Clear the registry to start with a clean slate
        clear_registry()
        
        # Register some test functions
        @register_function
        def add2(a, b):
            """Add two numbers."""
            return a + b
        
        @register_function(func_or_name="multiply", category="arithmetic", description="Multiply two numbers")
        def mult(a, b):
            """Multiply two numbers."""
            return a * b
        
        @register_function(func_or_name="square", category="arithmetic")
        def sqr(x):
            """Square a number."""
            return x * x
        
        @register_function(category="statistical", description="Calculate mean of array")
        def mean(x):
            """Calculate the mean of an array."""
            return jnp.mean(x)
        
        @register_function(category="statistical")
        def variance(x):
            """Calculate the variance of an array."""
            return jnp.var(x)
        
        # Function with default parameters
        @register_function(category="miscellaneous")
        def weighted_sum(values, weights=None):
            """Calculate weighted sum. If weights are None, use equal weights."""
            if weights is None:
                weights = jnp.ones_like(values)
            return jnp.sum(values * weights)
    
    def tearDown(self):
        """Clean up after tests by restoring original functions."""
        clear_registry()
        for name, func in self.original_functions.items():
            register_function(func_or_name=name)(func)

    def test_function_registration(self):
        """Test that functions can be registered and retrieved correctly."""
        # Check if our test functions are registered
        functions = get_registered_functions()
        self.assertIn('add2', functions, "Function 'add2' should be registered")
        self.assertIn('multiply', functions, "Function 'multiply' should be registered")
        self.assertIn('square', functions, "Function 'square' should be registered")
        self.assertIn('mean', functions, "Function 'mean' should be registered")
        self.assertIn('variance', functions, "Function 'variance' should be registered")
        
        # Verify function implementations
        self.assertEqual(functions['add2'](2, 3), 5, "add2(2, 3) should return 5")
        self.assertEqual(functions['multiply'](2, 3), 6, "multiply(2, 3) should return 6")
        self.assertEqual(functions['square'](3), 9, "square(3) should return 9")
        
        # Test mean and variance with arrays
        test_array = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(functions['mean'](test_array), 3.0, "mean([1,2,3,4,5]) should return 3.0")
        self.assertEqual(functions['variance'](test_array), 2.0, "variance([1,2,3,4,5]) should return 2.0")

    def test_function_categories(self):
        """Test that functions are correctly categorized."""
        categories = get_function_categories()
        
        # Check for expected categories
        self.assertIn('arithmetic', categories, "Should have 'arithmetic' category")
        self.assertIn('statistical', categories, "Should have 'statistical' category")
        self.assertIn('miscellaneous', categories, "Should have 'miscellaneous' category")
        
        # Check that functions are in the right categories
        self.assertIn('add2', categories['miscellaneous'], "add2 should be in 'miscellaneous'")
        self.assertIn('multiply', categories['arithmetic'], "multiply should be in 'arithmetic'")
        self.assertIn('square', categories['arithmetic'], "square should be in 'arithmetic'")
        self.assertIn('mean', categories['statistical'], "mean should be in 'statistical'")
        self.assertIn('variance', categories['statistical'], "variance should be in 'statistical'")
        self.assertIn('weighted_sum', categories['miscellaneous'], "weighted_sum should be in 'miscellaneous'")

    def test_function_metadata(self):
        """Test that function metadata is stored and retrieved correctly."""
        # Check metadata for add2 (default metadata)
        add2_meta = get_function_metadata('add2')
        self.assertEqual(add2_meta['category'], 'miscellaneous', "add2 should be in 'miscellaneous' category")
        self.assertIn('Add two numbers', add2_meta['description'], "add2 description should include docstring")
        
        # Check metadata for multiply (explicit metadata)
        multiply_meta = get_function_metadata('multiply')
        self.assertEqual(multiply_meta['category'], 'arithmetic', "multiply should be in 'arithmetic' category")
        self.assertEqual(multiply_meta['description'], 'Multiply two numbers', 
                        "multiply should have custom description")
        
        # Check metadata for a function with provided category but default description
        square_meta = get_function_metadata('square')
        self.assertEqual(square_meta['category'], 'arithmetic', "square should be in 'arithmetic' category")
        self.assertIn('Square a number', square_meta['description'], 
                      "square description should include docstring")
        
        # Check signature is recorded correctly
        self.assertIn('(a, b)', add2_meta['signature'], "add2 signature should include parameters a and b")
        self.assertIn('(x)', square_meta['signature'], "square signature should include parameter x")
        
        # Check signature with default parameters
        weighted_sum_meta = get_function_metadata('weighted_sum')
        self.assertIn('weights=None', weighted_sum_meta['signature'], 
                     "weighted_sum signature should include default parameter")

    def test_function_unregistration(self):
        """Test that functions can be unregistered."""
        # Unregister a function
        unregister_function('add2')
        
        # Verify it's gone
        functions = get_registered_functions()
        self.assertNotIn('add2', functions, "add2 should be unregistered")
        
        # Verify it's removed from categories
        categories = get_function_categories()
        self.assertNotIn('add2', categories['miscellaneous'], "add2 should be removed from categories")
        
        # Verify error is raised when trying to get metadata
        with self.assertRaises(ValueError):
            get_function_metadata('add2')
        
        # Verify error is raised when trying to unregister again
        with self.assertRaises(ValueError):
            unregister_function('add2')

    def test_function_context(self):
        """Test that function context is generated correctly for AST evaluation."""
        context = get_function_context()
        
        # Check context keys are formatted correctly
        self.assertIn('_func_add2', context, "Context should have '_func_add2' key")
        self.assertIn('_func_multiply', context, "Context should have '_func_multiply' key")
        
        # Verify functions in context work correctly
        self.assertEqual(context['_func_add2'](2, 3), 5, "Context function add2 should work")
        self.assertEqual(context['_func_multiply'](2, 3), 6, "Context function multiply should work")

    def test_formula_evaluation_with_functions(self):
        """Test evaluating formulas with registered functions."""
        # Get function context for evaluation
        context = get_function_context()
        
        # Add some variables to the context
        context.update({
            'x': jnp.array(2.0),
            'y': jnp.array(3.0),
            'z': jnp.array(4.0)
        })
        
        # Test formulas with functions
        test_cases = [
            ("add2(x, y)", 5.0),
            ("multiply(x, y)", 6.0),
            ("square(x)", 4.0),
            ("add2(x, multiply(y, z))", 14.0),
            ("square(add2(x, y))", 25.0)
        ]
        
        for formula, expected in test_cases:
            with self.subTest(formula=formula):
                ast = parse_formula(formula)
                result = ast.evaluate(context)
                self.assertAlmostEqual(float(result), expected, 
                    msg=f"Formula '{formula}' should evaluate to {expected}, got {result}")

    def test_extract_functions_from_formula(self):
        """Test extracting function names from formulas."""
        test_cases = [
            ("add2(x, y)", {'add2'}),
            ("multiply(x, y)", {'multiply'}),
            ("add2(x, multiply(y, z))", {'add2', 'multiply'}),
            ("square(add2(x, y))", {'square', 'add2'}),
            ("x + y", set()),
            ("mean(x) + variance(y)", {'mean', 'variance'})
        ]
        
        for formula, expected in test_cases:
            with self.subTest(formula=formula):
                functions = extract_functions(formula)
                self.assertEqual(functions, expected, 
                    f"Expected functions {expected} from formula '{formula}', got {functions}")

    def test_function_registration_errors(self):
        """Test error handling for function registration."""
        # Test registering with same name
        with self.assertRaises(ValueError):
            @register_function
            def add2(a, b):
                return a + b
        
        # Test registering with invalid category
        with self.assertRaises(ValueError):
            @register_function(category="invalid_category")
            def new_func(a):
                return a

if __name__ == "__main__":
    unittest.main() 