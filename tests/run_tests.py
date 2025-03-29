#!/usr/bin/env python
"""
Test runner for the hindsight library.

This script runs all tests in the tests directory and displays the results.
"""

import os
import sys
import unittest
import argparse
import time

# Add the project root to the path to make imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def get_test_modules(directory=None):
    """
    Discover all test modules in the given directory.
    
    Args:
        directory: Directory to search for test modules (None for all tests)
        
    Returns:
        List of test module names
    """
    if directory is None:
        # Run all tests
        return unittest.defaultTestLoader.discover(
            os.path.dirname(__file__), 
            pattern='test_*.py'
        )
    else:
        # Run tests in the specified directory
        return unittest.defaultTestLoader.discover(
            os.path.join(os.path.dirname(__file__), directory),
            pattern='test_*.py'
        )

def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description='Run hindsight library tests')
    parser.add_argument('--module', '-m', 
                        help='Run tests for a specific module (e.g., "data/ast")')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Display verbose output')
    args = parser.parse_args()
    
    # Determine verbosity level
    verbosity = 2 if args.verbose else 1
    
    # Create test suite
    test_suite = get_test_modules(args.module)
    
    # Run tests
    print("=== Running hindsight library tests ===")
    start_time = time.time()
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(test_suite)
    
    elapsed = time.time() - start_time
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Ran {result.testsRun} tests in {elapsed:.2f} seconds")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    # Set exit code based on test results
    sys.exit(len(result.failures) + len(result.errors))

if __name__ == "__main__":
    main() 