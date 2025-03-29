#!/usr/bin/env python
"""
Example runner for the hindsight library.

This script runs the specified examples from the examples directory.
"""

import os
import sys
import argparse
import importlib.util
import time

# Add the project root to the path to make imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def list_available_examples():
    """List all available example scripts in the examples directory."""
    examples = []
    
    for root, _, files in os.walk(os.path.dirname(__file__)):
        rel_path = os.path.relpath(root, os.path.dirname(__file__))
        
        for file in files:
            if file.endswith('.py') and file != '__init__.py' and file != 'run_examples.py':
                if rel_path == '.':
                    examples.append(file[:-3])  # Remove .py extension
                else:
                    examples.append(os.path.join(rel_path, file[:-3]))
    
    return sorted(examples)

def run_example(example_path):
    """
    Run the specified example script.
    
    Args:
        example_path: Path to the example script, relative to the examples directory
    """
    # Construct the full path to the example script
    if not example_path.endswith('.py'):
        example_path += '.py'
    
    full_path = os.path.join(os.path.dirname(__file__), example_path)
    
    if not os.path.exists(full_path):
        print(f"Error: Example '{example_path}' not found")
        return False
    
    # Import and run the example
    spec = importlib.util.spec_from_file_location("example_module", full_path)
    example_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example_module)
    
    print(f"\n=== Running example: {example_path} ===")
    
    # If the module has a main() function, call it
    if hasattr(example_module, 'main'):
        example_module.main()
    
    return True

def main():
    """Parse arguments and run examples."""
    parser = argparse.ArgumentParser(description='Run hindsight library examples')
    parser.add_argument('example', nargs='?', 
                       help='Name of the example to run (e.g., "data/ast/basic_usage")')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available examples')
    args = parser.parse_args()
    
    if args.list:
        print("Available examples:")
        for example in list_available_examples():
            print(f"  {example}")
        return
    
    if args.example:
        # Run a single example
        start_time = time.time()
        success = run_example(args.example)
        elapsed = time.time() - start_time
        
        if success:
            print(f"\nExample '{args.example}' completed in {elapsed:.2f} seconds")
    else:
        # List available examples if no example is specified
        print("Please specify an example to run, or use --list to see all available examples")
        print("\nAvailable examples:")
        for example in list_available_examples():
            print(f"  {example}")

if __name__ == "__main__":
    main() 