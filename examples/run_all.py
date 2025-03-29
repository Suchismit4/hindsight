#!/usr/bin/env python
"""
Script to run all examples in sequence.

This script automatically discovers and runs all example scripts
in the examples directory.
"""

import os
import sys
import time
import importlib.util
from pathlib import Path

# Add the project root to the path to make imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def find_example_files():
    """
    Find all example Python files in the examples directory.
    
    Returns:
        List of paths to example files, excluding __init__.py and runner scripts
    """
    examples_dir = Path(__file__).parent
    example_files = []
    
    for path in examples_dir.glob('**/*.py'):
        # Skip __init__.py and runner scripts
        if path.name.startswith('__') or path.name.startswith('run_'):
            continue
        
        # Only include files in the examples directory
        if examples_dir in path.parents:
            example_files.append(path)
    
    return sorted(example_files)

def run_example(example_path):
    """
    Run an example script and return success status.
    
    Args:
        example_path: Path to the example script
        
    Returns:
        Tuple of (success status, elapsed time)
    """
    rel_path = example_path.relative_to(Path(__file__).parent)
    print(f"\n{'='*80}")
    print(f"Running example: {rel_path}")
    print(f"{'='*80}")
    
    try:
        # Import and run the example
        start_time = time.time()
        
        spec = importlib.util.spec_from_file_location("example_module", example_path)
        example_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example_module)
        
        # If the module has a main() function, call it
        if hasattr(example_module, 'main'):
            example_module.main()
        
        elapsed = time.time() - start_time
        return True, elapsed
    except Exception as e:
        print(f"\nError running example {rel_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0

def main():
    """Find and run all examples."""
    print("=== Running all hindsight library examples ===\n")
    
    # Find all example files
    example_files = find_example_files()
    print(f"Found {len(example_files)} examples to run")
    
    # Run each example
    results = []
    for i, example_path in enumerate(example_files, 1):
        rel_path = example_path.relative_to(Path(__file__).parent)
        print(f"\n[{i}/{len(example_files)}] Running: {rel_path}")
        
        success, elapsed = run_example(example_path)
        results.append((rel_path, success, elapsed))
    
    # Print summary
    print("\n\n" + "="*80)
    print("= Summary of Example Runs")
    print("="*80)
    
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    print(f"\nSuccessful examples: {len(successful)}/{len(results)}")
    for path, _, elapsed in successful:
        print(f"  ✓ {path} - {elapsed:.2f}s")
    
    if failed:
        print(f"\nFailed examples: {len(failed)}/{len(results)}")
        for path, _, _ in failed:
            print(f"  ✗ {path}")
    
    total_time = sum(r[2] for r in results if r[1])
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main() 