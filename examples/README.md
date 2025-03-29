# Hindsight Library Examples

This directory contains example scripts demonstrating the usage of the hindsight library.

## Directory Structure

The examples are organized by module:

```
examples/
├── data/
│   ├── ast/                    # AST module examples
│   │   ├── basic_usage.py      # Basic usage examples
│   │   ├── advanced_usage.py   # Advanced financial formula examples
│   │   └── ...
│   └── ...
└── ...
```

## Running Examples

### Run a Specific Example

To run a specific example:

```bash
python examples/run_examples.py data/ast/basic_usage
```

### List All Available Examples

To see a list of all available examples:

```bash
python examples/run_examples.py --list
```

## Example Descriptions

### AST Module Examples

1. **Basic Usage** (`data/ast/basic_usage.py`):
   - Demonstrates core functionality of the AST system
   - Shows how to parse and evaluate simple formulas
   - Explains variable substitution and formula visualization
   - Includes examples of function registration and JIT compilation

2. **Advanced Usage** (`data/ast/advanced_usage.py`):
   - Shows advanced financial applications of the AST system
   - Includes complex financial formulas (CAPM, Sharpe ratio, etc.)
   - Demonstrates batch processing of formulas
   - Compares performance optimization techniques

## Creating New Examples

When adding new examples, follow these guidelines:

1. Create a new Python file in the appropriate subdirectory.
2. Include detailed docstrings and comments explaining the example.
3. Implement a `main()` function that runs the example.
4. Use meaningful section headers to organize the example.
5. Add visualizations where helpful.
6. Update this README with a description of your example.

Example template:

```python
#!/usr/bin/env python
"""
Description of what this example demonstrates.

This example shows...
"""

import os
import sys

# Add the project root to the path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.module import feature  # Import relevant features

def example_section_one():
    """First example section."""
    # Example code here
    print("This example demonstrates...")

def example_section_two():
    """Second example section."""
    # Example code here
    print("Another aspect of the feature...")

def main():
    """Run all examples in this script."""
    print("=== Example Name ===")
    example_section_one()
    example_section_two()
    print("Example completed.")

if __name__ == "__main__":
    main() 