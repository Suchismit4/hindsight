# AST Module for Financial Formula Parsing

This module provides a robust implementation of an Abstract Syntax Tree (AST) for parsing and evaluating financial formulas. It is designed to work seamlessly with JAX and xarray for efficient computation on large datasets.

## Core Components

### Node Types
- `Node`: Abstract base class for all AST nodes
- `BinaryOp`: Binary operations (+, -, *, /, ^)
- `UnaryOp`: Unary operations (-)
- `Literal`: Numeric constants
- `Variable`: Regular variables
- `DataVariable`: References to xarray DataArrays
- `FunctionCall`: Function applications

### Grammar
- Context-free grammar (CFG) for financial formulas
- Support for standard mathematical operations
- Function calls with multiple arguments
- Special syntax for DataArray references ($variable)

### Features
- JAX compatibility through Equinox
- Automatic differentiation support
- Efficient evaluation on large datasets
- Comprehensive error handling
- AST optimization passes

## Usage Examples

```python
from src.data.ast import parse_formula, evaluate_formula
import xarray as xr

# Simple arithmetic
formula = "2 * x + 1"
result = evaluate_formula(formula, {"x": 3.0})

# Working with DataArrays
ds = xr.Dataset({"price": [100, 110, 120]})
formula = "$price * (1 + rate)"
result = evaluate_formula(formula, {
    "_dataset": ds,
    "rate": 0.1
})

# Function calls
formula = "sma($close, 30)"
result = evaluate_formula(formula, {
    "_dataset": market_data,
    "_func_sma": moving_average_fn
})
```

## Module Organization

```
src/data/ast/
├── __init__.py      # Public API
├── nodes.py         # AST node definitions
├── grammar.py       # CFG and parsing rules
├── parser.py        # Formula parsing
├── functions.py     # Function registry
└── visualization.py # AST visualization
```

## Future Improvements

- [ ] Split nodes.py into smaller modules
- [ ] Add custom operator support
- [ ] Implement type validation system
- [ ] Add serialization support
- [ ] Create comprehensive test suite
- [ ] Add performance benchmarks
- [ ] Implement visitor pattern for AST transformations
- [ ] Add error recovery during parsing

## Contributing

When contributing to this module:
1. Maintain JAX compatibility
2. Preserve xarray integration
3. Keep detailed docstrings
4. Add tests for new features
5. Update documentation 