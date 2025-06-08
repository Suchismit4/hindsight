# FormulaManager Public API Documentation

## Overview

The `FormulaManager` class is the central component of the Hindsight AST (Abstract Syntax Tree) module for financial formula parsing, validation, and evaluation. This manager provides a comprehensive framework for defining financial formulas in YAML format, handling dependencies between formulas, and efficiently evaluating them on xarray datasets containing financial time series data.

The FormulaManager supports advanced features including:
- **Formula Dependencies**: Formulas can depend on other formulas either as functions (functional dependence) or as time series data (time series dependence)
- **Multiple Configuration Evaluation**: Single formulas can be evaluated with multiple parameter configurations in a single call
- **JAX Optimization**: Built-in JAX JIT compilation for high-performance computation
- **Schema Validation**: Comprehensive validation of formula definitions against JSON schema
- **Dependency Resolution**: Automatic topological sorting and dependency resolution
- **Module Integration**: Support for external weight generators and custom modules

## Class Definition

```python
class FormulaManager:
    """
    Manager for formula definitions.
    
    This class handles loading formula definitions from YAML files,
    validating them against the schema, and providing an interface
    for formula evaluation.
    
    Attributes:
        formulas: Dictionary mapping formula names to their definitions
        _schema: The JSON schema for formula definitions
        _registered_functions: Set of function names that have been registered
        _module_cache: Cache for loaded modules and functions
    """
```

## Constructor

### `__init__(definitions_dir: Optional[str] = None)`

Initialize the FormulaManager with optional formula definitions directory.

**Parameters:**
- `definitions_dir` (`Optional[str]`): Path to directory containing formula definition files. If `None`, uses the default built-in definitions directory containing standard financial formulas like technical indicators and market characteristics.

**Behavior:**
- Initializes internal data structures for formula storage, dependency tracking, and caching
- Loads JSON schema for formula validation from `definitions/schema.yaml`
- If `definitions_dir` is provided, loads all `.yaml` files from that directory
- If `definitions_dir` is `None`, automatically loads built-in formulas via `load_default_formulas()`
- Builds dependency graph between formulas after loading

**Example:**
```python
# Load default built-in formulas
manager = FormulaManager()

# Load formulas from custom directory
manager = FormulaManager("/path/to/custom/formulas")
```

## Formula Loading Methods

### `load_directory(directory: str) -> None`

Load all formula definition files from a specified directory.

**Parameters:**
- `directory` (`str`): Path to directory containing `.yaml` formula definition files

**Behavior:**
- Scans the directory for all files with `.yaml` extension
- Excludes `schema.yaml` file from loading (reserved for schema definition)
- Calls `load_file()` for each discovered YAML file
- Updates dependency graph after loading all files

**Raises:**
- `FileNotFoundError`: If the specified directory does not exist

**Example:**
```python
manager = FormulaManager()
manager.load_directory("/path/to/additional/formulas")
```

### `load_file(file_path: str) -> None`

Load formula definitions from a single YAML file.

**Parameters:**
- `file_path` (`str`): Path to the YAML file containing formula definitions

**Behavior:**
- Reads and parses the YAML file using `yaml.safe_load()`
- Iterates through all top-level keys in the YAML (each key is a formula name)
- Skips the special `example` key (used for documentation purposes)
- Calls `add_formula()` for each formula definition
- Automatically validates each formula against the schema

**Raises:**
- `FileNotFoundError`: If the specified file does not exist
- `ValueError`: If any formula definition in the file is invalid
- `yaml.YAMLError`: If the file contains invalid YAML syntax

**Example:**
```python
manager = FormulaManager()
manager.load_file("/path/to/custom_formulas.yaml")
```

### `load_default_formulas() -> None`

Load formula definitions from the built-in definitions directory.

**Parameters:** None

**Behavior:**
- Automatically locates the built-in definitions directory relative to the module location
- Loads all standard formula definitions including technical indicators, market characteristics, and composite formulas
- Calls `_rebuild_dependency_graph()` after loading to ensure all inter-formula dependencies are properly resolved
- Prints warning if the default definitions directory is not found

**Example:**
```python
manager = FormulaManager()
# Default formulas are loaded automatically, but can be explicitly reloaded
manager.load_default_formulas()
```

## Formula Management Methods

### `add_formula(name: str, definition: Dict[str, Any]) -> None`

Add a single formula definition after comprehensive validation.

**Parameters:**
- `name` (`str`): Unique name for the formula (used as identifier for evaluation and dependencies)
- `definition` (`Dict[str, Any]`): Formula definition dictionary conforming to the schema

**Behavior:**
- Checks for duplicate formula names and raises error if formula already exists
- Validates the formula definition against the JSON schema using `jsonschema.validate()`
- Parses the formula expression using the AST parser to verify syntax correctness
- Registers any function names defined in the formula's function definitions
- Updates the dependency graph by analyzing formula dependencies
- Stores the validated formula definition in the internal formulas dictionary

**Raises:**
- `KeyError`: If a formula with the same name already exists
- `ValueError`: If the formula definition is invalid according to the schema or if the expression contains syntax errors
- `jsonschema.exceptions.ValidationError`: If the definition fails schema validation

**Formula Definition Structure:**
The definition dictionary must contain:
- `description`: Human-readable description of what the formula computes
- `expression`: Formula expression string in the CFG syntax
- `return_type`: Expected return type (`"scalar"`, `"array"`, `"dataarray"`, `"dataset"`)
- `variables`: Dictionary defining all variables used in the formula
- Optional: `functions`, `modules`, `tags`, `notes`

**Example:**
```python
manager = FormulaManager()

definition = {
    "description": "Simple moving average",
    "expression": "sma($price, $window)",
    "return_type": "dataarray",
    "variables": {
        "price": {
            "type": "dataarray",
            "description": "Price time series"
        },
        "window": {
            "type": "number",
            "description": "Moving average window size",
            "default": 20
        }
    }
}

manager.add_formula("my_sma", definition)
```

### `get_formula(name: str) -> Dict[str, Any]`

Retrieve a formula definition by name.

**Parameters:**
- `name` (`str`): Name of the formula to retrieve

**Returns:**
- `Dict[str, Any]`: Complete formula definition dictionary containing all schema fields

**Raises:**
- `KeyError`: If the specified formula name is not found

**Example:**
```python
manager = FormulaManager()
formula_def = manager.get_formula("rsi")
print(formula_def["description"])  # "Relative Strength Index - measures momentum"
```

## Formula Evaluation Methods

### `evaluate(name, context, validate_inputs=True)`

Evaluate one or more formulas with the given context. This is the primary method for formula evaluation.

**Method Signature:**
```python
def evaluate(
    self,
    name: Union[str, List[str], Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]]],
    context: Dict[str, Any],
    validate_inputs: bool = True
) -> Union[Any, xr.Dataset]:
```

**Parameters:**
- `name`: Formula specification in one of three formats:
  - `str`: Single formula name for simple evaluation
  - `List[str]`: List of formula names to evaluate with shared context
  - `Dict[str, Union[Dict, List[Dict]]]`: Formula names mapped to configuration(s) for advanced multi-parameter evaluation

- `context` (`Dict[str, Any]`): Dictionary containing:
  - `'_dataset'`: xarray Dataset containing financial time series data (required for formulas using DataArray variables)
  - Variable values for formula parameters
  - Function implementations for custom functions

- `validate_inputs` (`bool`, default=`True`): Whether to validate input values against formula schemas

**Returns:**
- For single formula: Direct result of formula evaluation (preserves type from formula)
- For multiple formulas: xarray Dataset containing all results as data variables

**Behavior:**
- Normalizes all input formats to the dictionary format used internally
- Delegates to `evaluate_bulk()` for actual evaluation processing
- Handles dependency resolution automatically
- For single formula evaluation, extracts the result from the dataset for backward compatibility
- Supports both functional dependencies (formulas calling other formulas as functions) and time series dependencies (formulas using other formula results as input data)

**Raises:**
- `KeyError`: If any specified formula does not exist
- `ValueError`: If inputs are invalid, required variables are missing, or evaluation fails
- `TypeError`: If the `name` parameter is not of a supported type

**Example Usage:**

**Single Formula:**
```python
context = {
    '_dataset': financial_dataset,
    'window': 14
}
rsi_result = manager.evaluate("rsi", context)
```

**Multiple Formulas:**
```python
context = {
    '_dataset': financial_dataset
}
results = manager.evaluate(["rsi", "wma", "ema"], context)
# Returns Dataset with data variables: rsi, wma, ema
```

**Multi-Configuration Evaluation:**
```python
context = {
    '_dataset': financial_dataset,
    'price': 'close'
}

formula_configs = {
    "wma": {"window": 10},  # Single config
    "rsi": [{"window": 14}, {"window": 21}],  # Multiple configs
    "alma": [
        {"window": 10, "offset": 0.85},
        {"window": 20, "offset": 0.9}
    ]
}

results = manager.evaluate(formula_configs, context)
# Returns Dataset with variables: wma, rsi_w14, rsi_w21, alma_w10_o0.85, alma_w20_o0.9
```

### `evaluate_bulk(formula_names, context, validate_inputs=True, jit_compile=True)`

Efficiently evaluate multiple formulas in bulk with support for multiple configurations per formula.

**Method Signature:**
```python
def evaluate_bulk(
    self,
    formula_names: Union[List[str], Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]]],
    context: Dict[str, Any],
    validate_inputs: bool = True,
    jit_compile: bool = True
) -> xr.Dataset:
```

**Parameters:**
- `formula_names`: Formula specification in one of two formats:
  - `List[str]`: List of formula names (backward compatible)
  - `Dict[str, Union[Dict, List[Dict]]]`: Formula names mapped to configuration(s)

- `context` (`Dict[str, Any]`): Evaluation context containing dataset and variables
- `validate_inputs` (`bool`, default=`True`): Whether to validate inputs against schemas
- `jit_compile` (`bool`, default=`True`): Whether to enable JIT compilation optimizations

**Returns:**
- `xr.Dataset`: Dataset containing all formula results as data variables

**Behavior:**
- Normalizes input formats to consistent internal representation
- Performs dependency resolution using topological sorting
- Evaluates formulas in dependency order to handle inter-formula dependencies
- Supports multiple parameter configurations for single formulas
- Generates descriptive names for multi-configuration results
- Caches intermediate results for time series dependencies
- Integrates with JAX JIT compilation for performance optimization

**Result Naming Convention:**
- Single configuration: Formula name (e.g., "wma", "rsi")
- Multiple configurations: Formula name + parameter suffix (e.g., "wma_w10", "rsi_w14", "rsi_w21")

**Example:**
```python
formula_configs = {
    "rsi": [{"window": 14}, {"window": 21}],
    "bollinger_bands": {"window": 20, "std_dev": 2}
}

results = manager.evaluate_bulk(formula_configs, context)
```

### `evaluate_all_loaded(context, validate_inputs=True, jit_compile=True)`

Evaluate all formulas that have been loaded into the manager.

**Method Signature:**
```python
def evaluate_all_loaded(
    self,
    context: Dict[str, Any],
    validate_inputs: bool = True,
    jit_compile: bool = True
) -> xr.Dataset:
```

**Parameters:**
- `context` (`Dict[str, Any]`): Evaluation context containing dataset and variables
- `validate_inputs` (`bool`, default=`True`): Whether to validate inputs against schemas  
- `jit_compile` (`bool`, default=`True`): Whether to enable JIT compilation optimizations

**Returns:**
- `xr.Dataset`: Dataset containing results from all loaded formulas as data variables

**Behavior:**
- Retrieves list of all loaded formula names using `list_formulas()`
- Delegates to `evaluate_bulk()` with the complete formula list
- Useful for computing all available indicators or characteristics at once
- Respects formula dependencies and evaluation order

**Example:**
```python
context = {
    '_dataset': financial_dataset
}

# Evaluate all loaded formulas at once
all_results = manager.evaluate_all_loaded(context)
print(list(all_results.data_vars))  # Shows all computed formulas
```

## Formula Inspection Methods

### `list_formulas() -> List[str]`

Get a sorted list of all available formula names.

**Parameters:** None

**Returns:**
- `List[str]`: Alphabetically sorted list of formula names that have been loaded

**Example:**
```python
manager = FormulaManager()
formulas = manager.list_formulas()
print(formulas)  # ['alma', 'dema', 'ema', 'rsi', 'wma', ...]
```

### `get_formula_info(name: str) -> str`

Get a comprehensive, human-readable description of a formula.

**Parameters:**
- `name` (`str`): Name of the formula

**Returns:**
- `str`: Multi-line formatted string containing complete formula information including description, expression, variables, functions, and notes

**Raises:**
- `KeyError`: If the specified formula does not exist

**Output Format:**
- Formula name and description
- Expression syntax
- Return type
- Variable definitions with types, descriptions, and default values
- Function definitions with arguments
- Additional notes if present

**Example:**
```python
info = manager.get_formula_info("rsi")
print(info)
# Output:
# Formula: rsi
# Description: Relative Strength Index - measures momentum
# Expression: 100 - (100 / (1 + (rma(gain($price, 1), $window) / rma(loss($price, 1), $window))))
# Return Type: dataarray
# 
# Variables:
#   price:
#     Type: dataarray
#     Description: Price data (typically close price)
#   window:
#     Type: number
#     Description: Period for RSI calculation
#     Default: 14
# ...
```

## Dependency Analysis Methods

### `get_formula_dependencies(name: str) -> Set[str]`

Get the complete set of formula names that a formula depends on.

**Parameters:**
- `name` (`str`): Name of the formula to analyze

**Returns:**
- `Set[str]`: Set of formula names that the specified formula depends on (both functional and time series dependencies)

**Raises:**
- `KeyError`: If the specified formula does not exist

**Example:**
```python
deps = manager.get_formula_dependencies("compound_momentum")
print(deps)  # {'momentum', 'volatility', 'trend_strength'}
```

### `get_dependency_chain(name: str) -> List[str]`

Get the complete dependency chain for a formula in evaluation order.

**Parameters:**
- `name` (`str`): Name of the formula to analyze

**Returns:**
- `List[str]`: Formula names in dependency order (dependencies first, target formula last)

**Raises:**
- `KeyError`: If the specified formula does not exist
- `ValueError`: If circular dependencies are detected in the formula graph

**Behavior:**
- Performs topological sorting of the dependency graph
- Ensures that all dependencies are evaluated before the target formula
- Detects and reports circular dependency cycles

**Example:**
```python
chain = manager.get_dependency_chain("advanced_momentum")
print(chain)  # ['price_ret', 'momentum', 'volatility', 'advanced_momentum']
```

### `get_functional_dependencies(name: str) -> Set[str]`

Get the set of formula names that a formula depends on as functions (functional dependence).

**Parameters:**
- `name` (`str`): Name of the formula to analyze

**Returns:**
- `Set[str]`: Set of formula names that are called as functions within the formula expression

**Raises:**
- `KeyError`: If the specified formula does not exist

**Behavior:**
- Parses the formula expression to extract function calls
- Identifies which function calls correspond to other loaded formulas
- Returns only the subset that represents functional dependencies

**Example:**
```python
func_deps = manager.get_functional_dependencies("composite_signal")
print(func_deps)  # {'rsi', 'bollinger_bands'} - formulas called as functions
```

### `get_time_series_dependencies(name: str) -> Set[str]`

Get the set of formula names that a formula depends on as time series data (time series dependence).

**Parameters:**
- `name` (`str`): Name of the formula to analyze

**Returns:**
- `Set[str]`: Set of formula names that are used as DataArray variables in the formula

**Raises:**
- `KeyError`: If the specified formula does not exist

**Behavior:**
- Examines the formula's variable definitions
- Identifies DataArray variables that correspond to other formula names
- Excludes self-references and variables that don't match formula names

**Example:**
```python
ts_deps = manager.get_time_series_dependencies("momentum_divergence")
print(ts_deps)  # {'price_momentum', 'volume_momentum'} - formulas used as data
```

### `get_function_dependencies(name: str) -> Set[str]`

Get the set of built-in function names that a formula depends on.

**Parameters:**
- `name` (`str`): Name of the formula to analyze

**Returns:**
- `Set[str]`: Set of built-in function names (not formula names) used by the formula

**Raises:**
- `KeyError`: If the specified formula does not exist

**Behavior:**
- Extracts function names from the formula's function definitions section
- Returns functions that are defined in the schema but not other formulas
- Useful for understanding external function dependencies

**Example:**
```python
func_deps = manager.get_function_dependencies("wma")
print(func_deps)  # {'wma'} - built-in functions, not other formulas
```

## Advanced Features

### `compile_all_formulas_as_functions() -> None`

Pre-compile all loaded formulas as functions for performance optimization.

**Parameters:** None

**Behavior:**
- Iterates through all loaded formulas
- Compiles each formula into a callable function using `_compile_formula_as_function()`
- Caches compiled functions for use in functional dependencies
- Prints warnings for formulas that cannot be compiled
- Enables formulas to be used as functions in other formula expressions

**Use Case:**
- Call this method after loading all formulas to ensure optimal performance
- Particularly beneficial when formulas will be used as functions in other formulas
- Pre-compilation avoids compilation overhead during evaluation

**Example:**
```python
manager = FormulaManager()
manager.compile_all_formulas_as_functions()
# Now formulas can be used efficiently as functions in expressions
```

### `get_compiled_formula_function(name: str) -> Callable`

Get a compiled formula function that can be called directly.

**Parameters:**
- `name` (`str`): Name of the formula to compile

**Returns:**
- `Callable`: Compiled function that can be called with appropriate arguments

**Raises:**
- `KeyError`: If the specified formula does not exist
- `ValueError`: If the formula cannot be compiled into a function

**Behavior:**
- Compiles the formula into a standalone callable function
- The resulting function can be used independently of the FormulaManager
- Handles argument mapping and context preparation automatically

**Example:**
```python
rsi_func = manager.get_compiled_formula_function("rsi")
# Now rsi_func can be called directly
result = rsi_func(price_data=close_prices, window=14)
```

### `list_formula_functions() -> List[str]`

Get a list of formulas that have been compiled as functions.

**Parameters:** None

**Returns:**
- `List[str]`: Sorted list of formula names that are available as compiled functions

**Behavior:**
- Returns formulas that have been successfully compiled via `compile_all_formulas_as_functions()` or `get_compiled_formula_function()`
- Useful for checking which formulas are ready for use as functions in other formulas

**Example:**
```python
manager.compile_all_formulas_as_functions()
compiled = manager.list_formula_functions()
print(compiled)  # ['ema', 'rsi', 'sma', 'wma', ...]
```

## Context Requirements

### Dataset Context

The evaluation context must contain specific keys for proper formula evaluation:

**Required Keys:**
- `'_dataset'` (`xr.Dataset`): The primary financial dataset containing time series data with proper time coordinates (year, month, day) and asset identifiers

**Optional Keys:**
- Variable values matching formula variable definitions
- Function implementations for custom functions
- Override values for formula parameters

### Dataset Structure Requirements

The dataset must follow the Hindsight data structure conventions:

**Coordinates:**
- `year`, `month`, `day`: Multi-dimensional time coordinates for business day handling
- `asset`: Asset identifier coordinate (e.g., PERMNO, ticker symbols)
- `mask`: Boolean mask for valid business days
- `mask_indices`: Indices mapping business days to data positions

**Data Variables:**
- Financial time series data (prices, volumes, returns, etc.)
- Each data variable is an xarray DataArray with time and asset dimensions

**Example:**
```python
context = {
    '_dataset': financial_dataset,  # Required
    'window': 14,                   # Formula parameter
    'price': 'close',              # Variable mapping
}
```

## Error Handling

The FormulaManager provides comprehensive error handling with specific exception types:

### Schema Validation Errors
- `jsonschema.exceptions.ValidationError`: Formula definition doesn't conform to schema
- `ValueError`: Invalid formula expressions or configuration

### Formula Access Errors  
- `KeyError`: Accessing non-existent formulas
- `FileNotFoundError`: Loading from non-existent files or directories

### Evaluation Errors
- `ValueError`: Missing required variables, invalid data types, or evaluation failures
- `TypeError`: Incorrect parameter types in method calls

### Dependency Errors
- `ValueError`: Circular dependencies detected in formula graph

## Performance Considerations

### JAX Integration
- Built-in JAX JIT compilation for numerical operations
- Automatic optimization of rolling window operations
- Efficient handling of large financial datasets

### Caching
- Module caching for external weight generators
- Formula function compilation caching
- Intermediate result caching for time series dependencies

### Bulk Evaluation
- `evaluate_bulk()` is more efficient than multiple `evaluate()` calls
- Dependency resolution performed once for entire batch
- Optimized memory usage for large formula sets

## Best Practices

### Formula Definition
1. Use descriptive variable names and comprehensive descriptions
2. Provide default values for commonly used parameters
3. Include validation rules for numeric parameters
4. Document complex formulas with notes

### Performance Optimization
1. Call `compile_all_formulas_as_functions()` after loading formulas
2. Use `evaluate_bulk()` for multiple formula evaluation
3. Structure dependencies to minimize circular references
4. Leverage time series dependencies for complex indicators

### Error Prevention
1. Validate formula definitions before adding to production systems
2. Use `get_formula_info()` to verify formula specifications
3. Check dependencies with `get_dependency_chain()` before evaluation
4. Ensure dataset structure matches formula requirements

## Migration Guide

### From Individual Function Calls
Replace individual function calls with FormulaManager evaluation:

```python
# Old approach
rsi_result = calculate_rsi(prices, window=14)
ema_result = calculate_ema(prices, window=20)

# New approach with FormulaManager
context = {'_dataset': dataset, 'window': 14}
result = manager.evaluate(['rsi', 'ema'], context)
```

### From Static Configurations
Leverage multiple configuration support:

```python
# Old: Multiple separate calls
results = []
for window in [10, 20, 50]:
    results.append(calculate_sma(prices, window))

# New: Single call with multiple configs
configs = {"sma": [{"window": w} for w in [10, 20, 50]]}
results = manager.evaluate(configs, context)
``` 