"""
Formula management module.

This module provides functionality for loading and managing formula definitions
from YAML files. It handles validation, dependency resolution, and formula
registration.
"""

import os
from typing import Dict, Any, List, Set, Optional, Union
import yaml
import jsonschema
from pathlib import Path
import xarray as xr
import importlib
import functools

from .nodes import Node, DataVariable
from .parser import parse_formula, evaluate_formula
from .functions import register_function

import jax
import jax.numpy as jnp

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
    
    def __init__(self, definitions_dir: Optional[str] = None):
        """
        Initialize the formula manager.
        
        Args:
            definitions_dir: Path to directory containing formula definition files.
                           If None, uses the default 'definitions' directory.
        """
        self.formulas: Dict[str, Dict[str, Any]] = {}
        self._registered_functions: Set[str] = set()
        self._module_cache: Dict[str, Any] = {}  # Cache for loaded modules and functions
        
        # Load the schema
        schema_path = os.path.join(os.path.dirname(__file__), 'definitions', 'schema.yaml')
        with open(schema_path) as f:
            self._schema = yaml.safe_load(f)
        
        # Load formulas if directory provided, or load defaults
        if definitions_dir:
            self.load_directory(definitions_dir)
        else:
            self.load_default_formulas()
    
    def load_directory(self, directory: str) -> None:
        """
        Load all formula definition files from a directory.
        
        Args:
            directory: Path to directory containing .yaml files
            
        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If a formula definition is invalid
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Load all .yaml files
        for file_path in directory.glob('*.yaml'):
            if file_path.name != 'schema.yaml':
                self.load_file(str(file_path))
    
    def load_file(self, file_path: str) -> None:
        """
        Load formula definitions from a YAML file.
        
        Args:
            file_path: Path to the YAML file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If formula definitions are invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path) as f:
            definitions = yaml.safe_load(f)
        
        # Validate and add each formula
        for name, definition in definitions.items():
            if name != 'example':  # Skip example section
                self.add_formula(name, definition)
    
    def add_formula(self, name: str, definition: Dict[str, Any]) -> None:
        """
        Add a formula definition after validation.
        
        Args:
            name: Name of the formula
            definition: Formula definition dictionary
            
        Raises:
            ValueError: If formula definition is invalid
            KeyError: If formula name already exists
        """
        # Check for duplicate
        if name in self.formulas:
            raise KeyError(f"Formula '{name}' already exists")
        
        # Validate against schema
        try:
            jsonschema.validate(instance={"formula": definition}, schema=self._schema)
        except jsonschema.exceptions.ValidationError as e:
            raise ValueError(f"Invalid formula definition for '{name}': {str(e)}")
        
        # Validate the formula expression
        try:
            ast = parse_formula(definition['expression'])
            # TODO: Validate that AST only uses declared variables and functions
        except Exception as e:
            raise ValueError(f"Invalid formula expression for '{name}': {str(e)}")
        
        # Add to formulas dictionary
        self.formulas[name] = definition
        
        # Register any functions defined in the formula
        if 'functions' in definition:
            for func_name in definition['functions']:
                self._registered_functions.add(func_name)
    
    def get_formula(self, name: str) -> Dict[str, Any]:
        """
        Get a formula definition by name.
        
        Args:
            name: Name of the formula
            
        Returns:
            Formula definition dictionary
            
        Raises:
            KeyError: If formula doesn't exist
        """
        if name not in self.formulas:
            raise KeyError(f"Formula '{name}' not found")
        return self.formulas[name]
    
    def _load_module(self, module_path: str, function_name: str, cache_key: str) -> Any:
        """
        Load a function from a module with caching.
        
        Args:
            module_path: Python module path (e.g., 'src.data.generators.weights')
            function_name: Function name within the module
            cache_key: Key for caching the loaded function
            
        Returns:
            The loaded function
            
        Raises:
            ImportError: If module or function cannot be loaded
        """
        if cache_key in self._module_cache:
            return self._module_cache[cache_key]
        
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the function from the module
            if not hasattr(module, function_name):
                raise ImportError(f"Function '{function_name}' not found in module '{module_path}'")
            
            func = getattr(module, function_name)
            
            # Cache the function
            self._module_cache[cache_key] = func
            
            return func
            
        except ImportError as e:
            raise ImportError(f"Error loading {function_name} from {module_path}: {str(e)}")
    
    def _process_modules(self, formula_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process module definitions in a formula and execute generators to populate context.
        
        Args:
            formula_name: Name of the formula
            context: Current evaluation context
            
        Returns:
            Updated context with generated values
            
        Raises:
            ImportError: If module loading fails
            ValueError: If module execution fails
        """
        formula = self.get_formula(formula_name)
        modules = formula.get('modules', {})
        variables = formula.get('variables', {})
        
        if not modules:
            return context
        
        # Create a copy of context to avoid modifying the original
        updated_context = context.copy()
        
        # Process each variable that has a generator
        for var_name, var_def in variables.items():
            generator_alias = var_def.get('generator')
            if not generator_alias:
                continue
                
            if generator_alias not in modules:
                raise ValueError(f"Generator '{generator_alias}' referenced by variable '{var_name}' not found in modules for formula '{formula_name}'")
            
            module_def = modules[generator_alias]
            module_path = module_def['module_path']
            function_name = module_def['function_name']
            cache_result = module_def.get('cache_result', True)
            
            # Create cache key
            cache_key = f"{module_path}.{function_name}"
            
            # Load the generator function
            generator_func = self._load_module(module_path, function_name, cache_key)
            
            # Prepare arguments for the generator function
            # Pass all current context values as keyword arguments to the generator
            generator_kwargs = {}
            for key, value in updated_context.items():
                if not key.startswith('_'):  # Skip internal context keys like '_dataset'
                    generator_kwargs[key] = value
                    
            try:
                # Call the generator function
                generated_value = generator_func(**generator_kwargs)
                
                # Update context with generated value
                updated_context[var_name] = generated_value
                
            except Exception as e:
                raise ValueError(f"Error calling generator '{generator_alias}' for variable '{var_name}' in formula '{formula_name}': {str(e)}")
        
        return updated_context
    
    def evaluate(
        self,
        name: Union[str, List[str]],
        context: Dict[str, Any],
        validate_inputs: bool = True
    ) -> Union[Any, xr.Dataset]:
        """
        Evaluate one or more formulas with the given context.
        
        Args:
            name: Name of the formula or list of formula names
            context: Dictionary of variable values
            validate_inputs: Whether to validate inputs against the schema
            
        Returns:
            For single formula: Result of formula evaluation (same as before)
            For multiple formulas: xarray Dataset containing all results as data variables
            
        Raises:
            KeyError: If formula doesn't exist
            ValueError: If inputs are invalid or evaluation fails
        """
        # Handle single formula (backward compatibility)
        if isinstance(name, str):
            formula = self.get_formula(name)
                
            # Process modules first (if any) to generate dynamic values
            context = self._process_modules(name, context)
            
            if validate_inputs:
                self._validate_inputs(name, context)
            
            try:
                result, _ = evaluate_formula(formula['expression'], context, formula_name=name)
                # TODO: Validate result type matches formula's return_type
                return result
            except Exception as e:
                raise ValueError(f"Error evaluating formula '{name}': {str(e)}")
        
        # Handle multiple formulas
        elif isinstance(name, list):
            return self.evaluate_bulk(name, context, validate_inputs)
        
        else:
            raise TypeError(f"Expected str or List[str] for name, got {type(name)}")

    def evaluate_bulk(
        self,
        formula_names: List[str],
        context: Dict[str, Any],
        validate_inputs: bool = True,
        jit_compile: bool = True
    ) -> xr.Dataset:
        """
        Evaluate multiple formulas efficiently in bulk.
        
        This method parses all formulas upfront and evaluates them. The individual
        functions (like wma, sma, etc.) are already JIT-compiled, so we don't need
        to JIT compile the orchestration layer.
        
        Args:
            formula_names: List of formula names to evaluate
            context: Dictionary of variable values and functions
            validate_inputs: Whether to validate inputs against schemas
            jit_compile: Whether to enable JIT compilation (individual functions are already JIT compiled)
            
        Returns:
            xarray Dataset containing all formula results as data variables
            
        Raises:
            KeyError: If any formula doesn't exist
            ValueError: If inputs are invalid or evaluation fails
        """
        if not formula_names:
            raise ValueError("formula_names cannot be empty")

        if '_dataset' not in context or not isinstance(context['_dataset'], xr.Dataset):
            raise ValueError("Evaluation context must contain the input xarray Dataset under the key '_dataset'.")

        input_ds = context['_dataset']
        
        # Validate all formulas exist and parse their ASTs upfront
        parsed_formulas = {}
        for name in formula_names:
            if name not in self.formulas:
                raise KeyError(f"Formula '{name}' not found")
            
            formula_def = self.get_formula(name)
            
            # Validate inputs if requested
            if validate_inputs:
                self._validate_inputs(name, context)
            
            # Parse the formula into AST
            try:
                ast_node = parse_formula(formula_def['expression'])
                parsed_formulas[name] = ast_node
            except Exception as e:
                raise ValueError(f"Error parsing formula '{name}': {str(e)}")
        
        # Evaluate all formulas (no orchestration-level JIT compilation needed)
        # The individual functions (wma, sma, etc.) are already JIT compiled
        results = {}
        for name, ast_node in parsed_formulas.items():
            try:
                # Process modules for this formula to generate dynamic values
                formula_context = self._process_modules(name, context)
                result = ast_node.evaluate(formula_context)
                results[name] = result
            except Exception as e:
                raise ValueError(f"Error evaluating formula '{name}': {str(e)}")
        
        # Construct output dataset with all results
        output_ds = input_ds.copy()
        
        for name, result in results.items():
            if isinstance(result, xr.DataArray):
                # Set the name on the DataArray
                result.name = name
                # Add to the output dataset
                output_ds[name] = result
            elif isinstance(result, xr.Dataset):
                # If the result is a Dataset, merge its variables
                for var_name, var_data in result.data_vars.items():
                    # Prefix with formula name to avoid conflicts
                    prefixed_name = f"{name}_{var_name}"
                    output_ds[prefixed_name] = var_data
            else:
                # For scalar results, convert to DataArray
                scalar_da = xr.DataArray(result, name=name)
                output_ds[name] = scalar_da
        
        return output_ds

    def evaluate_all_loaded(
        self,
        context: Dict[str, Any],
        validate_inputs: bool = True,
        jit_compile: bool = True
    ) -> xr.Dataset:
        """
        Evaluate all loaded formulas in the manager.
        
        Convenience method to evaluate all formulas that have been loaded
        from definition files.
        
        Args:
            context: Dictionary of variable values and functions
            validate_inputs: Whether to validate inputs against schemas
            jit_compile: Whether to JIT compile the bulk evaluation
            
        Returns:
            xarray Dataset containing all formula results as data variables
        """
        all_formula_names = list(self.formulas.keys())
        return self.evaluate_bulk(all_formula_names, context, validate_inputs, jit_compile)
    
    def _validate_inputs(self, name: str, context: Dict[str, Any]) -> None:
        """
        Validate input values against the formula's schema and runtime expectations.
        
        Args:
            name: Name of the formula
            context: Dictionary of input values
            
        Raises:
            ValueError: If inputs are invalid
        """
        formula = self.get_formula(name)
        variables_schema = formula.get('variables', {})

        # Check if _dataset is required and valid if there are any dataarray type variables
        has_dataarray_var = any(
            var_def.get('type') == 'dataarray' 
            for var_def in variables_schema.values()
        )

        dataset_in_context = None
        if has_dataarray_var:
            if '_dataset' not in context:
                raise ValueError(f"Formula '{name}' uses DataArray variables, but '_dataset' was not found in the context.")
            dataset_in_context = context['_dataset']
            if not isinstance(dataset_in_context, xr.Dataset):
                raise ValueError(f"Context entry '_dataset' must be an xarray.Dataset for formula '{name}', but got {type(dataset_in_context)}.")

        for var_name, var_def in variables_schema.items():
            # Check if variable is provided or has a default
            if var_name not in context and 'default' not in var_def:
                raise ValueError(f"Missing required variable '{var_name}' for formula '{name}'.")

            # Determine the value to validate (either from context or default)
            # If var_name is not in context, var_def['default'] must exist (checked above for required vars)
            value_to_validate = context.get(var_name, var_def.get('default'))

            var_type = var_def.get('type')

            if var_type == 'dataarray':
                if not isinstance(value_to_validate, str):
                    raise ValueError(f"Variable '{var_name}' (type 'dataarray') for formula '{name}' expects a string key in the context, "
                                     f"but got {type(value_to_validate)} for value '{value_to_validate}'.")
                
                # This check assumes dataset_in_context is populated if has_dataarray_var is true.
                if dataset_in_context is None and has_dataarray_var:
                    # This case should ideally not be reached if the initial _dataset check is correct
                    raise ValueError("Internal error: _dataset not verified before dataarray variable check.")
                
                if value_to_validate not in dataset_in_context:
                    raise ValueError(f"Key '{value_to_validate}' (for variable '{var_name}') not found in the provided '_dataset' "
                                     f"for formula '{name}'. Available keys: {list(dataset_in_context.data_vars.keys())}.")
                
                # check if the resolved item is indeed an xr.DataArray
                if not isinstance(dataset_in_context[value_to_validate], xr.DataArray):
                    raise ValueError(f"Dataset entry for key '{value_to_validate}' (for variable '{var_name}') "
                                     f"was expected to be an xr.DataArray, but found type {type(dataset_in_context[value_to_validate])}.")
            
            elif var_type == 'number':
                if not isinstance(value_to_validate, (int, float)):
                    raise ValueError(f"Variable '{var_name}' (type 'number') for formula '{name}' expects an int or float, "
                                     f"but got {type(value_to_validate)} for value '{value_to_validate}'.")

            # TODO: Integrate further schema-based validation if present (e.g., min, max for numbers)
            if var_type == 'number' and 'validation' in var_def:
                rules = var_def['validation']
                if 'min' in rules and value_to_validate < rules['min']:
                    raise ValueError(f"Variable '{var_name}' value {value_to_validate} is less than minimum {rules['min']}.")
                if 'max' in rules and value_to_validate > rules['max']:
                    raise ValueError(f"Variable '{var_name}' value {value_to_validate} is greater than maximum {rules['max']}.")

    def get_formula_info(self, name: str) -> str:
        """
        Get a human-readable description of a formula.
        
        Args:
            name: Name of the formula
            
        Returns:
            Formatted string with formula information
            
        Raises:
            KeyError: If formula doesn't exist
        """
        formula = self.get_formula(name)
        
        info = [
            f"Formula: {name}",
            f"Description: {formula['description']}",
            f"Expression: {formula['expression']}",
            f"Return Type: {formula['return_type']}",
            "",
            "Variables:",
        ]
        
        for var_name, var_def in formula['variables'].items():
            info.append(f"  {var_name}:")
            info.append(f"    Type: {var_def['type']}")
            info.append(f"    Description: {var_def['description']}")
            if 'default' in var_def:
                info.append(f"    Default: {var_def['default']}")
        
        if 'functions' in formula:
            info.extend(["", "Functions:"])
            for func_name, func_def in formula['functions'].items():
                info.append(f"  {func_name}:")
                info.append(f"    Description: {func_def['description']}")
                info.append(f"    Arguments:")
                for arg in func_def['args']:
                    info.append(f"      - {arg['name']} ({arg['type']}): {arg['description']}")
        
        if 'notes' in formula:
            info.extend(["", f"Notes: {formula['notes']}"])
        
        return "\n".join(info)
    
    def list_formulas(self) -> List[str]:
        """
        Get a list of all available formula names.
        
        Returns:
            List of formula names
        """
        return sorted(self.formulas.keys())
    
    def get_formula_dependencies(self, name: str) -> Set[str]:
        """
        Get the set of function names that a formula depends on.
        
        Args:
            name: Name of the formula
            
        Returns:
            Set of function names used by the formula
            
        Raises:
            KeyError: If formula doesn't exist
        """
        formula = self.get_formula(name)
        return set(formula.get('functions', {}).keys()) 

    def load_default_formulas(self) -> None:
        """
        Load formula definitions from the default definitions directory.
        
        This method loads all .yaml files from the package's built-in definitions directory.
        """
        default_dir = os.path.join(os.path.dirname(__file__), 'definitions')
        if os.path.exists(default_dir):
            self.load_directory(default_dir)
        else:
            print(f"Warning: Default definitions directory not found at {default_dir}") 