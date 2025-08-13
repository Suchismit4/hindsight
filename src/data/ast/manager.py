"""
Formula management module.

This module provides functionality for loading and managing formula definitions
from YAML files. It handles validation, dependency resolution, and formula
registration.
"""

import os
from typing import Dict, Any, List, Set, Optional, Union, Callable
import yaml
import jsonschema
from pathlib import Path
import xarray as xr
import importlib
import functools

from .nodes import Node, DataVariable, FunctionCall, Literal
from .parser import parse_formula, evaluate_formula, extract_functions
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
        self._formula_functions: Dict[str, Any] = {}  # Cache for compiled formula functions
        self._dependency_graph: Dict[str, Set[str]] = {}  # Formula dependency tracking
        
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
                
        # Analyze dependencies and update dependency graph
        self._update_dependency_graph(name, definition)
    
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
            # Check if this is a built-in function that can be used as a formula
            if self._is_builtin_function(name):
                return self._create_builtin_formula_definition(name)
            raise KeyError(f"Formula '{name}' not found")
        return self.formulas[name]
    
    def _is_builtin_function(self, name: str) -> bool:
        """
        Check if a name corresponds to a built-in function.
        
        Args:
            name: Function name to check
            
        Returns:
            True if the name is a registered built-in function
        """
        from .functions import get_registered_functions
        return name in get_registered_functions()
    
    def _create_builtin_formula_definition(self, name: str) -> Dict[str, Any]:
        """
        Create a formula definition for a built-in function.
        
        Args:
            name: Name of the built-in function
            
        Returns:
            Formula definition dictionary that mimics YAML structure
        """
        from .functions import get_registered_functions, get_function_metadata
        
        functions_registry = get_registered_functions()
        if name not in functions_registry:
            raise KeyError(f"Built-in function '{name}' not found")
        
        func = functions_registry[name]
        
        # Get function metadata if available
        try:
            metadata = get_function_metadata(name)
            description = metadata.get('description', f"Built-in function: {name}")
        except:
            description = f"Built-in function: {name}"
        
        # Analyze function signature to create variable definitions
        import inspect
        sig = inspect.signature(func)
        variables = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'data':
                # Map 'data' parameter to 'price' to match existing formula conventions
                variables['price'] = {
                    'type': 'dataarray',
                    'description': f'Price data for {name}'
                }
            elif param_name in ['window', 'periods', 'period']:
                variables[param_name] = {
                    'type': 'number',
                    'description': f'Window size for {name}',
                    'validation': {'min': 1}
                }
            elif param_name in ['alpha', 'beta', 'gamma']:
                variables[param_name] = {
                    'type': 'number',
                    'description': f'Smoothing parameter for {name}',
                    'validation': {'min': 0, 'max': 1}
                }
            elif param_name == 'weights':
                variables[param_name] = {
                    'type': 'array',
                    'description': f'Weights array for {name}'
                }
            else:
                # Generic parameter
                variables[param_name] = {
                    'type': 'number',
                    'description': f'Parameter {param_name} for {name}'
                }
            
            # Add default value if available
            if param.default != inspect.Parameter.empty:
                variables[param_name]['default'] = param.default
        
        # Create the expression string - this calls the function with its parameters
        param_names = list(sig.parameters.keys())
        if param_names:
            # Build expression like "sma($price, $window)"
            # Map 'data' parameter to 'price' to match existing conventions
            param_refs = []
            for param in param_names:
                if param == 'data':
                    param_refs.append('$price')
                else:
                    param_refs.append(f'${param}')
            expression = f"{name}({', '.join(param_refs)})"
        else:
            expression = f"{name}()"
        
        formula_def = {
            'description': description,
            'expression': expression,
            'return_type': 'dataarray',
            'variables': variables,
            'functions': {
                name: {
                    'description': f'Built-in function {name}',
                    'args': [
                        {
                            'name': param_name,
                            'type': 'dataarray' if param_name == 'data' else 'number',
                            'description': f'Parameter {param_name}'
                        }
                        for param_name in param_names
                    ]
                }
            }
        }
        
        return formula_def
    
    def _update_dependency_graph(self, formula_name: str, definition: Dict[str, Any]) -> None:
        """
        Update the dependency graph by analyzing formula dependencies.
        
        Args:
            formula_name: Name of the formula
            definition: Formula definition dictionary
        """
        try:
            # Parse the formula to extract function calls
            ast = parse_formula(definition['expression'])
            functions_used = extract_functions(definition['expression'])
            
            # Find which of these functions are actually other formulas (functional dependence)
            formula_dependencies = set()
            for func_name in functions_used:
                if func_name in self.formulas and func_name != formula_name:
                    formula_dependencies.add(func_name)
            
            # Also check for time series dependencies (dataarray variables that reference other formulas)
            variables_schema = definition.get('variables', {})
            for var_name, var_def in variables_schema.items():
                if var_def.get('type') == 'dataarray':
                    # Check if this variable name matches another formula name
                    if var_name in self.formulas and var_name != formula_name:
                        formula_dependencies.add(var_name)
            
            self._dependency_graph[formula_name] = formula_dependencies
            
        except Exception as e:
            print(f"Warning: Could not analyze dependencies for formula '{formula_name}': {e}")
            self._dependency_graph[formula_name] = set()
    
    def _resolve_dependencies(self, formula_name: str, visited: Optional[Set[str]] = None) -> List[str]:
        """
        Resolve dependencies for a formula in topological order.
        
        Args:
            formula_name: Name of the formula to resolve dependencies for
            visited: Set of already visited formulas (for cycle detection)
            
        Returns:
            List of formula names in dependency order (dependencies first)
            
        Raises:
            ValueError: If circular dependencies are detected
        """
        if visited is None:
            visited = set()
            
        if formula_name in visited:
            raise ValueError(f"Circular dependency detected involving formula '{formula_name}'")
        
        if formula_name not in self._dependency_graph:
            return [formula_name]
        
        visited.add(formula_name)
        dependencies = []
        
        # First, resolve all dependencies
        for dep in self._dependency_graph[formula_name]:
            if dep in self.formulas:  # Only process formula dependencies
                dep_order = self._resolve_dependencies(dep, visited.copy())
                for dep_formula in dep_order:
                    if dep_formula not in dependencies:
                        dependencies.append(dep_formula)
        
        # Finally, add the formula itself
        if formula_name not in dependencies:
            dependencies.append(formula_name)
            
        return dependencies
    
    def _get_evaluation_order(self, formula_names: List[str]) -> List[str]:
        """
        Get the evaluation order for a list of formulas, including their dependencies.
        
        Args:
            formula_names: List of formula names to evaluate
            
        Returns:
            List of formula names in evaluation order (dependencies first)
            
        Raises:
            ValueError: If circular dependencies are detected
        """
        all_formulas = set()
        
        # Collect all formulas and their dependencies
        for formula_name in formula_names:
            if formula_name in self.formulas or self._is_builtin_function(formula_name):
                dependency_chain = self._resolve_dependencies(formula_name)
                all_formulas.update(dependency_chain)
        
        # Return the formulas in dependency order
        # We need to do a topological sort of all collected formulas
        result = []
        remaining = set(all_formulas)
        
        while remaining:
            # Find formulas with no remaining dependencies
            ready = set()
            for formula in remaining:
                dependencies = self._dependency_graph.get(formula, set())
                # Check if all dependencies are already in the result
                if dependencies.issubset(set(result)):
                    ready.add(formula)
            
            if not ready:
                # No formulas are ready - there must be a circular dependency
                raise ValueError(f"Circular dependency detected among formulas: {remaining}")
            
            # Add ready formulas to result (in sorted order for consistency)
            for formula in sorted(ready):
                result.append(formula)
                remaining.remove(formula)
        
        return result
    
    def _compile_formula_as_function(self, formula_name: str) -> Callable:
        """
        Compile a formula into a callable function that can be used by other formulas.
        
        Args:
            formula_name: Name of the formula to compile
            
        Returns:
            Callable function that evaluates the formula
        """
        if formula_name in self._formula_functions:
            return self._formula_functions[formula_name]
        
        formula_def = self.get_formula(formula_name)
        
        def compiled_formula(*args, **kwargs):
            """
            Compiled formula function.
            
            Args:
                *args: Positional arguments (for AST function calls)
                **kwargs: Keyword arguments (variables + context)
                
            Returns:
                Result of formula evaluation
            """
            # If called with positional arguments, match them to variable names
            variables_schema = formula_def.get('variables', {})
            var_names = list(variables_schema.keys())
            
            # Create evaluation context from kwargs first
            context = kwargs.copy()
            
            # Add positional arguments to context
            for i, arg in enumerate(args):
                if i < len(var_names):
                    context[var_names[i]] = arg
                    
            # If _dataset is not in context, we can't evaluate the formula
            if '_dataset' not in context:
                raise ValueError(f"Formula function '{formula_name}' requires '_dataset' in context but it was not found")
            
            # Add default values for missing variables
            for var_name, var_def in variables_schema.items():
                if var_name not in context and 'default' in var_def:
                    context[var_name] = var_def['default']
            
            # Process modules if any
            context = self._process_modules(formula_name, context)
            
            # Add formula functions to context (for dependencies)
            # Pass the current dataset if available, and exclude current formula to prevent recursion
            dataset = context.get('_dataset')
            exclude_formulas = {formula_name}  # Exclude current formula to prevent circular dependency
            formula_function_context = self._get_formula_function_context(dataset, exclude_formulas)
            context.update(formula_function_context)
            
            # Also include built-in functions
            from .functions import get_function_context
            builtin_context = get_function_context()
            context.update(builtin_context)
            
            # Evaluate the formula
            try:
                result, _ = evaluate_formula(formula_def['expression'], context, formula_name=formula_name)
                return result
            except Exception as e:
                raise ValueError(f"Error evaluating compiled formula '{formula_name}': {str(e)}")
        
        # Cache the compiled function
        self._formula_functions[formula_name] = compiled_formula
        return compiled_formula
    
    def _get_formula_function_context(self, dataset: Optional[xr.Dataset] = None, exclude_formulas: Optional[Set[str]] = None) -> Dict[str, Callable]:
        """
        Get a context dictionary containing all compiled formula functions.
        
        Args:
            dataset: Optional dataset to pass to formula functions
            exclude_formulas: Set of formula names to exclude (prevents circular dependencies)
        
        Returns:
            Dictionary mapping function names to compiled formulas
        """
        if exclude_formulas is None:
            exclude_formulas = set()
            
        context = {}
        
        # Ensure all formulas are compiled
        for formula_name in self.formulas:
            if formula_name in exclude_formulas:
                continue
                
            if formula_name not in self._formula_functions:
                try:
                    self._compile_formula_as_function(formula_name)
                except Exception as e:
                    print(f"Warning: Could not compile formula '{formula_name}' as function: {e}")
                    continue
        
        # Create wrapped functions that include the dataset context
        for formula_name, func in self._formula_functions.items():
            if formula_name in exclude_formulas:
                continue
                
            if dataset is not None:
                def create_wrapped_func(original_func, ds):
                    def wrapped_func(*args, **kwargs):
                        # Ensure _dataset is in the context
                        kwargs['_dataset'] = ds
                        return original_func(*args, **kwargs)
                    return wrapped_func
                
                wrapped_func = create_wrapped_func(func, dataset)
                context[f"_func_{formula_name}"] = wrapped_func  # Add with _func_ prefix for AST compatibility
            else:
                # No dataset provided - use original function
                context[f"_func_{formula_name}"] = func  # Add with _func_ prefix for AST compatibility
            
        return context
    
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
        name: Union[str, List[str], Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]]],
        context: Dict[str, Any],
        validate_inputs: bool = True
    ) -> Union[Any, xr.Dataset]:
        """
        Evaluate one or more formulas with the given context.
        
        This method is now streamlined to use evaluate_bulk for all cases,
        providing a unified evaluation pathway with formula function support.
        
        Args:
            name: Can be:
                - str: Single formula name (backward compatible)
                - List[str]: List of formula names with shared context (backward compatible)
                - Dict[str, Union[Dict, List[Dict]]]: Formula names mapped to config(s)
                  Example: {"wma": {"window": 10}, "rsi": [{"window": 14}, {"window": 21}]}
            context: Dictionary of variable values shared across all evaluations
            validate_inputs: Whether to validate inputs against the schema
            
        Returns:
            For single formula: Result of formula evaluation (same as before)
            For multiple formulas: xarray Dataset containing all results as data variables
            
        Raises:
            KeyError: If formula doesn't exist
            ValueError: If inputs are invalid or evaluation fails
        """
        # Normalize all inputs to the dictionary format used by evaluate_bulk
        if isinstance(name, str):
            # Single formula - convert to dict format
            formula_configs = {name: [{}]}
            result_ds = self.evaluate_bulk(formula_configs, context, validate_inputs)
            # Return the single result for backward compatibility
            if name in result_ds.data_vars:
                return result_ds[name]
            else:
                # If there's only one result, return it
                data_vars = list(result_ds.data_vars.keys())
                if len(data_vars) == 1:
                    return result_ds[data_vars[0]]
                else:
                    return result_ds
        
        elif isinstance(name, list):
            # List of formulas - convert to dict format
            formula_configs = {formula_name: [{}] for formula_name in name}
            return self.evaluate_bulk(formula_configs, context, validate_inputs)
        
        elif isinstance(name, dict):
            # Dictionary format - pass through to evaluate_bulk
            return self.evaluate_bulk(name, context, validate_inputs)
        
        else:
            raise TypeError(f"Expected str, List[str], or Dict for name, got {type(name)}")

    def evaluate_bulk(
        self,
        formula_names: Union[List[str], Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]]],
        context: Dict[str, Any],
        validate_inputs: bool = True,
        jit_compile: bool = True
    ) -> xr.Dataset:
        """
        Evaluate multiple formulas efficiently in bulk, with support for multiple configurations.
        
        Args:
            formula_names: Can be:
                - List[str]: List of formula names (backward compatible)
                - Dict[str, Union[Dict, List[Dict]]]: Formula names mapped to config(s)
                  Example: {
                      "wma": {"window": 10},  # Single config
                      "rsi": [{"window": 14}, {"window": 21}],  # Multiple configs
                      "alma": [
                          {"window": 10, "offset": 0.85},
                          {"window": 20, "offset": 0.9}
                      ]
                  }
            context: Dictionary of variable values and functions shared across all evaluations
            validate_inputs: Whether to validate inputs against schemas
            jit_compile: Whether to enable JIT compilation (individual functions are already JIT compiled)
            
        Returns:
            xarray Dataset containing all formula results as data variables.
            When multiple configs are used, results are named as:
            - Single config: formula_name (e.g., "wma", "rsi")
            - Multiple configs: formula_name + parameter suffix (e.g., "wma_w10", "rsi_w14_w21")
            
        Raises:
            KeyError: If any formula doesn't exist
            ValueError: If inputs are invalid or evaluation fails
        """
        # Normalize input to dictionary format
        if isinstance(formula_names, list):
            # Convert list to dict with empty configs for backward compatibility
            formula_configs = {name: [{}] for name in formula_names}
        else:
            # Normalize dict values to always be lists
            formula_configs = {}
            for formula_name, configs in formula_names.items():
                if isinstance(configs, dict):
                    formula_configs[formula_name] = [configs]
                elif isinstance(configs, list):
                    formula_configs[formula_name] = configs
                else:
                    raise ValueError(f"Invalid config type for formula '{formula_name}': {type(configs)}")

        if not formula_configs:
            raise ValueError("formula_names/configs cannot be empty")

        if '_dataset' not in context or not isinstance(context['_dataset'], xr.Dataset):
            raise ValueError("Evaluation context must contain the input xarray Dataset under the key '_dataset'.")

        input_ds = context['_dataset']
        
        # Add formula functions to context for dependency support
        formula_function_context = self._get_formula_function_context(input_ds)
        context.update(formula_function_context)
        
        # Add built-in functions to context  
        from .functions import get_function_context
        builtin_context = get_function_context()
        context.update(builtin_context)
        
        # Parse all formulas once (reuse for multiple configs)
        parsed_formulas = {}
        formula_definitions = {}
          
        for formula_name in formula_configs:
            formula_def = self.get_formula(formula_name)
            formula_definitions[formula_name] = formula_def
            
            # Parse the formula into AST
            try:
                ast_node = parse_formula(formula_def['expression'])
                parsed_formulas[formula_name] = ast_node
            except Exception as e:
                raise ValueError(f"Error parsing formula '{formula_name}': {str(e)}")
        
        # Determine evaluation order based on dependencies
        evaluation_order = self._get_evaluation_order(list(formula_configs.keys()))
        
        # Keep track of computed formula results that can be used as time series dependencies
        computed_results = {}
        
        # Evaluate all formula instances in dependency order
        results = {}
        
        for formula_name in evaluation_order:
            if formula_name not in formula_configs:
                # This formula was added due to dependencies but not explicitly requested
                # Evaluate it with default config to make it available for dependents
                config_list = [{}]
            else:
                config_list = formula_configs[formula_name]
            
            formula_def = formula_definitions.get(formula_name)
            if not formula_def:
                # Load formula definition if not already loaded (for dependencies)
                if formula_name not in self.formulas:
                    continue  # Skip if formula doesn't exist
                formula_def = self.get_formula(formula_name)
                formula_definitions[formula_name] = formula_def
                
                # Parse the formula AST if not already parsed
                if formula_name not in parsed_formulas:
                    try:
                        ast_node = parse_formula(formula_def['expression'])
                        parsed_formulas[formula_name] = ast_node
                    except Exception as e:
                        print(f"Warning: Could not parse dependency formula '{formula_name}': {e}")
                        continue
            
            ast_node = parsed_formulas[formula_name]
            
            for config_idx, config in enumerate(config_list):
                # Merge formula-specific config with base context
                formula_context = context.copy()
                formula_context.update(config)
                
                # Add computed formula results to the dataset for time series dependencies
                if computed_results:
                    enhanced_dataset = formula_context['_dataset'].copy()
                    for dep_name, dep_result in computed_results.items():
                        if isinstance(dep_result, xr.DataArray):
                            enhanced_dataset[dep_name] = dep_result
                    formula_context['_dataset'] = enhanced_dataset
                
                # Add default values for any missing variables
                variables_schema = formula_def.get('variables', {})
                for var_name, var_def in variables_schema.items():
                    if var_name not in formula_context and 'default' in var_def:
                        formula_context[var_name] = var_def['default']
                
                # Process modules for this formula to generate dynamic values
                formula_context = self._process_modules(formula_name, formula_context)
                
                # Validate inputs if requested
                if validate_inputs:
                    self._validate_inputs(formula_name, formula_context)
                
                try:
                    result = ast_node.evaluate(formula_context)
                    
                    # Generate result name based on configuration
                    if len(config_list) == 1 and not config:
                        # Single config with no overrides - use formula name
                        result_name = formula_name
                    else:
                        # Multiple configs or overrides - generate unique name
                        result_name = self._generate_result_name(
                            formula_name, 
                            config, 
                            formula_def.get('variables', {}),
                            config_idx if len(config_list) > 1 else None
                        )
                        
                        # Check if this config only contains default values
                        # If so, make it the primary version (no suffix)
                        variables_schema = formula_def.get('variables', {})
                        all_defaults = True
                        for param_name, param_value in config.items():
                            if param_name.startswith('_') or param_name == 'price' or param_name == 'lag':
                                continue
                            default_value = variables_schema.get(param_name, {}).get('default')
                            if default_value is None or param_value != default_value:
                                all_defaults = False
                                break
                        
                        # If all parameters match defaults and we haven't used the formula name yet
                        if all_defaults and formula_name not in results:
                            result_name = formula_name
                    
                    # Handle lag parameter if specified
                    lag_param = config.get('lag')
                    if lag_param is not None:
                        # Ensure lag is a list for consistent processing
                        lag_values = lag_param if isinstance(lag_param, list) else [lag_param]
                        
                        # Process each lag value by creating a shifted AST
                        for lag_val in lag_values:
                            # Create a new AST that wraps the original formula with shift()
                            # shift(original_expression, lag_value)
                            shifted_ast = FunctionCall(
                                name='shift',
                                args=[
                                    ast_node,  # The original formula AST
                                    Literal(value=float(lag_val))  # Lag as a static literal
                                ]
                            )
                            
                            # Evaluate the shifted AST in the same context
                            try:
                                lagged_result = shifted_ast.evaluate(formula_context)
                                
                                # Generate name with lag suffix
                                lag_suffix = f"_lag{lag_val}" if lag_val >= 0 else f"_lead{-lag_val}"
                                lagged_name = f"{result_name}{lag_suffix}"
                                
                                results[lagged_name] = lagged_result
                            except Exception as e:
                                raise ValueError(f"Error applying lag {lag_val} to formula '{formula_name}': {str(e)}")
                    else:
                        # No lag specified - add the result as is
                        results[result_name] = result
                        
                        # Track this result for potential use as time series dependency
                        # Use the formula name (not result_name) as the key for dependency resolution
                        if isinstance(result, xr.DataArray):
                            # Track all results as potential dependencies, but use the result_name
                            # as the key to ensure we're tracking the correct result instance
                            computed_results[formula_name] = result
                    
                except Exception as e:
                    config_str = f" with config {config}" if config else ""
                    raise ValueError(f"Error evaluating formula '{formula_name}'{config_str}: {str(e)}")
        
        # Construct output dataset with only computed results
        # Create a new dataset with same coordinates but no data variables from input
        output_ds = xr.Dataset(coords=input_ds.coords)
        
        for result_name, result in results.items():
            if isinstance(result, xr.DataArray):
                # Set the name on the DataArray
                result.name = result_name
                # Add to the output dataset
                output_ds[result_name] = result
            elif isinstance(result, xr.Dataset):
                # If the result is a Dataset, merge its variables
                for var_name, var_data in result.data_vars.items():
                    # Prefix with result name to avoid conflicts
                    prefixed_name = f"{result_name}_{var_name}"
                    output_ds[prefixed_name] = var_data
            else:
                # For scalar results, convert to DataArray
                scalar_da = xr.DataArray(result, name=result_name)
                output_ds[result_name] = scalar_da
        
        return output_ds
    
    def _generate_result_name(
        self, 
        formula_name: str, 
        config: Dict[str, Any], 
        variables_schema: Dict[str, Any],
        config_idx: Optional[int] = None
    ) -> str:
        """
        Generate a unique result name based on formula name and configuration.
        
        Args:
            formula_name: Base formula name
            config: Configuration overrides
            variables_schema: Schema of formula variables (to get defaults)
            config_idx: Optional index when multiple configs exist
            
        Returns:
            Unique result name
        """
        if not config:
            # No overrides, but multiple configs exist
            if config_idx is not None:
                return f"{formula_name}_{config_idx}"
            return formula_name
        
        # Build suffix from parameters that differ from defaults
        suffix_parts = []
        
        for param_name, param_value in sorted(config.items()):
            # Skip internal parameters, dataset references, and lag (handled separately)
            if param_name.startswith('_') or param_name == 'price' or param_name == 'lag':
                continue
                
            # Get default value from schema
            default_value = None
            if param_name in variables_schema:
                default_value = variables_schema[param_name].get('default')
            
            # Include parameter if it differs from default or no default exists
            if default_value is None or param_value != default_value:
                # Format the value appropriately
                if isinstance(param_value, (int, float)):
                    # For numbers, use simple representation
                    if isinstance(param_value, float) and param_value.is_integer():
                        value_str = str(int(param_value))
                    else:
                        value_str = str(param_value).replace('.', 'p')
                elif isinstance(param_value, bool):
                    value_str = 'T' if param_value else 'F'
                elif isinstance(param_value, str):
                    # For strings, use first few characters
                    value_str = param_value[:3]
                else:
                    # For other types, use a hash
                    value_str = str(hash(str(param_value)))[:4]
                
                # Create abbreviated parameter name
                param_abbrev = ''.join(c for c in param_name if c.isupper() or param_name.index(c) == 0).lower()
                if not param_abbrev:
                    param_abbrev = param_name[:1]
                    
                suffix_parts.append(f"{param_abbrev}{value_str}")
        
        if suffix_parts:
            return f"{formula_name}_{'_'.join(suffix_parts)}"
        elif config_idx is not None:
            return f"{formula_name}_{config_idx}"
        else:
            return formula_name

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
            # For time series dependencies (formulas), they will be resolved during evaluation
            is_ts_dependency = (var_def.get('type') == 'dataarray' and 
                              var_name in self.formulas and var_name != name)
            
            if var_name not in context and 'default' not in var_def and not is_ts_dependency:
                raise ValueError(f"Missing required variable '{var_name}' for formula '{name}'.")

            # Determine the value to validate (either from context or default)
            # If var_name is not in context, var_def['default'] must exist (checked above for required vars)
            value_to_validate = context.get(var_name, var_def.get('default'))
            
            # For time series dependencies, use the formula name as the key
            is_ts_dependency = (var_def.get('type') == 'dataarray' and 
                              var_name in self.formulas and var_name != name)
            if is_ts_dependency:
                value_to_validate = var_name  # Use formula name as the key

            var_type = var_def.get('type')

            if var_type == 'dataarray':
                if not isinstance(value_to_validate, str):
                    raise ValueError(f"Variable '{var_name}' (type 'dataarray') for formula '{name}' expects a string key in the context, "
                                     f"but got {type(value_to_validate)} for value '{value_to_validate}'.")
                
                # This check assumes dataset_in_context is populated if has_dataarray_var is true.
                if dataset_in_context is None and has_dataarray_var:
                    # This case should ideally not be reached if the initial _dataset check is correct
                    raise ValueError("Internal error: _dataset not verified before dataarray variable check.")
                
                # Check if this is a time series dependency (references another formula)
                if is_ts_dependency:
                    # This is a formula dependency - it will be resolved during evaluation
                    # Skip the dataset key check for now as the dependency will be computed
                    continue
                
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
        Get the set of formula names that a formula depends on.
        
        Args:
            name: Name of the formula
            
        Returns:
            Set of formula names used by the formula
            
        Raises:
            KeyError: If formula doesn't exist
        """
        if name not in self.formulas:
            raise KeyError(f"Formula '{name}' not found")
        
        return self._dependency_graph.get(name, set())
    
    def get_dependency_chain(self, name: str) -> List[str]:
        """
        Get the full dependency chain for a formula in evaluation order.
        
        Args:
            name: Name of the formula
            
        Returns:
            List of formula names in dependency order (dependencies first)
            
        Raises:
            KeyError: If formula doesn't exist
            ValueError: If circular dependencies are detected
        """
        if name not in self.formulas:
            raise KeyError(f"Formula '{name}' not found")
            
        return self._resolve_dependencies(name)
    
    def list_formula_functions(self) -> List[str]:
        """
        Get a list of formulas that are available as functions.
        
        Returns:
            List of formula names that can be used as functions
        """
        return sorted(self._formula_functions.keys())
    
    def compile_all_formulas_as_functions(self) -> None:
        """
        Pre-compile all formulas as functions for performance.
        
        This can be called to ensure all formulas are ready to be used
        as functions in other formulas.
        """
        for formula_name in self.formulas:
            try:
                self._compile_formula_as_function(formula_name)
            except Exception as e:
                print(f"Warning: Could not compile formula '{formula_name}' as function: {e}")
    
    def get_compiled_formula_function(self, name: str) -> Callable:
        """
        Get a compiled formula function by name.
        
        Args:
            name: Name of the formula
            
        Returns:
            Compiled formula function
            
        Raises:
            KeyError: If formula doesn't exist
            ValueError: If formula cannot be compiled
        """
        if name not in self.formulas:
            raise KeyError(f"Formula '{name}' not found")
            
        return self._compile_formula_as_function(name)
    
    def get_function_dependencies(self, name: str) -> Set[str]:
        """
        Get the set of built-in function names that a formula depends on.
        
        Args:
            name: Name of the formula
            
        Returns:
            Set of built-in function names used by the formula
            
        Raises:
            KeyError: If formula doesn't exist
        """
        formula = self.get_formula(name)
        return set(formula.get('functions', {}).keys())
    
    def get_time_series_dependencies(self, name: str) -> Set[str]:
        """
        Get the set of formula names that a formula depends on as time series (dataarray variables).
        
        Args:
            name: Name of the formula
            
        Returns:
            Set of formula names used as dataarray variables by the formula
            
        Raises:
            KeyError: If formula doesn't exist
        """
        if name not in self.formulas:
            raise KeyError(f"Formula '{name}' not found")
        
        formula = self.get_formula(name)
        variables_schema = formula.get('variables', {})
        ts_dependencies = set()
        
        for var_name, var_def in variables_schema.items():
            if var_def.get('type') == 'dataarray' and var_name in self.formulas and var_name != name:
                ts_dependencies.add(var_name)
        
        return ts_dependencies
    
    def get_functional_dependencies(self, name: str) -> Set[str]:
        """
        Get the set of formula names that a formula depends on as functions.
        
        Args:
            name: Name of the formula
            
        Returns:
            Set of formula names called as functions by the formula
            
        Raises:
            KeyError: If formula doesn't exist
        """
        if name not in self.formulas:
            raise KeyError(f"Formula '{name}' not found")
        
        formula = self.get_formula(name)
        
        try:
            functions_used = extract_functions(formula['expression'])
            functional_deps = set()
            
            for func_name in functions_used:
                if func_name in self.formulas:
                    functional_deps.add(func_name)
            
            return functional_deps
        except Exception:
            return set() 

    def load_default_formulas(self) -> None:
        """
        Load formula definitions from the default definitions directory.
        
        This method loads all .yaml files from the package's built-in definitions directory.
        """
        default_dir = os.path.join(os.path.dirname(__file__), 'definitions')
        if os.path.exists(default_dir):
            self.load_directory(default_dir)
            # After loading all formulas, update dependency graphs
            self._rebuild_dependency_graph()
        else:
            print(f"Warning: Default definitions directory not found at {default_dir}")
    
    def _rebuild_dependency_graph(self) -> None:
        """
        Rebuild the entire dependency graph after loading all formulas.
        
        This is needed because formula dependencies can only be resolved
        after all formulas are loaded.
        """
        # Clear existing dependency graph
        self._dependency_graph.clear()
        self._formula_functions.clear()
        
        # Rebuild dependency graph for all formulas
        for formula_name, definition in self.formulas.items():
            self._update_dependency_graph(formula_name, definition) 