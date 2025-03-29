"""
Formula execution module for financial characteristics.

This module provides functionality for executing formulas defined in
characteristic definition files. It handles dependencies between characteristics
and manages the execution pipeline.
"""

import xarray as xr
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Callable, Set, Tuple, Optional
import logging

from .formula import compiler

logger = logging.getLogger(__name__)

class FormulaExecutor:
    """
    Executor for financial characteristic formulas.
    
    This class manages the execution of formulas defined in characteristic
    definition files. It handles dependencies between characteristics and
    ensures that they are computed in the correct order.
    """
    
    def __init__(self, datasets: Dict[str, xr.Dataset]):
        """
        Initialize the formula executor.
        
        Args:
            datasets: Dictionary of datasets by source name
        """
        self.datasets = datasets
        self.computed_characteristics = {}
        self.execution_stack = []
        
    def get_dataset_variable(self, variable: str) -> xr.DataArray:
        """
        Get a variable from a dataset.
        
        Args:
            variable: Variable name in format 'dataset:variable'
            
        Returns:
            DataArray containing the variable data
            
        Raises:
            ValueError: If the variable or dataset does not exist
        """
        if ':' in variable:
            dataset_name, var_name = variable.split(':', 1)
            if dataset_name not in self.datasets:
                raise ValueError(f"Dataset '{dataset_name}' not found")
            if var_name not in self.datasets[dataset_name]:
                raise ValueError(f"Variable '{var_name}' not found in dataset '{dataset_name}'")
            return self.datasets[dataset_name][var_name]
        else:
            # Look for the variable in all datasets
            for dataset_name, dataset in self.datasets.items():
                if variable in dataset:
                    return dataset[variable]
            
            # If not found in any dataset, check if it's a computed characteristic
            if variable in self.computed_characteristics:
                return self.computed_characteristics[variable]
                
            raise ValueError(f"Variable '{variable}' not found in any dataset")
    
    def execute_formula(self, formula: str, 
                      characteristic_name: str, 
                      dependencies: Optional[Set[str]] = None) -> xr.DataArray:
        """
        Execute a formula and return the result.
        
        Args:
            formula: Formula string in the CFG syntax
            characteristic_name: Name of the characteristic
            dependencies: Set of dependencies for this formula
            
        Returns:
            DataArray containing the computed characteristic
            
        Raises:
            ValueError: If there is a circular dependency or a dependency cannot be resolved
            RuntimeError: If there is an error during formula execution
        """
        # Check for circular dependencies
        if characteristic_name in self.execution_stack:
            raise ValueError(f"Circular dependency detected for '{characteristic_name}'")
        
        # If already computed, return the result
        if characteristic_name in self.computed_characteristics:
            return self.computed_characteristics[characteristic_name]
        
        # If dependencies are not provided, extract them from the formula
        if dependencies is None:
            try:
                ast = compiler.FormulaAST.parse(formula)
                dependencies = ast.extract_dependencies()
            except Exception as e:
                raise ValueError(f"Failed to parse formula for '{characteristic_name}': {e}")
        
        # Add this characteristic to the execution stack
        self.execution_stack.append(characteristic_name)
        
        try:
            # Resolve dependencies
            resolved_deps = {}
            for dep in dependencies:
                if dep in self.computed_characteristics:
                    resolved_deps[dep] = self.computed_characteristics[dep]
                else:
                    try:
                        resolved_deps[dep] = self.get_dataset_variable(dep)
                    except ValueError:
                        # If dependency not found in datasets, it might be another characteristic
                        # that hasn't been computed yet
                        if dep in self.characteristic_definitions:
                            dep_def = self.characteristic_definitions[dep]
                            dep_result = self.execute_formula(
                                dep_def['formula'], 
                                dep, 
                                dep_def.get('dependencies')
                            )
                            resolved_deps[dep] = dep_result
                        else:
                            raise ValueError(f"Dependency '{dep}' for '{characteristic_name}' not found")
            
            # For now, since we haven't implemented the full compiler yet,
            # we'll just return a placeholder result
            # In the future, this will compile and execute the formula
            logger.warning(f"Formula execution not yet implemented for '{characteristic_name}'")
            result = xr.DataArray(
                np.full((len(self.datasets['crsp'].time), len(self.datasets['crsp'].asset)), np.nan),
                dims=['time', 'asset'],
                coords={
                    'time': self.datasets['crsp'].time,
                    'asset': self.datasets['crsp'].asset
                }
            )
            
            # Store the result
            self.computed_characteristics[characteristic_name] = result
            
            return result
        finally:
            # Remove this characteristic from the execution stack
            self.execution_stack.pop()
    
    def execute_characteristics(self, 
                              characteristic_definitions: Dict[str, Dict[str, Any]],
                              requested_characteristics: List[str]) -> Dict[str, xr.DataArray]:
        """
        Execute formulas for the requested characteristics.
        
        Args:
            characteristic_definitions: Dictionary of characteristic definitions
            requested_characteristics: List of characteristic names to compute
            
        Returns:
            Dictionary of computed characteristics
            
        Raises:
            ValueError: If a requested characteristic is not defined
        """
        self.characteristic_definitions = characteristic_definitions
        
        # Clear previously computed characteristics
        self.computed_characteristics = {}
        
        # Compute each requested characteristic
        results = {}
        for char_name in requested_characteristics:
            if char_name not in characteristic_definitions:
                raise ValueError(f"Characteristic '{char_name}' not defined")
            
            char_def = characteristic_definitions[char_name]
            formula = char_def['formula']
            dependencies = char_def.get('dependencies')
            
            result = self.execute_formula(formula, char_name, dependencies)
            results[char_name] = result
        
        return results 