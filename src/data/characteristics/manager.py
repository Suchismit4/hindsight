"""
Manager for financial characteristics computation.

This module provides the CharacteristicsManager class that coordinates the
computation of financial characteristics from raw financial data using a 
context-free grammar (CFG) approach for defining characteristic formulas.
"""

import xarray as xr
import yaml
import os
from typing import Dict, Any, List, Union, Optional, Set
from src.data.core.cache import CacheManager
from .executor import FormulaExecutor
from .parser import parse_formula, extract_variables

class CharacteristicsManager:
    """
    Manager for computing and caching financial characteristics.
    
    This class computes financial characteristics using a flexible grammar-based approach
    that allows for defining characteristics through equations in configuration files.
    It serves as a higher-level cache (L3) on top of the raw data cache.
    
    Attributes:
        cache_manager: Reference to the cache manager for storing computed characteristics
        _characteristic_definitions: Dictionary of characteristic definitions loaded from configuration
        _characteristic_dependencies: Graph of dependencies between characteristics
    """
    
    def __init__(self, cache_manager: CacheManager, definitions_dir: Optional[str] = None):
        """
        Initialize the CharacteristicsManager.
        
        Args:
            cache_manager: The cache manager to use for caching computed characteristics
            definitions_dir: Directory containing characteristic definition files
        """
        self.cache_manager = cache_manager
        self._characteristic_definitions = {}
        self._characteristic_dependencies = {}
        
        # Default definitions directory if not provided
        if definitions_dir is None:
            # Try to find definitions in package directory
            import src
            package_dir = os.path.dirname(os.path.abspath(src.__file__))
            definitions_dir = os.path.join(package_dir, 'data', 'characteristics', 'definitions')
        
        # Load characteristic definitions from the definitions directory if it exists
        if os.path.exists(definitions_dir):
            self._load_characteristic_definitions(definitions_dir)
    
    def _load_characteristic_definitions(self, definitions_dir: str) -> None:
        """
        Load characteristic definitions from YAML files.
        
        Args:
            definitions_dir: Directory containing characteristic definition files
        """
        for filename in os.listdir(definitions_dir):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                filepath = os.path.join(definitions_dir, filename)
                try:
                    with open(filepath, 'r') as file:
                        definitions = yaml.safe_load(file)
                        
                        # Validate and process definitions
                        if isinstance(definitions, dict):
                            for char_name, char_def in definitions.items():
                                self._add_characteristic_definition(char_name, char_def)
                except Exception as e:
                    print(f"Error loading characteristic definitions from {filepath}: {e}")
    
    def _add_characteristic_definition(self, name: str, definition: Dict[str, Any]) -> None:
        """
        Add a characteristic definition to the manager.
        
        Args:
            name: Name of the characteristic
            definition: Definition dictionary containing formula, dependencies, etc.
        """
        # Validate the definition
        if 'formula' not in definition:
            print(f"Warning: Characteristic {name} has no formula defined, skipping.")
            return
        
        # Add to definitions
        self._characteristic_definitions[name] = definition
        
        # Extract and store dependencies
        dependencies = self._extract_dependencies(definition['formula'])
        self._characteristic_dependencies[name] = dependencies
    
    def _extract_dependencies(self, formula: str) -> Set[str]:
        """
        Extract dependencies from a characteristic formula.
        
        Uses the parser module to extract variables from the formula,
        then filters to only include known characteristics.
        
        Args:
            formula: Formula string in the CFG syntax
            
        Returns:
            Set of characteristic names that this formula depends on
        """
        try:
            # Parse the formula and extract variables
            ast_node = parse_formula(formula)
            variables = extract_variables(ast_node)
            
            # Filter to only include known characteristics
            dependencies = {var for var in variables if var in self._characteristic_definitions}
            
            return dependencies
        except Exception as e:
            # If parsing fails, return an empty set
            import logging
            logging.getLogger(__name__).warning(f"Failed to extract dependencies from formula: {e}")
            return set()
    
    def register_characteristic(self, name: str, definition: Dict[str, Any]) -> None:
        """
        Register a new characteristic definition programmatically.
        
        Args:
            name: Name of the characteristic
            definition: Definition dictionary containing formula, dependencies, etc.
        """
        self._add_characteristic_definition(name, definition)
    
    def compute_characteristics(self, data: Dict[str, xr.Dataset], config: Dict[str, Any]) -> Dict[str, xr.Dataset]:
        """
        Compute financial characteristics based on the input data and configuration.
        
        The computation process follows these steps:
        1. Identify which characteristics are requested
        2. Determine dependencies between characteristics
        3. Compute characteristics in the correct order based on dependencies
        4. Cache computed characteristics for future use
        
        Args:
            data: Dictionary of raw datasets from different sources
            config: Configuration specifying which characteristics to compute, with structure:
                {
                    "characteristics": {
                        "accounting": ["assets", "sales", "book_equity", ...],
                        "market": ["ret_1_0", "ret_12_1", "chcsho_12m", ...],
                        ...
                    },
                    "cache_level": "L3",  # Optional, default is "L3"
                    "force_recompute": False  # Optional, default is False
                }
            
        Returns:
            Dictionary of datasets with computed characteristics
            
        Raises:
            ValueError: If an unknown characteristic is requested
        """
        # Extract the requested characteristics
        if "characteristics" not in config:
            return data
            
        requested_chars = []
        for category, chars in config["characteristics"].items():
            requested_chars.extend(chars)
            
        # Check if all requested characteristics are known
        unknown_chars = [c for c in requested_chars if c not in self._characteristic_definitions]
        if unknown_chars:
            raise ValueError(f"Unknown characteristics requested: {unknown_chars}")
            
        # Read cache configuration
        cache_level = config.get("cache_level", "L3")
        force_recompute = config.get("force_recompute", False)
        
        # Try to load characteristics from cache if not forcing recomputation
        if not force_recompute:
            # This would be implemented to check if the characteristics are in the cache
            # and load them if they are, avoiding recomputation
            pass
        
        # Create a formula executor with the raw datasets
        executor = FormulaExecutor(data)
        
        try:
            # Compute all characteristics
            computed_chars = executor.execute_characteristics(
                self._characteristic_definitions,
                requested_chars
            )
            
            # Add computed characteristics to a new dataset
            chars_dataset = xr.Dataset(computed_chars)
            
            # Update the original data dictionary with the characteristics dataset
            result = data.copy()
            result['characteristics'] = chars_dataset
            
            # Cache the computed characteristics for future use
            if cache_level != "none":
                # This would be implemented to store the characteristics in the cache
                pass
                
            return result
        except Exception as e:
            # Log the error and re-raise
            import logging
            logging.getLogger(__name__).error(f"Error computing characteristics: {e}")
            raise
    
    def get_available_characteristics(self) -> Dict[str, List[str]]:
        """
        Get information about all available financial characteristics.
        
        Returns:
            Dictionary mapping characteristic categories to lists of characteristics
        """
        # Group characteristics by category
        categorized = {
            "accounting": [],
            "market": [],
            "combined": []
        }
        
        # Assign each characteristic to its category based on the definition
        for char_name, char_def in self._characteristic_definitions.items():
            category = char_def.get("category", "other")
            if category in categorized:
                categorized[category].append(char_name)
            else:
                # If the category doesn't exist, create it
                categorized[category] = [char_name]
        
        return categorized
    
    def get_characteristic_definition(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get the definition for a specific characteristic.
        
        Args:
            name: Name of the characteristic
            
        Returns:
            Definition dictionary if found, None otherwise
        """
        return self._characteristic_definitions.get(name) 