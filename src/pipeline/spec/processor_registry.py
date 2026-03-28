"""
Processor registry for mapping YAML processor specifications to processor instances.

This module provides a registry system that enables instantiation of processor
objects from YAML configuration dictionaries. It acts as a bridge between the
declarative pipeline specifications and the concrete processor implementations.
"""

from typing import Any, Dict, Type
from src.pipeline.data_handler.processors import (
    Processor,
    CSZScore,
    PerAssetFFill,
    FormulaEval,
    CrossSectionalSort,
    PortfolioReturns,
    FactorSpread,
)


class ProcessorRegistry:
    """
    Registry for mapping processor type strings to processor classes.
    
    This registry enables dynamic processor instantiation from YAML specifications,
    converting processor configuration dictionaries into concrete processor instances.
    
    The registry follows a simple pattern:
    1. Map processor type strings to processor classes
    2. Extract processor-specific parameters from config
    3. Instantiate processor with appropriate parameters
    
    Example:
        >>> config = {'type': 'cs_zscore', 'vars': ['close', 'volume'], 'out_suffix': '_norm'}
        >>> processor = ProcessorRegistry.create_processor(config)
        >>> isinstance(processor, CSZScore)
        True
    """
    
    # Registry mapping processor type strings to classes
    _registry: Dict[str, Type[Processor]] = {
        'cs_zscore': CSZScore,
        'per_asset_ffill': PerAssetFFill,
        'formula_eval': FormulaEval,
        'cross_sectional_sort': CrossSectionalSort,
        'portfolio_returns': PortfolioReturns,
        'factor_spread': FactorSpread,
        # Short aliases used in ff3_model.yaml
        'sort': CrossSectionalSort,
        'port_ret': PortfolioReturns,
    }
    
    @classmethod
    def register(cls, type_name: str, processor_class: Type[Processor]) -> None:
        """
        Register a new processor type.
        
        Args:
            type_name: String identifier for the processor type
            processor_class: Processor class to register
            
        Example:
            >>> ProcessorRegistry.register('my_processor', MyProcessor)
        """
        cls._registry[type_name] = processor_class
    
    @classmethod
    def create_processor(cls, config: Dict[str, Any]) -> Processor:
        """
        Create a processor instance from a configuration dictionary.
        
        The configuration dictionary must contain a 'type' field that maps to a
        registered processor class. All other fields are passed as parameters to
        the processor constructor.
        
        Args:
            config: Configuration dictionary with 'type' and processor-specific params
            
        Returns:
            Instantiated processor object
            
        Raises:
            ValueError: If processor type is not found or config is invalid
            KeyError: If 'type' field is missing from config
            
        Example:
            >>> config = {
            ...     'type': 'cs_zscore',
            ...     'name': 'normalizer',
            ...     'vars': ['close', 'volume'],
            ...     'out_suffix': '_norm'
            ... }
            >>> processor = ProcessorRegistry.create_processor(config)
        """
        if 'type' not in config:
            raise KeyError("Processor config must contain 'type' field")
        
        proc_type = config['type']
        
        if proc_type not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ValueError(
                f"Unknown processor type: '{proc_type}'. "
                f"Available types: {available}"
            )
        
        processor_class = cls._registry[proc_type]
        
        # Extract processor-specific parameters (everything except 'type')
        params = {k: v for k, v in config.items() if k != 'type'}
        
        # Generate a default name if not provided
        if 'name' not in params:
            params['name'] = f"{proc_type}_{id(config) % 10000}"
        
        try:
            return processor_class(**params)
        except TypeError as e:
            raise ValueError(
                f"Invalid parameters for processor '{proc_type}': {e}. "
                f"Config: {params}"
            )
    
    @classmethod
    def get_available_types(cls) -> list:
        """
        Get list of available processor types.
        
        Returns:
            List of registered processor type strings
        """
        return list(cls._registry.keys())

