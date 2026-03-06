"""
YAML parser for pipeline specifications.

This module provides functionality to parse YAML files into PipelineSpec
objects, with validation and error handling.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

from .schema import (
    PipelineSpec,
    DataSourceSpec,
    FeaturesSpec,
    FormulaOperationSpec,
    PreprocessingSpec,
)


class SpecParser:
    """
    Parser for pipeline specification YAML files.
    
    This parser converts YAML configuration files into strongly-typed
    PipelineSpec objects, with validation and helpful error messages.
    """
    
    @staticmethod
    def load_from_yaml(spec_path: str) -> PipelineSpec:
        """
        Load and parse a pipeline specification from a YAML file.
        
        Args:
            spec_path: Path to the YAML specification file
            
        Returns:
            Parsed and validated PipelineSpec object
            
        Raises:
            FileNotFoundError: If the spec file doesn't exist
            ValueError: If the spec is invalid or malformed
            yaml.YAMLError: If the YAML syntax is invalid
            
        Example:
            >>> spec = SpecParser.load_from_yaml("configs/momentum_model.yaml")
            >>> print(spec.name, spec.version)
            momentum_strategy_v1 1.0
        """
        spec_path = Path(spec_path)
        
        if not spec_path.exists():
            raise FileNotFoundError(f"Specification file not found: {spec_path}")
        
        try:
            with open(spec_path, 'r') as f:
                raw_spec = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {spec_path}: {e}")
        
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Specification file must contain a dictionary, got {type(raw_spec)}")
        
        # Parse the specification
        try:
            spec = SpecParser._parse_spec(raw_spec)
            
            # Validate the parsed spec
            spec.validate()
            
            return spec
            
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Error parsing specification from {spec_path}: {e}")
    
    @staticmethod
    def _parse_spec(raw: Dict[str, Any]) -> PipelineSpec:
        """
        Parse the raw dictionary into a PipelineSpec.
        
        Args:
            raw: Raw dictionary from YAML file
            
        Returns:
            PipelineSpec object
        """
        # Extract basic fields
        name = raw.get('name')
        if not name:
            raise ValueError("Specification must include 'name' field")
        
        version = raw.get('spec_version', raw.get('version', '1.0'))
        
        # Parse time range
        time_range = raw.get('time_range', {})
        if not time_range:
            raise ValueError("Specification must include 'time_range' field")
        
        # Parse data sources
        data_raw = raw.get('data', {})
        if not data_raw:
            raise ValueError("Specification must include 'data' field with at least one source")
        
        data = SpecParser._parse_data_sources(data_raw)
        
        # Parse optional features
        features = None
        if 'features' in raw:
            features = SpecParser._parse_features(raw['features'])
        
        # Parse optional preprocessing
        preprocessing = None
        if 'preprocessing' in raw:
            preprocessing = SpecParser._parse_preprocessing(raw['preprocessing'])
        
        # Parse optional model
        model = None
        if 'model' in raw:
            model = SpecParser._parse_model(raw['model'])
        
        # Parse optional metadata
        metadata = raw.get('metadata', {})
        
        return PipelineSpec(
            name=name,
            version=version,
            data=data,
            time_range=time_range,
            features=features,
            preprocessing=preprocessing,
            model=model,
            metadata=metadata
        )
    
    @staticmethod
    def _parse_data_sources(raw: Dict[str, Any]) -> Dict[str, DataSourceSpec]:
        """
        Parse data sources section.
        
        Args:
            raw: Raw data sources dictionary
            
        Returns:
            Dictionary of DataSourceSpec objects
        """
        data_sources = {}
        
        for source_name, source_config in raw.items():
            if not isinstance(source_config, dict):
                raise ValueError(f"Data source '{source_name}' must be a dictionary")
            
            provider = source_config.get('provider')
            if not provider:
                raise ValueError(f"Data source '{source_name}' must specify 'provider'")
            
            dataset = source_config.get('dataset')
            if not dataset:
                raise ValueError(f"Data source '{source_name}' must specify 'dataset'")
            
            data_sources[source_name] = DataSourceSpec(
                provider=provider,
                dataset=dataset,
                frequency=source_config.get('frequency'),
                filters=source_config.get('filters', {}),
                processors=source_config.get('processors', [])
            )
        
        return data_sources
    
    @staticmethod
    def _parse_features(raw: Dict[str, Any]) -> FeaturesSpec:
        """
        Parse features section.
        
        Args:
            raw: Raw features dictionary
            
        Returns:
            FeaturesSpec object
        """
        operations_raw = raw.get('operations', [])
        
        if not isinstance(operations_raw, list):
            raise ValueError("Features 'operations' must be a list")
        
        operations = []
        for op_config in operations_raw:
            if not isinstance(op_config, dict):
                raise ValueError("Each operation must be a dictionary")
            
            name = op_config.get('name')
            if not name:
                raise ValueError("Each operation must have a 'name'")
            
            formulas = op_config.get('formulas', {})
            if not isinstance(formulas, dict):
                raise ValueError(f"Operation '{name}' formulas must be a dictionary")
            
            # Normalize formulas: ensure each formula maps to a list of configs
            normalized_formulas = {}
            for formula_name, formula_configs in formulas.items():
                if isinstance(formula_configs, dict):
                    # Single config: wrap in list
                    normalized_formulas[formula_name] = [formula_configs]
                elif isinstance(formula_configs, list):
                    # Already a list
                    normalized_formulas[formula_name] = formula_configs
                else:
                    raise ValueError(
                        f"Formula '{formula_name}' in operation '{name}' must be "
                        f"a dict or list, got {type(formula_configs)}"
                    )
            
            depends_on = op_config.get('depends_on', [])
            if isinstance(depends_on, str):
                # Single dependency: wrap in list
                depends_on = [depends_on]
            elif not isinstance(depends_on, list):
                raise ValueError(f"Operation '{name}' depends_on must be a string or list")
            
            operations.append(FormulaOperationSpec(
                name=name,
                formulas=normalized_formulas,
                depends_on=depends_on
            ))
        
        return FeaturesSpec(operations=operations)
    
    @staticmethod
    def _parse_preprocessing(raw: Dict[str, Any]) -> PreprocessingSpec:
        """
        Parse preprocessing section.
        
        Args:
            raw: Raw preprocessing dictionary
            
        Returns:
            PreprocessingSpec object
        """
        shared = raw.get('shared', [])
        learn = raw.get('learn', [])
        infer = raw.get('infer', [])
        mode = raw.get('mode', 'independent')
        
        # Validate that processor lists are actually lists
        if not isinstance(shared, list):
            raise ValueError("Preprocessing 'shared' must be a list")
        if not isinstance(learn, list):
            raise ValueError("Preprocessing 'learn' must be a list")
        if not isinstance(infer, list):
            raise ValueError("Preprocessing 'infer' must be a list")
        
        return PreprocessingSpec(
            shared=shared,
            learn=learn,
            infer=infer,
            mode=mode
        )
    
    @staticmethod
    def _parse_model(raw: Dict[str, Any]):
        """
        Parse model section.
        
        Args:
            raw: Raw model dictionary
            
        Returns:
            ModelSpec object
        """
        from .schema import ModelSpec
        
        # Adapter type (sklearn, lightgbm, xgboost, pytorch, etc.)
        adapter = raw.get('adapter', 'sklearn')  # Default to sklearn
        
        # Model type within the adapter
        model_type = raw.get('type')
        if not model_type:
            raise ValueError("Model specification must include 'type' field")
        
        params = raw.get('params', {})
        features = raw.get('features', [])
        target = raw.get('target', '')
        walk_forward = raw.get('walk_forward', {})
        adapter_params = raw.get('adapter_params', {})
        runner_params = raw.get('runner_params', {})
        
        # Validate features is a list
        if not isinstance(features, list):
            raise ValueError("Model 'features' must be a list")
        
        # Validate walk_forward is a dict
        if not isinstance(walk_forward, dict):
            raise ValueError("Model 'walk_forward' must be a dictionary")
        
        # Validate adapter_params is a dict
        if not isinstance(adapter_params, dict):
            raise ValueError("Model 'adapter_params' must be a dictionary")
        
        # Validate runner_params is a dict
        if not isinstance(runner_params, dict):
            raise ValueError("Model 'runner_params' must be a dictionary")
        
        return ModelSpec(
            adapter=adapter,
            type=model_type,
            params=params,
            features=features,
            target=target,
            walk_forward=walk_forward,
            adapter_params=adapter_params,
            runner_params=runner_params
        )
    
    @staticmethod
    def save_to_yaml(spec: PipelineSpec, output_path: str) -> None:
        """
        Save a PipelineSpec to a YAML file.
        
        Args:
            spec: PipelineSpec to save
            output_path: Path to output YAML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(spec.to_dict(), f, default_flow_style=False, sort_keys=False)

