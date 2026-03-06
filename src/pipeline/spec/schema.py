"""
Pipeline specification schema definitions.

This module defines the data structures for pipeline specifications, which
describe end-to-end data processing workflows from data loading through
feature engineering and preprocessing.

The specification format is designed to be:
- Human-readable (YAML-based)
- Version-controlled (declarative configuration)
- Cache-friendly (explicit dependencies and stages)
- Extensible (easy to add new stages and operations)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class DataSourceSpec:
    """
    Specification for a single data source.
    
    Attributes:
        provider: Data provider name (e.g., "wrds", "crypto", "openbb")
        dataset: Dataset name within the provider (e.g., "crsp", "binance_spot")
        frequency: Data frequency (e.g., "daily", "hourly", "monthly")
        filters: Dictionary of filters to apply at DataFrame level
        processors: List of processor configurations for xarray-level operations
    """
    
    provider: str
    dataset: str
    frequency: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    processors: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'provider': self.provider,
            'dataset': self.dataset,
            'frequency': self.frequency,
            'filters': self.filters,
            'processors': self.processors,
        }


@dataclass
class FormulaOperationSpec:
    """
    Specification for a formula evaluation operation.
    
    Formula operations are the building blocks of feature engineering. Each
    operation can depend on previous operations, enabling staged computation
    and cache reuse.
    
    Attributes:
        name: Unique name for this operation (used for caching and dependencies)
        formulas: Dictionary mapping formula names to lists of configurations
                 Example: {"rsi": [{"window": 14}, {"window": 21}]}
        depends_on: List of operation names this operation depends on
    """
    
    name: str
    formulas: Dict[str, List[Dict[str, Any]]]
    depends_on: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'formulas': self.formulas,
            'depends_on': self.depends_on,
        }


@dataclass
class FeaturesSpec:
    """
    Specification for feature engineering stage.
    
    The features stage consists of one or more formula operations that are
    executed in dependency order. Operations can depend on each other, enabling
    complex feature pipelines with intermediate caching.
    
    Attributes:
        operations: List of formula operations to execute
    """
    
    operations: List[FormulaOperationSpec] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operations': [op.to_dict() for op in self.operations]
        }
    
    def get_operation_order(self) -> List[str]:
        """
        Get operations in dependency order (topological sort).
        
        Returns:
            List of operation names in execution order
            
        Raises:
            ValueError: If circular dependencies are detected
        """
        # Build dependency graph: node -> set of nodes it depends on
        graph = {op.name: set(op.depends_on) for op in self.operations}
        
        # Calculate in-degree: how many nodes depend on each node
        in_degree = {name: 0 for name in graph}
        for name, deps in graph.items():
            for dep in deps:
                if dep not in in_degree:
                    raise ValueError(f"Unknown dependency: {dep}")
                # The current node depends on dep, so dep's in-degree increases
                in_degree[name] += 1
        
        # Start with nodes that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # For each node that depends on the current node
            for neighbor, deps in graph.items():
                if node in deps:
                    # Remove this dependency
                    in_degree[neighbor] -= 1
                    # If all dependencies satisfied, add to queue
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        if len(result) != len(graph):
            raise ValueError("Circular dependency detected in formula operations")
        
        return result


@dataclass
class PreprocessingSpec:
    """
    Specification for preprocessing stage.
    
    The preprocessing stage uses the DataHandler pipeline with three types
    of processors:
    - shared: Applied once to the full dataset
    - learn: Fit on training data, transform on both train and inference
    - infer: Transform-only on inference data
    
    Attributes:
        shared: List of shared processor configurations
        learn: List of learn processor configurations
        infer: List of infer processor configurations
        mode: Pipeline mode ("independent" or "append")
    """
    
    shared: List[Dict[str, Any]] = field(default_factory=list)
    learn: List[Dict[str, Any]] = field(default_factory=list)
    infer: List[Dict[str, Any]] = field(default_factory=list)
    mode: str = "independent"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'shared': self.shared,
            'learn': self.learn,
            'infer': self.infer,
            'mode': self.mode,
        }


@dataclass
class ModelSpec:
    """
    Specification for model training and prediction using ModelRunner.
    
    Integrates with the existing ModelAdapter/ModelRunner infrastructure for
    walk-forward validation. The system supports multiple adapter types.
    
    Attributes:
        adapter: Adapter type ("sklearn", "lightgbm", "xgboost", "pytorch", etc.)
        type: Model type within the adapter (e.g., "RandomForestRegressor" for sklearn)
        params: Model hyperparameters passed to model constructor
        features: List of feature variable names to use as inputs
        target: Target variable name for prediction
        walk_forward: Walk-forward validation configuration with SegmentConfig params
            - train_span_hours: Training window size in hours
            - infer_span_hours: Inference window size in hours
            - step_hours: Step size between segments in hours
            - gap_hours: Gap between train and infer windows (default: 0)
            - start: Optional start date (uses data bounds if not specified)
            - end: Optional end date (uses data bounds if not specified)
        adapter_params: Additional parameters for the adapter (output_var, use_proba, etc.)
        runner_params: Additional parameters for ModelRunner (overlap_policy, etc.)
    """
    
    adapter: str = "sklearn"  # Default to sklearn for backward compatibility
    type: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    target: str = ""
    walk_forward: Dict[str, Any] = field(default_factory=dict)
    adapter_params: Dict[str, Any] = field(default_factory=dict)
    runner_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'adapter': self.adapter,
            'type': self.type,
            'params': self.params,
            'features': self.features,
            'target': self.target,
            'walk_forward': self.walk_forward,
            'adapter_params': self.adapter_params,
            'runner_params': self.runner_params,
        }


@dataclass
class PipelineSpec:
    """
    Complete pipeline specification.
    
    A pipeline spec describes an end-to-end data processing workflow,
    including data loading, feature engineering, preprocessing, and modeling.
    
    Attributes:
        name: Unique name for this pipeline
        version: Version string (for tracking changes)
        data: Dictionary of data source specifications
        time_range: Dictionary with 'start' and 'end' date strings
        features: Optional feature engineering specification
        preprocessing: Optional preprocessing specification
        model: Optional model training specification
        metadata: Optional metadata dictionary for documentation
    """
    
    name: str
    version: str
    data: Dict[str, DataSourceSpec]
    time_range: Dict[str, str]
    features: Optional[FeaturesSpec] = None
    preprocessing: Optional[PreprocessingSpec] = None
    model: Optional[ModelSpec] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """
        Validate the pipeline specification.
        
        Checks:
        - Required fields are present
        - Time range is valid
        - Feature operation dependencies are valid
        - No circular dependencies
        - Data sources reference valid providers
        - Processor configurations are valid
        
        Raises:
            ValueError: If validation fails
        """
        # Validate name and version
        if not self.name:
            raise ValueError("Pipeline name is required")
        if not self.version:
            raise ValueError("Pipeline version is required")
        
        # Validate time range
        self.validate_time_range()
        
        # Validate data sources
        if not self.data:
            raise ValueError("At least one data source is required")
        
        self.validate_data_sources()
        
        # Validate features (if present)
        if self.features:
            # Check for circular dependencies
            try:
                self.features.get_operation_order()
            except ValueError as e:
                raise ValueError(f"Invalid feature specification: {e}")
            
            # Validate formulas exist
            self.validate_formulas()
        
        # Validate preprocessing (if present)
        if self.preprocessing:
            if self.preprocessing.mode not in ['independent', 'append']:
                raise ValueError(f"Invalid preprocessing mode: {self.preprocessing.mode}")
    
    def validate_time_range(self) -> None:
        """
        Validate that time range is properly specified.
        
        Raises:
            ValueError: If time range is invalid
        """
        if 'start' not in self.time_range or 'end' not in self.time_range:
            raise ValueError("Time range must include 'start' and 'end'")
        
        # Try to parse dates to ensure they're valid
        from datetime import datetime
        try:
            start = datetime.fromisoformat(self.time_range['start'])
            end = datetime.fromisoformat(self.time_range['end'])
            
            if start >= end:
                raise ValueError(f"Start date ({self.time_range['start']}) must be before end date ({self.time_range['end']})")
        except ValueError as e:
            raise ValueError(f"Invalid date format in time_range: {e}")
    
    def validate_data_sources(self) -> None:
        """
        Validate that data sources reference valid providers and datasets.
        
        This performs basic validation of provider names and dataset names.
        More comprehensive validation (checking against _PROVIDER_REGISTRY)
        would require importing the data module, which we avoid here to
        keep the spec module lightweight.
        
        Raises:
            ValueError: If data source configuration is invalid
        """
        for source_name, source_spec in self.data.items():
            if not source_spec.provider:
                raise ValueError(f"Data source '{source_name}' missing provider")
            if not source_spec.dataset:
                raise ValueError(f"Data source '{source_name}' missing dataset")
            
            # Validate processor configurations
            self.validate_processors(source_name, source_spec)
    
    def validate_processors(self, source_name: str, source_spec: DataSourceSpec) -> None:
        """
        Validate that processor configurations are properly formatted.
        
        Args:
            source_name: Name of the data source
            source_spec: Data source specification
            
        Raises:
            ValueError: If processor configuration is invalid
        """
        if not source_spec.processors:
            return
        
        for processor in source_spec.processors:
            if not isinstance(processor, dict):
                raise ValueError(
                    f"Data source '{source_name}': processor must be a dictionary, got {type(processor)}"
                )
            
            if 'type' not in processor:
                raise ValueError(
                    f"Data source '{source_name}': processor missing 'type' field"
                )
    
    def validate_formulas(self) -> None:
        """
        Validate that formulas referenced in operations exist.
        
        This checks that formula names are non-empty strings.
        More comprehensive validation (checking against FormulaManager)
        would require importing the AST module, which we avoid here.
        
        Raises:
            ValueError: If formula configuration is invalid
        """
        if not self.features:
            return
        
        for operation in self.features.operations:
            # operation.formulas is Dict[str, List[Dict[str, Any]]]
            for formula_name, formula_configs in operation.formulas.items():
                if not formula_name:
                    raise ValueError(
                        f"Operation '{operation.name}': formula name cannot be empty"
                    )
                
                if not isinstance(formula_configs, list):
                    raise ValueError(
                        f"Operation '{operation.name}', formula '{formula_name}': "
                        f"configs must be a list, got {type(formula_configs)}"
                    )
                
                # Validate each config is a dict
                for config in formula_configs:
                    if not isinstance(config, dict):
                        raise ValueError(
                            f"Operation '{operation.name}', formula '{formula_name}': "
                            f"each config must be a dict, got {type(config)}"
                        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'name': self.name,
            'version': self.version,
            'data': {name: spec.to_dict() for name, spec in self.data.items()},
            'time_range': self.time_range,
            'metadata': self.metadata,
        }
        
        if self.features:
            result['features'] = self.features.to_dict()
        
        if self.preprocessing:
            result['preprocessing'] = self.preprocessing.to_dict()
        
        if self.model:
            result['model'] = self.model.to_dict()
        
        return result

