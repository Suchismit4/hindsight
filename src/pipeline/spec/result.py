"""
Execution result containers for pipeline execution.

This module provides data classes for storing and reporting pipeline execution results,
including cache statistics, timing information, and operation-level details.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import xarray as xr
from datetime import datetime

from .schema import PipelineSpec


@dataclass
class OperationResult:
    """
    Result for a single feature operation.
    
    Attributes:
        name: Operation name
        formulas_computed: List of formula names computed in this operation
        cache_key: Cache key for this operation
        cache_hit: Whether this operation was loaded from cache
        execution_time: Time taken to execute (or load from cache) in seconds
    """
    name: str
    formulas_computed: List[str]
    cache_key: str
    cache_hit: bool
    execution_time: float


@dataclass
class ExecutionResult:
    """
    Container for pipeline execution results.
    
    This class holds all artifacts and metadata from a pipeline execution,
    including the final datasets, cache keys, timing information, and
    operation-level details for analysis and debugging.
    
    Attributes:
        spec: The pipeline specification that was executed
        data: Raw data by source name (after L2 processing)
        features_data: Dataset after feature engineering (L3)
        preprocessed_data: Dataset after preprocessing (L4)
        model_predictions: Dataset with model predictions (L5)
        learned_states: Learned processor states from preprocessing (for inference)
        fitted_model: Legacy alias for the model artifact (ModelRunnerResult)
        model_runner_result: Canonical model artifact (ModelRunnerResult or adapter)
        cache_keys: Mapping from stage name to cache key
        execution_time: Total execution time in seconds
        cache_hits: Mapping from stage name to cache hit/miss status
        operation_results: List of per-operation results (for feature engineering)
        timestamp: When this execution completed
    """
    spec: PipelineSpec
    data: Dict[str, xr.Dataset] = field(default_factory=dict)
    features_data: Optional[xr.Dataset] = None
    preprocessed_data: Optional[xr.Dataset] = None
    model_predictions: Optional[xr.Dataset] = None
    learned_states: Dict[str, Any] = field(default_factory=dict)
    fitted_model: Optional[Any] = None  # kept for backward compatibility
    model_runner_result: Optional[Any] = None
    cache_keys: Dict[str, str] = field(default_factory=dict)
    execution_time: float = 0.0
    cache_hits: Dict[str, bool] = field(default_factory=dict)
    operation_results: List[OperationResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_final_dataset(self) -> xr.Dataset:
        """
        Get the final dataset from the pipeline.
        
        Returns the most processed dataset available:
        1. Preprocessed data if available (L4)
        2. Features data if available (L3)
        3. First data source otherwise (L2)
        
        Returns:
            The final xarray Dataset
            
        Raises:
            ValueError: If no data is available
        """
        if self.preprocessed_data is not None:
            return self.preprocessed_data
        elif self.features_data is not None:
            return self.features_data
        elif self.data:
            # Return the first data source (or merged if multiple)
            return next(iter(self.data.values()))
        else:
            raise ValueError("No data available in execution result")
    
    def get_cache_efficiency(self) -> Dict[str, Any]:
        """
        Calculate cache hit rate and efficiency metrics.
        
        Returns:
            Dictionary with cache efficiency statistics:
            - total_operations: Total number of feature operations
            - cache_hits: Number of cache hits
            - cache_misses: Number of cache misses
            - hit_rate: Proportion of operations that were cache hits
        """
        total_ops = len(self.operation_results)
        if total_ops == 0:
            return {
                'total_operations': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'hit_rate': 0.0
            }
        
        hits = sum(1 for op in self.operation_results if op.cache_hit)
        return {
            'total_operations': total_ops,
            'cache_hits': hits,
            'cache_misses': total_ops - hits,
            'hit_rate': hits / total_ops
        }
    
    def summary(self) -> str:
        """
        Generate a human-readable execution summary.
        
        Returns:
            Multi-line string with execution statistics
        """
        lines = [
            "="*80,
            f"Pipeline Execution Summary: {self.spec.name}",
            "="*80,
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total execution time: {self.execution_time:.2f}s",
            "",
            "Data Sources:",
        ]
        
        for source_name in self.data.keys():
            cache_status = "HIT" if self.cache_hits.get(f"data_{source_name}", False) else "MISS"
            lines.append(f"  - {source_name}: {cache_status}")
        
        if self.operation_results:
            lines.append("")
            lines.append("Feature Operations:")
            for op_result in self.operation_results:
                cache_status = "HIT" if op_result.cache_hit else "MISS"
                formulas_str = ", ".join(op_result.formulas_computed)
                lines.append(f"  - {op_result.name} ({cache_status}, {op_result.execution_time:.2f}s)")
                lines.append(f"    Formulas: {formulas_str}")
        
        if self.operation_results:
            efficiency = self.get_cache_efficiency()
            lines.append("")
            lines.append("Cache Efficiency:")
            lines.append(f"  Total operations: {efficiency['total_operations']}")
            lines.append(f"  Cache hits: {efficiency['cache_hits']}")
            lines.append(f"  Cache misses: {efficiency['cache_misses']}")
            lines.append(f"  Hit rate: {efficiency['hit_rate']:.1%}")
        
        lines.append("="*80)
        return "\n".join(lines)
    
    def get_operation_by_name(self, name: str) -> Optional[OperationResult]:
        """
        Get operation result by name.
        
        Args:
            name: Operation name
            
        Returns:
            OperationResult if found, None otherwise
        """
        for op_result in self.operation_results:
            if op_result.name == name:
                return op_result
        return None

