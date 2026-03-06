"""
Pipeline executor for end-to-end spec execution.

This module provides the PipelineExecutor class that orchestrates the execution
of pipeline specifications, coordinating between data loading, feature engineering,
preprocessing, and caching.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xarray as xr

from src.data.managers.data_manager import DataManager
from src.data.ast.manager import FormulaManager
from src.data.ast.functions import get_function_context
from src.data.core.jit import prepare_for_jit, restore_from_jit
from src.pipeline.cache import GlobalCacheManager, CacheStage
from src.pipeline.data_handler.merge import merge_datasets

from .schema import PipelineSpec, DataSourceSpec, FormulaOperationSpec
from .result import ExecutionResult, OperationResult


class PipelineExecutor:
    """
    Executor for pipeline specifications.
    
    This class orchestrates end-to-end execution of pipeline specifications,
    handling data loading (L1/L2), feature engineering (L3), and preprocessing (L4)
    with intelligent caching at each stage.
    
    Attributes:
        cache_manager: Global cache manager for content-addressable caching
        data_manager: Data manager for loading raw data
        formula_manager: Formula manager for evaluating formulas (lazy-initialized)
    """
    
    def __init__(
        self, 
        cache_manager: Optional[GlobalCacheManager] = None,
        data_manager: Optional[DataManager] = None
    ):
        """
        Initialize the pipeline executor.
        
        Args:
            cache_manager: Global cache manager. If None, creates a new one.
            data_manager: Data manager for loading data. If None, creates a new one.
        """
        self.cache_manager = cache_manager or GlobalCacheManager()
        self.data_manager = data_manager or DataManager()
        self._formula_manager: Optional[FormulaManager] = None
        
        # Track operation cache keys for dependency resolution
        self._operation_cache_keys: Dict[str, str] = {}
    
    @property
    def formula_manager(self) -> FormulaManager:
        """Lazy-initialize formula manager."""
        if self._formula_manager is None:
            self._formula_manager = FormulaManager()
        return self._formula_manager
    
    def execute(self, spec: PipelineSpec) -> ExecutionResult:
        """
        Execute a complete pipeline from specification.
        
        This is the main entry point for pipeline execution. It executes
        all stages defined in the spec (data loading, features, preprocessing)
        and returns a comprehensive result object.
        
        Args:
            spec: Pipeline specification to execute
            
        Returns:
            ExecutionResult containing all artifacts and metadata
            
        Raises:
            ValueError: If spec validation fails or execution encounters errors
        """
        start_time = time.time()
        
        # Validate the spec
        spec.validate()
        
        # Create result container
        result = ExecutionResult(spec=spec)
        
        # Stage 1: Data loading (L2)
        print(f"\n{'='*80}")
        print(f"Executing Pipeline: {spec.name}")
        print(f"{'='*80}\n")
        
        data_dict, data_key = self._execute_data_stage(spec)
        result.data = data_dict
        result.cache_keys['data'] = data_key
        
        # Merge data sources into single dataset for feature engineering
        merged_data = self._merge_data_sources(spec, data_dict)
        
        # Stage 2: Feature engineering (L3)
        if spec.features and spec.features.operations:
            print(f"\n{'='*80}")
            print("Feature Engineering Stage")
            print(f"{'='*80}\n")
            
            features_data, features_key = self._execute_features_stage(
                spec, merged_data, data_key
            )
            result.features_data = features_data
            result.cache_keys['features'] = features_key
        else:
            # No features defined, use merged data as-is
            result.features_data = merged_data
        
        # Stage 3: Preprocessing (L4)
        preprocessing_handler = None
        if spec.preprocessing and (spec.preprocessing.shared or spec.preprocessing.learn or spec.preprocessing.infer):
            print(f"\n{'='*80}")
            print("Preprocessing Stage")
            print(f"{'='*80}\n")
            
            preprocessed_data, preprocessing_key, preprocessing_handler = self._execute_preprocessing_stage(
                spec, result.features_data or merged_data, result.cache_keys.get('features', data_key)
            )
            result.preprocessed_data = preprocessed_data
            result.cache_keys['preprocessing'] = preprocessing_key
            # Store handler (contains learned states) for model stage
            result.learned_states = preprocessing_handler
        else:
            # No preprocessing defined, use features data as-is
            result.preprocessed_data = result.features_data
        
        # Stage 4: Model Training/Predictions (L5)
        if spec.model:
            print(f"\n{'='*80}")
            print("Model Training/Prediction Stage")
            print(f"{'='*80}\n")
            
            model_predictions, model_key, runner_result = self._execute_model_stage(
                spec, 
                result.preprocessed_data or result.features_data or merged_data,
                result.cache_keys.get('preprocessing', result.cache_keys.get('features', data_key)),
                preprocessing_handler
            )
            result.model_predictions = model_predictions
            result.cache_keys['model'] = model_key
            result.fitted_model = runner_result
            if hasattr(result, "model_runner_result"):
                result.model_runner_result = runner_result
        
        # Record total execution time
        result.execution_time = time.time() - start_time
        
        print(f"\n{result.summary()}")
        
        return result
    
    def _execute_data_stage(self, spec: PipelineSpec) -> Tuple[Dict[str, xr.Dataset], str]:
        """
        Execute data loading stage with L2 caching.
        
        Loads all data sources defined in the spec, using the cache manager
        to avoid redundant loading when the same data is requested.
        
        Args:
            spec: Pipeline specification
            
        Returns:
            Tuple of (datasets_dict, cache_key) where:
            - datasets_dict: Mapping from source name to loaded dataset
            - cache_key: Combined cache key for all data sources
        """
        print("Loading data sources...")
        
        datasets = {}
        source_cache_keys = []
        
        for source_name, source_spec in spec.data.items():
            print(f"\n  Source: {source_name}")
            print(f"    Provider: {source_spec.provider}")
            print(f"    Dataset: {source_spec.dataset}")
            
            # Build config for this data source
            config = self._build_data_config(source_name, source_spec, spec.time_range)
            
            # Compute function for loading data
            def load_data_fn():
                return self._load_single_source(source_name, source_spec, spec.time_range)
            
            # Use cache manager
            dataset, cache_key = self.cache_manager.get_or_compute(
                stage=CacheStage.L2_POSTPROCESSED,
                config=config,
                parent_keys=[],
                compute_fn=load_data_fn
            )
            
            datasets[source_name] = dataset
            source_cache_keys.append(cache_key)
            
            print(f"    Shape: {dict(dataset.dims)}")
            print(f"    Variables: {list(dataset.data_vars)}")
        
        # Compute combined cache key for all data sources
        # This is used as parent key for feature operations
        import hashlib
        import json
        combined_key = hashlib.sha256(
            json.dumps(sorted(source_cache_keys)).encode()
        ).hexdigest()[:16]
        
        return datasets, combined_key
    
    def _build_data_config(
        self, 
        source_name: str, 
        source_spec: DataSourceSpec,
        time_range: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Convert DataSourceSpec to cache config format.
        
        Args:
            source_name: Name of the data source
            source_spec: Data source specification
            time_range: Global time range from pipeline spec
            
        Returns:
            Configuration dictionary for cache key computation
        """
        # Use global time range (DataSourceSpec doesn't have time_range attribute)
        start_date = time_range.get('start')
        end_date = time_range.get('end')
        
        return {
            'source_name': source_name,
            'provider': source_spec.provider,
            'dataset': source_spec.dataset,
            'frequency': source_spec.frequency,
            'start_date': start_date,
            'end_date': end_date,
            'filters': source_spec.filters,
            'external_tables': source_spec.external_tables,
            'columns': source_spec.columns,
            'processors': source_spec.processors,
        }
    
    def _load_single_source(
        self,
        source_name: str,
        source_spec: DataSourceSpec,
        time_range: Dict[str, str]
    ) -> xr.Dataset:
        """
        Load a single data source using DataManager.
        
        Args:
            source_name: Name of the data source
            source_spec: Data source specification
            time_range: Global time range from pipeline spec
            
        Returns:
            Loaded xarray Dataset
        """
        # Build data_path for DataManager
        data_path = f"{source_spec.provider}/{source_spec.dataset}"
        
        # Use global time range (DataSourceSpec doesn't have time_range attribute)
        start_date = time_range.get('start')
        end_date = time_range.get('end')
        
        # Build config for DataManager
        config = {
            'start_date': start_date,
            'end_date': end_date,
        }
        
        # Add frequency if specified
        if source_spec.frequency:
            config['frequency'] = source_spec.frequency
        
        # Add filters
        if source_spec.filters:
            config['filters'] = source_spec.filters
        if source_spec.external_tables:
            config['external_tables'] = source_spec.external_tables
        if source_spec.columns:
            config['columns_to_read'] = source_spec.columns
        if source_spec.processors:
            config['processors'] = source_spec.processors
        
        # Create data request for DataManager
        data_request = [{
            'data_path': data_path,
            'config': config
        }]
        
        # Load data
        result = self.data_manager._get_raw_data(data_request)
        
        # Return the dataset
        return result[data_path]
    
    def _merge_data_sources(self, spec: PipelineSpec, datasets: Dict[str, xr.Dataset]) -> xr.Dataset:
        """
        Merge multiple data sources into a single dataset.
        
        For now, this is a simple implementation that just returns the first
        dataset if there's only one, or merges them if there are multiple.
        
        Args:
            datasets: Dictionary of datasets by source name
            
        Returns:
            Merged xarray Dataset
        """
        if len(datasets) == 0:
            raise ValueError("No datasets to merge")

        if spec.merges:
            base_name = spec.merge_base or next(iter(datasets.keys()))
            if base_name not in datasets:
                raise ValueError(f"Merge base '{base_name}' not found in loaded datasets")

            referenced = {base_name} | {merge_cfg["right_name"] for merge_cfg in spec.merges}
            unmerged = set(datasets.keys()) - referenced
            if unmerged:
                raise ValueError(
                    f"Data sources {sorted(unmerged)} are loaded but not included in ordered merges"
                )

            print(f"\nMerging data sources with declared merge plan (base={base_name})...")
            merged = merge_datasets(
                base=datasets[base_name],
                datasets=datasets,
                merge_config=spec.merges,
            )
            print(f"  Merged shape: {dict(merged.dims)}")
            print(f"  Merged variables: {list(merged.data_vars)}")
            return merged

        if len(datasets) == 1:
            # Single dataset, return as-is
            return next(iter(datasets.values()))
        
        # Multiple datasets - merge them
        # For now, use xarray's merge which aligns on coordinates
        print("\nMerging multiple data sources...")
        merged = xr.merge(list(datasets.values()))
        print(f"  Merged shape: {dict(merged.dims)}")
        print(f"  Merged variables: {list(merged.data_vars)}")
        
        return merged
    
    def _execute_features_stage(
        self,
        spec: PipelineSpec,
        data: xr.Dataset,
        data_cache_key: str
    ) -> Tuple[xr.Dataset, str]:
        """
        Execute feature engineering stage with per-operation caching.
        
        Executes all feature operations defined in the spec in dependency order,
        caching each operation independently to enable reuse across specs.
        
        Args:
            spec: Pipeline specification
            data: Input dataset (merged from data sources)
            data_cache_key: Cache key of the input data
            
        Returns:
            Tuple of (features_dataset, features_cache_key)
        """
        if not spec.features or not spec.features.operations:
            return data, data_cache_key
        
        # Get operation execution order (topological sort)
        operation_order = spec.features.get_operation_order()
        print(f"Operation execution order: {operation_order}")
        
        # Execute each operation with caching
        current_data = data
        operation_cache_keys = []
        
        # Clear operation cache keys for this execution
        self._operation_cache_keys = {}
        
        for op_name in operation_order:
            operation = self._get_operation(spec.features, op_name)
            
            # Execute operation with caching
            current_data, op_key, op_result = self._execute_operation(
                operation,
                current_data,
                data_cache_key
            )
            
            # Store cache key for this operation
            self._operation_cache_keys[op_name] = op_key
            operation_cache_keys.append(op_key)
        
        # Compute overall features cache key from operation keys
        import hashlib
        import json
        features_key = hashlib.sha256(
            json.dumps(sorted(operation_cache_keys)).encode()
        ).hexdigest()[:16]
        
        return current_data, features_key
    
    def _get_operation(self, features_spec, op_name: str) -> FormulaOperationSpec:
        """Get operation by name from features spec."""
        for op in features_spec.operations:
            if op.name == op_name:
                return op
        raise ValueError(f"Operation '{op_name}' not found in features spec")
    
    def _execute_operation(
        self,
        operation: FormulaOperationSpec,
        data: xr.Dataset,
        data_cache_key: str
    ) -> Tuple[xr.Dataset, str, OperationResult]:
        """
        Execute a single formula operation with caching.
        
        Args:
            operation: Formula operation specification
            data: Input dataset
            data_cache_key: Cache key of the input data
            
        Returns:
            Tuple of (updated_dataset, cache_key, operation_result)
        """
        print(f"\n  Operation: {operation.name}")
        
        # Compute parent keys (data + dependencies)
        parent_keys = [data_cache_key]
        for dep in operation.depends_on:
            if dep in self._operation_cache_keys:
                parent_keys.append(self._operation_cache_keys[dep])
            else:
                raise ValueError(
                    f"Dependency '{dep}' not found for operation '{operation.name}'. "
                    f"Available: {list(self._operation_cache_keys.keys())}"
                )
        
        # Build config for this operation
        # operation.formulas is already a dict: Dict[str, List[Dict[str, Any]]]
        config = {
            'operation_name': operation.name,
            'formulas': operation.formulas,
            'depends_on': operation.depends_on
        }
        
        # Track timing
        op_start_time = time.time()
        
        # Define compute function
        def compute_fn():
            return self._compute_formulas(operation.formulas, data)
        
        # Use cache manager
        result_ds, cache_key = self.cache_manager.get_or_compute(
            stage=CacheStage.L3_FEATURES,
            config=config,
            parent_keys=parent_keys,
            compute_fn=compute_fn
        )
        
        op_execution_time = time.time() - op_start_time
        
        # Determine if this was a cache hit
        # (cache manager prints "Cache hit" or "Cache miss")
        # For now, we'll check if execution was very fast (< 0.1s suggests cache hit)
        cache_hit = op_execution_time < 0.1
        
        # Merge results into dataset
        merged_data = data.assign(**{var: result_ds[var] for var in result_ds.data_vars})
        
        # Create operation result
        formulas_computed = list(result_ds.data_vars.keys())
        op_result = OperationResult(
            name=operation.name,
            formulas_computed=formulas_computed,
            cache_key=cache_key,
            cache_hit=cache_hit,
            execution_time=op_execution_time
        )
        
        print(f"    Formulas computed: {formulas_computed}")
        print(f"    Cache key: {cache_key}")
        print(f"    Time: {op_execution_time:.2f}s")
        
        return merged_data, cache_key, op_result
    
    def _compute_formulas(
        self,
        formulas: Dict[str, List[Dict[str, Any]]],
        data: xr.Dataset
    ) -> xr.Dataset:
        """
        Compute formulas using FormulaManager.
        
        Args:
            formulas: Dictionary mapping formula names to config lists
            data: Input dataset
            
        Returns:
            Dataset with computed formula results
        """
        # Prepare data for JIT
        data_jit, recover = prepare_for_jit(data)
        
        # Build context
        context = {
            '_dataset': data_jit,
            'price': 'close',  # Default mapping
            **get_function_context()
        }
        
        # Evaluate formulas
        result = self.formula_manager.evaluate_bulk(
            formulas,
            context,
            validate_inputs=True,
            jit_compile=True
        )
        
        # Restore from JIT
        return restore_from_jit(result, recover)
    
    def _execute_preprocessing_stage(
        self,
        spec: PipelineSpec,
        features_data: xr.Dataset,
        features_cache_key: str
    ) -> Tuple[xr.Dataset, str, Any]:
        """
        Execute preprocessing stage with L4 caching using DataHandler.
        
        Creates a DataHandler from the preprocessing spec and uses it to apply
        shared/learn/infer processors. This ensures consistency with the existing
        DataHandler infrastructure used throughout the codebase.
        
        Args:
            spec: Pipeline specification
            features_data: Input dataset from features stage
            features_cache_key: Cache key of the input features
            
        Returns:
            Tuple of (preprocessed_dataset, cache_key, data_handler)
        """
        if not spec.preprocessing:
            return features_data, features_cache_key, None
        
        # Build config for caching
        config = {
            'mode': spec.preprocessing.mode,
            'shared': spec.preprocessing.shared,
            'learn': spec.preprocessing.learn,
            'infer': spec.preprocessing.infer,
        }
        
        # Define compute function
        def compute_fn():
            return self._compute_preprocessing(spec.preprocessing, features_data)
        
        # Use cache manager with state support
        # Returns (dataset, handler) where handler contains learned states
        preprocessed_data, handler, cache_key = self.cache_manager.get_or_compute_with_state(
            stage=CacheStage.L4_PREPROCESSED,
            config=config,
            parent_keys=[features_cache_key],
            compute_fn=compute_fn
        )
        
        print(f"\nPreprocessing completed")
        print(f"  Mode: {spec.preprocessing.mode}")
        print(f"  Shared processors: {len(spec.preprocessing.shared)}")
        print(f"  Learn processors: {len(spec.preprocessing.learn)}")
        print(f"  Infer processors: {len(spec.preprocessing.infer)}")
        if handler and getattr(handler, "learn_states", None):
            state_types = [type(st).__name__ for st in handler.learn_states]
            print(f"  Learned states: {state_types}")
        
        return preprocessed_data, cache_key, handler
    
    def _compute_preprocessing(
        self,
        preprocessing_spec,
        data: xr.Dataset
    ) -> Tuple[xr.Dataset, Any]:
        """
        Compute preprocessing transformations using DataHandler.
        
        Creates a DataHandler from the preprocessing spec and uses it to apply
        processors. This ensures we use the same infrastructure as the rest of
        the codebase (example.py, ModelRunner, etc.).
        
        Args:
            preprocessing_spec: Preprocessing specification
            data: Input dataset
            
        Returns:
            Tuple of (preprocessed_dataset, data_handler)
        """
        from src.pipeline.spec.processor_registry import ProcessorRegistry
        from src.pipeline.data_handler import DataHandler, HandlerConfig, PipelineMode
        
        # Instantiate processors from spec
        shared_processors = [ProcessorRegistry.create_processor(cfg) for cfg in preprocessing_spec.shared]
        learn_processors = [ProcessorRegistry.create_processor(cfg) for cfg in preprocessing_spec.learn]
        infer_processors = [ProcessorRegistry.create_processor(cfg) for cfg in preprocessing_spec.infer]
        
        # Map mode string to PipelineMode enum
        mode_map = {
            'independent': PipelineMode.INDEPENDENT,
            'append': PipelineMode.APPEND,
        }
        mode = mode_map.get(preprocessing_spec.mode.lower(), PipelineMode.INDEPENDENT)
        
        # Create DataHandler config
        handler_config = HandlerConfig(
            shared=shared_processors,
            learn=learn_processors,
            infer=infer_processors,
            mode=mode,
            feature_cols=[],  # Not used in preprocessing-only mode
            label_cols=[]     # Not used in preprocessing-only mode
        )
        
        # Create and build handler
        print(f"  Building DataHandler with {len(shared_processors)} shared, "
              f"{len(learn_processors)} learn, {len(infer_processors)} infer processors")
        handler = DataHandler(base=data, config=handler_config)
        handler.build()
        
        # Get the learn view (applies shared + learn processors)
        # This is what we want for model training - data with all preprocessing applied
        from src.pipeline.data_handler import View
        preprocessed_data = handler.view(View.LEARN)
        
        print(f"  Preprocessing complete")
        print(f"  Output variables: {list(preprocessed_data.data_vars)[:10]}...")  # Show first 10
        
        return preprocessed_data, handler
    
    def _execute_model_stage(
        self,
        spec: PipelineSpec,
        input_data: xr.Dataset,
        input_cache_key: str,
        preprocessing_handler: Any = None
    ) -> Tuple[xr.Dataset, str, Any]:
        """
        Execute model training/prediction stage with L5 caching.
        
        Trains a model on the input data using ModelRunner walk-forward execution.
        Caches both the predictions dataset and the ModelRunnerResult artifact.
        
        Args:
            spec: Pipeline specification
            input_data: Input dataset (preprocessed or features)
            input_cache_key: Cache key of the input data
            
        Returns:
            Tuple of (predictions_dataset, cache_key, fitted_model)
        """
        if not spec.model:
            return input_data, input_cache_key, None
        
        # Build config for caching
        config = {
            'adapter': spec.model.adapter,
            'model_type': spec.model.type,
            'params': spec.model.params,
            'features': spec.model.features,
            'target': spec.model.target,
            'walk_forward': spec.model.walk_forward,
            'adapter_params': spec.model.adapter_params,
            'runner_params': spec.model.runner_params,
        }
        
        # Define compute function
        def compute_fn():
            return self._compute_model(spec.model, input_data, preprocessing_handler)
        
        # Use cache manager with state support (ModelRunnerResult is the state)
        predictions, runner_result, cache_key = self.cache_manager.get_or_compute_with_state(
            stage=CacheStage.L5_MODEL,
            config=config,
            parent_keys=[input_cache_key],
            compute_fn=compute_fn
        )
        
        print(f"\nModel training/prediction completed")
        print(f"  Model type: {spec.model.adapter}/{spec.model.type}")
        print(f"  Features: {len(spec.model.features)}")
        print(f"  Target: {spec.model.target}")
        if spec.model.walk_forward:
            wf = spec.model.walk_forward
            print(f"  Walk-forward: train={wf.get('train_span_hours', 'N/A')}h, "
                  f"infer={wf.get('infer_span_hours', 'N/A')}h, "
                  f"step={wf.get('step_hours', 'N/A')}h")
        
        return predictions, cache_key, runner_result
    
    def _compute_model(
        self,
        model_spec,
        data: xr.Dataset,
        preprocessing_handler: Any = None
    ) -> Tuple[xr.Dataset, Any]:
        """
        Train model and generate predictions using ModelRunner.
        
        This properly integrates with the existing ModelRunner infrastructure for
        walk-forward validation, following the pattern in example.py.
        
        Args:
            model_spec: Model specification
            data: Input dataset (preprocessed)
            preprocessing_handler: DataHandler from preprocessing (used if provided)
            
        Returns:
            Tuple of (predictions_dataset, ModelRunnerResult)
        """
        import importlib
        import numpy as np
        from src.pipeline.data_handler import DataHandler, HandlerConfig, PipelineMode
        from src.pipeline.walk_forward import SegmentConfig, make_plan
        from src.pipeline.model.runner import ModelRunner
        
        print(f"  Training {model_spec.adapter}/{model_spec.type} model...")
        
        # Validate inputs
        feature_vars = model_spec.features
        target_var = model_spec.target
        
        if not feature_vars:
            raise ValueError("Model specification must include 'features' list")
        if not target_var:
            raise ValueError("Model specification must include 'target' variable")
        
        # Check that all variables exist
        for var in feature_vars:
            if var not in data.data_vars:
                raise ValueError(f"Feature variable '{var}' not found in dataset. Available: {list(data.data_vars)}")
        if target_var not in data.data_vars:
            raise ValueError(f"Target variable '{target_var}' not found in dataset. Available: {list(data.data_vars)}")
        
        # Create or reuse DataHandler
        if preprocessing_handler is not None:
            # Use the preprocessing handler (already has learned states)
            handler = preprocessing_handler
            print(f"    Using preprocessing handler with {len(handler.config.learn)} learn processors")
        else:
            # Create a minimal handler for the model
            handler_config = HandlerConfig(
                shared=[],
                learn=[],
                infer=[],
                mode=PipelineMode.INDEPENDENT,
                feature_cols=feature_vars,
                label_cols=[target_var]
            )
            handler = DataHandler(base=data, config=handler_config)
            handler.build()
            print(f"    Created minimal handler for model")
        
        # Create walk-forward plan
        wf_config = model_spec.walk_forward
        if wf_config:
            # Extract walk-forward parameters
            train_span_hours = wf_config.get('train_span_hours', 24 * 5)  # Default: 5 days
            infer_span_hours = wf_config.get('infer_span_hours', 24 * 1)  # Default: 1 day
            step_hours = wf_config.get('step_hours', 24 * 1)  # Default: 1 day
            gap_hours = wf_config.get('gap_hours', 0)  # Default: no gap
            
            # Get time bounds from data or config
            start = wf_config.get('start')
            end = wf_config.get('end')
            
            if start is None or end is None:
                # Extract from data
                time_coord = data.coords.get('time')
                if time_coord is not None:
                    time_vals = time_coord.values
                    if start is None:
                        start = np.datetime64(time_vals.min(), 'ns')
                    if end is None:
                        end = np.datetime64(time_vals.max(), 'ns')
                else:
                    raise ValueError("Cannot determine time bounds: no 'time' coordinate in data and no start/end in config")
            else:
                start = np.datetime64(start)
                end = np.datetime64(end)
            
            # Create segment config
            segment_config = SegmentConfig(
                start=start,
                end=end,
                train_span=np.timedelta64(train_span_hours, 'h'),
                infer_span=np.timedelta64(infer_span_hours, 'h'),
                step=np.timedelta64(step_hours, 'h'),
                gap=np.timedelta64(gap_hours, 'h'),
                clip_to_data=True
            )
            
            # Generate plan
            plan = make_plan(segment_config, ds_for_bounds=data)
            print(f"    Created walk-forward plan with {len(plan)} segments")
            print(f"    Train: {train_span_hours}h, Infer: {infer_span_hours}h, Step: {step_hours}h")
        else:
            raise ValueError("Model specification must include 'walk_forward' configuration")
        
        # Create model instance based on adapter type
        adapter_type = model_spec.adapter.lower()
        model_type = model_spec.type
        
        if adapter_type == 'sklearn':
            # Try common sklearn modules
            sklearn_modules = [
                'sklearn.linear_model',
                'sklearn.ensemble',
                'sklearn.tree',
                'sklearn.svm',
                'sklearn.neighbors',
            ]
            
            model_class = None
            for module_name in sklearn_modules:
                try:
                    module = importlib.import_module(module_name)
                    if hasattr(module, model_type):
                        model_class = getattr(module, model_type)
                        break
                except ImportError:
                    continue
            
            if model_class is None:
                raise ValueError(
                    f"Could not find sklearn model '{model_type}'. "
                    f"Tried modules: {sklearn_modules}"
                )
            
            print(f"    Created {model_type} with params: {model_spec.params}")
        else:
            # Future: support for other adapters (lightgbm, xgboost, pytorch, etc.)
            raise ValueError(
                f"Adapter type '{adapter_type}' not yet implemented. "
                f"Currently supported: sklearn. "
                f"Future: lightgbm, xgboost, pytorch, etc."
            )
        
        # Get adapter params (output_var, use_proba, etc.)
        adapter_params = model_spec.adapter_params.copy()
        if 'output_var' not in adapter_params:
            adapter_params['output_var'] = f'{target_var}_pred'
        output_var = adapter_params['output_var']
        
        # Create model factory (fresh model for each segment)
        def make_adapter():
            """Create a fresh model adapter for each segment."""
            model = model_class(**model_spec.params)
            if adapter_type == 'sklearn':
                from src.pipeline.model.adapter import SklearnAdapter
                return SklearnAdapter(
                    model=model,
                    handler=handler,
                    **adapter_params
                )
            else:
                raise ValueError(f"Adapter type '{adapter_type}' not implemented")
        
        # Get runner params
        runner_params = model_spec.runner_params.copy()
        overlap_policy = runner_params.get('overlap_policy', 'last')
        
        # Create and run ModelRunner
        print(f"    Running ModelRunner with {len(plan)} segments...")
        runner = ModelRunner(
            handler=handler,
            plan=plan,
            model_factory=make_adapter,
            feature_cols=feature_vars,
            label_col=target_var,
            overlap_policy=overlap_policy,
            output_var=output_var,
            return_model_states=True
        )
         
        result = runner.run()
        
        print(f"    Model trained successfully across {len(plan)} segments")
        print(f"    Predictions added as '{output_var}'")
        
        return result.pred_ds, result
