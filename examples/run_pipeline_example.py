"""
Pipeline Execution Example

This script demonstrates the complete pipeline system with:
1. Data loading (L2 cache)
2. Feature engineering (L3 cache)
3. Preprocessing (L4 cache)
4. Model training/prediction (L5 cache)

It shows cache reuse across different pipeline specifications.
"""

import shutil
from pathlib import Path
import time

from src.pipeline.cache import GlobalCacheManager
from src.pipeline.spec import SpecParser, PipelineExecutor
from src.data.managers.data_manager import DataManager


def setup_cache(cache_root: str = "~/data/hindsight_example_cache"):
    """Setup cache directory."""
    cache_path = Path(cache_root).expanduser()
    if cache_path.exists():
        print(f"Cleaning existing cache: {cache_path}")
        shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")


def print_result_summary(result, execution_time: float):
    """Print a summary of execution results."""
    print(f"\nExecution Summary:")
    print(f"  Total time: {execution_time:.2f}s")
    print(f"  Pipeline: {result.spec.name} v{result.spec.version}")
    
    print(f"\n  Data sources: {list(result.data.keys())}")
    
    if result.features_data:
        print(f"  Features computed: {len(result.features_data.data_vars)} variables")
        print(f"    Sample features: {list(result.features_data.data_vars)[:5]}")
    
    if result.preprocessed_data:
        print(f"  Preprocessed variables: {len(result.preprocessed_data.data_vars)}")
        if result.learned_states and hasattr(result.learned_states, 'learn_states'):
            print(f"    Learned states: {len(result.learned_states.learn_states)} processor(s)")
        elif result.learned_states:
            print(f"    DataHandler created with preprocessing")
    
    if result.model_predictions:
        print(f"  Model type: {result.spec.model.type}")
        print(f"  Features used: {len(result.spec.model.features)}")
        print(f"  Predictions: {result.spec.model.target}_pred")
    
    print(f"\n  Cache keys:")
    for stage, key in result.cache_keys.items():
        print(f"    {stage}: {key[:16]}...")


def main():
    """Run the pipeline example."""
    
    print_section("Pipeline System Example")
    print("This example demonstrates:")
    print("  1. End-to-end pipeline execution (data → features → preprocessing → model)")
    print("  2. Hierarchical caching (L2 → L3 → L4 → L5)")
    print("  3. Cache reuse across different pipeline specifications")
    
    # Setup
    cache_root = setup_cache()
    cache_manager = GlobalCacheManager(cache_root=str(cache_root))
    data_manager = DataManager()
    executor = PipelineExecutor(cache_manager=cache_manager, data_manager=data_manager)
    
    # Spec paths
    baseline_spec_path = Path(__file__).parent / "pipeline_specs" / "crypto_momentum_baseline.yaml"
    enhanced_spec_path = Path(__file__).parent / "pipeline_specs" / "crypto_momentum_enhanced.yaml"
    
    # =========================================================================
    # Part 1: Execute Baseline Pipeline
    # =========================================================================
    print_section("Part 1: Execute Baseline Pipeline")
    print(f"Loading spec: {baseline_spec_path.name}")
    
    baseline_spec = SpecParser.load_from_yaml(str(baseline_spec_path))
    print(f"\nPipeline: {baseline_spec.name} v{baseline_spec.version}")
    print(f"  Data: {list(baseline_spec.data.keys())}")
    print(f"  Features: {len(baseline_spec.features.operations)} operation(s)")
    print(f"  Preprocessing: {len(baseline_spec.preprocessing.learn)} learn processor(s)")
    print(f"  Model: {baseline_spec.model.type}")
    
    print("\nExecuting baseline pipeline (all cache misses expected)...")
    start_time = time.time()
    baseline_result = executor.execute(baseline_spec)
    baseline_time = time.time() - start_time
    
    print_result_summary(baseline_result, baseline_time)
    
    # =========================================================================
    # Part 2: Re-execute Baseline Pipeline (Cache Hits)
    # =========================================================================
    print_section("Part 2: Re-execute Baseline Pipeline")
    print("Executing same pipeline again (all cache hits expected)...")
    
    start_time = time.time()
    baseline_result2 = executor.execute(baseline_spec)
    baseline_time2 = time.time() - start_time
    
    print_result_summary(baseline_result2, baseline_time2)
    
    print(f"\n  Performance comparison:")
    print(f"    First execution:  {baseline_time:.2f}s")
    print(f"    Second execution: {baseline_time2:.2f}s")
    print(f"    Speedup: {baseline_time/baseline_time2:.1f}x")
    
    # Verify cache hits
    print(f"\n  Cache verification:")
    for stage in ['data', 'features', 'preprocessing', 'model']:
        match = baseline_result.cache_keys[stage] == baseline_result2.cache_keys[stage]
        status = "✓ MATCH" if match else "✗ MISMATCH"
        print(f"    {stage}: {status}")
    
    # =========================================================================
    # Part 3: Execute Enhanced Pipeline (Partial Cache Reuse)
    # =========================================================================
    print_section("Part 3: Execute Enhanced Pipeline")
    print(f"Loading spec: {enhanced_spec_path.name}")
    
    enhanced_spec = SpecParser.load_from_yaml(str(enhanced_spec_path))
    print(f"\nPipeline: {enhanced_spec.name} v{enhanced_spec.version}")
    print(f"  Data: {list(enhanced_spec.data.keys())} (SAME as baseline)")
    print(f"  Features: {len(enhanced_spec.features.operations)} operation(s) (EXTENDED)")
    print(f"  Preprocessing: {len(enhanced_spec.preprocessing.learn)} learn processor(s) (EXTENDED)")
    print(f"  Model: {enhanced_spec.model.type} (DIFFERENT)")
    
    print("\nExecuting enhanced pipeline...")
    print("  Expected cache behavior:")
    print("    - Data (L2): HIT (same data source)")
    print("    - Features (L3): PARTIAL (reuse baseline features, compute new ones)")
    print("    - Preprocessing (L4): MISS (different feature set)")
    print("    - Model (L5): MISS (different model type)")
    
    start_time = time.time()
    enhanced_result = executor.execute(enhanced_spec)
    enhanced_time = time.time() - start_time
    
    print_result_summary(enhanced_result, enhanced_time)
    
    # =========================================================================
    # Part 4: Cache Analysis
    # =========================================================================
    print_section("Part 4: Cache Analysis")
    
    # Compare cache keys
    print("Cache key comparison (baseline vs enhanced):")
    print(f"  Data (L2):")
    print(f"    Baseline: {baseline_result.cache_keys['data'][:16]}...")
    print(f"    Enhanced: {enhanced_result.cache_keys['data'][:16]}...")
    print(f"    Match: {'✓ YES (cache reused)' if baseline_result.cache_keys['data'] == enhanced_result.cache_keys['data'] else '✗ NO'}")
    
    print(f"\n  Features (L3):")
    print(f"    Baseline: {baseline_result.cache_keys['features'][:16]}...")
    print(f"    Enhanced: {enhanced_result.cache_keys['features'][:16]}...")
    print(f"    Match: {'✓ YES' if baseline_result.cache_keys['features'] == enhanced_result.cache_keys['features'] else '✗ NO (different features)'}")
    
    print(f"\n  Preprocessing (L4):")
    print(f"    Baseline: {baseline_result.cache_keys['preprocessing'][:16]}...")
    print(f"    Enhanced: {enhanced_result.cache_keys['preprocessing'][:16]}...")
    print(f"    Match: {'✓ YES' if baseline_result.cache_keys['preprocessing'] == enhanced_result.cache_keys['preprocessing'] else '✗ NO (different inputs)'}")
    
    print(f"\n  Model (L5):")
    print(f"    Baseline: {baseline_result.cache_keys['model'][:16]}...")
    print(f"    Enhanced: {enhanced_result.cache_keys['model'][:16]}...")
    print(f"    Match: {'✓ YES' if baseline_result.cache_keys['model'] == enhanced_result.cache_keys['model'] else '✗ NO (different model)'}")
    
    # Cache statistics
    print("\nCache statistics:")
    stats = cache_manager.get_stats()
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")
    
    if 'by_stage' in stats:
        print(f"\n  By stage:")
        for stage, stage_stats in stats['by_stage'].items():
            if stage_stats['count'] > 0:
                print(f"    {stage}: {stage_stats['count']} entries, {stage_stats['size_mb']:.2f} MB")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_section("Summary")
    print("Key Takeaways:")
    print("  1. First execution computes all stages (data → features → preprocessing → model)")
    print("  2. Second execution of same spec hits all caches (6-18x speedup)")
    print("  3. Different specs reuse shared stages automatically")
    print("  4. Cache keys are content-addressable (based on config + dependencies)")
    print("  5. Learned states (preprocessing) and fitted models are cached")
    
    print(f"\nExecution times:")
    print(f"  Baseline (first):  {baseline_time:.2f}s")
    print(f"  Baseline (second): {baseline_time2:.2f}s")
    print(f"  Enhanced (first):  {enhanced_time:.2f}s")
    
    print(f"\nCache directory: {cache_root}")
    print("  (Delete this directory to clear cache)")


if __name__ == "__main__":
    main()

