"""
Minimal pipeline example: one spec, two runs (second run exercises cache hits).

For a multi-spec cache walkthrough, use ``dev/examples/run_pipeline_example.py``
if you maintain a local ``dev/`` tree.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from src.data.managers.data_manager import DataManager
from src.pipeline.cache import GlobalCacheManager
from src.pipeline.spec import PipelineExecutor, SpecParser


def _setup_cache(cache_root: str = "~/data/hindsight_example_cache") -> Path:
    cache_path = Path(cache_root).expanduser()
    if cache_path.exists():
        print(f"Cleaning existing cache: {cache_path}")
        shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def main() -> None:
    spec_path = Path(__file__).parent / "pipeline_specs" / "crypto_momentum_baseline.yaml"
    cache_root = _setup_cache()
    cache_manager = GlobalCacheManager(cache_root=str(cache_root))
    data_manager = DataManager()
    executor = PipelineExecutor(cache_manager=cache_manager, data_manager=data_manager)

    spec = SpecParser.load_from_yaml(str(spec_path))
    print(f"Pipeline: {spec.name} v{spec.version}  (spec: {spec_path})")

    print("\nFirst run (cache misses expected)...")
    t0 = time.time()
    r1 = executor.execute(spec)
    first_s = time.time() - t0

    print("\nSecond run (cache hits expected)...")
    t0 = time.time()
    r2 = executor.execute(spec)
    second_s = time.time() - t0

    print(f"\n  First:  {first_s:.2f}s")
    print(f"  Second: {second_s:.2f}s")
    if second_s > 0:
        print(f"  Speedup: {first_s / second_s:.1f}x")
    for stage in ("data", "features", "preprocessing", "model"):
        match = r1.cache_keys[stage] == r2.cache_keys[stage]
        print(f"  {stage} keys match: {'yes' if match else 'no'}")
    print(f"\nCache directory: {cache_root}")


if __name__ == "__main__":
    main()
