# Testing

Commands, coverage requirements, test templates, and anti-patterns for the Hindsight test suite.

---

## Commands

```bash
# Run full test suite
~/conda/envs/jax/bin/pytest tests/ -v

# Run a single test file
~/conda/envs/jax/bin/pytest tests/test_pipeline.py -v

# Run a single test by name
~/conda/envs/jax/bin/pytest tests/test_pipeline.py::test_function_name -v

# Lint — errors only (CI-blocking)
~/conda/envs/jax/bin/flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Lint — warnings (non-blocking)
~/conda/envs/jax/bin/flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Import check — verifies module structure is intact
~/conda/envs/jax/bin/python -c "import src"
```

**Baseline:** 109 tests passed, 4 skipped, 0 failed.

**Pre-existing lint issues (do not fix, do not block on):**
- `src/backtester/core.py:436,457` — F821: undefined `execution_timestamp`
- `src/data/ast/functions.py:186` — F824: unused `nonlocal`

---

## Required Coverage by Change Type

| Change type | Required tests |
|---|---|
| New `@register_function` | Basic case, NaN case, all-NaN slice, correct dim output |
| New processor (`ProcessorContract`) | Single-asset, multi-asset, shape preservation, stateful fit/transform separation |
| New YAML formula definition | Dependency resolution order, evaluation result, variable substitution |
| Statistical model adapter | Fit convergence, transform shape, no future leakage |
| ML model adapter | Train/test split correctness, prediction shape, no leakage |
| Multi-frequency merge | Date alignment, as-of correctness, no future data |
| Cache change | Hit, miss, date-range subsetting |
| Portfolio stage | Bucket count, VW vs EW, spread direction (long minus short) |
| Rolling kernel | Known value (e.g., `mean([1,2,3], w=3) == 2.0`), NaN propagation |

---

## NaN Test Template

Every new cross-sectional or rolling function must include a NaN correctness test:

```python
import numpy as np
import xarray as xr

def test_my_function_all_nan():
    """All-NaN input must produce all-NaN output."""
    da = xr.DataArray(
        np.full((1, 1, 1, 1, 5), np.nan),
        dims=["year", "month", "day", "hour", "asset"],
    )
    result = my_function(da)
    assert result.isnull().all(), "All-NaN input must yield all-NaN output"


def test_my_function_partial_nan():
    """NaN assets must receive NaN output; valid assets must be computed."""
    data = np.array([[[[[1.0, 2.0, np.nan, 4.0, 5.0]]]]])
    da = xr.DataArray(data, dims=["year", "month", "day", "hour", "asset"])
    result = my_function(da)
    assert np.isnan(result.values[0, 0, 0, 0, 2]), "NaN input must yield NaN output"
    assert not np.isnan(result.values[0, 0, 0, 0, 0]), "Valid input must yield valid output"
```

---

## Leakage Test Pattern

For any operation that uses lagged or forward-looking data, assert no future information leaks:

```python
import numpy as np
import xarray as xr

def test_no_future_leakage():
    """Verify that learn processor does not see inference data."""
    # Create dataset with distinct train and infer values
    train_ds = make_dataset(values=1.0, time_steps=100)
    infer_ds = make_dataset(values=999.0, time_steps=20)  # sentinel value

    # Fit on train only
    state = processor.fit(train_ds)

    # Apply to infer
    result = processor.transform(infer_ds, state)

    # Assert no sentinel value leaked into fitted state
    assert 999.0 not in extract_state_values(state), \
        "Inference data must not appear in learned state"
```

---

## Walk-Forward Segment Test Pattern

Assert that a `learn` processor fitted on segment N is not re-fitted using inference data from segment N+1:

```python
def test_segment_isolation():
    """Each segment must produce independent learned state."""
    plan = make_plan(SegmentConfig(
        start=start, end=end,
        train_span=train_span, infer_span=infer_span,
        step=step,
    ))
    runner = WalkForwardRunner(handler=handler, plan=plan)
    result = runner.run()

    # Verify per-segment states are independent
    for i, seg_state in enumerate(result.segment_states):
        # State from segment i should only reflect training data from segment i
        assert seg_state is not None, f"Segment {i} must produce state"
        # Verify training bounds match expected segment boundaries
        assert seg_state.train_end <= plan.segments[i].train_end
```

---

## Synthetic Data Utilities

Always use synthetic data in unit tests. Never require external data connections.

```python
from src.data.loaders.table import Loader

# Generate a synthetic dataset
ds = Loader.load_simulated_data(
    num_assets=10,
    num_timesteps=100,
    num_vars=3,
    freq=FrequencyType.DAILY,
    start_date="2020-01-01",
)
```

Or build minimal xr.Datasets directly:

```python
import numpy as np
import xarray as xr

ds = xr.Dataset(
    {"close": xr.DataArray(
        np.random.randn(2, 1, 5, 1, 10),
        dims=["year", "month", "day", "hour", "asset"],
    )},
)
```

---

## What Never to Do in Tests

| ❌ Anti-Pattern | Why |
|----------------|-----|
| `assert result is not None` without checking the value | Passes even when the result is wrong |
| Test only the happy path | Misses NaN, empty, and edge-case failures |
| Require a live external connection (WRDS, API, etc.) | Tests must run offline in CI |
| Use `print()` inside test code | Clutters output; use `assert` messages instead |
| Pipe test output through `\| tail` or `\| head` | Masks the real exit code; a failing test reports success |
| Hardcode absolute file paths in tests | Breaks portability across environments |
| Skip the NaN test for cross-sectional functions | NaN handling is the most common source of silent bugs |

---

## See also

- [ANTI_BLOAT.md](./ANTI_BLOAT.md) — Pre-merge checklist that includes "does pytest still pass?"
- [KNOWN_BUGS.md](./KNOWN_BUGS.md) — u_roll NaN bug that affects rolling test expectations
- [WALK_FORWARD.md](./WALK_FORWARD.md) — Leakage contract details for walk-forward segment tests
- [PIPELINE_SYSTEM.md](./PIPELINE_SYSTEM.md) — Processor types and their required test coverage
