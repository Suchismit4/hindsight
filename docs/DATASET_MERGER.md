# Dataset Merger

## What This Page Covers

This page explains how Hindsight merges multiple `xarray` datasets when they do
not share the same frequency, asset coverage, or publication timing. It covers
the merge configuration surface, the alignment modes implemented in
`src/pipeline/data_handler/merge.py`, and the point-in-time behavior that makes
these merges usable in research workflows.

## When To Read It

Read this page when you are combining multiple sources in a pipeline, especially
when one source is lower frequency than another or when publication lag matters.
If you are working on CRSP plus Compustat, quarterly fundamentals, earnings
surprises, or any similar workflow, this page is the right entry point.

## Core Ideas

### Why `xr.merge()` is not enough

The merger exists because Hindsight datasets are not plain two-dimensional
tables. They typically carry explicit calendar dimensions:

```text
(year, month, day, hour, asset)
```

That creates real merge problems:

- lower-frequency data must be expanded onto a richer time grid
- asset universes differ between sources
- variables need namespacing to avoid collisions
- point-in-time correctness matters when one dataset is only available after a reporting lag

### The configuration surface

`MergeSpec` is the unit of merge configuration. The important fields are:

| Field | Purpose |
| --- | --- |
| `right_name` | Which named dataset to merge in |
| `on` | Join dimension or dimensions, usually `asset` |
| `time_alignment` | Alignment mode such as `ffill` or `as_of` |
| `time_offset_months` | Shift availability forward or backward in calendar months |
| `ffill_limit` | Optional forward-fill cap |
| `prefix` / `suffix` | Rename incoming variables to avoid collisions |
| `variables` / `drop_vars` | Include or exclude selected variables |

### Alignment modes

The merger currently exposes five alignment modes:

| Mode | Meaning |
| --- | --- |
| `exact` | Require exact alignment on the existing time grid |
| `ffill` | Carry lower-frequency values forward onto later periods |
| `bfill` | Backfill from later values |
| `nearest` | Use the nearest available observation |
| `as_of` | Point-in-time style merge that uses the latest available value before the target period |

In real research workflows, `ffill` and `as_of` are the important modes. `as_of`
is the one to reach for when the data should not appear before it was
economically available.

### What the merger actually does

A merge runs in three broad steps:

1. normalize and optionally rename the right-hand dataset
2. expand it onto the left dataset's calendar
3. align on the join dimensions and merge

In practice, that means the right dataset may be:

- reindexed to the left dataset's `year`
- broadcast across `month`, `day`, or `hour`
- shifted by `time_offset_months`
- reindexed to the left dataset's asset coordinate

### Point-in-time behavior

The most important nontrivial field is `time_offset_months`.

For annual fundamentals merged into monthly returns, a positive offset means the
current year's data should only become visible after that lag has elapsed. That
is what keeps a factor workflow from using information before it would have been
known.

This is why the FF3 example uses:

```yaml
merges:
  - right_name: "compustat"
    on: "asset"
    time_alignment: "as_of"
    time_offset_months: 6
```

The example is not special, but it is a good demonstration of the intended
research discipline: define data availability in the merge layer, not as a
hand-waved assumption downstream.

## Practical Examples

### Merge annual fundamentals into monthly returns

```python
from src.pipeline.data_handler import DatasetMerger, MergeSpec, TimeAlignment

merger = DatasetMerger()

spec = MergeSpec(
    right_name="compustat",
    on="asset",
    time_alignment=TimeAlignment.AS_OF,
    time_offset_months=6,
    prefix="comp_",
    variables=["seq", "txditc", "at"],
)

merged = merger.merge(crsp_monthly, compustat_annual, spec)
```

### Use declarative merge config in a pipeline spec

```yaml
merge_base: "crsp"
merges:
  - right_name: "compustat"
    on: "asset"
    time_alignment: "as_of"
    time_offset_months: 6
    variables: ["seq", "txditc", "ps"]
```

That is the same pattern used by [`examples/ff3_model.yaml`](../examples/ff3_model.yaml).

### Merge multiple sources in order

```python
specs = [
    MergeSpec(
        right_name="compustat",
        on="asset",
        time_alignment=TimeAlignment.AS_OF,
        time_offset_months=6,
        prefix="comp_",
    ),
    MergeSpec(
        right_name="ibes",
        on="asset",
        time_alignment=TimeAlignment.FFILL,
        prefix="ibes_",
    ),
]

merged = merger.merge_multiple(base=crsp, datasets=datasets, specs=specs)
```

## Common Pitfalls

- Treating `as_of` and `ffill` as interchangeable. They can look similar in a toy example, but `as_of` is the safer signal of intent when publication lag matters.
- Forgetting to namespace incoming variables. A merge that silently collides with existing variable names is hard to reason about later.
- Assuming identical asset dtypes across sources. The merger normalizes join-coordinate dtypes because real datasets often do not arrive in the same type.
- Thinking the merge layer is only about convenience. In this codebase it is part of the research contract: publication lag and availability belong here.
- Building the whole workflow around FF3 terminology. FF3 is just one example of a broader multi-source, point-in-time merge pattern.

## Read Next

- [PIPELINE_SYSTEM.md](./PIPELINE_SYSTEM.md) for how merges fit into a full pipeline spec
- [ARCHITECTURE.md](./ARCHITECTURE.md) for the wider system model and stage boundaries
- [INDEX.md](./INDEX.md) for the documentation map

