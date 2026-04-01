# Hindsight

Hindsight is a YAML-first quantitative research library built around `xarray`
datasets and JAX-backed numerical workflows. It is designed for researchers and
engineers who want a consistent way to move from raw market data to features,
preprocessing, walk-forward model execution, and portfolio-style analysis
without rewriting the same pipeline glue for each project.

The core system model is:

```text
provider -> loader -> xr.Dataset -> formulas/processors -> walk-forward/model
```

Most datasets are normalized into a panel layout over:

```text
(year, month, day, hour, asset)
```

That shape contract is what lets the same library support multi-frequency data
loading, feature computation, cross-sectional sorting, factor construction, and
walk-forward evaluation.

## Who This Is For

- Engineers onboarding to the repository and trying to understand how the pieces fit.
- Quant researchers authoring YAML pipelines for data, features, preprocessing, and models.
- Contributors extending loaders, formulas, processors, or pipeline execution.

## Start Here

- [Documentation index](docs/INDEX.md)
- [System architecture](docs/ARCHITECTURE.md)
- [Pipeline and YAML guide](docs/PIPELINE_SYSTEM.md)
- [Dataset merger reference](docs/DATASET_MERGER.md)

## Examples

**Tracked (minimal):** see [`examples/README.md`](examples/README.md).

- [`examples/run_minimal_example.py`](examples/run_minimal_example.py) — runs [`examples/pipeline_specs/crypto_momentum_baseline.yaml`](examples/pipeline_specs/crypto_momentum_baseline.yaml) twice to demonstrate caching.

**Local `dev/examples/`** (gitignored) can hold larger demos: multi-spec cache walkthrough (`run_pipeline_example.py`), `ff3_model.yaml` on WRDS, and other scratch.

## What The Repo Emphasizes

- Declarative YAML pipeline specs instead of workflow-specific helper APIs.
- Reusable data and feature pipelines with content-addressable caching.
- A single data model that works for research features, portfolio construction, and walk-forward modeling.

## Notes For Automation

`AGENTS.md` is the thin router for coding agents. Optional local workspaces such
as `.agent-docs/` (ignored by git) can hold compact agent-facing notes; human
documentation is under `docs/` and this README.
