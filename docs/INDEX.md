# Documentation Index

This documentation set is the human-readable guide to the tracked Hindsight
codebase. It focuses on the system model, YAML pipeline authoring, and the
implementation boundaries that matter when you are adopting or extending the
library.

## Start Here

- Read [README.md](../README.md) for the project-level overview.
- Read [ARCHITECTURE.md](./ARCHITECTURE.md) for the layer map, data flow, and core invariants.
- Read [PIPELINE_SYSTEM.md](./PIPELINE_SYSTEM.md) before authoring or debugging a YAML pipeline.

## Core Guides

| Page | Use it for |
| --- | --- |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | System model, layer boundaries, dimension contract, and how the main subsystems fit together |
| [PIPELINE_SYSTEM.md](./PIPELINE_SYSTEM.md) | YAML structure, pipeline stages, supported processors, model execution, and common authoring pitfalls |
| [DATASET_MERGER.md](./DATASET_MERGER.md) | Multi-source joins, time alignment, and point-in-time merge behavior |

## Example Entry Points

**Tracked (clone-friendly):**

- [`examples/run_minimal_example.py`](../examples/run_minimal_example.py) — minimal runner; see [`examples/README.md`](../examples/README.md).
- [`examples/pipeline_specs/crypto_momentum_baseline.yaml`](../examples/pipeline_specs/crypto_momentum_baseline.yaml) — smallest end-to-end modeling spec.

**Local `dev/examples/`** (gitignored): multi-spec cache demo (`run_pipeline_example.py`), `ff3_model.yaml`, and other extended samples.

## How To Read The Docs

- Start with architecture if the repository is new to you.
- Jump to the pipeline guide if you already understand the data model and want to work in YAML.
- Use the dataset merger reference when you are combining sources with different frequencies or publication lags.

## Notes For Agents

Human-facing docs live here and in [README.md](../README.md). [AGENTS.md](../AGENTS.md)
routes automation; optional local `.agent-docs/` (gitignored) may exist beside
this tree for agent workflows.
