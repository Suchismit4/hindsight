# Hindsight Codex Router

## Identity / Doctrine
Hindsight is a general quantitative research library.

Core doctrine:
- YAML-first pipeline semantics.
- xarray/JAX-native data flow.
- Minimal API surface.
- Preserve behavior unless a path/import fix is required.
- Optimize for generic quant primitives, not one workflow.
- [`examples/pipeline_specs/crypto_momentum_baseline.yaml`](examples/pipeline_specs/crypto_momentum_baseline.yaml) is the tracked minimal pipeline example; richer local demos may live under `dev/examples/`. Neither is a design constraint.

Canonical data flow:
`provider -> from_table() -> xr.Dataset(year, month, day, hour, asset) -> .dt -> FormulaManager/DataHandler -> walk-forward/model`

## Read-First Order
1. [`docs/INDEX.md`](docs/INDEX.md) for human-facing documentation.
2. Optional local workspaces (gitignored): `.agent/audit/RESUME_STATUS.md` if present for mission state.
3. Optional local `.agent-docs/INDEX.md` if present for compact agent routing (human docs live in `docs/`).
4. Source of truth for behavior/signatures:
   - [`src/pipeline/spec/schema.py`](src/pipeline/spec/schema.py)
   - [`src/data/ast/functions.py`](src/data/ast/functions.py)
   - [`src/pipeline/spec/executor.py`](src/pipeline/spec/executor.py)
   - [`src/pipeline/data_handler/processors.py`](src/pipeline/data_handler/processors.py)
   - [`src/pipeline/walk_forward/execution.py`](src/pipeline/walk_forward/execution.py)
   - [`src/pipeline/data_handler/merge.py`](src/pipeline/data_handler/merge.py)
5. Example contract surface: [`examples/pipeline_specs/crypto_momentum_baseline.yaml`](examples/pipeline_specs/crypto_momentum_baseline.yaml); optional local `dev/examples/ff3_model.yaml` for factor-style workflows.

## Resume Locations
Use these for interrupted missions and transient execution state (local / gitignored):
- `.factory/` (Factory artifacts and validation traces)
- `.agent/audit/RESUME_STATUS.md`, `STRUCTURAL_AUDIT.md`, `REORG_PLAN.md`

## Durable Knowledge Locations
Primary human documentation: [`docs/`](docs/INDEX.md). Optional local `.agent-docs/` (if present) may duplicate or extend routing notes; keep durable prose aligned with `docs/` where it matters for contributors.

## Anti-Bloat Rules
- Do not add thin wrappers, duplicate registries, or one-caller abstractions.
- Do not add workflow-specific shortcut APIs to core layers.
- Prefer moving/re-homing/trimming over refactoring behavior.
- Keep pipeline semantics declarative; do not bypass YAML with imperative glue when avoidable.
- If a change requires behavior drift beyond import/path repair, stop and document blocker.

## Update Policy
- Prefer updating `docs/` for durable contributor-facing knowledge.
- Update source comments only for code-local, non-obvious invariants.
- Optional local `.agent/audit/*` for mission status when using agent workflows.
- Keep this file thin and routing-focused; it is not the architecture dump.

## Token Efficiency Rule
Do not restate architecture already captured in `docs/`; route to it.

Also:
- Do not duplicate long architecture prose across multiple instruction files.
- If other instruction files exist, keep them as short pointers or tool-specific notes.
