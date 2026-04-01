# Hindsight Codex Router

## Identity / Doctrine
Hindsight is a general quantitative research library.

Core doctrine:
- YAML-first pipeline semantics.
- xarray/JAX-native data flow.
- Minimal API surface.
- Preserve behavior unless a path/import fix is required.
- Optimize for generic quant primitives, not one workflow.
- `examples/ff3_model.yaml` is an example pipeline, not a design constraint.

Canonical data flow:
`provider -> from_table() -> xr.Dataset(year, month, day, hour, asset) -> .dt -> FormulaManager/DataHandler -> walk-forward/model`

## Read-First Order
1. [`.agent/audit/RESUME_STATUS.md`](.agent/audit/RESUME_STATUS.md) (if present) for current mission state.
2. [`.agent-docs/INDEX.md`](.agent-docs/INDEX.md) for durable architecture routing.
3. Source of truth for behavior/signatures:
   - [`src/pipeline/spec/schema.py`](src/pipeline/spec/schema.py)
   - [`src/data/ast/functions.py`](src/data/ast/functions.py)
   - [`src/pipeline/spec/executor.py`](src/pipeline/spec/executor.py)
   - [`src/pipeline/data_handler/processors.py`](src/pipeline/data_handler/processors.py)
   - [`src/pipeline/walk_forward/execution.py`](src/pipeline/walk_forward/execution.py)
   - [`src/pipeline/data_handler/merge.py`](src/pipeline/data_handler/merge.py)
4. Example contract surface:
   - [`examples/ff3_model.yaml`](examples/ff3_model.yaml)

## Resume Locations
Use these for interrupted missions and transient execution state:
- [`.factory/`](.factory/) (Factory artifacts and validation traces)
- [`.agent/audit/RESUME_STATUS.md`](.agent/audit/RESUME_STATUS.md)
- [`.agent/audit/STRUCTURAL_AUDIT.md`](.agent/audit/STRUCTURAL_AUDIT.md)
- [`.agent/audit/REORG_PLAN.md`](.agent/audit/REORG_PLAN.md)

## Durable Knowledge Locations
All durable architecture and implementation guidance belongs in [`.agent-docs/`](.agent-docs/):
- [`.agent-docs/INDEX.md`](.agent-docs/INDEX.md) (entrypoint)
- [`.agent-docs/ARCHITECTURE.md`](.agent-docs/ARCHITECTURE.md)
- [`.agent-docs/PIPELINE_SYSTEM.md`](.agent-docs/PIPELINE_SYSTEM.md)
- [`.agent-docs/YAML_REFERENCE.md`](.agent-docs/YAML_REFERENCE.md)
- [`.agent-docs/QUANT_PRIMITIVES.md`](.agent-docs/QUANT_PRIMITIVES.md)
- [`.agent-docs/KNOWN_BUGS.md`](.agent-docs/KNOWN_BUGS.md)
- [`.agent-docs/KNOWN_GAPS.md`](.agent-docs/KNOWN_GAPS.md)
- [`.agent-docs/ANTI_BLOAT.md`](.agent-docs/ANTI_BLOAT.md)
- [`.agent-docs/TESTING.md`](.agent-docs/TESTING.md)

## Anti-Bloat Rules
- Do not add thin wrappers, duplicate registries, or one-caller abstractions.
- Do not add workflow-specific shortcut APIs to core layers.
- Prefer moving/re-homing/trimming over refactoring behavior.
- Keep pipeline semantics declarative; do not bypass YAML with imperative glue when avoidable.
- If a change requires behavior drift beyond import/path repair, stop and document blocker.

## Update Policy
- Update `.agent-docs/` when knowledge is durable and likely needed repeatedly.
- Update source comments only for code-local, non-obvious invariants.
- Update `.agent/audit/*` for mission status, plans, and temporary execution notes.
- Keep this file thin and routing-focused; it is not the architecture dump.

## Token Efficiency Rule
"Do not restate architecture already captured in `.agent-docs/`; route to it."

Also:
- Do not duplicate long architecture prose across multiple instruction files.
- If other instruction files exist, keep them as short pointers or tool-specific notes.
