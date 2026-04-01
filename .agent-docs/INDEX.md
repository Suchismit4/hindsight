# Hindsight — Agent Documentation Index

**Start here.** Hindsight is a JAX-accelerated, xarray-native quantitative research library. It uses declarative YAML pipelines for data loading, feature engineering, preprocessing, modeling, portfolio construction, and walk-forward evaluation over `(year, month, day, hour, asset)` panels. `examples/ff3_model.yaml` is a workflow example, not a special-case architecture target.

**Fast read order:** `ARCHITECTURE` -> `PIPELINE_SYSTEM` -> `YAML_REFERENCE` -> `KNOWN_BUGS` -> `ANTI_BLOAT`.

---

## Navigation

| File | What it answers | Read if... |
| --- | --- | --- |
| [ARCHITECTURE](./ARCHITECTURE.md) | Data flow, dimension contract, layer boundaries | You need system-level placement and ownership |
| [DATA_LAYER](./DATA_LAYER.md) | `from_table()`, providers, cache, `.dt` accessor | You are loading data or debugging data shape/caching |
| [FORMULA_SYSTEM](./FORMULA_SYSTEM.md) | Formula YAML grammar, registry, execution flow | You are adding/debugging formula functions |
| [PIPELINE_SYSTEM](./PIPELINE_SYSTEM.md) | `PipelineSpec`, processor stages, executor flow | You are editing YAML pipeline behavior |
| [WALK_FORWARD](./WALK_FORWARD.md) | Segment planning/execution and leakage boundaries | You are touching temporal evaluation logic |
| [QUANT_PRIMITIVES](./QUANT_PRIMITIVES.md) | Cross-sectional sort/returns/spread primitives | You are building factor or portfolio workflows |
| [YAML_REFERENCE](./YAML_REFERENCE.md) | Key-by-key YAML contract across repo surfaces | You need exact schema keys/defaults/types |
| [KNOWN_GAPS](./KNOWN_GAPS.md) | Missing capabilities and blocked workflows | You are planning new capability work |
| [KNOWN_BUGS](./KNOWN_BUGS.md) | Active defects and known workarounds | Behavior looks wrong and you need triage |
| [ANTI_BLOAT](./ANTI_BLOAT.md) | Structural anti-bloat merge checklist | You are adding public surface or abstractions |
| [TESTING](./TESTING.md) | Commands, required coverage, test anti-patterns | You are validating a code or YAML change |

---

## See also

- [ARCHITECTURE.md](./ARCHITECTURE.md) — Primary system map
- [YAML_REFERENCE.md](./YAML_REFERENCE.md) — Canonical YAML contract
- [TESTING.md](./TESTING.md) — Validation requirements by change type
