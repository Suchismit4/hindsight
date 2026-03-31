---
name: doc-writer
description: Reads source materials and writes comprehensive, linked agent documentation files in .agent-docs/.
---

# Doc Writer

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Features that require reading source code, existing documentation, and reference materials to produce structured markdown documentation files for `.agent-docs/`. These are LLM-optimized knowledge base files, not human narrative docs.

## Required Skills

None.

## Work Procedure

1. **Read source materials (in this order).** Before writing ANY documentation:
   - `.agent/DUMP.md` — SIG, BLOAT, GAP, ARCH, CRITICAL, INTEGRATION NOTES sections
   - `.agent/CLAUDE.md` — Layer Map, Key Patterns
   - `.agent/AGENTS.md` — Operating priorities (these are in .agent/ after Milestone 2)
   - `docs/ARCHITECTURE.md`, `docs/PIPELINE_SYSTEM.md`, `docs/DATASET_MERGER.md` — existing human docs
   - `examples/ff3_model.yaml` — concrete YAML reference
   - `examples/pipeline_specs/` — more YAML examples
   - `src/pipeline/spec/schema.py` — authoritative PipelineSpec schema
   - `src/data/ast/functions.py` — authoritative registered functions list
   - `src/data/ast/definitions/` — formula YAML examples
   - `src/data/configs/` — data source config YAML examples
   - Browse `src/` directory structure to understand the module hierarchy

   Do NOT copy content blindly from `source/` (Sphinx tree) — it predates the pipeline layer and is structurally outdated.

2. **Create .agent-docs/ directory** if it doesn't exist:
   ```bash
   mkdir -p .agent-docs
   ```

3. **Write each file following its specification.** The feature description will specify WHICH files to write. For each file:
   - Follow the content rules in the feature description precisely
   - Write for an LLM audience: structured, scannable, with tables and code blocks
   - Keep content factual and sourced from the codebase — do not speculate
   - Include a `## See also` section at the bottom with relative links to related files

4. **INDEX.md special rules:** If writing INDEX.md:
   - First 10 lines must be a "Start here" paragraph explaining what the library is
   - Must contain a navigation table: `| File | What it answers | Read if... |`
   - Must link to every other file in .agent-docs/ using relative links
   - Verify every linked file exists

5. **YAML_REFERENCE.md special rules:** If writing YAML_REFERENCE.md:
   - Extract ALL YAML keys from `examples/**/*.yaml`, `src/data/ast/definitions/*.yaml`, `src/data/configs/*.yaml`
   - For each key: document type, default, description, example
   - Cross-reference with `src/pipeline/spec/schema.py` for authoritative types

6. **KNOWN_BUGS.md special rules:** Must contain at minimum these 5 known bugs:
   - u_roll NaN issue
   - Rolling.std issue
   - OpenBB broken methods
   - config_schema leak
   - Hardcoded CCM path
   Source from `.agent/DUMP.md` BLOAT/bug sections.

7. **QUANT_PRIMITIVES.md special rules:** Must document ALL of:
   - cs_rank, cs_quantile, assign_bucket (cross-sectional functions)
   - CrossSectionalSort, PortfolioReturns, FactorSpread (pipeline processors)
   For each: purpose, YAML usage, Python signature, NaN behavior, known edge cases.

8. **Fama-French constraint:** No file may mention Fama-French as a design constraint or special case built into the library. It may appear ONLY as a workflow example (e.g., "the FF3 pipeline example demonstrates...").

9. **Verify cross-links.** After writing all files in this feature:
   - Check that every `[text](./FILE.md)` link points to an existing file
   - Check that every file has a `## See also` section
   - Verify ARCHITECTURE.md links to DATA_LAYER.md, PIPELINE_SYSTEM.md, WALK_FORWARD.md
   - Verify QUANT_PRIMITIVES.md links to FORMULA_SYSTEM.md and PIPELINE_SYSTEM.md
   - Verify KNOWN_GAPS.md links to QUANT_PRIMITIVES.md and PIPELINE_SYSTEM.md

10. **Commit.** Stage all `.agent-docs/` files and commit.

## Example Handoff

```json
{
  "salientSummary": "Wrote 6 .agent-docs/ files: INDEX.md (navigation table with 11 entries), ARCHITECTURE.md (ASCII data-flow diagram, dimension contract table, layer map), DATA_LAYER.md (from_table, DataManager, CacheManager, .dt accessor signatures), FORMULA_SYSTEM.md (grammar, 28 registered functions, FormulaManager), PIPELINE_SYSTEM.md (3-stage model, PipelineSpec schema, 8 processors), WALK_FORWARD.md (SegmentConfig, runners, leakage contract). All cross-links verified, See also sections present.",
  "whatWasImplemented": "Created 6 comprehensive .agent-docs/ files from source materials. INDEX.md has Start-here paragraph and full navigation table. ARCHITECTURE.md has ASCII data-flow diagram and dimension contract. All files have See also sections with verified relative links.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      { "command": "ls -la .agent-docs/", "exitCode": 0, "observation": "6 files, all non-empty" },
      { "command": "head -10 .agent-docs/INDEX.md", "exitCode": 0, "observation": "Start-here paragraph present in first 10 lines" },
      { "command": "grep -c '## See also' .agent-docs/*.md", "exitCode": 0, "observation": "6/6 files have See also section" },
      { "command": "grep -c 'DATA_LAYER\\|PIPELINE_SYSTEM\\|WALK_FORWARD' .agent-docs/ARCHITECTURE.md", "exitCode": 0, "observation": "All 3 required links present" }
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": []
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Source material files (.agent/DUMP.md, .agent/CLAUDE.md) don't exist or are empty
- Cannot find authoritative information for a required documentation topic
- Feature description specifies writing a file but the content requirements are ambiguous
- Cross-link targets don't exist yet (files assigned to a different feature)
