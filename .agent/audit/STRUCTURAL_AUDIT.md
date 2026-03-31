# Structural Audit — Hindsight Repository

**Date:** 2026-03-30
**Tracked files:** 166 (via `git ls-files | wc -l`)
**Root-level items:** 35+ files and directories

---

## File & Directory Classification

### Root-Level Items

| Path | Category | Action | Notes |
|------|----------|--------|-------|
| `.agent/` | AI_ARTIFACT | gitignore | Agent audit workspace (local only) |
| `.agents/` | AI_ARTIFACT | gitignore | Agent config directory (not tracked) |
| `.claude/` | AI_ARTIFACT | gitignore | Claude AI config (not tracked) |
| `.codex/` | AI_ARTIFACT | gitignore | Codex config (not tracked) |
| `.factory/` | AI_ARTIFACT | keep tracked | Mission infrastructure — must remain tracked |
| `.github/` | PRODUCTION (mixed) | keep; move copilot-instructions.md | CI workflow is PRODUCTION; copilot-instructions.md is AI_ARTIFACT |
| `.gitignore` | PRODUCTION | rewrite | Current version is overly aggressive (see analysis below) |
| `.ipynb_checkpoints/` | GENERATED | git rm --cached, gitignore | 4 files tracked that should not be |
| `.pytest_cache/` | GENERATED | gitignore | Already gitignored, not tracked |
| `.ruff_cache/` | GENERATED | gitignore | Already gitignored, not tracked |
| `.vscode/` | DEV_SCRATCH | gitignore | Editor-specific config, not tracked |
| `AGENTS.md` | AI_ARTIFACT | move to .agent/ | Agent instructions; NOT currently tracked (gitignored by `*.md`) |
| `CLAUDE.md` | AI_ARTIFACT | move to .agent/ | Claude context file; NOT currently tracked |
| `DOCS.md` | AI_ARTIFACT | move to .agent/, git rm --cached | AI doc aggregation; IS tracked |
| `DUMP.md` | AI_ARTIFACT | move to .agent/ | Exploration dump; NOT currently tracked |
| `Makefile` | AI_ARTIFACT | move to .agent/ | Sphinx doc build helper; IS tracked |
| `README.md` | PRODUCTION | keep | Project readme; tracked |
| `SKILLS.md` | AI_ARTIFACT | move to .agent/ | Skills documentation; NOT currently tracked |
| `Untitled.ipynb` | DEV_SCRATCH | git rm --cached | Scratch notebook; IS tracked |
| `__pycache__/` | GENERATED | gitignore | Already gitignored, but nested .pyc ARE tracked |
| `build/` | GENERATED | gitignore | Sphinx build output; not tracked |
| `config.toml` | AI_ARTIFACT | move to .agent/ | Sphinx config for doc build; NOT tracked (gitignored by `*.toml`) |
| `conftest.py` | PRODUCTION | keep | Root conftest with `sys.path.insert` for `src` importability |
| `dev/` | mixed | see dev/ table below | Contains tests, AI artifacts, and scratch scripts |
| `docs/` | PRODUCTION | keep | Human-authored documentation (3 files tracked) |
| `example.py` | PRODUCTION | move to examples/ | Working runnable example; IS tracked |
| `examples/` | PRODUCTION | keep | Example scripts + pipeline specs |
| `ff3_v2.ipynb` | DEV_SCRATCH | git rm --cached | Scratch notebook; IS tracked |
| `full_docs.md` | DEV_SCRATCH | git rm --cached | Auto-generated full docs; IS tracked |
| `full_docs_api.md` | DEV_SCRATCH | git rm --cached | Auto-generated API docs; IS tracked |
| `make.bat` | AI_ARTIFACT | move to .agent/ | Windows Sphinx build; NOT tracked (gitignored by `*.bat`) |
| `open_docs.py` | AI_ARTIFACT | move to .agent/ | Doc viewer utility; IS tracked |
| `pipeline_example.log` | DEV_SCRATCH | gitignore | Log file output; NOT tracked |
| `prediction_comparison.png` | DEV_SCRATCH | git rm --cached | Output plot; IS tracked |
| `prediction_comparison_\`.png` | DEV_SCRATCH | git rm --cached | Output plot; IS tracked |
| `price_and_predictions.png` | DEV_SCRATCH | git rm --cached | Output plot; IS tracked |
| `price_and_predictions_\`.png` | DEV_SCRATCH | git rm --cached | Output plot; IS tracked |
| `project_paths.txt` | DEV_SCRATCH | gitignore | Generated file listing; NOT tracked |
| `project_structure.txt` | DEV_SCRATCH | gitignore | Generated file listing; NOT tracked |
| `pytest.ini` | PRODUCTION | keep | Pytest configuration; tracked |
| `requirements.txt` | PRODUCTION | keep | Dependencies; tracked |
| `run_full.log` | DEV_SCRATCH | gitignore | Log output; NOT tracked |
| `serve_docs.py` | AI_ARTIFACT | move to .agent/ | Doc server utility; IS tracked |
| `source/` | DOCS_STALE | gitignore | Sphinx RST source tree (not used in CI, stale); NOT tracked |
| `src/` | PRODUCTION | keep | Main source code; tracked (but contains tracked .pyc files) |
| `wrdsfactorexample/` | DEV_SCRATCH | delete | Empty directory, serves no purpose |

### dev/ Directory Items

| Path | Category | Action | Notes |
|------|----------|--------|-------|
| `dev/CURSOR_PROMPT.md` | AI_ARTIFACT | move to .agent/ | Cursor AI prompt |
| `dev/CURSOR_PROMPT_FINAL.md` | AI_ARTIFACT | move to .agent/ | Cursor AI prompt (final version) |
| `dev/INC3_PLAN.md` | AI_ARTIFACT | move to .agent/ | Incremental development plan |
| `dev/SESSION_CHANGELOG.md` | AI_ARTIFACT | move to .agent/ | Session changelog |
| `dev/__pycache__/` | GENERATED | gitignore | Bytecode cache |
| `dev/archive/` | DEV_SCRATCH | gitignore | Archived scratch scripts (check_merge_asset.py, crypto_data_pull.py, pipe.py, test.py, test_external_tables.py, test_merge.py, test_merge_real_data.py, test_portfolio_returns.py, test_sort.py, val.py, verify_compustat_load.py, verify_ff3.py) |
| `dev/conftest.py` | CANONICAL_TEST | move to tests/conftest.py | Shared test fixtures (simulated_daily_ds, simulated_monthly_ds, ohlcv_ds) |
| `dev/test_backtester.py` | CANONICAL_TEST | move to tests/ | Backtester test file |
| `dev/test_cross_sectional.py` | CANONICAL_TEST | move to tests/ | Cross-sectional ops tests |
| `dev/test_data_core.py` | CANONICAL_TEST | move to tests/ | Data core tests |
| `dev/test_ff3_methodology.py` | CANONICAL_TEST | move to tests/ | FF3 methodology tests |
| `dev/test_ff3_pipeline.py` | CANONICAL_TEST | move to tests/ | FF3 pipeline tests |
| `dev/test_ff3_wrds.py` | CANONICAL_TEST | move to tests/ | FF3 WRDS tests |
| `dev/test_formula_ast.py` | CANONICAL_TEST | move to tests/ | Formula AST tests |
| `dev/test_pipeline.py` | CANONICAL_TEST | move to tests/ | Pipeline tests |
| `dev/test_rolling.py` | CANONICAL_TEST | move to tests/ | Rolling operations tests |
| `dev/test_walk_forward.py` | CANONICAL_TEST | move to tests/ | Walk-forward tests |

### .github/ Items

| Path | Category | Action | Notes |
|------|----------|--------|-------|
| `.github/copilot-instructions.md` | AI_ARTIFACT | move to .agent/ | Copilot instructions; IS tracked |
| `.github/workflows/python-app.yml` | PRODUCTION | keep | CI workflow; tracked |

---

## .gitignore Analysis

### Current .gitignore Problems

The current `.gitignore` is **overly aggressive**. It uses blanket extension patterns that prevent production files from being trackable:

| Pattern | Problem |
|---------|---------|
| `*.md` | Blocks ALL markdown files — prevents tracking AGENTS.md, docs/*.md, src/data/README.md, etc. Existing tracked .md files are preserved only because they were added before this rule. |
| `*.yaml` / `*.yml` | Blocks ALL YAML files — prevents tracking `examples/ff3_model.yaml`, `examples/pipeline_specs/*.yaml`. Source YAML files under `src/` survive only because they were tracked before this rule. |
| `*.toml` | Blocks config.toml and any future .toml files. |
| `*.txt` | Blocks ALL text files — `requirements.txt` only survives because it was tracked before. |
| `*.bat` | Blocks make.bat. |
| `*.config` | Too broad. |
| `source/` | Ignores the entire Sphinx source directory. |

### Files Currently Tracked That Should NOT Be

| File(s) | Count | Reason |
|---------|-------|--------|
| `src/**/__pycache__/*.pyc` | 51 | Compiled bytecode — should never be tracked |
| `.ipynb_checkpoints/*` | 4 | Jupyter checkpoint files |
| `prediction_comparison.png` | 1 | Output plot |
| `prediction_comparison_\`.png` | 1 | Output plot |
| `price_and_predictions.png` | 1 | Output plot |
| `price_and_predictions_\`.png` | 1 | Output plot |
| `.ipynb_checkpoints/apple_tsla_ema-checkpoint.png` | 1 | Checkpoint PNG |
| `Untitled.ipynb` | 1 | Scratch notebook |
| `ff3_v2.ipynb` | 1 | Scratch notebook |
| `full_docs.md` | 1 | Auto-generated docs |
| `full_docs_api.md` | 1 | Auto-generated docs |
| `DOCS.md` | 1 | AI artifact (to be moved) |
| `Makefile` | 1 | AI artifact (to be moved) |
| `serve_docs.py` | 1 | AI artifact (to be moved) |
| `open_docs.py` | 1 | AI artifact (to be moved) |
| `.github/copilot-instructions.md` | 1 | AI artifact (to be moved) |
| `example.py` | 1 | Moving to examples/ |
| **Total** | **~70** | |

### Files NOT Tracked That SHOULD Be (Due to Aggressive .gitignore)

| File | Why it should be tracked |
|------|--------------------------|
| `examples/ff3_model.yaml` | Production example pipeline spec |
| `examples/pipeline_specs/crypto_momentum_baseline.yaml` | Production example pipeline spec |
| `examples/pipeline_specs/crypto_momentum_enhanced.yaml` | Production example pipeline spec |

**Note:** Source YAML files (`src/data/ast/definitions/*.yaml`, `src/data/configs/*.yaml`) and source markdown files (`src/data/README.md`, `src/data/ast/README.md`) are currently still tracked because they were added before the aggressive `.gitignore` rules. However, any NEW `.yaml` or `.md` files would be silently ignored.

### Recommended .gitignore Approach

Replace blanket extension patterns with targeted directory/file patterns:
- **MUST ignore:** `.agent/`, `.factory/`, `build/`, `__pycache__/`, `*.pyc`, `*.log`, `*.png`, `dev/`, `.ipynb_checkpoints/`, `*.ipynb`, `.pytest_cache/`, `.ruff_cache/`, `.vscode/`, `.claude/`, `.codex/`, `.agents/`, `venv/`, `*.parquet`, `*.csv`, `*.zip`, `.DS_Store`, `.env`
- **MUST NOT ignore:** `*.md`, `*.yaml`, `*.yml`, `*.toml`, `*.txt` (these contain production files)

---

## Summary Statistics

| Category | Count |
|----------|-------|
| PRODUCTION | 13 items (src/, examples/, docs/, README.md, conftest.py, pytest.ini, requirements.txt, .github/workflows/, .gitignore, example.py) |
| AI_ARTIFACT | 16 items (AGENTS.md, CLAUDE.md, DUMP.md, SKILLS.md, DOCS.md, config.toml, Makefile, make.bat, serve_docs.py, open_docs.py, .github/copilot-instructions.md, dev/CURSOR_PROMPT.md, dev/CURSOR_PROMPT_FINAL.md, dev/INC3_PLAN.md, dev/SESSION_CHANGELOG.md, .agent/) |
| DEV_SCRATCH | 14 items (Untitled.ipynb, ff3_v2.ipynb, full_docs.md, full_docs_api.md, 4 PNGs, pipeline_example.log, run_full.log, project_paths.txt, project_structure.txt, wrdsfactorexample/, dev/archive/) |
| GENERATED | 5 items (.ipynb_checkpoints/, __pycache__/, build/, .pytest_cache/, .ruff_cache/) |
| DOCS_STALE | 1 item (source/) |
| CANONICAL_TEST | 11 items (dev/conftest.py, 10 dev/test_*.py files) |
