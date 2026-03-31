# Reorganization Plan — Hindsight Repository

**Date:** 2026-03-30
**Based on:** STRUCTURAL_AUDIT.md

---

## 1. File Moves

All moves listed as `FROM → TO`. Every FROM path verified to exist.

### Root AI Artifacts → .agent/

| # | FROM | TO |
|---|------|----|
| 1 | `AGENTS.md` | `.agent/AGENTS.md` |
| 2 | `CLAUDE.md` | `.agent/CLAUDE.md` |
| 3 | `DUMP.md` | `.agent/DUMP.md` |
| 4 | `SKILLS.md` | `.agent/SKILLS.md` |
| 5 | `DOCS.md` | `.agent/DOCS.md` |
| 6 | `config.toml` | `.agent/config.toml` |
| 7 | `Makefile` | `.agent/Makefile` |
| 8 | `make.bat` | `.agent/make.bat` |
| 9 | `serve_docs.py` | `.agent/serve_docs.py` |
| 10 | `open_docs.py` | `.agent/open_docs.py` |

### Dev AI Artifacts → .agent/

| # | FROM | TO |
|---|------|----|
| 11 | `dev/CURSOR_PROMPT.md` | `.agent/CURSOR_PROMPT.md` |
| 12 | `dev/CURSOR_PROMPT_FINAL.md` | `.agent/CURSOR_PROMPT_FINAL.md` |
| 13 | `dev/INC3_PLAN.md` | `.agent/INC3_PLAN.md` |
| 14 | `dev/SESSION_CHANGELOG.md` | `.agent/SESSION_CHANGELOG.md` |

### GitHub AI Artifact → .agent/

| # | FROM | TO |
|---|------|----|
| 15 | `.github/copilot-instructions.md` | `.agent/copilot-instructions.md` |

### Example File → examples/

| # | FROM | TO |
|---|------|----|
| 16 | `example.py` | `examples/example.py` |

### Canonical Tests → tests/

| # | FROM | TO |
|---|------|----|
| 17 | `dev/test_backtester.py` | `tests/test_backtester.py` |
| 18 | `dev/test_cross_sectional.py` | `tests/test_cross_sectional.py` |
| 19 | `dev/test_data_core.py` | `tests/test_data_core.py` |
| 20 | `dev/test_ff3_methodology.py` | `tests/test_ff3_methodology.py` |
| 21 | `dev/test_ff3_pipeline.py` | `tests/test_ff3_pipeline.py` |
| 22 | `dev/test_ff3_wrds.py` | `tests/test_ff3_wrds.py` |
| 23 | `dev/test_formula_ast.py` | `tests/test_formula_ast.py` |
| 24 | `dev/test_pipeline.py` | `tests/test_pipeline.py` |
| 25 | `dev/test_rolling.py` | `tests/test_rolling.py` |
| 26 | `dev/test_walk_forward.py` | `tests/test_walk_forward.py` |
| 27 | `dev/conftest.py` | `tests/conftest.py` |

**Total: 27 moves**

---

## 2. Deletions

| Path | Reason |
|------|--------|
| `wrdsfactorexample/` | Empty directory, serves no purpose |

**Note:** Tracked `.pyc` files and `.ipynb_checkpoints/` are handled in the Git Untrack section below (they are removed from git tracking but the on-disk files are handled by `.gitignore`).

---

## 3. Git Untrack (`git rm --cached`)

Files currently tracked by git that must be removed from tracking. These files remain on disk but are gitignored going forward.

### .pyc Files (51 files)

```
src/__pycache__/__init__.cpython-312.pyc
src/data/__pycache__/__init__.cpython-312.pyc
src/data/__pycache__/manager.cpython-312.pyc
src/data/__pycache__/processors_registry.cpython-312.pyc
src/data/__pycache__/provider.cpython-312.pyc
src/data/__pycache__/registry.cpython-312.pyc
src/data/ast/__pycache__/__init__.cpython-312.pyc
src/data/ast/__pycache__/functions.cpython-312.pyc
src/data/ast/__pycache__/grammar.cpython-312.pyc
src/data/ast/__pycache__/manager.cpython-312.pyc
src/data/ast/__pycache__/nodes.cpython-312.pyc
src/data/ast/__pycache__/parser.cpython-312.pyc
src/data/ast/__pycache__/test.cpython-312.pyc
src/data/ast/__pycache__/visualization.cpython-312.pyc
src/data/configs/__pycache__/__init__.cpython-312.pyc
src/data/core/__pycache__/__init__.cpython-312.pyc
src/data/core/__pycache__/cache.cpython-312.pyc
src/data/core/__pycache__/computations.cpython-312.pyc
src/data/core/__pycache__/data.cpython-312.pyc
src/data/core/__pycache__/operations.cpython-312.pyc
src/data/core/__pycache__/provider.cpython-312.pyc
src/data/core/__pycache__/struct.cpython-312.pyc
src/data/core/__pycache__/util.cpython-312.pyc
src/data/core/operations/__pycache__/__init__.cpython-312.pyc
src/data/core/operations/__pycache__/standard.cpython-312.pyc
src/data/filters/__pycache__/__init__.cpython-312.pyc
src/data/filters/__pycache__/filters.cpython-312.pyc
src/data/generators/__pycache__/__init__.cpython-312.pyc
src/data/generators/__pycache__/weights.cpython-312.pyc
src/data/generators/__pycache__/window.cpython-312.pyc
src/data/loaders/__pycache__/__init__.cpython-312.pyc
src/data/loaders/__pycache__/base.cpython-312.pyc
src/data/loaders/abstracts/__pycache__/__init__.cpython-312.pyc
src/data/loaders/abstracts/__pycache__/base.cpython-312.pyc
src/data/loaders/crypto/__pycache__/__init__.cpython-312.pyc
src/data/loaders/crypto/__pycache__/local.cpython-312.pyc
src/data/loaders/open_bb/__pycache__/__init__.cpython-312.pyc
src/data/loaders/open_bb/__pycache__/generic.cpython-312.pyc
src/data/loaders/wrds/__pycache__/__init__.cpython-312.pyc
src/data/loaders/wrds/__pycache__/compustat.cpython-312.pyc
src/data/loaders/wrds/__pycache__/crsp.cpython-312.pyc
src/data/loaders/wrds/__pycache__/fundv.cpython-312.pyc
src/data/loaders/wrds/__pycache__/generic.cpython-312.pyc
src/data/loaders/wrds/__pycache__/helpers.cpython-312.pyc
src/data/managers/__pycache__/__init__.cpython-312.pyc
src/data/managers/__pycache__/config_schema.cpython-312.pyc
src/data/managers/__pycache__/data_manager.cpython-312.pyc
src/data/processors/__pycache__/__init__.cpython-312.pyc
src/data/processors/__pycache__/processors.cpython-312.pyc
src/data/processors/__pycache__/registry.cpython-312.pyc
```

**Shortcut command:** `git ls-files '*.pyc' | xargs git rm --cached`

### .ipynb_checkpoints/ (4 files)

```
.ipynb_checkpoints/Untitled-checkpoint.ipynb
.ipynb_checkpoints/apple_tsla_ema-checkpoint.png
.ipynb_checkpoints/bug-checkpoint.ipynb
.ipynb_checkpoints/example_ff3-checkpoint.ipynb
```

**Shortcut command:** `git rm --cached -r .ipynb_checkpoints/`

### Output PNGs (4 files)

```
prediction_comparison.png
prediction_comparison_`.png
price_and_predictions.png
price_and_predictions_`.png
```

### Notebooks (2 files)

```
Untitled.ipynb
ff3_v2.ipynb
```

### Generated Docs (2 files)

```
full_docs.md
full_docs_api.md
```

### AI Artifact (1 file)

```
DOCS.md
```

**Note:** `DOCS.md` is both moved to `.agent/DOCS.md` and untracked from its root location. Since `.agent/` will be gitignored, the move effectively removes it from tracking. Use `git mv` for history preservation, or manually copy + `git rm --cached`.

### Log Files

No `.log` files are currently tracked (`git ls-files '*.log'` returns empty). The new `.gitignore` will prevent future log files from being tracked.

**Total files to untrack: ~64**

---

## 4. .gitignore Section

### Current .gitignore Issues

The current `.gitignore` uses blanket extension patterns (`*.md`, `*.yaml`, `*.yml`, `*.toml`, `*.txt`, `*.bat`) that prevent production files from being tracked. This must be replaced with targeted patterns.

### Proposed .gitignore Content

```gitignore
# === Python ===
__pycache__/
**/__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
dist/
*.egg

# === Virtual Environments ===
venv/
.venv/

# === Data Files ===
*.parquet
*.csv
*.zip

# === IDE / Editor ===
.vscode/
.idea/

# === OS ===
.DS_Store
Thumbs.db

# === Environment ===
.env
.env.*

# === Build Artifacts ===
build/

# === Caches ===
.pytest_cache/
**/.pytest_cache/
.ruff_cache/
**/.ruff_cache/

# === Jupyter ===
*.ipynb
.ipynb_checkpoints/

# === Logs ===
*.log

# === Output Plots ===
*.png

# === Dev Scratch ===
dev/
source/
full_docs.md
full_docs_api.md
project_paths.txt
project_structure.txt

# === AI / Agent Directories ===
.agent/
.agents/
.claude/
.codex/

# === Mission Infrastructure ===
.factory/
```

### What This .gitignore Does NOT Ignore (Preserved for Production)

- `*.md` — markdown files (README.md, docs/*.md, src/data/README.md, etc.)
- `*.yaml` / `*.yml` — YAML files (examples/*.yaml, src/data/ast/definitions/*.yaml, src/data/configs/*.yaml)
- `*.toml` — TOML files (if any future production configs)
- `*.txt` — text files (requirements.txt)
- `*.py` — Python source files
- `*.rst` — reStructuredText (if used)

---

## 5. Final Folder Structure

ASCII tree of the proposed repository structure after all moves, deletions, and .gitignore changes:

```
hindsight/
├── .github/
│   └── workflows/
│       └── python-app.yml
├── .gitignore
├── README.md
├── conftest.py                          # Root conftest (sys.path.insert for src)
├── pytest.ini
├── requirements.txt
│
├── docs/                                # Human-authored documentation
│   ├── ARCHITECTURE.md
│   ├── DATASET_MERGER.md
│   └── PIPELINE_SYSTEM.md
│
├── examples/                            # Runnable examples + pipeline specs
│   ├── example.py                       # (moved from root)
│   ├── ff3_model.yaml
│   ├── run_pipeline_example.py
│   └── pipeline_specs/
│       ├── crypto_momentum_baseline.yaml
│       └── crypto_momentum_enhanced.yaml
│
├── src/                                 # Main source code (unchanged)
│   ├── __init__.py
│   ├── backtester/
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── struct.py
│   │   └── metrics/
│   │       ├── __init__.py
│   │       └── standard.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── ast/
│   │   │   ├── __init__.py
│   │   │   ├── README.md
│   │   │   ├── definitions/
│   │   │   │   ├── composite.yaml
│   │   │   │   ├── marketchars.yaml
│   │   │   │   ├── schema.yaml
│   │   │   │   ├── technical.yaml
│   │   │   │   └── ts_dependence.yaml
│   │   │   ├── functions.py
│   │   │   ├── grammar.py
│   │   │   ├── manager.py
│   │   │   ├── nodes.py
│   │   │   ├── parser.py
│   │   │   └── visualization.py
│   │   ├── configs/
│   │   │   ├── __init__.py
│   │   │   ├── crypto_standard.yaml
│   │   │   └── equity_standard.yaml
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── cache.py
│   │   │   ├── jit.py
│   │   │   ├── operations/
│   │   │   │   ├── __init__.py
│   │   │   │   └── standard.py
│   │   │   ├── provider.py
│   │   │   ├── rolling.py
│   │   │   ├── struct.py
│   │   │   └── types.py
│   │   ├── filters/
│   │   │   ├── __init__.py
│   │   │   └── filters.py
│   │   ├── generators/
│   │   │   ├── __init__.py
│   │   │   ├── weights.py
│   │   │   └── window.py
│   │   ├── loaders/
│   │   │   ├── __init__.py
│   │   │   ├── abstracts/
│   │   │   │   ├── __init__.py
│   │   │   │   └── base.py
│   │   │   ├── crypto/
│   │   │   │   ├── __init__.py
│   │   │   │   └── local.py
│   │   │   ├── open_bb/
│   │   │   │   ├── __init__.py
│   │   │   │   └── generic.py
│   │   │   ├── table.py
│   │   │   └── wrds/
│   │   │       ├── __init__.py
│   │   │       ├── compustat.py
│   │   │       ├── crsp.py
│   │   │       └── generic.py
│   │   ├── managers/
│   │   │   ├── __init__.py
│   │   │   ├── config_schema.py
│   │   │   └── data_manager.py
│   │   └── processors/
│   │       ├── __init__.py
│   │       ├── processors.py
│   │       └── registry.py
│   └── pipeline/
│       ├── __init__.py
│       ├── cache/
│       │   ├── __init__.py
│       │   ├── manager.py
│       │   ├── metadata.py
│       │   └── stages.py
│       ├── data_handler/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── core.py
│       │   ├── handler.py
│       │   ├── merge.py
│       │   └── processors.py
│       ├── model/
│       │   ├── __init__.py
│       │   ├── adapter.py
│       │   └── runner.py
│       ├── spec/
│       │   ├── __init__.py
│       │   ├── executor.py
│       │   ├── parser.py
│       │   ├── processor_registry.py
│       │   ├── result.py
│       │   └── schema.py
│       └── walk_forward/
│           ├── __init__.py
│           ├── execution.py
│           ├── planning.py
│           └── segments.py
│
└── tests/                               # Canonical test suite (moved from dev/)
    ├── conftest.py                      # (moved from dev/conftest.py)
    ├── test_backtester.py
    ├── test_cross_sectional.py
    ├── test_data_core.py
    ├── test_ff3_methodology.py
    ├── test_ff3_pipeline.py
    ├── test_ff3_wrds.py
    ├── test_formula_ast.py
    ├── test_pipeline.py
    ├── test_rolling.py
    └── test_walk_forward.py
```

### Gitignored Directories (exist on disk, not tracked)

```
.agent/                  # AI artifacts, audit docs, moved files
.agents/                 # Agent config
.claude/                 # Claude config
.codex/                  # Codex config
.factory/                # Mission infrastructure
.vscode/                 # Editor config
build/                   # Sphinx build output
dev/                     # Remaining scratch files (archive/, scripts)
source/                  # Sphinx RST source (stale)
```

---

## Self-Validation Checklist

- [x] Every FROM path exists (verified via `ls`)
- [x] No TO path overwrites an existing unrelated file (`tests/` doesn't exist yet; `examples/example.py` doesn't exist)
- [x] No `src/**/*.py` files appear in any move operation (all moves are .md, .py utility scripts, .toml, .bat, or test files from dev/)
- [x] No `examples/**/*.yaml` files appear in any delete operation (only deletion is empty `wrdsfactorexample/`)
- [x] Plan has a .gitignore section with complete proposed content (Section 4)
- [x] Plan has an ASCII tree of the proposed final folder structure (Section 5)
