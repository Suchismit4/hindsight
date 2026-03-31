# User Testing

Testing surface, required testing skills/tools, and resource cost classification.

---

## Validation Surface

This mission has a single validation surface: **command-line shell**.

There is no web UI, no TUI, no API server. All validation is done via shell commands:
- `pytest` for test suite execution
- `flake8` for lint checks
- `python -c "import src"` for import verification
- `git ls-files`, `git status` for git state verification
- `ls`, `cat`, `grep` for file existence and content checks
- Markdown link resolution checks via script

**Tools:** All validation uses the shell Execute tool. No agent-browser or tuistory needed.

## Validation Concurrency

**Surface: shell commands**
- Max concurrent validators: **5**
- Rationale: Each shell command uses negligible resources (~50MB RAM, brief CPU). Machine has 64 CPUs and ~500GB RAM. 5 concurrent validators are well within budget.
- No shared state conflicts between validators (read-only checks against filesystem and git)

## Known Pre-Existing Test Behavior

- Baseline: 109 passed, 4 skipped in `dev/` test suite
- Backtester module tests may have issues — not blocking
- 3 pre-existing flake8 errors (F821 x2 in backtester/core.py, F824 x1 in ast/functions.py)
- Various DeprecationWarning and FutureWarning in test output — cosmetic, not failures
