---
name: cleanup-worker
description: Executes the reorganization plan — file moves, .gitignore rewrite, test relocation, git cleanup.
---

# Cleanup Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Features that require executing file moves, updating .gitignore, relocating tests, running `git rm --cached`, and verifying the repository is in a clean state after reorganization.

## Required Skills

None.

## Work Procedure

1. **Read the plan.** Read `.agent/audit/REORG_PLAN.md` thoroughly before making any changes. Understand every move, every gitignore change, and the target structure.

2. **Create target directories.** Before any moves, create all destination directories:
   ```bash
   mkdir -p .agent/audit .agent/memory tests/
   ```

3. **Execute file moves in dependency order.** Move files that nothing depends on first. Use `git mv` where possible to preserve history. For each move:
   - Verify the source exists: `ls <FROM>`
   - Verify destination dir exists: `ls <TO_DIR>`
   - Execute: `git mv <FROM> <TO>` (or `mv` + `git add` if `git mv` fails)
   - Verify the move: `ls <TO>`

4. **Move canonical tests to tests/.** Move all test files from `dev/` to `tests/`:
   - Move `dev/conftest.py` → `tests/conftest.py`
   - Move each `dev/test_*.py` → `tests/test_*.py`
   - Verify: `ls tests/` shows all test files

5. **Rewrite .gitignore.** Replace the current overly-aggressive .gitignore with a targeted one. Reference the `.gitignore Rewrite Rules` section in AGENTS.md. Critical: do NOT add `*.md`, `*.yaml`, `*.yml`, `*.toml`, or `*.txt` patterns. Source YAML files in `src/` must remain trackable.

6. **Untrack stale files.** Run `git rm --cached` for files that are now gitignored:
   ```bash
   git rm --cached -r __pycache__/ .ipynb_checkpoints/ 2>/dev/null || true
   git rm --cached '*.pyc' 2>/dev/null || true
   ```
   Also untrack: output PNGs, notebooks (if moved/gitignored), full_docs*.md, log files, etc. Only untrack files that are moving to gitignored paths or match new gitignore patterns.

7. **Update pytest.ini.** Ensure `norecursedirs` is correct for the new layout. Tests are now in `tests/`, not `dev/`.

8. **Run validation suite.** Execute ALL of these and verify:
   ```bash
   ~/conda/envs/jax/bin/pytest tests/ -v                   # must pass (109 passed, 4 skipped baseline)
   ~/conda/envs/jax/bin/flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics  # only 3 pre-existing errors
   ~/conda/envs/jax/bin/python -c "import src"              # must succeed
   git ls-files '*.pyc'                                      # must be empty
   git ls-files '.ipynb_checkpoints/'                        # must be empty
   git ls-files '*.png'                                      # must be empty
   git ls-files '*.ipynb'                                    # must be empty
   git ls-files | grep -E 'CURSOR_PROMPT|SESSION_CHANGELOG|INC.*_PLAN'  # must be empty
   ```

9. **Write CHANGELOG_REORG.md.** Create `.agent/audit/CHANGELOG_REORG.md` listing every file that moved and where it went.

10. **Commit.** Stage all changes and commit with a clear message describing the reorganization.

## Example Handoff

```json
{
  "salientSummary": "Executed full repo reorganization: moved 14 AI artifacts to .agent/, relocated 10 test files + conftest.py to tests/, moved example.py to examples/, rewrote .gitignore (from 50-line aggressive to 25-line targeted), ran git rm --cached on 80+ stale tracked files (.pyc, .ipynb_checkpoints, PNGs). All 109 tests pass from tests/ directory. Import check passes. Git state clean.",
  "whatWasImplemented": "Complete repository reorganization per REORG_PLAN.md: 14 AI artifact moves to .agent/, 11 test file moves to tests/, .gitignore rewrite, git rm --cached for .pyc/.ipynb_checkpoints/PNGs/notebooks/logs, pytest.ini updated, CHANGELOG_REORG.md written.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      { "command": "~/conda/envs/jax/bin/pytest tests/ -v", "exitCode": 0, "observation": "109 passed, 4 skipped" },
      { "command": "~/conda/envs/jax/bin/flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics", "exitCode": 1, "observation": "3 pre-existing errors only (2x F821 backtester, 1x F824 ast)" },
      { "command": "~/conda/envs/jax/bin/python -c 'import src'", "exitCode": 0, "observation": "Import succeeds" },
      { "command": "git ls-files '*.pyc' | wc -l", "exitCode": 0, "observation": "0 — no .pyc tracked" },
      { "command": "git ls-files '.ipynb_checkpoints/' | wc -l", "exitCode": 0, "observation": "0 — no checkpoints tracked" },
      { "command": "git ls-files '*.png' | wc -l", "exitCode": 0, "observation": "0 — no PNGs tracked" },
      { "command": "git ls-files '*.ipynb' | wc -l", "exitCode": 0, "observation": "0 — no notebooks tracked" },
      { "command": "git ls-files | grep -E 'CURSOR_PROMPT|SESSION_CHANGELOG|INC.*_PLAN' | wc -l", "exitCode": 0, "observation": "0 — no AI prompt files tracked" },
      { "command": "git ls-files | grep full_docs | wc -l", "exitCode": 0, "observation": "0 — no full_docs tracked" }
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

- A file move breaks `import src` or causes test failures that cannot be resolved by path adjustments
- REORG_PLAN.md contains a move that would overwrite an important file
- Git state is inconsistent after moves (merge conflicts, unexpected deletions)
- Tests fail for reasons unrelated to the reorganization
