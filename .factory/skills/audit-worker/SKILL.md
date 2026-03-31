---
name: audit-worker
description: Reads the repository structure, classifies files, and produces a structural audit and reorganization plan.
---

# Audit Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Features that require reading the repository structure, classifying files into categories, and producing written audit/plan documents. No file moves — only analysis and documentation.

## Required Skills

None.

## Work Procedure

1. **Read the directory tree.** Use `ls`, `find`, and `git ls-files` to enumerate every file and directory in the repository. Pay special attention to:
   - Root-level files and their purposes
   - The `dev/` directory (tests + AI artifacts + archive)
   - Files tracked by git that shouldn't be (`.pyc`, `.ipynb_checkpoints/`, output PNGs, notebooks)
   - The current `.gitignore` and its patterns

2. **Classify every file/folder.** Assign each item to one of these categories:
   - **PRODUCTION** — ships to GitHub, used by human developers
   - **AI_ARTIFACT** — agent context files not for human consumption
   - **DEV_SCRATCH** — developer-only WIP, not production-ready
   - **GENERATED** — build artifacts
   - **DOCS_STALE** — documentation that is outdated or misplaced
   - **CANONICAL_TEST** — tests that belong in the proper test suite

   Reference the classification guide in AGENTS.md for specific file assignments.

3. **Analyze .gitignore gaps.** Compare what is currently tracked (`git ls-files`) against what should be tracked. Identify:
   - Files tracked that should be gitignored
   - Patterns in .gitignore that are too aggressive (e.g., `*.md`, `*.yaml`)
   - Missing patterns that should be added

4. **Write STRUCTURAL_AUDIT.md.** Create `.agent/audit/STRUCTURAL_AUDIT.md` with:
   - Complete file/folder classification table
   - .gitignore analysis (current vs. proposed)
   - Notes on any ambiguous files

5. **Write REORG_PLAN.md.** Create `.agent/audit/REORG_PLAN.md` with:
   - Every file move as `FROM → TO` (explicit paths)
   - Every file that should be gitignored but currently isn't
   - Every file that should be deleted (e.g., stale `.pyc` in tracked paths, empty directories)
   - A `.gitignore` section with the complete proposed `.gitignore` content
   - The proposed final folder structure as an ASCII tree

6. **Self-validate.** Before completing:
   - Verify every FROM path actually exists: `ls <path>` for each
   - Verify no TO path would overwrite an unrelated existing file
   - Verify no `src/**/*.py` files appear in any move operation
   - Verify no `examples/**/*.yaml` files appear in any delete operation
   - Verify the plan has both a .gitignore section and a folder structure tree

7. **Commit.** Stage and commit `.agent/audit/STRUCTURAL_AUDIT.md` and `.agent/audit/REORG_PLAN.md`.

## Example Handoff

```json
{
  "salientSummary": "Audited 45 root-level items and classified all files. Found 23 tracked files that should be gitignored (including 70+ .pyc files, 4 .ipynb_checkpoints, 4 PNGs, 2 notebooks). Wrote STRUCTURAL_AUDIT.md and REORG_PLAN.md with 18 file moves, 5 deletions, and complete .gitignore rewrite. Validated all FROM paths exist and no src/*.py files are in moves.",
  "whatWasImplemented": "Created .agent/audit/STRUCTURAL_AUDIT.md with complete file classification (45 items across 6 categories) and .agent/audit/REORG_PLAN.md with all file moves, .gitignore proposal, and target folder structure tree.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      { "command": "git ls-files | wc -l", "exitCode": 0, "observation": "140 tracked files found" },
      { "command": "ls .agent/audit/STRUCTURAL_AUDIT.md", "exitCode": 0, "observation": "File exists, 4.2KB" },
      { "command": "ls .agent/audit/REORG_PLAN.md", "exitCode": 0, "observation": "File exists, 6.8KB" },
      { "command": "grep -c 'FROM.*→.*TO' .agent/audit/REORG_PLAN.md", "exitCode": 0, "observation": "18 move operations listed" },
      { "command": "grep 'src/.*\\.py' .agent/audit/REORG_PLAN.md | grep -i 'move\\|rename'", "exitCode": 1, "observation": "No src/*.py in moves — correct" }
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

- Cannot determine whether a file is production or scratch (ambiguous purpose)
- File classification conflicts with AGENTS.md guidance
- Found files that were not anticipated in the mission plan
