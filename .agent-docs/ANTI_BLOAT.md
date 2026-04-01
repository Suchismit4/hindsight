# Anti-Bloat Checklist

Rules for preventing unnecessary complexity in the Hindsight codebase. Treat this as a repository-wide pre-merge gate for all src/pipeline/examples changes.

---

## Structural — Do Not Add

| ❌ Do not add | Why | What to do instead |
|--------------|-----|---------------------|
| Public class wrapping a stateless function | Adds indirection with no benefit; inflates API surface | Convert to a plain function |
| Re-export shim duplicating another `__init__.py` | Creates two sources of truth for the same export | Remove the shim; fix callers to import from the canonical location |
| Thin wrapper class (1–2 method calls, no state) | Extra abstraction that only obscures the underlying call | Delete the wrapper; have callers use the underlying function directly |
| Abstract base class with exactly one concrete subclass | Premature abstraction; the ABC adds complexity without flexibility | Merge the ABC into the concrete class, or remove the ABC |
| Getter/setter with no validation or transformation | Ceremony without purpose | Access the attribute directly |
| New processor class when a `@post_processor` function suffices | Classes are heavier to maintain and test than decorated functions | Use `@post_processor` decorator for simple stateless transforms |

---

## Deprecation Leaks — Do Not Import

| ❌ Do not do | Why | What to do instead |
|-------------|-----|---------------------|
| `from src.data.managers.config_schema import ...` in new code | Module is deprecated; triggers `DeprecationWarning` on every import | Use `src.pipeline.spec` (PipelineSpec, DataSourceSpec, etc.) |
| Leave `DeprecationWarning` on import unsuppressed or unmigrated | Users see warnings on every `import src` | Migrate callers or suppress at the import site |

---

## Dispatch Duplication — Do Not Repeat

| ❌ Do not do | Why | What to do instead |
|-------------|-----|---------------------|
| Copy the `isinstance(data, xr.Dataset) / isinstance(data, xr.DataArray)` pattern into another function | Already duplicated 20+ times in `ast/functions.py` | Extract a `_dispatch(data, fn_da, fn_ds)` helper and reuse it |

---

## API Surface Creep — Do Not Expose

| ❌ Do not do | Why | What to do instead |
|-------------|-----|---------------------|
| Make a function public when only one internal caller uses it | Inflates the API surface with unstable internals | Prefix with `_` to mark private |
| Create an abstraction that serves only one workflow | One-off abstractions become maintenance debt when the workflow changes | Document it as workflow-specific; do not promote to first-class API |
| Add a new YAML key without a test that exercises it | Untested keys silently break; users discover failures at runtime | Write a test that parses a YAML spec with the new key and verifies behavior |

---

## YAML-Contract Violations — Do Not Hardcode

| ❌ Do not do | Why | What to do instead |
|-------------|-----|---------------------|
| Write imperative Python glue for something expressible as a processor stage or YAML formula | Breaks the declarative pipeline contract; can't be composed or cached | Implement as a registered processor or formula function |
| Hardcode a column name, file path, or frequency in Python | Not portable across datasets or environments | Add a YAML config key and read from the spec |
| Hardcode model hyperparameters in Python | Prevents experiment tracking and reproducibility | Move to `model.params` in the YAML spec |

---

## General Operating Principles

Repository doctrine:

1. **Prefer architectural clarity over cleverness.** If it takes a paragraph to explain why the abstraction is needed, it probably is not.
2. **Minimize API surface area.** Every public export is a maintenance commitment.
3. **Avoid abstractions that serve one narrow workflow.** Think in terms of general quantitative research workflows.
4. **Treat YAML pipeline semantics as a first-class interface.** If a user can't express it in YAML, it might not belong in the pipeline.
5. **Frame gaps in terms of the workflow they block.** "Rolling std is missing" is better than "add std to Rolling class."
6. **One gap, one change.** Never bundle unrelated fixes in a single PR.

---

## Quick Self-Check Before Merging

```
[ ] No new public class wrapping a stateless function?
[ ] No re-export shim duplicating __init__.py?
[ ] No thin wrapper class with no state?
[ ] No import from config_schema.py?
[ ] No duplicated isinstance dispatch pattern?
[ ] No public export used by only one internal caller?
[ ] No hardcoded column name, path, or frequency?
[ ] No new YAML key without a test?
[ ] Does pytest still pass?
```

---

## See also

- [KNOWN_BUGS.md](./KNOWN_BUGS.md) — Active bugs, several of which are bloat-related (e.g., config_schema leak)
- [TESTING.md](./TESTING.md) — Required test coverage by change type
- [ARCHITECTURE.md](./ARCHITECTURE.md) — Layer boundaries that constrain where new code belongs
