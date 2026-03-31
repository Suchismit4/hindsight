# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Python Environment

- **Path:** `~/conda/envs/jax/` (Python 3.12.7)
- **Key binaries:** `~/conda/envs/jax/bin/python`, `~/conda/envs/jax/bin/pytest`, `~/conda/envs/jax/bin/flake8`
- **No conda activation needed** — use full paths to binaries
- JAX configured globally for float64 at import time (`src/__init__.py`)

## Dependencies

- All Python dependencies listed in `requirements.txt`
- Key libraries: jax, jaxlib, equinox, xarray, numpy, pandas, scikit-learn
- flake8 installed separately in the conda environment

## Platform Notes

- Running on Linux (Docker container)
- 64 CPUs, ~500GB RAM
- No network access needed for this mission
