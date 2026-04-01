# Hindsight Sphinx documentation

This directory is the Sphinx source for the Hindsight docs (`*.rst` under `source/`).

## Prerequisites

Work in the **jax** conda environment:

```bash
conda activate jax
```

Install Sphinx and the Read the Docs theme once in that env (if not already installed):

```bash
pip install sphinx sphinx-rtd-theme
```

## Build HTML

From the **repository root** (not `source/`):

```bash
conda activate jax
cd /path/to/hindsight
PYTHONPATH=. sphinx-build -b html source build/html
```

`PYTHONPATH=.` is required so autodoc can import `src.*` (including JAX-backed code).

Open the site:

```bash
xdg-open build/html/index.html   # Linux
# or open build/html/index.html in a browser
```

## Live preview (optional)

Serve the built HTML on a port:

```bash
conda activate jax
cd /path/to/hindsight
python -m http.server 8000 --directory build/html
```

Then open `http://localhost:8000/`.

## Documentation layout

- **Getting started**: `source/getting_started/` (overview, data loading, YAML pipeline, handlers, features, walk-forward, models, execution).
- **Examples**: `source/examples/`.
- **API**: `source/api/` (autosummary + autodoc into `source/api/generated/`).
