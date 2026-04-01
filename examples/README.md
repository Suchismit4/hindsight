# Tracked examples (minimal)

This directory is the smallest **version-controlled** sample of the pipeline system:

- `pipeline_specs/crypto_momentum_baseline.yaml` — end-to-end spec (data → features → preprocessing → model).
- `run_minimal_example.py` — runs that spec twice to show cache reuse.

Larger or workflow-specific samples (FF3, multi-spec cache demo, legacy scripts) live under `dev/examples/` in a local checkout (`dev/` is gitignored).
