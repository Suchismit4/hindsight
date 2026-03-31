#!/bin/bash
set -e

# Ensure we're in the project root
cd /home/ubuntu/projects/hindsight

# Verify conda environment is accessible
if [ ! -f ~/conda/envs/jax/bin/python ]; then
    echo "ERROR: conda jax environment not found at ~/conda/envs/jax/"
    exit 1
fi

# Verify key tools
~/conda/envs/jax/bin/python --version
~/conda/envs/jax/bin/pytest --version 2>/dev/null || true
~/conda/envs/jax/bin/flake8 --version 2>/dev/null || true

# Ensure .agent directory exists (for audit artifacts)
mkdir -p .agent/audit

echo "Environment ready."
