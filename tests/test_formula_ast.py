"""
Tests for the formula AST system: parsing, evaluation, variable extraction,
function registry, and dataset-aware evaluation.
"""

import numpy as np
import pytest
import xarray as xr

from src.data.ast.parser import (
    evaluate_formula,
    extract_functions,
    extract_variables,
    parse_formula,
)
from src.data.ast.functions import get_function_context, get_registered_functions


# ---------------------------------------------------------------------------
# Arithmetic parsing
# ---------------------------------------------------------------------------

def test_parse_arithmetic():
    node = parse_formula("2 + 3 * 4")
    result = node.evaluate({})
    assert float(result) == pytest.approx(14.0)


def test_parse_with_variables():
    node = parse_formula("x + y")
    result = node.evaluate({"x": 1.0, "y": 2.0})
    assert float(result) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Variable / function extraction
# ---------------------------------------------------------------------------

def test_extract_variables():
    variables = extract_variables("x + y * z")
    assert variables == {"x", "y", "z"}


def test_extract_data_variables():
    variables = extract_variables("$close / $volume")
    assert variables == {"$close", "$volume"}


def test_extract_functions():
    funcs = extract_functions("sma($close, 30)")
    assert "sma" in funcs


# ---------------------------------------------------------------------------
# Function registry
# ---------------------------------------------------------------------------

def test_function_registry_builtins():
    funcs = get_registered_functions()
    for expected in ("sma", "ema", "returns", "shift"):
        assert expected in funcs, f"Expected '{expected}' in registry"


# ---------------------------------------------------------------------------
# Dataset-aware evaluation
# ---------------------------------------------------------------------------

def test_evaluate_with_dataset(simulated_daily_ds):
    ds = simulated_daily_ds
    ctx = {
        "_dataset": ds,
        "var_1": ds["var_1"],
        "var_2": ds["var_2"],
        **get_function_context(),
    }
    result, out_ds = evaluate_formula("var_1 + var_2", ctx)
    assert isinstance(result, xr.DataArray)
    assert result.shape == ds["var_1"].shape


def test_sma_on_dataset(simulated_daily_ds):
    """sma applied to a Dataset (not $variable DataArray) should work."""
    ds = simulated_daily_ds
    # Call sma directly on the Dataset (bypasses the DataArray-without-parent issue)
    from src.data.ast.functions import sma
    result = sma(ds, 5)
    assert isinstance(result, xr.Dataset)
    # Same total number of elements per variable
    assert result["var_1"].size == ds["var_1"].size
    # After warmup (window=5) there should be some finite values
    flat = result.dt.to_time_indexed()["var_1"].values
    finite_count = int(np.isfinite(flat).sum())
    assert finite_count > 0, "Expected finite values after warmup"
