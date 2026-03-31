"""
WRDS-backed validation tests for FF3 data loading and merging.
"""

from pathlib import Path

import numpy as np
import pytest

from src.data.managers.data_manager import DataManager
from src.pipeline.data_handler.merge import DatasetMerger, MergeSpec, TimeAlignment


pytestmark = pytest.mark.wrds

CRSP_MONTHLY_PATH = Path("/wrds/crsp/sasdata/a_stock/msf.sas7bdat")
CRSP_NAMES_PATH = Path("/wrds/crsp/sasdata/a_stock/msenames.sas7bdat")
CRSP_DELIST_PATH = Path("/wrds/crsp/sasdata/a_stock/msedelist.sas7bdat")
COMPUSTAT_ANNUAL_PATH = Path("/wrds/comp/sasdata/d_na/funda.sas7bdat")
CCM_LINK_PATH = Path("/wrds/crsp/sasdata/a_ccm/ccmxpf_linktable.sas7bdat")

WRDS_REQUIRED_PATHS = [
    CRSP_MONTHLY_PATH,
    CRSP_NAMES_PATH,
    CRSP_DELIST_PATH,
    COMPUSTAT_ANNUAL_PATH,
    CCM_LINK_PATH,
]
WRDS_AVAILABLE = all(path.exists() for path in WRDS_REQUIRED_PATHS)


@pytest.fixture(autouse=True)
def _require_wrds_selection(request):
    if not WRDS_AVAILABLE:
        pytest.skip("Local WRDS SAS files not available")

    markexpr = request.config.option.markexpr or ""
    if "wrds" not in markexpr:
        pytest.skip("WRDS tests run only when selected with -m wrds")


def _load_crsp_monthly():
    manager = DataManager()
    request = [
        {
            "data_path": "wrds/equity/crsp",
            "config": {
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
                "frequency": "M",
                "filters": {
                    "shrcd__in": [10, 11],
                    "exchcd__in": [1, 2, 3],
                },
                "external_tables": [
                    {
                        "path": str(CRSP_DELIST_PATH),
                        "type": "replace",
                        "on": "permno",
                        "time_column": "dlstdt",
                        "from_column": "dlret",
                        "to_column": "ret",
                    },
                    {
                        "path": str(CRSP_NAMES_PATH),
                        "type": "lookup",
                        "on": "permno",
                        "columns": ["exchcd", "shrcd"],
                    },
                ],
                "processors": {
                    "transforms": [
                        {"type": "set_coordinates", "coord_type": "permco"},
                        {"type": "fix_market_equity"},
                    ]
                },
            },
        }
    ]
    return manager._get_raw_data(request)["wrds/equity/crsp"]


def _load_compustat_annual():
    manager = DataManager()
    request = [
        {
            "data_path": "wrds/equity/compustat",
            "config": {
                "start_date": "2019-01-01",
                "end_date": "2020-12-31",
                "frequency": "Y",
                "columns_to_read": [
                    "gvkey",
                    "datadate",
                    "seq",
                    "txditc",
                    "pstkrv",
                    "pstkl",
                    "pstk",
                    "indfmt",
                    "datafmt",
                    "popsrc",
                    "consol",
                ],
                "filters": {
                    "indfmt": "INDL",
                    "datafmt": "STD",
                    "popsrc": "D",
                    "consol": "C",
                },
                "processors": {
                    "transforms": [
                        {"type": "preferred_stock"},
                    ]
                },
            },
        }
    ]
    return manager._get_raw_data(request)["wrds/equity/compustat"]


def test_load_crsp_monthly():
    ds = _load_crsp_monthly()

    assert ds.sizes["asset"] > 0
    assert ds.sizes["hour"] == 1
    assert "ret" in ds.data_vars
    assert "me" in ds.data_vars
    assert "exchcd" in ds.data_vars
    assert "permco" in ds.coords


def test_load_compustat_annual():
    ds = _load_compustat_annual()

    assert ds.sizes["asset"] > 0
    assert "seq" in ds.data_vars
    assert "txditc" in ds.data_vars
    assert "ps" in ds.data_vars
    assert np.isfinite(ds["ps"].values).any()


def test_merge_crsp_compustat():
    crsp = _load_crsp_monthly()
    compustat = _load_compustat_annual()

    merged = DatasetMerger().merge(
        left=crsp,
        right=compustat,
        spec=MergeSpec(
            right_name="compustat",
            on="asset",
            time_alignment=TimeAlignment.AS_OF,
            time_offset_months=6,
            variables=["seq", "txditc", "ps"],
        ),
    )

    assert merged.sizes["asset"] == crsp.sizes["asset"]
    assert "seq" in merged.data_vars
    assert "txditc" in merged.data_vars
    assert "ps" in merged.data_vars
    assert np.isfinite(merged["seq"].values).any()


def test_ff3_factors_vs_published():
    pytest.skip("Published FF benchmark source is not configured locally yet")
