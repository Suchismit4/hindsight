"""
Tests for cross-sectional operations:
  - cs_rank, cs_quantile, cs_demean (ast/functions.py)
  - gt, lt, ge, le, eq, nan_const, where (ast/functions.py)
  - assign_bucket (ast/functions.py)
  - CrossSectionalSort with custom quantiles (pipeline/data_handler/processors.py)
  - ProcessorRegistry wiring for CrossSectionalSort and PortfolioReturns
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.ast.functions import (
    assign_bucket,
    cs_demean,
    cs_quantile,
    cs_rank,
    eq,
    ge,
    get_registered_functions,
    gt,
    le,
    lt,
    nan_const,
    where,
)
from src.data.ast.manager import FormulaManager
from src.data.loaders.table import from_table
from src.data.core.types import FrequencyType
from src.pipeline.data_handler.processors import CrossSectionalSort, FactorSpread, PortfolioReturns
from src.pipeline.spec.processor_registry import ProcessorRegistry


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_monthly_da(values, assets=None, n_months=1):
    """Build a minimal (year=1, month=n_months, day=1, hour=1, asset=N) DataArray."""
    values = np.asarray(values, dtype=np.float64)
    if assets is None:
        assets = list(range(len(values)))
    n_assets = len(assets)
    # broadcast values across months
    data = np.broadcast_to(values, (1, n_months, 1, 1, n_assets)).copy()
    return xr.DataArray(
        data,
        dims=['year', 'month', 'day', 'hour', 'asset'],
        coords={
            'year': [2020],
            'month': list(range(1, n_months + 1)),
            'day': [1],
            'hour': [0],
            'asset': assets,
        },
    )


def _make_ds_from_values(signal_values, scope_values=None, n_months=1):
    """Build a minimal xr.Dataset suitable for CrossSectionalSort."""
    n_assets = len(signal_values)
    assets = list(range(n_assets))
    data = np.broadcast_to(
        np.array(signal_values, dtype=np.float64),
        (1, n_months, 1, 1, n_assets),
    ).copy()
    ds = xr.Dataset(
        {'signal': xr.DataArray(
            data,
            dims=['year', 'month', 'day', 'hour', 'asset'],
            coords={'year': [2020], 'month': list(range(1, n_months + 1)),
                    'day': [1], 'hour': [0], 'asset': assets},
        )}
    )
    if scope_values is not None:
        scope = np.broadcast_to(
            np.array(scope_values, dtype=np.float64),
            (1, n_months, 1, 1, n_assets),
        ).copy()
        ds['scope'] = xr.DataArray(
            scope,
            dims=['year', 'month', 'day', 'hour', 'asset'],
            coords=ds.coords,
        )
    return ds


# ===========================================================================
# cs_rank
# ===========================================================================

class TestCsRank:
    def test_basic_rank(self):
        """[10, 30, 20, 40, 50] → [0.0, 0.5, 0.25, 0.75, 1.0]"""
        da = _make_monthly_da([10, 30, 20, 40, 50])
        result = cs_rank(da)
        expected = np.array([0.0, 0.5, 0.25, 0.75, 1.0])
        np.testing.assert_allclose(result.values[0, 0, 0, 0, :], expected)

    def test_nan_assets_get_nan_rank(self):
        """NaN assets are excluded from ranking; others re-ranked."""
        da = _make_monthly_da([10, np.nan, 20, np.nan, 50])
        result = cs_rank(da)
        vals = result.values[0, 0, 0, 0, :]
        assert np.isnan(vals[1])
        assert np.isnan(vals[3])
        # remaining [10, 20, 50] → ranks [0.0, 0.5, 1.0]
        np.testing.assert_allclose(vals[[0, 2, 4]], [0.0, 0.5, 1.0])

    def test_all_nan_returns_all_nan(self):
        da = _make_monthly_da([np.nan, np.nan, np.nan])
        result = cs_rank(da)
        assert np.all(np.isnan(result.values))

    def test_single_valid_returns_all_nan(self):
        """Fewer than 2 valid values → all NaN."""
        da = _make_monthly_da([np.nan, np.nan, 10.0, np.nan, np.nan])
        result = cs_rank(da)
        assert np.all(np.isnan(result.values))

    def test_preserves_dims(self):
        da = _make_monthly_da([1, 2, 3, 4, 5])
        result = cs_rank(da)
        assert result.dims == da.dims
        assert result.shape == da.shape

    def test_multi_month_independent(self):
        """Each time slice ranked independently."""
        da = _make_monthly_da([10, 30, 20, 40, 50], n_months=2)
        result = cs_rank(da)
        # Both months have the same values → same ranks
        np.testing.assert_allclose(result.values[0, 0, 0, 0, :],
                                   result.values[0, 1, 0, 0, :])

    def test_registered(self):
        assert 'cs_rank' in get_registered_functions()


# ===========================================================================
# cs_quantile
# ===========================================================================

class TestCsQuantile:
    def test_median(self):
        da = _make_monthly_da([10, 20, 30, 40, 50])
        result = cs_quantile(da, 0.5)
        assert 'asset' not in result.dims
        assert float(result.values[0, 0, 0, 0]) == pytest.approx(30.0)

    def test_30th_percentile(self):
        da = _make_monthly_da([10, 20, 30, 40, 50])
        result = cs_quantile(da, 0.3)
        expected = float(np.nanquantile([10, 20, 30, 40, 50], 0.3))
        assert float(result.values[0, 0, 0, 0]) == pytest.approx(expected)

    def test_skipna(self):
        da = _make_monthly_da([10, np.nan, 30, np.nan, 50])
        result = cs_quantile(da, 0.5)
        # median of [10, 30, 50]
        expected = float(np.nanquantile([10, 30, 50], 0.5))
        assert float(result.values[0, 0, 0, 0]) == pytest.approx(expected)

    def test_all_nan_returns_nan(self):
        da = _make_monthly_da([np.nan, np.nan, np.nan])
        result = cs_quantile(da, 0.5)
        assert np.isnan(float(result.values.flat[0]))

    def test_registered(self):
        assert 'cs_quantile' in get_registered_functions()


# ===========================================================================
# cs_demean
# ===========================================================================

class TestCsDemean:
    def test_basic_demean(self):
        da = _make_monthly_da([10, 20, 30])
        result = cs_demean(da)
        np.testing.assert_allclose(result.values[0, 0, 0, 0, :], [-10, 0, 10])

    def test_nan_preserved(self):
        da = _make_monthly_da([10, np.nan, 30])
        result = cs_demean(da)
        vals = result.values[0, 0, 0, 0, :]
        # mean of [10, 30] = 20
        assert vals[0] == pytest.approx(-10.0)
        assert np.isnan(vals[1])
        assert vals[2] == pytest.approx(10.0)

    def test_preserves_dims(self):
        da = _make_monthly_da([1, 2, 3, 4, 5])
        result = cs_demean(da)
        assert result.dims == da.dims
        assert result.shape == da.shape

    def test_registered(self):
        assert 'cs_demean' in get_registered_functions()


# ===========================================================================
# Comparison helpers
# ===========================================================================

class TestComparisonHelpers:
    def _da(self, values):
        return _make_monthly_da(values)

    def test_gt(self):
        result = gt(self._da([1, 2, 3]), 2)
        expected = np.array([False, False, True])
        np.testing.assert_array_equal(result.values[0, 0, 0, 0, :], expected)

    def test_lt(self):
        result = lt(self._da([1, 2, 3]), 2)
        np.testing.assert_array_equal(result.values[0, 0, 0, 0, :],
                                      [True, False, False])

    def test_ge(self):
        result = ge(self._da([1, 2, 3]), 2)
        np.testing.assert_array_equal(result.values[0, 0, 0, 0, :],
                                      [False, True, True])

    def test_le(self):
        result = le(self._da([1, 2, 3]), 2)
        np.testing.assert_array_equal(result.values[0, 0, 0, 0, :],
                                      [True, True, False])

    def test_eq(self):
        result = eq(self._da([1, 2, 3]), 2)
        np.testing.assert_array_equal(result.values[0, 0, 0, 0, :],
                                      [False, True, False])

    def test_nan_const(self):
        val = nan_const()
        assert np.isnan(val)

    def test_all_registered(self):
        funcs = get_registered_functions()
        for name in ('gt', 'lt', 'ge', 'le', 'eq', 'nan_const'):
            assert name in funcs, f"{name} not registered"


# ===========================================================================
# where
# ===========================================================================

class TestWhere:
    def _da(self, values):
        return _make_monthly_da(values)

    def test_basic_where(self):
        data = self._da([-1, 2, -3, 4, 0])
        result = where(gt(data, 0), data, 0)
        np.testing.assert_allclose(result.values[0, 0, 0, 0, :],
                                   [0, 2, 0, 4, 0])

    def test_where_nan_replacement(self):
        data = self._da([-1, 2, -3])
        result = where(gt(data, 0), data, nan_const())
        vals = result.values[0, 0, 0, 0, :]
        assert np.isnan(vals[0])
        assert vals[1] == pytest.approx(2.0)
        assert np.isnan(vals[2])

    def test_where_broadcast_scalar_condition(self):
        """Condition with no asset dim broadcasts correctly."""
        data = self._da([10, 20, 30, 40, 50])
        # threshold = median (scalar, no asset dim)
        threshold = cs_quantile(data, 0.5)
        result = where(gt(data, threshold), 1.0, 0.0)
        vals = result.values[0, 0, 0, 0, :]
        # median is 30; assets > 30 should get 1.0
        assert vals[3] == pytest.approx(1.0)  # 40
        assert vals[4] == pytest.approx(1.0)  # 50
        assert vals[0] == pytest.approx(0.0)  # 10

    def test_registered(self):
        assert 'where' in get_registered_functions()


# ===========================================================================
# assign_bucket
# ===========================================================================

class TestAssignBucket:
    def _da(self, values):
        return _make_monthly_da(values)

    def test_two_bins(self):
        """bp=25 → 0 for <=25, 1 for >25"""
        data = self._da([10, 20, 30, 40, 50])
        result = assign_bucket(data, 25)
        expected = np.array([0, 0, 1, 1, 1], dtype=np.float64)
        np.testing.assert_allclose(result.values[0, 0, 0, 0, :], expected)

    def test_three_bins(self):
        """bp1=30, bp2=60 → 0 below 30, 1 between, 2 above 60"""
        data = self._da([10, 25, 35, 50, 80])
        result = assign_bucket(data, 30, 60)
        expected = np.array([0, 0, 1, 1, 2], dtype=np.float64)
        np.testing.assert_allclose(result.values[0, 0, 0, 0, :], expected)

    def test_four_bins(self):
        """Three breakpoints → four bins"""
        data = self._da([5, 15, 25, 35])
        result = assign_bucket(data, 10, 20, 30)
        expected = np.array([0, 1, 2, 3], dtype=np.float64)
        np.testing.assert_allclose(result.values[0, 0, 0, 0, :], expected)

    def test_nan_data_propagates(self):
        data = self._da([10, np.nan, 30, 40, np.nan])
        result = assign_bucket(data, 25)
        vals = result.values[0, 0, 0, 0, :]
        assert vals[0] == pytest.approx(0.0)
        assert np.isnan(vals[1])
        assert vals[2] == pytest.approx(1.0)
        assert vals[3] == pytest.approx(1.0)
        assert np.isnan(vals[4])

    def test_broadcast_scalar_breakpoint(self):
        """Breakpoint from cs_quantile (no asset dim) broadcasts over data."""
        data = self._da([10, 20, 30, 40, 50])
        bp = cs_quantile(data, 0.5)   # 30.0, no asset dim
        result = assign_bucket(data, bp)
        vals = result.values[0, 0, 0, 0, :]
        # 10, 20, 30 are <= 30, so bucket 0; 40, 50 are > 30, bucket 1
        assert vals[0] == pytest.approx(0.0)
        assert vals[1] == pytest.approx(0.0)
        assert vals[2] == pytest.approx(0.0)  # 30 not > 30
        assert vals[3] == pytest.approx(1.0)
        assert vals[4] == pytest.approx(1.0)

    def test_preserves_dims(self):
        data = self._da([1, 2, 3, 4, 5])
        result = assign_bucket(data, 3)
        assert result.dims == data.dims
        assert result.shape == data.shape

    def test_registered(self):
        assert 'assign_bucket' in get_registered_functions()


# ===========================================================================
# CrossSectionalSort – custom quantiles
# ===========================================================================

class TestCrossSectionalSortCustomQuantiles:
    def test_equal_bins_unchanged(self):
        """n_bins=2 without custom quantiles → median split."""
        ds = _make_ds_from_values([10, 20, 30, 40, 50])
        proc = CrossSectionalSort(signal='signal', n_bins=2, name='test')
        out = proc.transform(ds)
        vals = out['signal_port'].values[0, 0, 0, 0, :]
        # median of [10,20,30,40,50]=30; 10,20,30 → bin 0, 40,50 → bin 1
        assert vals[3] == pytest.approx(1.0)  # 40
        assert vals[0] == pytest.approx(0.0)  # 10

    def test_custom_quantiles_30_70(self):
        """quantiles=[0.3, 0.7] → 3 bins at 30th and 70th percentile."""
        values = list(range(10, 110, 10))  # 10..100, 10 assets
        ds = _make_ds_from_values(values)
        proc = CrossSectionalSort(
            signal='signal', n_bins=0,
            quantiles=[0.3, 0.7], name='test'
        )
        out = proc.transform(ds)
        vals = out['signal_port'].values[0, 0, 0, 0, :]
        # bp30 ≈ 37, bp70 ≈ 73
        # assets [10,20,30] → bin 0; [40,50,60,70] → bin 1; [80,90,100] → bin 2
        assert vals[0] == pytest.approx(0.0)   # 10
        assert vals[4] == pytest.approx(1.0)   # 50
        assert vals[9] == pytest.approx(2.0)   # 100

    def test_scope_uses_subset_for_breakpoints(self):
        """Breakpoints computed on scope=True assets, assigned to all.

        CrossSectionalSort uses np.digitize (left-closed intervals), so a
        value *equal to* the median breakpoint falls into the upper bin.
        """
        # scope: first 5 are NYSE (value 1), last 5 are not (value 0)
        signal = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        scope  = [1,  1,  1,  1,  1,  0,  0,  0,  0,  0]
        ds = _make_ds_from_values(signal, scope_values=scope)
        proc = CrossSectionalSort(
            signal='signal', n_bins=2,
            scope='scope', name='test'
        )
        out = proc.transform(ds)
        vals = out['signal_port'].values[0, 0, 0, 0, :]
        # NYSE median of [10,20,30,40,50] = 30.
        # np.digitize with left-closed bins: 30 >= 30 → upper bin (1).
        assert vals[0] == pytest.approx(0.0)   # 10 → bin 0
        assert vals[1] == pytest.approx(0.0)   # 20 → bin 0
        assert vals[2] == pytest.approx(1.0)   # 30 → bin 1 (at boundary, upper)
        assert vals[3] == pytest.approx(1.0)   # 40 → bin 1
        assert vals[9] == pytest.approx(1.0)   # 100 (non-NYSE, still binned)

    def test_nan_signal_gets_nan_bucket(self):
        ds = _make_ds_from_values([10, np.nan, 30, np.nan, 50])
        proc = CrossSectionalSort(signal='signal', n_bins=2, name='test')
        out = proc.transform(ds)
        vals = out['signal_port'].values[0, 0, 0, 0, :]
        assert np.isnan(vals[1])
        assert np.isnan(vals[3])


# ===========================================================================
# ProcessorRegistry – wiring
# ===========================================================================

class TestProcessorRegistry:
    def test_cross_sectional_sort_registered(self):
        proc = ProcessorRegistry.create_processor(
            {'type': 'cross_sectional_sort', 'signal': 'me', 'n_bins': 2, 'name': 'sz'}
        )
        assert isinstance(proc, CrossSectionalSort)
        assert proc.signal == 'me'
        assert proc.n_bins == 2

    def test_portfolio_returns_registered(self):
        proc = ProcessorRegistry.create_processor(
            {'type': 'portfolio_returns', 'groupby': ['sz_port'], 'returns_var': 'ret', 'name': 'pr'}
        )
        assert isinstance(proc, PortfolioReturns)
        assert proc.returns_var == 'ret'

    def test_cross_sectional_sort_with_quantiles_from_registry(self):
        proc = ProcessorRegistry.create_processor({
            'type': 'cross_sectional_sort',
            'signal': 'bm',
            'n_bins': 0,
            'quantiles': [0.3, 0.7],
            'name': 'bm_sort',
        })
        assert isinstance(proc, CrossSectionalSort)
        assert proc.quantiles == [0.3, 0.7]

    def test_available_types_include_new_processors(self):
        types = ProcessorRegistry.get_available_types()
        assert 'cross_sectional_sort' in types
        assert 'portfolio_returns' in types


# ===========================================================================
# FormulaManager – YAML definitions
# ===========================================================================

class TestFormulaManagerCrossSection:
    def _make_ds(self, n_assets=5, n_months=3):
        np.random.seed(42)
        assets = list(range(n_assets))
        me_data = np.random.uniform(1e6, 1e9, (1, n_months, 1, 1, n_assets))
        exchcd_data = np.zeros((1, n_months, 1, 1, n_assets))
        exchcd_data[..., :3] = 1  # first 3 assets are NYSE
        ret_data = np.random.normal(0, 0.05, (1, n_months, 1, 1, n_assets))
        coords = {'year': [2020], 'month': list(range(1, n_months + 1)),
                  'day': [1], 'hour': [0], 'asset': assets}
        dims = ['year', 'month', 'day', 'hour', 'asset']
        return xr.Dataset({
            'me': xr.DataArray(me_data, dims=dims, coords=coords),
            'exchcd': xr.DataArray(exchcd_data, dims=dims, coords=coords),
            'ret': xr.DataArray(ret_data, dims=dims, coords=coords),
        })

    def test_yaml_definitions_load(self):
        mgr = FormulaManager()
        # cross_sectional.yaml should be loaded automatically
        assert 'sz_bucket' in mgr.formulas
        assert 'bm_bucket' in mgr.formulas
        assert 'is_nyse' in mgr.formulas

    def test_is_nyse_formula(self):
        mgr = FormulaManager()
        ds = self._make_ds()
        from src.data.ast.functions import get_function_context
        ctx = {'_dataset': ds, 'exchcd': 'exchcd', **get_function_context()}
        result = mgr.evaluate('is_nyse', ctx)
        # result may be (DataArray, Dataset) tuple or plain DataArray depending on version
        if isinstance(result, tuple):
            result = result[0]
        # first 3 assets have exchcd=1 → True (1), last 2 have exchcd=0 → False (0)
        vals = result.values[0, 0, 0, 0, :]
        assert bool(vals[0])   # NYSE
        assert not bool(vals[4])  # non-NYSE

    def test_me_nyse_formula(self):
        mgr = FormulaManager()
        ds = self._make_ds()
        from src.data.ast.functions import get_function_context
        ctx = {'_dataset': ds, 'me': 'me', 'exchcd': 'exchcd', **get_function_context()}
        result = mgr.evaluate('me_nyse', ctx)
        if isinstance(result, tuple):
            result = result[0]
        vals = result.values[0, 0, 0, 0, :]
        # Non-NYSE assets should be NaN
        assert not np.isnan(vals[0])
        assert np.isnan(vals[4])


# ===========================================================================
# Integration: FF3 size sort (partial)
# ===========================================================================

class TestFF3SizeSort:
    def test_nyse_median_split(self):
        """~50% of NYSE assets in each bin; non-NYSE also binned."""
        np.random.seed(0)
        n_assets = 200
        n_months = 12
        assets = list(range(n_assets))
        dims = ['year', 'month', 'day', 'hour', 'asset']
        coords = {'year': [2020], 'month': list(range(1, n_months + 1)),
                  'day': [1], 'hour': [0], 'asset': assets}
        me_data = np.random.uniform(1e6, 1e9, (1, n_months, 1, 1, n_assets))
        # First 100 assets are NYSE
        exchcd_data = np.zeros((1, n_months, 1, 1, n_assets))
        exchcd_data[..., :100] = 1

        ds = xr.Dataset({
            'me': xr.DataArray(me_data, dims=dims, coords=coords),
            'exchcd': xr.DataArray(exchcd_data, dims=dims, coords=coords),
        })

        # Step 1: NYSE-only ME
        me_nyse = where(eq(ds['exchcd'], 1), ds['me'], nan_const())
        # Step 2: breakpoint
        bp = cs_quantile(me_nyse, 0.5)
        # Step 3: bucket assignment
        sz = assign_bucket(ds['me'], bp)

        # ~50% of NYSE assets should be in bin 0 and 1 each month
        for m in range(n_months):
            sz_month = sz.values[0, m, 0, 0, :]
            nyse_buckets = sz_month[:100]
            n_small = int(np.sum(nyse_buckets == 0))
            n_big = int(np.sum(nyse_buckets == 1))
            assert abs(n_small - 50) <= 10, f"Month {m}: n_small={n_small}"
            assert abs(n_big - 50) <= 10, f"Month {m}: n_big={n_big}"

        # Non-NYSE assets are also assigned
        non_nyse_buckets = sz.values[0, 0, 0, 0, 100:]
        assert not np.any(np.isnan(non_nyse_buckets))

        # Output has correct dims
        assert sz.dims == ('year', 'month', 'day', 'hour', 'asset')

    def test_crosssectionalsort_custom_quantiles_integration(self):
        """CrossSectionalSort with quantiles=[0.3, 0.7] via ProcessorRegistry."""
        np.random.seed(1)
        n_assets = 100
        n_months = 6
        assets = list(range(n_assets))
        dims = ['year', 'month', 'day', 'hour', 'asset']
        coords = {'year': [2020], 'month': list(range(1, n_months + 1)),
                  'day': [1], 'hour': [0], 'asset': assets}
        me_data = np.random.uniform(0, 1, (1, n_months, 1, 1, n_assets))
        is_nyse = np.zeros((1, n_months, 1, 1, n_assets))
        is_nyse[..., :60] = 1  # first 60 are NYSE

        ds = xr.Dataset({
            'me': xr.DataArray(me_data, dims=dims, coords=coords),
            'is_nyse': xr.DataArray(is_nyse, dims=dims, coords=coords),
        })

        proc = ProcessorRegistry.create_processor({
            'type': 'cross_sectional_sort',
            'signal': 'me',
            'n_bins': 0,
            'quantiles': [0.3, 0.7],
            'scope': 'is_nyse',
            'name': 'bm_sort',
        })
        out = proc.transform(ds)

        assert 'me_port' in out
        vals = out['me_port'].values[0, 0, 0, 0, :]
        # Should have 3 distinct bins: 0, 1, 2
        unique_bins = set(float(v) for v in vals if not np.isnan(v))
        assert unique_bins == {0.0, 1.0, 2.0}


# ---------------------------------------------------------------------------
# TestFactorSpreadProcessor
# ---------------------------------------------------------------------------

def _build_ff3_dataset(n_assets=100, n_months=12, seed=42):
    """Build a simulated dataset with me, beme, ret for FF3 testing."""
    rng = np.random.default_rng(seed)

    me_vals = rng.lognormal(mean=3.0, sigma=1.0, size=(1, n_months, 1, 1, n_assets))
    beme_vals = rng.lognormal(mean=0.0, sigma=0.8, size=(1, n_months, 1, 1, n_assets))
    ret_vals = rng.normal(loc=0.005, scale=0.05, size=(1, n_months, 1, 1, n_assets))

    coords = {
        'year': [2020],
        'month': list(range(1, n_months + 1)),
        'day': [1],
        'hour': [0],
        'asset': list(range(n_assets)),
    }
    dims = ['year', 'month', 'day', 'hour', 'asset']
    return xr.Dataset({
        'me': xr.DataArray(me_vals, dims=dims, coords=coords),
        'beme': xr.DataArray(beme_vals, dims=dims, coords=coords),
        'ret': xr.DataArray(ret_vals, dims=dims, coords=coords),
    })


class TestFactorSpreadProcessor:
    """Tests for FactorSpread processor: long-short factor construction."""

    def _run_full_pipeline(self, ds):
        """Run CrossSectionalSort x2 → PortfolioReturns → FactorSpread."""
        # Size sort: 2 bins, median breakpoint
        ds = CrossSectionalSort(signal='me', n_bins=2, name='sz').transform(ds)
        # B/M sort: 3 bins, 30/70 breakpoints (FF standard)
        ds = CrossSectionalSort(
            signal='beme', n_bins=3, quantiles=[0.3, 0.7], name='bm'
        ).transform(ds)
        # Portfolio returns
        ds = PortfolioReturns(
            groupby=['me_port', 'beme_port'], returns_var='ret', weights_var='me',
            name='port'
        ).transform(ds)
        # Factor spreads
        proc = FactorSpread(
            source='port_ret_me_port_beme_port',
            factors={
                'SMB': {'long': {'me_port_dim': 0}, 'short': {'me_port_dim': 1},
                        'average_over': 'beme_port_dim'},
                'HML': {'long': {'beme_port_dim': 2}, 'short': {'beme_port_dim': 0},
                        'average_over': 'me_port_dim'},
            },
            name='ff_factors',
        )
        return proc.transform(ds)

    def test_smb_hml_exist(self):
        ds = _build_ff3_dataset()
        out = self._run_full_pipeline(ds)
        assert 'SMB' in out.data_vars
        assert 'HML' in out.data_vars

    def test_output_has_only_time_dims(self):
        ds = _build_ff3_dataset()
        out = self._run_full_pipeline(ds)
        time_dims = {'year', 'month', 'day', 'hour'}
        for factor in ('SMB', 'HML'):
            factor_dims = set(out[factor].dims)
            assert factor_dims == time_dims, (
                f"{factor} has dims {factor_dims}, expected {time_dims}"
            )

    def test_not_all_nan(self):
        ds = _build_ff3_dataset()
        out = self._run_full_pipeline(ds)
        assert not np.all(np.isnan(out['SMB'].values)), "SMB is all NaN"
        assert not np.all(np.isnan(out['HML'].values)), "HML is all NaN"

    def test_smb_matches_manual_computation(self):
        """SMB should equal mean(small portfolios) - mean(big portfolios)."""
        ds = _build_ff3_dataset()
        out = self._run_full_pipeline(ds)

        port_ret = out['port_ret_me_port_beme_port']
        # Small (me_port_dim=0), Big (me_port_dim=1)
        small_avg = port_ret.sel(me_port_dim=0).mean(dim='beme_port_dim', skipna=True)
        big_avg = port_ret.sel(me_port_dim=1).mean(dim='beme_port_dim', skipna=True)
        expected_smb = small_avg - big_avg

        np.testing.assert_allclose(
            out['SMB'].values, expected_smb.values, rtol=1e-6, equal_nan=True
        )

    def test_hml_matches_manual_computation(self):
        """HML should equal mean(high B/M portfolios) - mean(low B/M portfolios)."""
        ds = _build_ff3_dataset()
        out = self._run_full_pipeline(ds)

        port_ret = out['port_ret_me_port_beme_port']
        # High B/M (beme_port_dim=2), Low B/M (beme_port_dim=0)
        high_avg = port_ret.sel(beme_port_dim=2).mean(dim='me_port_dim', skipna=True)
        low_avg = port_ret.sel(beme_port_dim=0).mean(dim='me_port_dim', skipna=True)
        expected_hml = high_avg - low_avg

        np.testing.assert_allclose(
            out['HML'].values, expected_hml.values, rtol=1e-6, equal_nan=True
        )

    def test_missing_source_raises(self):
        ds = xr.Dataset({'ret': xr.DataArray(np.ones((1, 2, 1, 1, 3)),
                                              dims=['year', 'month', 'day', 'hour', 'asset'])})
        proc = FactorSpread(source='nonexistent', factors={}, name='x')
        with pytest.raises(KeyError, match="nonexistent"):
            proc.transform(ds)

    def test_registry_instantiation(self):
        """FactorSpread can be instantiated via ProcessorRegistry."""
        config = {
            'type': 'factor_spread',
            'source': 'port_ret_me_port_beme_port',
            'factors': {
                'SMB': {'long': {'me_port_dim': 0}, 'short': {'me_port_dim': 1},
                        'average_over': 'beme_port_dim'},
            },
            'name': 'ff_factors',
        }
        proc = ProcessorRegistry.create_processor(config)
        assert isinstance(proc, FactorSpread)
        assert proc.source == 'port_ret_me_port_beme_port'

    def test_aliases_sort_and_port_ret(self):
        """'sort' and 'port_ret' aliases resolve to the correct classes."""
        from src.pipeline.data_handler.processors import CrossSectionalSort, PortfolioReturns
        sort_proc = ProcessorRegistry.create_processor(
            {'type': 'sort', 'signal': 'me', 'n_bins': 2, 'name': 'sz'}
        )
        assert isinstance(sort_proc, CrossSectionalSort)

        port_proc = ProcessorRegistry.create_processor(
            {'type': 'port_ret', 'groupby': ['me_port'], 'returns_var': 'ret',
             'weights_var': 'me', 'name': 'p'}
        )
        assert isinstance(port_proc, PortfolioReturns)
