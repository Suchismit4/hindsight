# Known Gaps

Missing functionality in Hindsight that blocks concrete quantitative research workflows, based on current source capabilities.

---

## Gap Table

| Gap | Workflow It Blocks | Suggested Location | Status |
|-----|--------------------|--------------------|--------|
| No fiscal year grouping or month-of-year filtering | June/December portfolio rebalancing, calendar-conditional formulas | `src/data/core/struct.py` (`.dt` accessor) | OPEN |
| No multi-output formulas | Bollinger Bands (3 outputs: upper, middle, lower), MACD (signal, histogram, line) | `src/data/ast/nodes.py`, `manager.py` | OPEN |
| `Rolling.std()` raises `NotImplementedError` | Bollinger Bands, volatility measures, risk metrics | `src/data/core/rolling.py` | OPEN |
| Processors use `print()` warnings instead of `logging` | No structured error tracking, difficult to capture pipeline diagnostics | `src/pipeline/data_handler/processors.py` | OPEN |
| `CacheManager` cannot subset cached data by date range | Loading less data than what is cached requires a full re-fetch | `src/data/core/cache.py` | OPEN |
| `evaluate_bulk` cannot handle inline `expression:` from YAML | Inline formula definitions in pipeline specs require manual `add_formula()` calls | `src/data/ast/manager.py` | OPEN |
| No IC / ICIR / factor decay / turnover metrics | Alpha research signal evaluation, factor quality diagnostics | `src/data/ast/functions.py` (register as formula functions) or `src/pipeline/data_handler/processors.py` | OPEN |
| No mean-variance or risk-parity portfolio optimization | Portfolio optimization workflows beyond simple long-short sorts | `src/pipeline/data_handler/processors.py` (new processor) | OPEN |
| No announcement-date offset or event-time alignment | Earnings-based strategies, event studies, point-in-time earnings data | `src/data/core/struct.py` (`.dt` accessor) | OPEN |

---

## Resolved Gaps (for reference)

These gaps were previously identified and have been implemented:

| Gap | Resolution | Location |
|-----|------------|----------|
| No cross-sectional rank/quantile/percentile | `cs_rank`, `cs_quantile` registered functions | `src/data/ast/functions.py` |
| No groupby-reduce across asset dim | `cs_demean` registered function | `src/data/ast/functions.py` |
| No where/mask/conditional in formula grammar | `where`, `gt`, `lt`, `ge`, `le`, `eq`, `nan_const` functions | `src/data/ast/functions.py` |
| No breakpoint/bucket assignment | `assign_bucket` function + `CrossSectionalSort` processor | `src/data/ast/functions.py`, `src/pipeline/data_handler/processors.py` |
| No value-weighted mean across assets | `PortfolioReturns` processor | `src/pipeline/data_handler/processors.py` |
| No long-short spread construction | `FactorSpread` processor | `src/pipeline/data_handler/processors.py` |
| No comparison operators in formulas | `gt`, `lt`, `ge`, `le`, `eq` registered functions | `src/data/ast/functions.py` |

---

## See also

- [QUANT_PRIMITIVES.md](./QUANT_PRIMITIVES.md) — The six core quantitative primitives that resolved many earlier gaps
- [PIPELINE_SYSTEM.md](./PIPELINE_SYSTEM.md) — Processor stage model where new gap-filling processors should be added
- [FORMULA_SYSTEM.md](./FORMULA_SYSTEM.md) — Formula function registry where new gap-filling functions should be registered
