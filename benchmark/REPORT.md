# AusEFlux New Pipeline vs. Reference — Benchmark Report

**Date:** 2026-06-04
**New pipeline:** `/home/prl900/Downloads/auseflux_pipeline/` (step1/step2/step3)
**Reference:** `/home/prl900/github/AusEFlux/` (Burton et al. 2023) + published v2.1 product on NCI THREDDS
**Test region:** SE Australia, 600×600 px @ 500 m (148.00–151.00°E, 33.00–36.00°S), target year **2020**
**Climatology harmonised:** 2003–2022 (full, for the step2 anomaly baseline)

---

## 1. Summary

The new pipeline was run end-to-end on a small bbox using the **reference's own trained ensemble**
(30 models/flux = 15 LightGBM + 15 RandomForest from `gadi:xc0/.../models/ensemble/v2.1/`) and the
reference's static rasters, then compared band-by-band against the **published v2.1 product**
(OPeNDAP-subset from NCI THREDDS). The new output lands on the **identical canonical 500 m grid** as
the product (verified: max coordinate diff = 0.0°), so the comparison is exact 1:1 with no regridding.

The benchmark surfaced and fixed **two real bugs** (one a blocker, one consequential) and one
defensive guard. **After the fixes, all four fluxes reproduce the published product at R² ≈ 0.98–0.997
with near-zero bias** — strong numerical equivalence. The residual ≤3% is inherent to the
reimplementation (reprojection backend + quantile method).

---

## 2. Final results (after fixes) — new vs published v2.1, 2020

| flux | band | R² | RMSE | MAE | mean bias | mask Jaccard |
|---|---|---|---|---|---|---|
| GPP | 25th | 0.9964 | 3.43 | – | −0.97 | 0.9987 |
| GPP | median | **0.9967** | 3.35 | – | −0.83 | 0.9987 |
| GPP | 75th | 0.9961 | 3.77 | – | −1.65 | 0.9987 |
| ER | 25th | 0.9861 | 3.97 | – | −1.00 | 0.9987 |
| ER | median | **0.9848** | 4.15 | – | −0.93 | 0.9987 |
| ER | 75th | 0.9756 | 5.31 | – | −2.03 | 0.9987 |
| NEE | 25th | 0.9967 | 1.95 | – | 0.54 | 0.9987 |
| NEE | median | **0.9966** | 1.87 | – | 0.66 | 0.9987 |
| NEE | 75th | 0.9956 | 1.97 | – | 0.01 | 0.9987 |
| ET | 25th | 0.9963 | 1.20 | – | −0.17 | 0.9987 |
| ET | median | **0.9970** | 1.10 | – | −0.09 | 0.9987 |
| ET | 75th | 0.9955 | 1.38 | – | −0.41 | 0.9987 |

Units gC/m²/month (GPP/ER/NEE), mm/month (ET). Full table: `LEVEL4_2020.md`; plots: `report_plots/`.

### Before vs after the SILO-flip fix (median band)
| flux | R² before | R² after | RMSE before → after |
|---|---|---|---|
| GPP | 0.928 | **0.997** | 15.6 → 3.4 |
| ER  | 0.917 | **0.985** | 9.7 → 4.1 |
| NEE | 0.917 | **0.997** | 9.3 → 1.9 |
| ET  | 0.840 | **0.997** | 8.1 → **1.1** |

ET (water-driven) improved the most, as expected once the flipped rain/SRAD/VPD were corrected.

---

## 3. Findings

### Finding 1 — `scikit-learn` not pinned → RandomForest models fail to load *(blocker, fixed)*
step3 left `scikit-learn` unpinned; uv resolved ≥1.4, which cannot unpickle the reference RF joblibs
(trained on 1.3.2): `AttributeError: 'RandomForestRegressor' object has no attribute 'monotonic_cst'`.
The 15 LightGBM members load; the 15 RF members crash → half the ensemble lost.
**Fix:** pinned `scikit-learn==1.3.2`, `numpy<2`, `requires-python <3.13` in both step3 scripts.

### Finding 2 — SILO inputs reprojected upside-down in step1 *(root cause, fixed)*
SILO sources (rain, radiation, vp) have **ascending** latitude; MODIS/OzWALD/land-mask are
**descending**. `_affine()` always builds a north-up transform, so `_make_warper` reprojected the
SILO grids **vertically flipped**. Evidence: interim SRAD corr vs raw = 0.35 (normal) / **1.000
(flipped)**; interim rain at a productive northern point = **0 mm** vs raw **1124 mm** (flipped
ocean-zeros landed on northern land). GPP still scored ~0.93 because vegetation features
(MODIS/OzWALD) dominate it — but **ET (water-driven) collapsed to R²=0.84**, and the flipped
ocean-zeros produced `rain_anom = 0/0 = NaN`, masking ~6% of productive pixels.
**Fix:** in `_make_warper` (`step1_harmonisation.py`), flip ascending-lat / descending-lon source
arrays to north-up before reprojecting (MODIS/OzWALD untouched). Verified: SRAD corr→1.000, NE
rain→1129 mm. This single fix moved all fluxes to R² ≈ 0.98–0.997.

### Finding 3 — `rain_anom` 0/0 guard *(secondary, applied)*
`_fractional_anomalies` (step2) computed `rain/climatology`, which is 0/0=NaN (or x/0=Inf) in
zero-rainfall months. Added a guard filling non-finite anomalies with 1.0 at valid-data pixels.
Originally aimed at the masking gap, but Finding 2 was the true cause; kept as a defensive measure.

### Finding 4 — residual ≤3% *(inherent, not fixed)*
Remaining unbiased scatter comes from (a) **reprojection backend**: new `rasterio.reproject` vs
reference `odc.reproject`; (b) **quantile method**: new `np.nanquantile` (linear interpolation) vs
reference nearest-rank (`_percentile.np_percentile`). Both are small; matching them exactly would
require adopting odc-geo and nearest-rank quantiles.

---

## 4. Performance (Level 6)
| stage | scope | time (8-core box) |
|---|---|---|
| step1 harmonisation | per year (full) | ~1m53s/yr |
| step1 (SILO-only regen) | per year | ~6s/yr |
| step2 feature datasets | 20 yr, once | ~6m40s |
| step3 ensemble (30 models predict + combine) | per flux | ~25 min/flux old path; **~11–12 min fused** (now default; bit-identical). Cost ∝ model count; LightGBM ≈ RandomForest per-model. See `EFFICIENCY.md`. |

The new pipeline is memory-bounded (latitude-band tiling) and ran the whole bbox comfortably. The
RandomForest members dominate step3 runtime — the novel quantile-distillation path is designed to
replace this with 3 models (~10× faster) but could not be benchmarked (see below).

---

## 5. Level 5 — quantile-distillation path (built and validated)

No distilled quantile models existed anywhere on gadi (confirmed by an exhaustive read-only search:
xc0, os22/chad_tmp, scratch, the user's gdata) — the path is novel to the rewrite and the
target-generation step was not among the four scripts. It was **built faithful to the README**
(`benchmark/distill_quantile_models.py`): sample valid feature pixels across 2004/09/14/19/20 (all
months, ~400k samples) → run each through the 30 ensemble models → take the empirical 25/50/75
percentiles (`np.nanquantile`, matching the ensemble-combine definition) as targets → fit 3
quantile-loss LightGBM models/flux → saved to `…/models/quantile/v2.1/{flux}/`. Then `--mode predict`
produced the product (to `results/AusEFlux_quantile/`).

### Results (median band)
| flux | distilled vs **ensemble path** (fidelity) | distilled vs **published** |
|---|---|---|
| GPP | R²=0.994, bias −0.12 | R²=0.992, bias −0.94 |
| ER  | R²=0.990, bias −0.03 | R²=0.978, bias −0.96 |
| NEE | R²=0.991, bias −0.04 | R²=0.989, bias +0.61 |
| ET  | R²=0.993, bias −0.07 | R²=0.991, bias −0.16 |

- **The distilled median faithfully reproduces the 30-model ensemble median** (R² ≈ 0.99, ~0 bias).
- **The 25th/75th bands are slightly over-dispersed**: vs the ensemble the 25th is biased low (≈ −2 to
  −3) and the 75th high (≈ +2), i.e. the distilled interquartile spread is a few units wider than the
  ensemble's (most pronounced for ER). A known quantile-regression artifact; acceptable for an
  uncertainty envelope but worth noting if exact band widths matter.
- **Speed: ~13×.** Quantile predict = **7m34s for all 4 fluxes (~1.9 min/flux)** vs the ensemble path's
  ~25 min/flux (≈100 min) — confirms the README's ~10× claim. (One-off distillation training: ~12 min.)

### Levels status
- **Levels 0–4 (ensemble path):** complete; all 4 fluxes R² ≈ 0.98–0.997 vs published.
- **Level 5 (quantile path):** complete; median R² ≈ 0.99 vs both ensemble and published, ~13× faster,
  bands mildly over-dispersed.

## 6. Recommendations
1. **Keep the two code fixes** (sklearn pin + SILO orientation) — both are correct and necessary.
2. **Add a regression check** that step1 interim SILO variables correlate with the raw source in the
   correct orientation (the flip was silent and only hurt water-driven outputs).
3. To reach bit-closer parity, optionally adopt the reference's nearest-rank quantile and/or
   odc-geo reprojection (Finding 4).
4. Provide or generate the distilled quantile models to complete Level 5.

## 7. Reproduce
```
cd benchmark
uv run fetch_inputs.py --years 2002-2022 --target-year 2020   # OPeNDAP bbox subset
uv run python prep_static.py                                  # subset static grid to bbox
bash run_pipeline.sh 2020 2003 2022 4                         # step1 (×20) → step2 → step3 (×4)
uv run python level4_report.py --year 2020 --plots            # Level-4 metrics + plots
```
Static rasters + ensemble models were copied from `gadi:/g/data/xc0/project/AusEFlux/`.
