# AusEFlux Benchmarking — Comparison Methodology & Similarity Statistics

This document specifies **exactly what was compared, how, and the resulting
similarity statistics** between the refactored AusEFlux pipeline and the published
v2.1 reference product. It complements [`REPORT.md`](REPORT.md) (narrative findings &
recommendations) and [`README.md`](README.md) (how to run).

All statistics below are over the **SE-Australia test tile** (600×600 px @ 500 m,
148.00–151.00°E, 33.00–36.00°S), **target year 2020**, unless stated otherwise. The
continental 2025 results (§7) are appended when that run completes.

> **Superseded by the Gadi experiments (2026-10-27).** The statistics below remain
> valid for what they measured (the pipeline with its previous defaults: linear
> quantiles, mask without `LAI_anom`, anomaly guard, clipped bare fraction), but some
> conclusions changed. Current results: [`GADI_REPORT.md`](GADI_REPORT.md); Gadi
> methodology and tools: [`gadi/README.md`](gadi/README.md).
>
> | Section | What changed |
> |---|---|
> | §5 "inherent reimplementation residual", §8 "Residual after fixes (~3 %, inherent, not bugs)" | **Not inherent.** The Gadi ablations attribute it to semantic deviations, chiefly the quantile method and the no-data mask. Recombining the 30 reference member files with nearest-rank is bit-identical to the published product (E1), and the new step3 with nearest-rank, the reference mask (incl. `LAI_anom`) and the NaN-only mask test reproduces the published 2025 product bit-for-bit from the reference features (E2a). The rasterio vs odc difference is at float-noise level for 11 of 13 interim variables (E4). |
> | §8 item 2: linear vs nearest-rank "slightly different 25th/75th bands" | Measured continentally (E1): linear biases the median high (+0.17 to +0.34 per month) and the 25th/75th inward; annual GPP +0.6 %, ER +0.8 %, ET +0.5 %, NEE sink 2.8 to 3.3 % weaker (fails the tolerance). Nearest-rank is now the default. |
> | §6, §7: distilled quantile path, band over-dispersion | The §7 continental numbers used pinball-loss models trained on linear-quantile targets. E7 shows L2 loss on nearest-rank targets removes the over-dispersion (IQR +0.3 % vs +42 %) and matches annual totals within 0.02 %, but per-pixel RMSE stays at 6.4 % (GPP) and 12 % (NEE) of std, above the 5 % threshold. |
> | §5 Level-4 table, §6, §7 statistics | Unchanged as historical measurements. |

---

## 1. What is being compared

| Side | Definition |
|---|---|
| **New (test)** | Output of the refactored pipeline (`../step1`→`step2`→`step3`), fed the reference's **own** 30-member ensemble (15 LightGBM + 15 RandomForest) and static rasters. |
| **Reference (truth)** | The published operational product `AusEFlux_{flux}_500m_quantiles_{year}_v2.1.nc` from NCI THREDDS `ub8/au/AusEFlux/v2.1/monthly`. |

Because the **models and static inputs are identical** to the reference's, any
divergence isolates the rewrite's own stages (harmonisation, feature engineering,
prediction, quantile combination). Each flux product has three bands:
`{flux}_25th_percentile`, `{flux}_median`, `{flux}_75th_percentile`.

Fluxes & units: GPP, ER, NEE in gC·m⁻²·month⁻¹; ET in mm·month⁻¹.

---

## 2. Grid handling (how the two arrays are aligned)

- The refactored pipeline takes its target grid from the **land/sea mask**; subsetting
  that mask to the test region puts the output on the **identical canonical 500 m
  grid** as the published product. Verified: **max coordinate difference = 0.0°** in
  both latitude and longitude — a true 1:1 comparison, **no regridding/interpolation**.
- The static grid is subset to a small **buffer** beyond the comparison region so
  reprojection edge pixels fall outside it; the published product sits strictly inside
  the buffer, so all 600×600 published pixels have a counterpart.
- Metrics use only pixels **finite (valid) in both** products; the valid-pixel count
  `n_pix` is reported so coverage differences are explicit (never silently dropped).

---

## 3. Metric definitions

For new values `a` and reference values `b` over the jointly-valid pixels (`d = a−b`):

| Metric | Definition | Meaning |
|---|---|---|
| **R²** | `1 − Σd² / Σ(b−b̄)²` | variance of the reference explained by the new output |
| **RMSE** | `sqrt(mean(d²))` | typical magnitude of error (same units as flux) |
| **MAE** | `mean(\|d\|)` | median-robust error magnitude |
| **mean bias** | `mean(d)` | systematic offset (new − reference) |
| **max \|Δ\|** | `max(\|d\|)` | worst-case single-pixel difference |
| **% within 1% range** | `% of \|d\| < 0.01·(max b − min b)` | fraction effectively identical |
| **mask Jaccard** | `\|valid_a ∩ valid_b\| / \|valid_a ∪ valid_b\|` | agreement of valid-data coverage |

Pearson r, Spearman, residual histograms, density scatter, difference maps and
regional monthly-mean time series are also produced (`compare.py --plots`,
`report_plots/`).

---

## 4. Comparison design (levels)

| Level | Compares | Purpose |
|---|---|---|
| **0 — Grid & mask** | new grid ↔ published grid | gate: confirm exact 1:1 alignment before trusting any metric |
| **1 — Harmonisation** | step1 interim ↔ raw source (orientation/range/NaN) | catch reprojection / units / orientation errors |
| **2 — Features** | step2 features (veg fractions, anomalies, cml rain, ΔT) | catch feature-engineering divergence |
| **3 — Prediction & combine** | per-model preds & quantile combination | isolate the prediction engine and quantile method |
| **4 — Final product** | new product ↔ published v2.1 (per flux/band) | **headline equivalence test** |
| **5 — Quantile path** | distilled product ↔ ensemble path **and** ↔ published | distillation fidelity + the fast path's accuracy/scalability |

Levels 1–2 are diagnostic (the reference's intermediates are not published); the
decisive numbers are Level 4 (and Level 5 for the novel path). Level 3 found the SILO
orientation bug via the per-variable orientation check (`corr(interim, raw)` vs
`corr(interim, raw flipped)`).

---

## 5. Level 4 — final product vs published v2.1 (2020, ensemble path)

Statistics over the shared 600×600 grid (`n_pix` = jointly-valid pixel-months,
3,823,837 across 12 months):

| flux | band | R² | RMSE | MAE | mean bias | max \|Δ\| | % within 1% | mask Jaccard |
|---|---|---|---|---|---|---|---|---|
| GPP | 25th | 0.99639 | 3.432 | 1.926 | −0.969 | 65.49 | 80.5 | 0.9987 |
| GPP | median | **0.99669** | 3.351 | 1.893 | −0.832 | 63.18 | 82.2 | 0.9987 |
| GPP | 75th | 0.99607 | 3.770 | 1.983 | −1.650 | 65.41 | 82.7 | 0.9987 |
| ER | 25th | 0.98609 | 3.975 | 1.851 | −1.003 | 98.03 | 78.2 | 0.9987 |
| ER | median | **0.98479** | 4.148 | 2.098 | −0.932 | 103.10 | 76.3 | 0.9987 |
| ER | 75th | 0.97561 | 5.310 | 2.479 | −2.028 | 99.90 | 76.9 | 0.9987 |
| NEE | 25th | 0.99674 | 1.952 | 1.058 | +0.536 | 58.53 | 83.7 | 0.9987 |
| NEE | median | **0.99664** | 1.875 | 1.101 | +0.664 | 55.02 | 81.6 | 0.9987 |
| NEE | 75th | 0.99563 | 1.969 | 1.073 | +0.009 | 46.72 | 80.7 | 0.9987 |
| ET | 25th | 0.99633 | 1.200 | 0.667 | −0.169 | 25.23 | 83.4 | 0.9987 |
| ET | median | **0.99703** | 1.098 | 0.619 | −0.093 | 29.06 | 86.8 | 0.9987 |
| ET | 75th | 0.99551 | 1.385 | 0.694 | −0.407 | 39.83 | 86.1 | 0.9987 |

**Reading:** R² ≥ 0.975 for every flux/band, RMSE a few % of the regional means,
near-zero bias, mask agreement 0.999. The new pipeline reproduces the published
product to within the reimplementation residual (§8; *superseded: not inherent, see the note at the top*). ER is the least tight
(R² ≈ 0.975–0.986), consistent with respiration being the hardest flux to upscale.

---

## 6. Level 5 — quantile-distillation path (2020)

The quantile models were built by distilling the 30-member ensemble (sample feature
pixels → 30 predictions → empirical 25/50/75 percentiles as training targets → 3
quantile-loss LightGBM/flux; `distill_quantile_models.py`).

| flux | band | vs **ensemble path** R² (bias) | vs **published** R² (bias) |
|---|---|---|---|
| GPP | 25th | 0.9863 (−3.16) | 0.9817 (−4.14) |
| GPP | median | **0.9941 (−0.12)** | 0.9915 (−0.94) |
| GPP | 75th | 0.9923 (+2.55) | 0.9916 (+0.91) |
| ER | 25th | 0.9836 (−2.12) | 0.9682 (−3.14) |
| ER | median | **0.9899 (−0.03)** | 0.9775 (−0.96) |
| ER | 75th | 0.9796 (+2.27) | 0.9650 (+0.27) |
| NEE | 25th | 0.9880 (−1.85) | 0.9871 (−1.32) |
| NEE | median | **0.9910 (−0.04)** | 0.9887 (+0.61) |
| NEE | 75th | 0.9803 (+2.05) | 0.9775 (+2.04) |
| ET | 25th | 0.9877 (−1.11) | 0.9848 (−1.27) |
| ET | median | **0.9926 (−0.07)** | 0.9906 (−0.16) |
| ET | 75th | 0.9893 (+1.01) | 0.9872 (+0.60) |

**Reading:**
- The **median is reproduced faithfully** (R² ≈ 0.99, bias ≈ 0 vs the ensemble).
- The **25th/75th bands are mildly over-dispersed**: the 25th is biased low (≈ −1 to
  −3) and the 75th high (≈ +1 to +2.5), i.e. the distilled interquartile spread is a
  few units wider than the ensemble's (largest for ER). A known quantile-regression
  behaviour — acceptable for an uncertainty envelope; note it where exact band widths
  matter.
- **Speedup:** quantile predict = **7m34s for all 4 fluxes (~1.9 min/flux)** vs the
  ensemble path's ~25 min/flux ⇒ **~13×** (confirms the README's ~10× claim).

---

## 7. Continental full-domain consistency & scalability (GPP, NEE)

**Method.** The target-year slice of the reference's continental 500 m features
(6800×8400) is streamed read-only off gadi (`extract_continental.py`, per-timestep
to preserve layout), quantile models are distilled on a continental sample, GPP+NEE
are predicted over the whole continent (quantile path), and compared per-time-step to
the published product over OPeNDAP (`compare_continental.py`). The output is on the
same canonical grid as the product, so the comparison is 1:1 (coords checked to 1e-3).
Each continental quantile predict ran **~4h43m for GPP+NEE** (657% CPU, 2.9 GB peak
RSS) — the quantile path is what makes the continent tractable (the 30-model ensemble
would be ~tens of hours/flux).

> **Data-integrity note.** A first attempt scored R² ≈ −0.4/−0.8 — traced to a reshape
> bug in `extract_continental.py` (lat-band stream vs time-major reshape) that spatially
> scrambled the features. Fixed (stream per-timestep) and **verified the re-extracted
> features match the gadi source exactly** (e.g. kNDVI nanmean 0.110019 vs 0.110020)
> before re-running. The numbers below are post-fix. Lesson: spatially validate
> extracted arrays, not just their dimensions.

### Continental 2025 — quantile vs published v2.1 (n = 328,162,471 pixel-months)
| flux | band | R² | RMSE | MAE | mean bias | max \|Δ\| |
|---|---|---|---|---|---|---|
| GPP | 25th | 0.9897 | 4.543 | 2.720 | −1.704 | 107.1 |
| GPP | median | **0.9932** | 3.830 | 2.222 | +0.284 | 108.9 |
| GPP | 75th | 0.9911 | 4.523 | 2.806 | +1.716 | 122.5 |
| NEE | 25th | 0.9725 | 3.150 | 1.938 | −1.218 | 96.5 |
| NEE | median | **0.9797** | 2.556 | 1.548 | +0.264 | 80.6 |
| NEE | 75th | 0.9664 | 3.102 | 1.967 | +1.257 | 80.6 |

### Continental 2020 — quantile vs published v2.1 (n = 327,982,001 pixel-months)
| flux | band | R² | RMSE | MAE | mean bias | max \|Δ\| |
|---|---|---|---|---|---|---|
| GPP | 25th | 0.9878 | 4.785 | 2.846 | −1.654 | 132.8 |
| GPP | median | **0.9918** | 4.065 | 2.389 | +0.524 | 122.7 |
| GPP | 75th | 0.9886 | 4.962 | 3.209 | +2.083 | 130.1 |
| NEE | 25th | 0.9573 | 3.829 | 2.210 | −1.267 | 106.6 |
| NEE | median | **0.9656** | 3.235 | 1.897 | +0.256 | 96.0 |
| NEE | 75th | 0.9507 | 3.669 | 2.349 | +1.439 | 77.0 |

**Reading.** At full continental scale (~328 M pixel-months, both years) the distilled
quantile product reproduces the published median at **R² ≈ 0.99 (GPP) / 0.97–0.98 (NEE)**
with near-zero median bias — matching the small-tile result, confirming the pipeline and
the fast path scale consistently across the whole continent and across years. The 25th/75th
bands show the same mild over-dispersion as in §6 (25th biased low ≈ −1.7, 75th high ≈ +2;
median ≈ 0). NEE is consistently looser than GPP (net flux is the hardest, smallest-magnitude
target), and slightly looser in 2020 than 2025.

---

## 8. Impact of the bug fixes (provenance of the numbers)

The §5/§6 statistics are **after** the two fixes the benchmark surfaced. The fixes'
effect, measured on the same comparison (GPP/median unless noted):

| Stage | R² (median band) before → after | RMSE before → after |
|---|---|---|
| GPP | 0.928 → **0.997** | 15.6 → 3.4 |
| ER  | 0.917 → **0.985** | 9.7 → 4.1 |
| NEE | 0.917 → **0.997** | 9.3 → 1.9 |
| ET  | 0.840 → **0.997** | 8.1 → 1.1 |

- **Fix 1 (scikit-learn pin):** without it, the 15 RandomForest members fail to load
  and half the ensemble is silently lost.
- **Fix 2 (SILO latitude flip in step1):** the decisive fix. Diagnosed at Level 3 by
  per-variable orientation correlation — interim SRAD correlated **0.35** with the raw
  source but **1.000** with the *vertically-flipped* raw. ET (water-driven) was worst
  hit pre-fix (R²=0.84) and improved most.

**Bulk vs affected-pixel decomposition (pre-fix GPP/median, illustrating the flip):**
- 94 % of pixels (unaffected): R² = 0.979, RMSE = 8.3, bias = −0.17.
- 6 % of pixels (in the flipped/zero-rain band): R² = −0.28, RMSE = 56, bias = −44.

**Residual after fixes (~3%; originally described as inherent. Superseded: the Gadi experiments trace it to semantic deviations, see the note at the top):**
1. **Reprojection backend** — new uses `rasterio.reproject` (hand-built affine); the
   reference uses `odc.reproject` on labelled coordinates. Sub-pixel resampling
   differences propagate through 30 models.
2. **Quantile method** — new uses `np.nanquantile` (linear interpolation); the
   reference uses nearest-rank (`src/_percentile.py`, `np.round(p·(n−1))`). For n = 30
   members these give slightly different 25th/75th bands even from identical
   predictions.

**Error-distribution context (GPP/median, after fixes):** median |Δ| ≈ 1.9, p90 small
relative to the regional mean (~110 gC·m⁻²·month⁻¹); the large `max |Δ|` values are
isolated edge/cropland/water-margin pixels, not a systematic shift (bias ≈ 0).

---

## 9. Reproducing these statistics

```bash
# Level 4 (ensemble path) table:
uv run python level4_report.py --year 2020 --plots      # → LEVEL4_2020.md + report_plots/

# Level 5 (quantile path) fidelity + vs published — see README §6.
# Continental 2025 (§7):
uv run python compare_continental.py \
    --new-root /home/prl900/Downloads/wald_eflux/results/AusEFlux_quantile \
    --fluxes GPP NEE --year 2025
```
