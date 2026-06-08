# AusEFlux

Monthly 500 m gridded estimates of Australian terrestrial carbon and water fluxes, produced by empirically upscaling the OzFlux eddy covariance network with machine learning and remote sensing.

| Flux | Description | Units |
|------|-------------|-------|
| GPP  | Gross Primary Productivity | gC/m²/month |
| NEE  | Net Ecosystem Exchange | gC/m²/month |
| ER   | Ecosystem Respiration | gC/m²/month |
| ET   | Evapotranspiration | mm/month |

Each flux is delivered as a NetCDF file with 25th percentile, median, and 75th percentile bands covering Australia at 500 m (EPSG:4326).

## Pipeline

```
Raw satellite & climate data
      │
      ▼
Step 1 – Spatiotemporal Harmonisation
      │
      ▼
Step 2 – Feature Dataset Construction
      │
      ▼
Step 3 – Ensemble Prediction  ──or──  Step 3 (Quantile) – Direct Quantile Prediction
      │                                       │
      ▼                                       │
Step 4 – Ensemble Combination                 │
      │                                       │
      ▼                                       ▼
AusEFlux_{flux}_500m_quantiles_{year}_v2.1.nc
```

### Step 1 – Spatiotemporal Harmonisation

Reprojects raw MODIS, OzWALD, and SILO datasets to the 500 m target grid and resamples to monthly time steps. One NetCDF per variable per year. Existing files are skipped on re-run.

```
uv run step1_harmonisation.py --year 2024
```

### Step 2 – Feature Dataset Construction

Concatenates per-year interim files into multi-year feature datasets. Applies gap-filling, computes vegetation fractions, cumulative rainfall, fractional anomalies, ΔT, and C4 grass fraction. Produces 25 feature files consumed by the ML models.

```
uv run step2_feature_datasets.py
```

### Step 3 – Ensemble Prediction

Applies each pre-trained LightGBM ensemble model to the feature data for the target year, writing one prediction file per model per flux. Processes data in latitude bands to bound memory.

```
uv run step3_ensemble_prediction.py --year 2024
uv run step3_ensemble_prediction.py --year 2024 --fluxes GPP ET
```

This is followed by **Step 4 – Ensemble Combination**, which stacks all per-model predictions and computes the 25th, 50th, and 75th percentile across the ensemble for each pixel and time step. Both steps are handled by the same script.

### Step 3 (Quantile) – Direct Quantile Prediction

An alternative to the standard Step 3 + Step 4 pipeline. Instead of running ~30 ensemble models and computing empirical quantiles afterward, this script uses 3 quantile-regression LightGBM models (alpha = 0.25, 0.50, 0.75) that predict the percentile bands directly in a single pass.

```
uv run step3_ensemble_prediction_quantile.py --year 2024 --mode predict
uv run step3_ensemble_prediction_quantile.py --year 2024 --mode train --training_data data/training_csvs
```

The quantile models are distilled from the full ensemble: sampled pixels are run through all ensemble models, the resulting per-pixel quantiles become the training targets, and 3 compact LightGBM models are fitted with quantile loss. At inference time this replaces 30 model passes + a stacking/quantile step with just 3 model passes writing the final output directly, giving roughly a 10x speedup and significantly lower memory usage (no intermediate prediction files, no ensemble stacking in RAM).

The output format and variable names are identical to the standard pipeline, so downstream consumers do not need to change.

## Output

```
results/AusEFlux/{flux}/AusEFlux_{flux}_500m_quantiles_{year}_v2.1.nc
```

Each file contains three variables (`{flux}_25th_percentile`, `{flux}_median`, `{flux}_75th_percentile`) with dimensions `(time, latitude, longitude)`.

## Data Sources

* **OzFlux** eddy covariance flux tower data
* **MODIS** MCD43A4 (surface reflectance → kNDVI, NDWI), MYD11A1 (LST)
* **OzWALD** NDVI, LAI, temperature (Tmin, Tmax, kTavg)
* **SILO** rainfall, solar radiation, vapour pressure
* Static: vegetation height, C4 grass cover, urban/water mask

## Validation & benchmarking

This pipeline is a refactor of the published AusEFlux reference
([`cbur24/AusEFlux`](https://github.com/cbur24/AusEFlux), Burton et al. 2023,
[doi:10.5194/bg-20-4109-2023](https://doi.org/10.5194/bg-20-4109-2023)). The
[`benchmark/`](benchmark/) directory holds a reproducible harness that validates this
rewrite against the **published v2.1 product** on NCI THREDDS and measures its
performance. Headline result: it reproduces the published product at **R² ≈ 0.98–0.997**
(per flux/band), and at **full continental scale** (~328 M pixel-months) the fast
quantile path holds **R² ≈ 0.99 (GPP) / 0.97–0.98 (NEE)**.

See `benchmark/README.md` (how to run, large-scale), `benchmark/BENCHMARKING.md`
(comparison methodology + all similarity statistics), `benchmark/REPORT.md` (findings),
and `benchmark/EFFICIENCY.md` (optimization results).

### Fixes & improvements applied (found via the benchmark)
1. **`scikit-learn==1.3.2` / `numpy<2` pinned** in `step3_*` — the reference's RandomForest
   joblibs cannot be unpickled by newer scikit-learn (`monotonic_cst`); unpinned, half the
   ensemble silently failed to load.
2. **SILO reprojection orientation fix** in `step1_harmonisation.py` (`_make_warper`) —
   ascending-latitude sources (SILO rain/SRAD/VPD) were reprojected vertically flipped; the
   fix flips ascending-lat / descending-lon sources to north-up. This was the dominant source
   of error (ET R² 0.84 → 0.997).
3. **`_fractional_anomalies` 0/0 guard** in `step2_feature_datasets.py` (defensive).
4. **Fused predict + combine** in `step3_ensemble_prediction.py` — features are read once per
   tile, all ensemble members predicted, quantiles written directly (no per-model intermediate
   files). Bit-identical to the old path, ~1.9× faster (2.1× with treelite-compiled LightGBM
   members via `benchmark/compile_models.py`).
