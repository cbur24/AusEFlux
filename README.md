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
Step 1: Spatiotemporal Harmonisation
      │
      ▼
Step 2: Feature Dataset Construction
      │
      ▼
Step 3: Ensemble Prediction + Combination   ──or──   Step 3 (Quantile): Direct Quantile Prediction
   (predict 30 members & take 25/50/75                   (3 distilled LightGBM models predict
    percentiles in one fused pass)                        the bands directly)
      │                                                          │
      ▼                                                          ▼
AusEFlux_{flux}_500m_quantiles_{year}_v2.1.nc
```

> **Note on "Step 4".** Earlier versions split prediction (Step 3) and ensemble
> combination (Step 4) into two passes. They are now fused into a single pass inside
> `step3_ensemble_prediction.py` (function `_predict_and_combine`): features for each
> tile are read once, all members predict, and the 25/50/75 percentiles are written
> straight to the final product. There is no separate Step 4 script.

All scripts are single-file `uv run` scripts with PEP 723 headers; every flag is
listed by `uv run <script> --help`.

> **Picking this up?** Read [`HANDOVER.md`](HANDOVER.md): state of the code, validation
> evidence, how to run an annual update on NCI Gadi, efficiency against the reference,
> and the next steps.

### Paths

Defaults come from environment variables, and every path can also be given on the
command line.

| Variable | Default | Used for |
|---|---|---|
| `AUSEFLUX_ROOT` | `/home/prl900/Downloads/wald_eflux` | `interim_500m/` (step1 output), `features_500m/` (step2 output), `results/` (models, `variables.txt`, products) |
| `AUSEFLUX_INPUT` | `$AUSEFLUX_ROOT/input` | raw inputs in the `/g/data/ub8/au` layout (`MODIS/`, `OzWALD/`, `SILO/`) |
| `AUSEFLUX_STATIC` | `$AUSEFLUX_INPUT/static` | land/sea and urban/water masks, bare-soil NDVI, VegH, C4 grass |

On Gadi: `AUSEFLUX_INPUT=/g/data/ub8/au`, `AUSEFLUX_STATIC=/g/data/xc0/project/AusEFlux/data`
(see `benchmark/gadi/README.md`).

### Step 1: Spatiotemporal Harmonisation

Reprojects raw MODIS, OzWALD and SILO data to the 500 m target grid and resamples to
monthly time steps. One NetCDF per variable per year, `{interim}/{var}/{var}_500m_{year}.nc`.
Existing files are skipped on re-run. Writes are atomic: each file is streamed to
`{file}.part.{pid}` and renamed when complete, so an existing file is always complete.

```
uv run step1_harmonisation.py --year 2025 --threads 8
uv run step1_harmonisation.py --year 2025 --vars climate,SILO,rain_prior   # one shard
```

| Flag | Meaning |
|---|---|
| `--lst_qc {ref,new}` | MYD11A1 QC rule. `ref` (default) applies QC every year except 2022 and 2023, the reference rule used for all years of the published v2.1. `new` (earlier refactor rule, kept for comparison) applies QC only before 2022; for 2025 it changes 8 to 75 % of LST pixels (RMSE 0.43 °C vs the reference interim). |
| `--vars` | Subset of variable groups (`NDWI,kNDVI,NDVI,LAI,LST,VegH,climate,SILO,rain_prior`) for sharding a year across jobs. |
| `--engine {rasterio,odc}` | Reprojection engine. `rasterio` (default) is faster and gives the same result for same-grid variables; `odc` is kept for comparison. |
| `--modis_nodata {poison,exclude}` | NaN handling when averaging NDWI/kNDVI to 500 m. `poison` (default) reproduces the reference's mask (about 0.3 % more NaN pixel-months, an artefact of the reference's odc call losing its nodata attribute). `exclude` skips NaN sub-pixels, which is arguably more correct but changes the wet-season gap-fill in the north (E5: north-window GPP/NEE RMSE 6–10 % of std vs 1–3 %). |
| `--threads` | Threads for dask and the rasterio warp. |
| `--orientation_check` | Prints corr(output, source) and corr(output, flipped source) at a few land pixels (guard for the SILO flip bug). |

### Step 2: Feature Dataset Construction

Builds the 28 multi-year feature stacks `{features}/{var}_500m.nc` from the interim
files: gap-filling, vegetation fractions, cumulative rainfall, fractional anomalies,
ΔT and C4 grass fraction.

```
uv run step2_feature_datasets.py --years_out 2024-2025 --threads 8
uv run step2_feature_datasets.py --stages gapfill --vars NDWI,kNDVI --years_out 2024-2025   # one shard
```

| Flag | Meaning |
|---|---|
| `--stages` | Subset of `gapfill,vegfrac,cmlrain,deltat,anom,c4` (run in that order; missing gap-filled inputs are recomputed inline). |
| `--vars` | Variable subset for the `gapfill` and `anom` stages, for PBS sharding. |
| `--years_out YYYY-YYYY` | Write only these years; computation still uses the full history. Bit-identical to the same slice of a full run, and about 45 GB instead of 1.7 TB for an annual update. |
| `--reuse_clim` / `--no_reuse_clim` | Reuse stored monthly climatologies (default, bit-identical) or recompute them. |
| `--threads`, `--pool` | Latitude tiles computed concurrently (forked processes by default). Bit-identical. A worker killed by the OOM killer aborts the stage with an error (no hang). Peak RSS ≈ 4 × threads × tile; auto tiles are 200/100/50/25 rows (floor 25), see `--help`. |
| `--lat_rows START:END` | Process only a latitude window; outputs are bit-identical to those rows of a full run. Used for windowed validation. An existing output from another window in the same `--results_path` is a fatal error. |
| `--no_bare_clip` (default) / `--bare_clip` | Reference behaviour: `bare = 1 - trees - grass`, unclipped (negative at some pixels, as in the reference). |
| `--anom_guard none` (default) / `one` | Reference behaviour: non-finite fractional anomalies stay NaN/Inf, so step3 masks NaN and sets Inf to 0 as the reference does. `one` fills them with 1.0. |
| `--bottleneck`, `--flox` | Off by default. They speed up rolling windows and groupby but move results away from the reference at float32 rounding level; step2 is I/O-bound, so they buy little. |

### Step 3: Ensemble Prediction + Combination (fused)

Applies every pre-trained ensemble member (15 LightGBM + 15 RandomForest per flux) to
the features of the target year and combines them into the final product in one pass.
Features of each (time, latitude-band) tile are read once, all members predict, and the
25th/50th/75th percentiles across members are written directly.

```
uv run step3_ensemble_prediction.py --year 2025 --fluxes GPP NEE --threads 4 --months 1-3
uv run step3_ensemble_prediction.py --year 2025 --fluxes GPP NEE --merge
```

Defaults reproduce the reference semantics:

| Flag (default) | Meaning |
|---|---|
| `--quantile_method nearest` | Reference nearest-rank percentile (`src/_percentile.py`: sort members, index `round_half_even(q·(n−1))`, ≥ 3 valid members). Exact against the published v2.1 product and about 200× faster than `linear` (`np.nanquantile`), which is kept for comparison only. |
| `--mask_vars LAI_anom,VegH,NDWI,rain_anom,Tavg` | Reference no-data mask. |
| `--mask_test nan` | Only NaN marks no-data, like the reference's `isnull()`; ±Inf features are predicted with Inf set to 0. `nonfinite` also masks ±Inf. |
| `--valid_only` | Predict only valid, non-urban pixels and scatter back. Bit-identical to `--all_pixels`, about 1.8× faster predict. |
| `--threads N` | `n_jobs` for member prediction and tl2cgen. On Gadi 4 threads gives the lowest SU per product. |
| `--months`, `--merge` | Shard a year by month (part files under `{output}/{flux}/parts/`), then assemble the 12 parts. |
| `--merge_partial` | Like `--merge` but accepts fewer than 12 parts (current, incomplete year); the months present are listed in `auseflux_partial_months` and a WARNING names the missing ones. |
| `--lat_rows START:END` | Process a latitude window only (windowed validation). |
| `--dump_members DIR` | Also write each member's prediction, for parity checks. |
| `--n_models N` | First N members only. Not for production: N = 20 and N = 10 fail the tolerance (see `benchmark/GADI_REPORT.md`, E1). |

Products, month parts and member dumps carry the run configuration as global
attributes: `quantile_method`, `mask_vars`, `mask_test`, `n_models`, `valid_only` and
`lat_rows`. Month parts must agree on all of them to be merged, and an existing output
whose attributes differ from the request is skipped with a WARNING.

**Compiled LightGBM.** If `<models>/{flux}/compiled/<member>.so` exists (built with
`benchmark/compile_models.py`, needs `tl2cgen` and gcc), step3 uses it instead of the
joblib. Bit-identical including the dart members, about 1.85× faster on the LightGBM
half. RandomForest members always run on scikit-learn.

### Step 3 (Quantile): Direct Quantile Prediction

An alternative to the 30-member ensemble: 3 LightGBM models predict the 25th/50th/75th
bands directly. Output format and variable names are identical to the ensemble path.

```
uv run step3_ensemble_prediction_quantile.py --year 2025 --mode predict --threads 24
```

The models are distilled from the ensemble: the targets are the nearest-rank quantiles
of the 30 members at sampled pixels. On Gadi (E7), L2 loss on these targets removes the
band over-dispersion of the pinball loss, and the continental 2025 product matches the
published annual GPP within 0.02 %, but per-pixel RMSE is 6 to 7 % of std (GPP) and 12 %
(NEE), so it **does not pass** the per-pixel tolerance. It costs about 21 SU per
flux-year. Treat it as a fast approximation, not a reproduction. `--mode train` fits to
tower data, which is a different model, not a distillation.

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
[doi:10.5194/bg-20-4109-2023](https://doi.org/10.5194/bg-20-4109-2023)), validated on
NCI Gadi against the published v2.1 products for 2024 and 2025
(`benchmark/GADI_REPORT.md`):

- **Prediction and combination.** Fed the reference's own features, the fused step3
  reproduces the published v2.1 product **bit-for-bit** (E2a: 3 continental months of
  2025, GPP and NEE; every LightGBM member bit-identical, RandomForest members within
  one float32 ulp at ≤ 2 pixels, no percentile affected).
- **Harmonisation.** From raw inputs, step1 reproduces 11 of 13 interim variables to
  float noise (E4, 2024 and 2025). NDWI/kNDVI match the reference's NaN mask with the
  default `--modis_nodata poison` (`exclude` has 0.3 % fewer NaN); LST/VegH differ at RMSE
  0.004 °C / 0.002 m because of the reference's GeoBox affine.
- **Features.** step2 on the reference interim reproduces 16 of 28 features
  bit-identically, the rest at float32 rounding (E3, 3 latitude windows, 2024 and 2025).
- **End to end** (new step1 + step2 + step3 from raw inputs, continental 2025, GPP and
  NEE): every band passes the scorecard against the published v2.1 product (E5). R² ≥
  0.9995, RMSE ≤ 2.2 % of the reference std, 98.2–98.7 % of pixel-months bit-equal,
  continental annual sums within 0.05 %, every major vegetation group within 0.4 %. The
  2024 latitude windows pass too once `--modis_nodata poison` (now the default) is used;
  with `exclude` the tropical north fails on the wet-season months. Full-year cost for
  GPP + NEE ≈ 285 SU (step1 13, step2 114, step3 72 + 86) against ≈ 3,100 SU for the
  reference's step3 alone. See `benchmark/GADI_REPORT.md`.

The earlier bbox and continental results (`benchmark/REPORT.md`, `benchmark/BENCHMARKING.md`)
were obtained with the previous defaults (linear quantiles, mask without `LAI_anom`).
Their ≤ 3 % residual was not inherent: the Gadi ablations attribute it to those
semantic deviations (chiefly the quantile method and the mask), not to the rewrite
itself. The bbox was not re-run with the new defaults.

See `benchmark/README.md` (bbox harness), `benchmark/gadi/README.md` (Gadi workflow and
experiment commands), `benchmark/GADI_REPORT.md` (experiment ledger) and
`benchmark/EFFICIENCY.md` (performance).

### Fixes & improvements (found via the benchmarks)

Correctness:
1. **`scikit-learn==1.3.2` / `numpy<2` pinned** in `step3_*`. The reference's
   RandomForest joblibs cannot be unpickled by newer scikit-learn (`monotonic_cst`);
   unpinned, half the ensemble silently failed to load.
2. **SILO reprojection orientation fix** in `step1_harmonisation.py` (`_make_warper`).
   Ascending-latitude sources (SILO rain/SRAD/VPD) were reprojected vertically flipped
   (ET R² 0.84 → 0.997 on the bbox).
3. **Nearest-rank quantiles** are the reference method and the default. They are exact
   against the published product (E1: recombining the 30 reference member files is
   bit-identical for all 4 fluxes, 2024 and 2025). Linear interpolation biased the
   median high and the 25th/75th bands inward; NEE annual sink 2.8 to 3.3 % weaker,
   which fails the tolerance on its own.
4. **`LAI_anom` restored to the no-data mask** (the reference list). Dropping it
   predicted 0.41 % extra pixels per month.
5. **NaN-only mask test** (`--mask_test nan`), like the reference's `isnull()`: ±Inf
   pixels are predicted (Inf → 0), not masked (0.05 to 0.09 % of valid pixels per month).
6. **Unclipped bare fraction** (`--no_bare_clip`) and **no anomaly guard**
   (`--anom_guard none`), both reference behaviour. The clip and the 1.0 fill changed
   inputs at 579 k and about 900 k pixel-months in the 3 test windows.
7. **LST QC rule**: `--lst_qc ref` (now the default) matches the reference for 2024/2025.
8. **`.ipynb_checkpoints`** and other hidden directories in the interim tree are
   ignored by step2 (the reference excluded them explicitly).
9. **`--engine odc` fix**: the source GeoBox was built from float32 coordinates
   (up to 4 to 5 px shift in the east for 500 m OzWALD).
10. **Atomic step1 writes** (`{file}.part.{pid}`, renamed when complete): an interrupted
    or concurrent shard never leaves a truncated file that a re-run would skip.
11. **step3 run-configuration attributes** on every output (see Step 3), checked
    on merge and on skip, and `--merge_partial` for an incomplete year.

Efficiency (all exact):
1. **Fused predict + combine**: features read once per tile, no per-model files.
2. **step2 chunked reads**: interim files are opened with latitude-tile chunks; with
   `chunks={}` every tile re-read whole 2.7 GB files (68 s → 0.55 s per tile).
3. **step3 `--valid_only`** (1.8× faster predict) and **compiled LightGBM** via tl2cgen
   (1.85× on the LightGBM half, exact including dart).
4. **`np.nanquantile` cost trap**: in numpy 1.26 it is a per-pixel Python loop,
   about 30 min per continental month vs about 10 s for nearest-rank sort-and-take.
5. **Sharding and windows**: step1 `--vars`, step2 `--stages/--vars/--years_out`,
   step3 `--months/--merge`, and `--lat_rows` in step2/step3. step1 takes 21 min and
   about 13 SU per year on Gadi (reference about 3.5 h); step3 about 60 to 80 SU per
   flux-year with 4 threads (see `benchmark/EFFICIENCY.md`).
