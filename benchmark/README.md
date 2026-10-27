# AusEFlux Benchmark & Validation Harness

A reproducible harness that validates the **refactored AusEFlux pipeline**
(`../step1_harmonisation.py`, `../step2_feature_datasets.py`,
`../step3_ensemble_prediction.py`, `../step3_ensemble_prediction_quantile.py`)
against the **published AusEFlux v2.1 product**, and measures its performance and
scalability — from a small test tile up to the full Australian continent.

It exists to answer one question: *does the efficient, modular rewrite produce
results equivalent to the original implementation, and does it scale?*

---

## 1. Context

AusEFlux is a monthly, 500 m gridded estimate of Australian terrestrial carbon and
water fluxes (GPP, ER, NEE, ET), produced by empirically upscaling the OzFlux eddy
covariance network with machine learning and remote sensing.

- **Original implementation / reference:** Chad Burton et al. — repository
  [`cbur24/AusEFlux`](https://github.com/cbur24/AusEFlux). The scientific method is
  described in Burton, C.A., Renzullo, L.J., Rifai, S.W., & Van Dijk, A.I. (2023),
  *Empirical upscaling of OzFlux eddy covariance for high-resolution monitoring of
  terrestrial carbon uptake in Australia*, **Biogeosciences** 20(19): 4109–4134,
  [doi:10.5194/bg-20-4109-2023](https://doi.org/10.5194/bg-20-4109-2023).
  The reference is notebook-driven (`notebooks/Historical.ipynb`,
  `notebooks/Annual_update.ipynb`) over the modules in `src/` (`_harmonisation.py`,
  `_feature_datasets.py`, `_ensemble_modelling.py`, `_ensemble_prediction.py`,
  `_combine_ensemble.py`, `_percentile.py`).
- **Refactored pipeline (this repo's parent):** a flat, four-script,
  [`uv`](https://docs.astral.sh/uv/)-runnable rewrite designed to be faster,
  memory-bounded (latitude-band tiling), and scalable, while producing equivalent
  output. Each stage maps onto the reference:

  | Refactored script | Reference equivalent |
  |---|---|
  | `step1_harmonisation.py` | `src/_harmonisation.py` |
  | `step2_feature_datasets.py` | `src/_feature_datasets.py` |
  | `step3_ensemble_prediction.py` (predict + combine) | `src/_ensemble_prediction.py` + `src/_combine_ensemble.py` |
  | `step3_ensemble_prediction_quantile.py` (novel fast path) | *(no equivalent — new)* |

- **Published product (ground truth):** the operational v2.1 product on the NCI
  THREDDS server, [`ub8/au/AusEFlux`](https://thredds.nci.org.au/thredds/catalog/ub8/au/AusEFlux/catalog.html)
  (500 m, monthly, per-flux `AusEFlux_{flux}_500m_quantiles_{year}_v2.1.nc`, with
  `25th_percentile` / `median` / `75th_percentile` bands). A 5 km version is on
  [Zenodo](https://zenodo.org/records/14942139).

The validation feeds the **reference's own trained ensemble** (30 models/flux =
15 LightGBM + 15 RandomForest) and **static rasters** into the refactored pipeline,
so any difference from the published product is attributable to the rewrite (its
harmonisation, feature engineering, prediction, or quantile combination), not to
different models — making this a genuine equivalence test.

---

## 2. Results in brief

After two bug fixes that the benchmark surfaced (see §7 and `REPORT.md`), the
refactored pipeline reproduces the published product closely:

| Flux (median band) | R² vs published (2020) |
|---|---|
| GPP | 0.997 |
| ER  | 0.985 |
| NEE | 0.997 |
| ET  | 0.997 |

The novel quantile-distillation path reproduces the ensemble median at R² ≈ 0.99
with a **~13× speedup**. Full write-up, per-band tables, and before/after numbers
are in **[`REPORT.md`](REPORT.md)** (and the generated table `LEVEL4_2020.md`).

---

## 3. Repository layout

```
benchmark/
├── README.md                  # this file
├── REPORT.md                  # findings, metrics, before/after, recommendations
├── BENCHMARKING.md            # comparison methodology + full similarity statistics
├── EFFICIENCY.md              # speed-vs-equivalence results for the optimizations
├── LEVEL4_2020.md             # generated per-flux/-band metrics table
├── pyproject.toml / uv.lock   # local uv environment for the tooling
│
├── fetch_inputs.py            # download bbox-subset raw inputs + published product (OPeNDAP)
├── prep_static.py             # subset the grid-aligned static rasters to the test bbox
├── run_pipeline.sh            # orchestrate step1 (×N years) → step2 → step3
├── compare.py                 # 2-file metrics + diagnostic plots (in-memory)
├── level4_report.py           # run compare across all fluxes/bands → markdown + plots
├── distill_quantile_models.py # build the quantile path's models by distilling the ensemble
├── extract_continental.py     # stream continental feature slices off NCI gadi (read-only)
├── compare_continental.py     # memory-streamed continental comparison vs OPeNDAP product
├── compile_models.py          # treelite-compile LightGBM members to native .so (exact, ~1.5x)
└── step3_fused.py             # fused-path prototype (folded into ../step3); subsampling sweep
```

> **Note:** `../step3_ensemble_prediction.py` now uses a **fused predict+combine** (features
> read once, all members predicted per tile, quantiles written directly — no 30 intermediate
> files), 1.9–2.1× faster and bit-identical to the previous path. It auto-uses
> treelite-compiled LightGBM members from `compile_models.py` when present. See `EFFICIENCY.md`.

The pipeline reads/writes under a working tree (default
`/home/prl900/Downloads/wald_eflux/`, set by constants near the top of each
`../step*.py`):

```
wald_eflux/
├── input/                     # raw inputs (mirrors NCI ub8/au layout) + static/
│   └── static/                # land/sea mask (defines target grid), VegH, C4, etc.
├── interim_500m/              # step1 output (one var per year)
├── features_500m/             # step2 output (25 feature stacks)
├── results/
│   ├── variables.txt          # ordered feature list (from reference training)
│   ├── models/ensemble/v2.1/  # reference 30-model ensemble (per flux)
│   ├── models/quantile/v2.1/  # distilled quantile models (per flux)
│   ├── AusEFlux/{flux}/        # ensemble-path final product
│   └── AusEFlux_quantile/{flux}/  # quantile-path final product
└── reference/AusEFlux_v2.1/   # downloaded published product (comparison target)
```

---

## 4. Prerequisites

- **`uv`** (the tooling and all `../step*.py` scripts are `uv run`-able; deps are
  pinned inline via PEP 723 or in `pyproject.toml`).
- **NCI access** for the inputs and the reference models/static rasters:
  - Raw inputs and the published product are public on **NCI THREDDS** (`ub8/au`)
    via OPeNDAP — no login needed.
  - The trained ensemble models, `variables.txt`, and static ancillary rasters live
    in the project space `/g/data/xc0/project/AusEFlux/` and were copied via `scp`
    from gadi (requires an NCI account with `xc0` membership). See §6.

Set up the local tooling env once:

```bash
cd benchmark
uv sync          # creates .venv from pyproject.toml / uv.lock
```

---

## 5. Data sources

| Item | Source | Obtained by |
|---|---|---|
| Raw inputs (MODIS MCD43A4/MYD11A1, OzWALD, SILO) | NCI THREDDS `dodsC/ub8/au/...` (OPeNDAP) | `fetch_inputs.py` (spatial subset) |
| Published v2.1 product | NCI THREDDS `ub8/au/AusEFlux/v2.1/monthly` | `fetch_inputs.py` / `compare_continental.py` |
| Trained ensemble (30/flux) + `variables.txt` | `gadi:/g/data/xc0/project/AusEFlux/results/models/ensemble/v2.1/` and `.../results/variables.txt` | `scp` |
| Static rasters (land/sea mask, urban/water mask, bare-soil NDVI, VegH, C4 grass) | `gadi:/g/data/xc0/project/AusEFlux/data/` | `scp`, then `prep_static.py` |
| Precomputed continental features (for large-scale runs) | `gadi:/g/data/xc0/project/AusEFlux/data/500m/` | `extract_continental.py` (read-only stream) |

> **Note on the trained models:** the reference RandomForest members were pickled
> with **scikit-learn 1.3.2**; newer scikit-learn cannot unpickle them. `step3` is
> pinned accordingly (see §7, Fix 1).

---

## 6. Running the benchmark — quick start (small test tile)

A small bounding box (SE Australia, ~640×640 px @ 500 m) validates numerical
equivalence quickly. Because step2 computes anomalies against a **2003–2022
climatology**, the *full climatology period* must be harmonised — but only over the
small bbox (cheap via OPeNDAP).

```bash
cd benchmark

# 1. Download raw inputs (2002–2022 for the climatology + target year) and the
#    published product, spatially subset to the bbox:
uv run fetch_inputs.py --years 2002-2022 --target-year 2020

# 2. Copy the reference models + static rasters from gadi (one-off):
scp gadi:/g/data/xc0/project/AusEFlux/results/variables.txt \
    /home/prl900/Downloads/wald_eflux/results/
scp -r gadi:/g/data/xc0/project/AusEFlux/results/models/ensemble/v2.1/{GPP,ER,NEE,ET} \
    /home/prl900/Downloads/wald_eflux/results/models/ensemble/v2.1/
scp gadi:/g/data/xc0/project/AusEFlux/data/{land_sea_mask_500m.nc,VegH_1km_2007_2010.nc,ndvi_of_baresoil_500m.nc,Aust_C4_grass_cover_percentage.tif,urban_water_mask_500m.nc} \
    /home/prl900/Downloads/wald_eflux/input/static/

# 3. Subset the grid-aligned static rasters to the bbox (defines the target grid):
uv run python prep_static.py

# 4. Run the pipeline: step1 (per year, 4 in parallel) → step2 → step3 (4 fluxes):
bash run_pipeline.sh 2020 2003 2022 4

# 5. Compare the ensemble-path product to the published product:
uv run python level4_report.py --year 2020 --plots
#    → LEVEL4_2020.md + report_plots/

# 6. (Optional) Build & validate the quantile-distillation path:
uv run distill_quantile_models.py --fluxes GPP ER NEE ET
uv run ../step3_ensemble_prediction_quantile.py --year 2020 --mode predict --fluxes GPP ER NEE ET
```

`fetch_inputs.py` accepts `--west/--east/--south/--north` to change the bbox and
`--skip-product` / `--overwrite`. The default bbox is in the script header.

### Why a buffer?
`prep_static.py` subsets the static grid to a small buffer beyond the comparison
region so that reprojection edge effects fall *outside* the area being compared; the
published product sits strictly inside it, giving an exact 1:1 grid (verified to
0.0° coordinate difference).

---

## 7. Bug fixes applied to the pipeline

The benchmark surfaced two real defects; both are fixed in the parent `../step*.py`
and documented here for the original authors:

1. **`scikit-learn` not pinned → RandomForest models fail to load (blocker).**
   `step3` left `scikit-learn` unpinned; uv resolved ≥1.4, which cannot unpickle the
   reference RF joblibs (`AttributeError: ... 'monotonic_cst'`), silently losing half
   the ensemble. **Fixed:** pinned `scikit-learn==1.3.2`, `numpy<2`, `requires-python
   <3.13` in both `step3*` PEP 723 headers. *(Reference trained with sklearn 1.3.2 —
   see `cbur24/AusEFlux` `requirements.txt`.)*

2. **SILO inputs reprojected upside-down (root cause of most residual error).**
   SILO sources (rain/radiation/vp) have *ascending* latitude while MODIS/OzWALD/the
   land mask are *descending*; `_affine()`/`_make_warper` in `step1` assumed north-up,
   so only the SILO variables were vertically flipped (confirmed: interim SRAD
   correlated 1.000 with the *flipped* raw). Vegetation features masked the error in
   GPP, but ET (water-driven) collapsed to R²≈0.84 and a flipped band of ocean-zeros
   masked productive pixels. **Fixed:** `_make_warper` now flips ascending-latitude /
   descending-longitude sources to north-up before reprojecting. This single fix moved
   all fluxes to R² ≈ 0.98–0.997. *(The reference avoids this by reprojecting with
   `odc.geo`/`odc.reproject` on labelled coordinates rather than a hand-built affine.)*

A `0/0`→`1.0` guard was also added to `_fractional_anomalies` in `step2`; it is now
off by default (`--anom_guard none`, reference behaviour).

The remaining ≤3% differences on the bbox were first described as inherent. The Gadi
experiments (2026-10-27, [`GADI_REPORT.md`](GADI_REPORT.md)) showed otherwise: they
come from semantic deviations, chiefly linear quantiles instead of the reference's
nearest-rank and a no-data mask without `LAI_anom`. With the current defaults (nearest
rank, reference mask, `--mask_test nan`), step3 reproduces the published 2025 product
bit-for-bit from the reference features. The bbox numbers in this harness were measured
with the old defaults and have not been re-run.

---

## 8. Running at large scale / full continent

A from-scratch continental run is impractical on a single workstation (the 2003–2022
climatology at 500 m needs >1 TB of raw downloads and days of compute). Two scalable
strategies are provided.

### 8a. Continental prediction from the reference's precomputed features
The reference's stacked continental feature datasets live at
`gadi:/g/data/xc0/project/AusEFlux/data/500m/` (~59 GB each). `extract_continental.py`
streams **only the target-year slice** of each feature off gadi **read-only** — using
the project's miniconda (`/g/data/xc0/software/python/miniconda3/bin/python`, which has
xarray+netCDF4) to read in latitude bands and stream raw `float32` bytes to stdout,
which are reassembled into local netCDFs. This keeps gadi-side RAM low (~200 MB),
writes nothing on gadi, and transfers ~68 GB (vs 1.7 TB for the whole files).

```bash
# 1. Stream the target-year (e.g. 2025) continental features (~30 min, ~68 GB):
uv run python extract_continental.py        # → wald_eflux/features_500m_cont/

# 2. Re-distill quantile models on a CONTINENTAL sample (the bbox-trained models do
#    not generalise to arid/tropical Australia):
uv run distill_quantile_models.py --fluxes GPP NEE \
    --features-dir /home/prl900/Downloads/wald_eflux/features_500m_cont \
    --years 2025 --n-per-year 500000

# 3. Predict continentally with the fast quantile path (~4–5 h/flux):
uv run ../step3_ensemble_prediction_quantile.py --year 2025 --mode predict \
    --fluxes GPP NEE \
    --features_path /home/prl900/Downloads/wald_eflux/features_500m_cont \
    --static_path  /home/prl900/Downloads/wald_eflux/input/static/_full

# 4. Compare to the published 2025 product, streamed per time-step over OPeNDAP:
uv run python compare_continental.py \
    --new-root /home/prl900/Downloads/wald_eflux/results/AusEFlux_quantile \
    --fluxes GPP NEE --year 2025
```

> The **ensemble** path also runs continentally but is ~58 h/flux (30 models, RF
> dominated); the **quantile** path (3 models) is the one that scales — this is the
> point of the distillation approach.

### 8b. Larger region via the full pipeline from raw inputs
To exercise `step1`+`step2`+`step3` (including the fixes) end-to-end at a larger but
tractable scale, re-run §6 with a bigger bounding box (disk- and time-bounded by the
climatology download). `prep_static.py` and `fetch_inputs.py` both take explicit
`--west/--east/--south/--north`.

### Scaling notes
- **Memory:** every stage is latitude-band tiled (`TILE_ROWS`, `MAX_TILE_MEM`), so
  peak RSS stays bounded regardless of domain size — the continental run uses the same
  code path as the bbox.
- **Time (observed):** step1 ≈ 1m53s/yr (full) or ≈6 s/yr (SILO-only re-run); step2
  ≈ 6m40s (20 yr); step3 ensemble ≈ 25 min/flux, quantile ≈ 1.9 min/flux — at bbox
  scale; multiply by the pixel-count ratio (continental ≈ 140× the bbox).
- **No silent caps:** comparisons restrict to pixels valid in *both* products and
  report the count; coordinate alignment is verified, not assumed.

---

## 9. The tools (reference)

| Script | Purpose | Key options |
|---|---|---|
| `fetch_inputs.py` | OPeNDAP-subset raw inputs + published product into the pipeline's input layout | `--years`, `--target-year`, `--west/east/south/north`, `--skip-product`, `--overwrite` |
| `prep_static.py` | Subset land/sea mask + bare-soil NDVI + urban/water mask to the bbox (originals kept in `static/_full/`) | `--west/east/south/north` |
| `run_pipeline.sh` | Orchestrate step1 (parallel) → step2 → step3 | `TARGET_YEAR CLIM_START CLIM_END N_PARALLEL` |
| `compare.py` | Two-NetCDF metrics (R², RMSE, MAE, bias, mask Jaccard, %-within-tol) + maps/scatter/histograms; crops to overlapping extent | `--new-var/--ref-var`, `--plots`, `--json` |
| `level4_report.py` | Run `compare` across all fluxes/bands → `LEVEL4_<year>.md` + plots | `--year`, `--plots` |
| `distill_quantile_models.py` | Distil the ensemble into 3 quantile-loss LightGBM models/flux (sample → 30-model preds → 25/50/75 percentiles as targets) | `--fluxes`, `--years`, `--n-per-year`, `--features-dir` |
| `extract_continental.py` | Read-only stream of a year's continental feature slices off gadi | `--year`, `--out` |
| `compare_continental.py` | Memory-streamed continental comparison vs the OPeNDAP product | `--new-root`, `--fluxes`, `--year` |
| `compile_models.py` | Compile LightGBM ensemble members to native `.so` (treelite/tl2cgen) for ~1.5× exact predict | `--models-folder` |
| `step3_fused.py` | Benchmark prototype of the fused predict+combine (now folded into `../step3_ensemble_prediction.py`); supports `--n-models` for the subsampling sweep | `--flux`, `--year`, `--n-models` |
| `efficiency / EFFICIENCY.md` | Speed-vs-equivalence results for the optimizations | — |

---

## 10. Notes for the original authors (`cbur24/AusEFlux`)

- The **SILO latitude flip** (Fix 2) only manifests when reprojecting with a manually
  built affine; the notebook/`src` path using `odc.reproject` on labelled coordinates
  is not affected. Worth a regression check that harmonised SILO variables correlate
  with the raw source in the *correct* orientation, since the symptom is subtle
  (vegetation-dominated GPP looked fine; ET did not).
- The **scikit-learn 1.3.2** model-pickle dependency is a portability hazard for the
  saved ensemble; consider re-exporting models in a version-robust format (ONNX / a
  plain tree dump) for long-term reuse.
- The refactored `step3_ensemble_prediction_quantile.py` ships a `--mode train` that
  trains on the OzFlux tower CSVs (`data/training_data/`), whereas its README describes
  *distillation from the ensemble*. These produce different uncertainty bands; this
  harness implements the distillation interpretation in `distill_quantile_models.py`.
  Reconciling the script with its docstring would remove the ambiguity.

---

## 11. References

- Repository: <https://github.com/cbur24/AusEFlux>
- Paper: Burton et al. (2023), *Biogeosciences* 20(19):4109–4134,
  <https://doi.org/10.5194/bg-20-4109-2023>
- Product (500 m, NCI THREDDS): <https://thredds.nci.org.au/thredds/catalog/ub8/au/AusEFlux/catalog.html>
- Product (5 km, Zenodo): <https://zenodo.org/records/14942139>
- OzFlux / TERN eddy covariance data: <https://dap.tern.org.au/thredds/catalog/ecosystem_process/ozflux/catalog.html>
