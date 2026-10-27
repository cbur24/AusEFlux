# AusEFlux refactor: handover (branch `code-refactor`, 2026-10-28)

Read this first when picking the work up. It records where the code stands, what
has been proven against the published v2.1 product, how to run an annual update on
NCI Gadi, what it costs compared with the canonical implementation, and what to do
next. Detailed evidence lives in `benchmark/GADI_REPORT.md` (experiment ledger),
`benchmark/gadi/README.md` (Gadi workflow and every command used) and the step
scripts' `--help`.

## 1. Status in one paragraph

The four flat scripts on this branch (`step1_harmonisation.py`,
`step2_feature_datasets.py`, `step3_ensemble_prediction.py`,
`step3_ensemble_prediction_quantile.py`) replace the notebook-driven reference
pipeline in `master:src/` (Chad Burton's `cbur24/AusEFlux`). Run end to end from
raw inputs on Gadi, they reproduce the published v2.1 GPP and NEE products for
2025 across the whole continent within tolerance (R² ≥ 0.9995, RMSE ≤ 2.2 % of the
reference std, 98 % of pixel-months bit-equal, annual sums within 0.05 %). Fed the
reference's own features, step3 reproduces the published product bit-for-bit. The
defaults of every script now follow the reference semantics; every earlier
deviation is still available as a flag. A GPP + NEE annual update costs about
285 SU against roughly 3,100 SU for the reference's prediction stage alone. Nothing
remains queued on Gadi; the whole programme used 1.14 KSU of project `ea89`.

## 2. State of the code

### Scripts

| Script | Role | Defaults (reference semantics) | Notable options |
|---|---|---|---|
| `step1_harmonisation.py` | Raw MODIS / OzWALD / SILO → monthly 500 m interim, one file per variable-year | `--engine rasterio`, `--lst_qc ref` (MYD11A1 QC applied except 2022/2023), `--modis_nodata poison` (reference NDWI/kNDVI NaN mask) | `--vars` sharding, `--threads`, `--orientation_check`, `--engine odc` (comparison only), atomic `.part.<pid>` writes, skip-if-complete |
| `step2_feature_datasets.py` | Interim 2003→year → 28 feature stacks (gap-fill, veg fractions, cumulative rain, ΔT, anomalies, C4) | `--no_bare_clip`, `--anom_guard none`, bottleneck/flox off | `--stages`, `--vars`, `--years_out`, `--reuse_clim` (stored climatologies in `{results}/climatology/`), `--threads`/`--pool`, `--tile_rows`, `--lat_rows S:E` windows |
| `step3_ensemble_prediction.py` | 30-member ensemble predict + nearest-rank 25/50/75 percentiles in one fused pass | `--quantile_method nearest`, `--mask_vars LAI_anom,VegH,NDWI,rain_anom,Tavg`, `--mask_test nan`, `--valid_only` | `--months` sharding + `--merge`/`--merge_partial`, `--threads`, `--models_path` (tl2cgen-compiled LightGBM accepted), `--dump_members`, `--lat_rows`, `--selftest`, run-config attributes checked on merge |
| `step3_ensemble_prediction_quantile.py` | Distilled 3-model fast path (one LightGBM per band) | same mask and path conventions | `--mode train` (distil from member targets; `--loss l2` recommended) / `predict` |

Every script is a single file with a PEP 723 header (`uv run script.py --help`).
Paths resolve from `AUSEFLUX_ROOT`, `AUSEFLUX_INPUT`, `AUSEFLUX_STATIC` and can be
overridden per script (`--input_path`, `--static_path`, `--interim_path`,
`--features_path`, `--results_path`, `--models_path`, `--variables_file`,
`--output_path`). `step3_*` pin `scikit-learn==1.3.2` and `numpy<2`, which the
reference RandomForest joblibs require.

### Decisions on the deviations found in the review

Rule agreed with the project owner: a deviation from the reference is kept only if
it is cheaper **and** within tolerance.

| Deviation | Measured effect | Decision |
|---|---|---|
| Linear-interpolated quantiles | NEE annual sink 2.8–3.3 % weaker (fails) | reverted: nearest-rank, which is also 200× faster |
| `LAI_anom` dropped from mask | +0.41 % predicted pixels | reverted |
| Mask on `~isfinite` instead of `isnull` | 0.05–0.09 % of pixels | reverted (`--mask_test nan`) |
| Anomaly 0/0 → 1.0 guard | ~0.9 M pixel-months in 3 windows | reverted (`--anom_guard none`) |
| Bare fraction clipped at 0 | 579 k pixel-months | reverted (`--no_bare_clip`) |
| LST QC skipped ≥ 2022 | LST RMSE 0.43 °C | reverted (`--lst_qc ref`) |
| NaN sub-pixels excluded when averaging NDWI/kNDVI | north-window 2024 fails (GPP RMSE 6 %, NEE 9.5 % of std) | reverted (`--modis_nodata poison`) |
| rasterio instead of odc-geo warp | exact for same-grid variables; ≤ 3e-4 elsewhere | kept (faster, simpler) |
| flox / bottleneck | move results away from the reference | kept off |

### Benchmark tooling

- `benchmark/gadi/`: comparison tools (`compare_products.py` scorecard with
  per-month, per-MVG and annual sums; `member_parity.py`; `feature_parity.py`;
  `recombine_members.py`; `merge_months.py`), PBS templates in `pbs/` and the
  production submitter `pbs/submit_step3_year.sh`. `e5/` holds the exact E5 scripts.
- `benchmark/smoke/`: local synthetic-data harnesses for each step (see its README;
  baselines must be regenerated once).
- `benchmark/GADI_REPORT.md`: experiment ledger E0–E7 with the deviation table,
  timing table and per-experiment notes. `benchmark/REPORT.md`,
  `benchmark/BENCHMARKING.md` and `benchmark/EFFICIENCY.md` are the earlier local
  bbox results, superseded where they disagree.

### Validation evidence (all on Gadi, 2026-10-27/28)

| Check | Result |
|---|---|
| E1 recombine 30 reference member files with nearest-rank | bit-identical to published, 4 fluxes × 2024/2025 |
| E2a step3 on reference features, 3 continental months | all LightGBM members bit-identical (incl. dart); RF ≤ 1 ulp at ≤ 2 px; product bit-identical |
| E2b reference's own drift (2024 features vs its product) | ≤ 3 % of pixels differ (Dec-2024 gap-fill); sets the tolerance floor |
| E3 step2 on reference interim (3 windows) | 16/28 features bit-identical, rest float32 rounding |
| E4 step1 from raw (2024, 2025) | 11/13 interim vars exact or float noise; LST/VegH RMSE 0.004 °C / 0.002 m (reference GeoBox affine) |
| E5 end to end, continental 2025 GPP + NEE | PASS all bands; 2024 windows PASS with `poison` |
| E6 efficiency | every optimisation bit-identical; thread knee 4–12 |
| E7 distilled fast path (L2 loss) | GPP R² 0.996 / NEE 0.986, annual sums within 0.02 %, per-pixel RMSE 6–12 % of std (fails the 5 % check); 21 SU/flux-yr |

ER and ET were verified through E1 (exact recombination) only, not end to end.

## 3. Running an annual update on NCI Gadi

Read `benchmark/gadi/README.md` for the full workflow; this is the short form.
Constraints: compute on project `ea89` (`-q normal`, 2 SU per core-hour); never write
under `/g/data/xc0` (over quota); work under `/scratch/ub8/<user>/auseflux` (100-day
purge); durable copies under `/g/data/ub8/au/blobs/auseflux_refactor/`. Login nodes
have a 30-minute CPU limit and very slow numerics, so every computation goes through
PBS.

```bash
# workstation → Gadi (code only; .venv and .git excluded)
rsync -av --exclude .git --exclude '.venv' --exclude '__pycache__' \
      /home/prl900/github/AusEFlux/ gadi:/scratch/ub8/pl5189/auseflux/code/AusEFlux/

# on Gadi
W=/scratch/ub8/pl5189/auseflux; C=$W/code/AusEFlux; P=$C/benchmark/gadi/pbs
XC=/g/data/xc0/project/AusEFlux            # reference: read only
source $W/envs/activate_step12.sh          # or activate_step3.sh for step3 jobs
```

The two uv-built venvs (`$W/envs/{step12,step3}`) and their activation scripts exist;
they export `AUSEFLUX_INPUT=/g/data/ub8/au`,
`AUSEFLUX_STATIC=$XC/data`, `AUSEFLUX_ROOT=$W`. If they are purged, rebuild them on a
login node with uv from the scripts' PEP 723 headers (compute nodes have no network),
or fall back to the reference env `$XC/env/py310`.

1. **step1** (≈ 21 min wall, 13 SU per year), sharded by variable group:
   ```bash
   for v in climate:SILO:rain_prior NDWI:kNDVI NDVI:LAI LST VegH; do
     qsub -N s1_${v//:/_} -v YEAR=2026,VARS=$v,EXTRA="--orientation_check" $P/step1_year.pbs
   done
   ```
   Output `$W/interim_500m/{var}/{var}_500m_2026.nc`. The interim tree must hold every
   year from 2003 (rain from 2002); symlink the reference years from
   `$XC/data/interim_500m` as E5 did (`results/e5/interim/`).
2. **step2** (≈ 2.5 h critical path, 114 SU with fresh climatologies; est. 15–25 SU
   with `--reuse_clim` once climatologies are stored), sharded by stage and variable
   with `--years_out 2026-2026`; see the 22-job layout in `benchmark/gadi/e5/submit_step2.sh`.
   `vegfrac` is CPU-bound (1 h 46 min at 8 cpus); everything else is I/O-bound.
3. **step3** (GPP 72 SU, NEE 86 SU per flux-year): compile the LightGBM members once per
   flux with `benchmark/compile_models.py` into `$W/results/models_compiled/{flux}`
   (RF members are symlinked joblibs), then
   ```bash
   DRY_RUN=0 $P/submit_step3_year.sh GPP 2026 \
     --features_path $W/features_500m --static_path $XC/data \
     --variables_file $XC/results/variables.txt --output_path $W/results/AusEFlux
   ```
   Defaults: 4 threads, 3 months per job, dependent merge. Three E5 jobs landed on nodes
   2–3× slower than the rest and cost about 50 SU extra; `MONTHS_PER_JOB=1` limits that
   exposure. Month parts are skip-if-exists, so rerunning a failed month is safe.
4. **Score** against the previous product or the reference:
   `qsub -v AUSEF_ENV=step12,... $P/compare.pbs` running
   `benchmark/gadi/compare_products.py NEW REF --flux GPP --mvg --plots`
   (use `benchmark/gadi/e5/th_nee.json` thresholds for NEE).

## 4. Efficiency against the canonical implementation

| Stage | Reference v2.1 (2025 run) | This branch (measured, E4/E5) | Why |
|---|---|---|---|
| step1 per year | ≈ 3.5 h on a dask cluster | 21 min wall, 13 SU, RSS < 4 GB | rasterio warps with `num_threads`, chunked writes, MODIS band read once, VegH warped once |
| step2 | ≈ 6.5 h for full 2003–2025 stacks (1.7 TB written) | 2.5 h, 114 SU, 45 GB written (`--years_out`) | annual-update mode writes only the new year; tile-chunked interim reads (68 s → 0.55 s per tile); stored climatologies |
| step3 per flux-year | 30 jobs × 26 cpus × ≤ 1 h ≈ 1,560 SU, plus ~1 h combine | 72 (GPP) / 86 (NEE) SU, 10 GB peak | fused predict + combine, valid-pixel-only prediction (−52 % rows), tl2cgen-compiled LightGBM (1.85×), nearest-rank sort-and-take quantiles, 4-thread jobs at the scaling knee |
| GPP + NEE annual update | ≈ 3,100 SU + step1/step2 | ≈ 285 SU (≈ 190 with stored climatologies) | |

Every speed-up above is bit-identical to the unoptimised path (E6). Not adopted:
ONNX / Hummingbird RF (inexact or wrong), GTIL (4–9× slower), member subsampling
(fails tolerance), `normalsr` queue (+20 % per SU only for LightGBM).

## 5. Next steps (in priority order)

1. **Make the step3 submitter resilient to slow nodes**: default `MONTHS_PER_JOB=1`
   or add a per-job walltime watchdog that resubmits; E5 lost ≈ 50 SU this way.
2. **ER and ET end to end**: run step3 for ER and ET on the E5 2025 feature set
   (`/scratch/ub8/pl5189/auseflux/features_500m_e5`, until purged) and score them;
   ≈ 160 SU. Their member recombination is already exact (E1).
3. **Stored-climatology annual update**: run step2 2025 with `--reuse_clim` against the
   E5 climatologies and confirm the 15–25 SU estimate and bit-identity with E5.
4. **2024 full-year acceptance** if a complete scorecard for both years is wanted
   (≈ 300 SU for GPP + NEE); the three windows already pass.
5. **Deduplicate the two step3 scripts** (shared tile reader, mask, writer); keep
   both CLIs.
6. **step1 completeness check**: skip-if-exists cannot detect a partially written
   legacy file (pre-atomic-write); add a size or attribute check.
7. **Distilled fast path**: it fails only the per-pixel RMSE check. If a cheaper
   product is wanted for reporting of continental and MVG sums, it is already within
   0.02 % on annual totals; otherwise park it.
8. **Merge to `master`** once the owner agrees with the defaults; the reference `src/`
   stays untouched on `master`.

## 6. Where things are

| Item | Location |
|---|---|
| Code on Gadi (synced 2026-10-28, identical to this commit's parent tree) | `/scratch/ub8/pl5189/auseflux/code/AusEFlux/` |
| E5 products (2025 GPP/NEE), window products, scorecards, logs, scripts | `/scratch/ub8/pl5189/auseflux/results/e5/` and durable copy `/g/data/ub8/au/blobs/auseflux_refactor/e5/` |
| Other experiment outputs (E0–E7) | `/scratch/ub8/pl5189/auseflux/results/{e0,e1,e2a,e2b,e3w,e4,e6w,e7}/` (100-day purge; `e2a` member dumps ≈ 100 GB can be deleted) |
| New interim 2024/2025, E5 features | `/scratch/ub8/pl5189/auseflux/interim_500m/`, `/scratch/ub8/pl5189/auseflux/features_500m_e5/` |
| Compiled models | `/scratch/ub8/pl5189/auseflux/results/models_compiled/{GPP,NEE}/` |
| Reference code, models, products, interim, features | `/g/data/xc0/project/AusEFlux/` (read only); published products `/g/data/ub8/au/AusEFlux/v2.1/monthly/` |
| Raw inputs | `/g/data/ub8/au/{MODIS/mosaic/MCD43A4.006,MODIS/mosaic/MYD11A1.006,OzWALD/8day,OzWALD/daily/meteo,SILO}` |
| Original review plan | `~/.claude/plans/review-the-code-in-glistening-salamander.md` on the workstation |

Known data quirks: 2025 MCD43A4 files store reflectance ×10000 without a scale
factor and four composites are entirely fill; the 2024 reference product uses
`days since 2003-01-15` while 2025 uses `days since 2025-01-15` (compare on
year-month, which every tool here does); the reference interim tree contains an
empty `.ipynb_checkpoints` directory that step2 now ignores.
