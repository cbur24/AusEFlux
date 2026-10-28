# benchmark/gadi: running and validating the refactored pipeline on Gadi

Tools and PBS templates for Phase B of the plan
(`~/.claude/plans/review-the-code-in-glistening-salamander.md`): experiments
E0 to E7 that compare the refactored pipeline with the published AusEFlux v2.1
products for 2024 and 2025. Results and the deviation-ablation table are in
[`../GADI_REPORT.md`](../GADI_REPORT.md); this file covers the workflow and the
commands that produced them.

**Windowed design.** Per-pixel questions (engine parity, feature parity, drift
floor, efficiency, distillation) were answered on three stratified latitude
windows instead of the whole continent, via `--lat_rows START:END` in step2 and
step3 (about 6 % of the pixels, about 1/15 of the cost):

| window | rows | region |
|---|---|---|
| north | `600:1000` | tropical savanna |
| centre | `2800:3200` | arid centre |
| south | `5000:5400` | temperate south |

Windowed outputs carry an `_rSTART-END` suffix and a global attribute
`auseflux_lat_rows`; per-pixel computations make them bit-identical to those
rows of a full run. step2 needs a separate `--results_path` per window (its
climatologies are window-specific). Continental runs were kept for step1
(E4), the combine checks (E1), three step3 months (E2a), the distilled path
(E7) and the end-to-end acceptance (E5).

**Defaults now follow the reference.** step3: `--quantile_method nearest`,
`--mask_vars LAI_anom,VegH,NDWI,rain_anom,Tavg`, `--mask_test nan`,
`--valid_only`. step2: `--no_bare_clip`, `--anom_guard none`, bottleneck/flox
off. step1: `--lst_qc ref` (the reference rule) and `--modis_nodata poison` are now the defaults; the explicit flags below are kept. E5 ran step1 with `--modis_nodata exclude` (the default at the time); the `poison` rerun of NDWI/kNDVI is in `results/e5/poison/`.
Older commands below spell these flags out; they are no-ops now.

Every tool streams the continental grid (12 × 6800 × 8400) in latitude chunks
of about 400 rows and accumulates statistics in float64, so no tool holds a
whole band in memory. Peak RSS for `compare_products.py` is about 6 GB.
Products are aligned on (year, month), not on raw time values, and on
coordinates within 1e-4°.

| file | purpose |
|---|---|
| `compare_products.py NEW REF --flux F` | Product vs product, per band (25th/median/75th). Reports R², RMSE, MAE, bias, max abs Δ, % within 1 % of range, % bit-exact and mask Jaccard, overall, per month and per Major Vegetation Group (MVG). Also gives continental monthly means (cos-lat weighted), annual sums (continental and per MVG, TgC/yr or km³/yr), the tolerance scorecard, and with `--plots` a diff map, density scatter and monthly series. Writes `metrics.json` and `metrics.md`. |
| `recombine_members.py DIR --flux F --method nearest\|linear` | E1. Recombines the 30 member files into a product (`--out`), or with `--check REF` runs the exactness gate (exit 0 only if bit-identical). |
| `member_parity.py NEW REF --flux F` | E2a. Per-member, per-month max/mean abs Δ, NaN-mask mismatches and the 10 largest differences. Reads whole member files or `parts/{member}_mMM.nc`. |
| `feature_parity.py NEW REF --years 2024-2025` | E3. Compares `{var}_500m.nc` stacks (R², RMSE, bias, max abs Δ, mask mismatches, NaN/Inf fractions per month). With `--interim` (E4) it compares the `{var}/{var}_500m_{year}.nc` layout and adds the orientation diagnostic. |
| `merge_months.py PARTS --flux F --year Y --out FILE` | Assembles 12 month parts from a sharded step3 run, checks that every month is present, and verifies a bit-exact read-back. |
| `make_fixture.py --out DIR` | Builds a small synthetic fixture (24×30 px, 12 months, 4 members, fake MVG GeoTIFF in EPSG:3577, features and interim files) for local tests. |
| `_common.py` | Shared helpers: chunked reader, alignment, accumulators and netCDF writer. Every tool imports it from its own directory. |
| `pbs/*.pbs`, `pbs/submit_step3_year.sh` | PBS templates (project ea89, queue normal) and a submitter for a full step3 year (production recipe by default: 4-cpu jobs of 3 months + dependent merge). |

Every script has a PEP 723 header (`uv run tool.py --help`). Each one also runs
with any python ≥ 3.10 that has numpy, xarray, netCDF4 and pandas, plus
matplotlib/rasterio for plots and MVG. The local test suite also passes with
python 3.10, numpy 1.26 and xarray 2024.10, the versions in the reference env.

## Tolerance scorecard (defaults, from the plan)

Per flux and band, over continental jointly-valid pixel-months:

- R² ≥ 0.99 (GPP, ET) or ≥ 0.97 (NEE, ER)
- |bias| ≤ 1 % of mean |ref|
- RMSE ≤ 5 % of std(ref)
- mask Jaccard ≥ 0.999
- every continental monthly mean within 2 %
- annual continental total within 1 %
- annual total of each MVG within 2 %. Only MVGs covering ≥ 0.1 % of the area count; classes 0, 28 and 99 are excluded.

To override any of these, pass a JSON file with `--thresholds`, for example:

```json
{"r2_min": {"NEE": 0.98}, "mask_jaccard_min": 0.9995,
 "monthly_rel_denominator": "mean_abs", "mvg_exclude": [0, 28, 99, 24]}
```

`monthly_rel_denominator: "mean_abs"` divides each monthly difference by the
mean |ref| of the series instead of that month's value. Use it for NEE, whose
monthly means can be close to zero. E2b sets the drift floor, and the
thresholds are calibrated against it.

## One-time setup

```bash
# from the workstation
rsync -av --exclude .venv --exclude .git /home/prl900/github/AusEFlux/ \
      gadi:/scratch/ub8/pl5189/auseflux/code/AusEFlux/

# on gadi
W=/scratch/ub8/pl5189/auseflux
C=$W/code/AusEFlux; G=$C/benchmark/gadi; P=$G/pbs
XC=/g/data/xc0/project/AusEFlux          # read only, never write here
mkdir -p $W/{logs,interim_500m,features_500m,results,envs,e0,e1,e2a,e2b,e3,e4,e5,e6,e7}
```

The templates source `$W/envs/activate_step12.sh` (step1/2) or
`$W/envs/activate_step3.sh` (step3, merge, recombine, member_parity).
`compare.pbs` defaults to the step3 env; pass **`AUSEF_ENV=step12`** for
`compare_products.py`, which needs rasterio for the MVG raster (the step3 env
has none). Both files must
exist before you submit anything. Each one activates a venv built with uv on the login
node from the scripts' PEP 723 headers (compute nodes have no outbound
network): step12 is python 3.12 with xarray, rasterio/GDAL 3.12, odc-geo 0.5.3,
dask; step3 is python 3.11 with scikit-learn 1.3.2, numpy 1.26.4, lightgbm
4.7.0 (the reference used 4.2.0; members are still bit-identical) and tl2cgen
1.0.0. The fallback is the reference env:

```bash
# $W/envs/activate_step3.sh (fallback; step3 needs sklearn==1.3.2 and numpy<2)
module load python3/3.10.0
source /g/data/xc0/project/AusEFlux/env/py310/bin/activate
```

All templates set `AUSEFLUX_INPUT=/g/data/ub8/au`,
`AUSEFLUX_STATIC=/g/data/xc0/project/AusEFlux/data` and `AUSEFLUX_ROOT=$W`.
They put `/usr/bin/time -v` around the python call, so every log has the wall
time and max RSS. The last lines of each log print the elapsed time and an SU
estimate: `ncpus × hours × 2` and the Gadi charge including memory. Logs go to
`$W/logs/`.

### Passing parameters

Use `qsub -v NAME=value,...`. `qsub -v` splits on commas, so:

- list variables (`VARS`, `STAGE`, `MONTH`, `MONTHS`) take `:` or `+` as the separator, for example `VARS=LST:NDVI`;
- `EXTRA` takes extra flags separated by spaces, and must not contain commas;
- `ARGS_FILE` is a file with one CLI argument per line. Use it whenever an argument contains a comma, for example `--mask_vars LAI_anom,VegH,...`. `submit_step3_year.sh` writes this file for you.

Common reference paths used below:

```bash
REFP=$XC/results/AusEFlux/v2.1                        # working-copy products {flux}/AusEFlux_{flux}_500m_quantiles_{yr}_v2.1.nc
PUBP=/g/data/ub8/au/AusEFlux/v2.1/monthly             # published products
MEM25=$XC/results/predictions/annual_update/v2.1/2025 # 2025 members {flux}/{flux}_{lgbm,rf}_{i}.nc
REFF=$XC/data/500m                                    # reference features (step2 output)
REFI=$XC/data/interim_500m                            # reference interim (step1 output)
MODELS=$XC/results/models/ensemble/v2.1
VARS_TXT=$XC/results/variables.txt
REFMASK=LAI_anom,VegH,NDWI,rain_anom,Tavg             # reference no-data mask
```

## Production workflow (one annual update)

The scripted form of this section is `annual_update/run_year.sh YEAR` (see
`annual_update/README.md`): input check, hybrid interim, step1, step2 with the
stored climatologies, step3 for all fluxes, merges and a post-check, chained with
PBS dependencies. The manual steps below are what it runs.

### step1: one year, sharded by variable group (≈ 13 SU)

```bash
for v in climate:SILO:rain_prior NDWI:kNDVI NDVI:LAI LST VegH; do
  qsub -N s1_${v//:/_}_2025 -v YEAR=2025,VARS=$v,EXTRA="--lst_qc ref --orientation_check" $P/step1_year.pbs
done
```

The template requests 12 cpus; the jobs use 1.2 to 2.1 cores effectively, so
`-l ncpus=4,mem=16GB` is enough for every group except climate:SILO (13 to
15 GB OzWALD daily files). Output: `$W/interim_500m/{var}/{var}_500m_2025.nc`.

### step2: annual update with `--years_out`

```bash
F=$W/features_500m
common="--interim_path $W/interim_500m --results_path $F --years_out 2024-2025"
g1=$(qsub -N s2_gf1 -v STAGE=gapfill,VARS=NDWI:kNDVI:NDVI:LAI,EXTRA="$common" $P/step2_stage.pbs)
g2=$(qsub -N s2_gf2 -v STAGE=gapfill,VARS=LST:Tavg:SRAD:VPD,EXTRA="$common" $P/step2_stage.pbs)
g3=$(qsub -N s2_gf3 -v STAGE=gapfill,VARS=rain:VegH,EXTRA="$common" $P/step2_stage.pbs)
qsub -N s2_rest -W depend=afterok:$g1:$g2:$g3 -v STAGE=vegfrac:cmlrain:deltat:anom:c4,EXTRA="$common" $P/step2_stage.pbs
```

The interim tree must hold every year from 2003 (rain from 2002): the
climatologies, rolling windows and interpolation use the full history.
step2 is I/O-bound (145 to 490 % CPU of 800 %); see `../EFFICIENCY.md`.
With 24 threads the automatic tile height drops to the 25-row floor (peak
RSS ≈ 4 × threads × 25-row tile, ≈ 22 GB on the full grid). Stages that read
step2 outputs back (deltat, and anom with stored climatologies, in this
annual update) then decompress each 200-row chunk 8×; step2 prints a NOTE.
If that stage group is slow, run it with `--tile_rows 200` and
`-l ncpus=4,mem=48GB` (4 × 4 × ~1.9 GB tiles + headroom).

### step3: production recipe (E5 measured: GPP 72, NEE 86 SU per flux-year)

4 threads per job, 3 months per job, compiled LightGBM members, reference
defaults. Compile once per flux into a writable model folder (the xc0 models
are read-only), with symlinks to the joblibs:

```bash
MC=$W/results/models_compiled
for f in GPP NEE ER ET; do
  mkdir -p $MC/$f; ln -sf $MODELS/$f/*.joblib $MC/$f/
done
# in a job (needs gcc; RF members are skipped):
python $C/benchmark/compile_models.py --models-folder $MC/GPP --parallel 8
```

Then submit the year with `submit_step3_year.sh`. Its defaults are this
recipe: `THREADS=4` (ncpus = `--threads` = 4, mem = 4 GB × `THREADS`),
`MONTHS_PER_JOB=3` (four jobs with `--months 1-3`, `4-6`, ...), `WALLTIME=04:00:00`,
`MODELS_PATH` = `$W/results/models_compiled` when it holds the flux (else the
plain xc0 models, with a note), and a merge job that depends on all of them
(`afterok`). Flags go to every job through an args file, so commas are safe:

```bash
FLUX=GPP; YEAR=2025; OUT=$W/results/AusEFlux
DRY_RUN=1 $P/submit_step3_year.sh $FLUX $YEAR \
  --features_path $W/features_500m --static_path $XC/data \
  --variables_file $VARS_TXT --output_path $OUT      # prints the qsub commands
$P/submit_step3_year.sh $FLUX $YEAR \
  --features_path $W/features_500m --static_path $XC/data \
  --variables_file $VARS_TXT --output_path $OUT
```

Other knobs: `MONTHS="1 2 3"` (a subset; the merge expects the same months),
`QUEUE=normalsr` (if the project has access; about +20 % per SU),
`--models_path` in the flags overrides `MODELS_PATH`. The old behaviour, one
24-cpu job per month, is `THREADS=24 MONTHS_PER_JOB=1`. `step3_month.pbs`
itself defaults to 4 cpus / 16 GB and passes `--threads ${THREADS:-$NCPUS}`
and the `ARGS_FILE` lines. The ≈ 3 h per quarterly job is an extrapolation
from the windows (±30 %).

### Scoring a product

```bash
qsub -N cmp_GPP_2025 -v AUSEF_ENV=step12,NEW=$OUT/GPP/AusEFlux_GPP_500m_quantiles_2025_v2.1.nc,REF=$PUBP/GPP/AusEFlux_GPP_500m_quantiles_2025_v2.1.nc,FLUX=GPP,OUT=$W/results/cmp_GPP_2025,EXTRA=--plots $P/compare.pbs
```

For NEE pass `--thresholds` with `"monthly_rel_denominator": "mean_abs"`
(monthly means close to zero).

## E5: end-to-end acceptance (hybrid interim)

E5 runs the new pipeline from raw inputs where it matters and reuses the
reference interim for the climatology years. The interim tree combines the
reference years 2003 to 2023 with the new 2024/2025 from E4, built with
symlinks so nothing is copied:

```bash
I5=$W/e5/interim_500m
for d in $REFI/*/; do v=$(basename $d); mkdir -p $I5/$v
  for f in $d/*.nc; do ln -sf $f $I5/$v/; done
  for yr in 2024 2025; do
    [ -e $W/interim_500m/$v/${v}_500m_$yr.nc ] && ln -sf $W/interim_500m/$v/${v}_500m_$yr.nc $I5/$v/
  done
done
```

The new 2024/2025 files replace the reference ones of the same name (same
`ln -sf` target name). Then: one continental step2 run on `$I5` with
`--results_path $W/e5/features_500m --years_out 2024-2025` (sharded as in
the production step2 above), step3 with the production recipe for GPP and
NEE 2025 into `$W/e5/AusEFlux`, and `compare.pbs` (`AUSEF_ENV=step12`)
against `$PUBP`. Gate: scorecard PASS for every band of GPP and NEE. 2024 and
ER/ET are checked on the windows only. Status and results: `../GADI_REPORT.md`.

## Reproducing the experiments

The commands below are the ones the experiment jobs ran (taken from the
`+ python ...` lines that `run_timed` writes to each log), with
`W=/scratch/ub8/pl5189/auseflux`, `C=$W/code/AusEFlux`, `G=$C/benchmark/gadi`,
`XC=/g/data/xc0/project/AusEFlux`. `$W/results/models` is a symlink to
`$XC/results/models`. Each experiment's helper scripts and job files are kept
next to its results under `$W/results/<exp>/`. Results are under
`/scratch/ub8/pl5189/auseflux/results/` (100-day purge applies).

### E0: integrity gate

Ad hoc scripts (value and mask diff xc0 vs ub8, grid and member inventory).
Outputs: `$W/results/e0_reference_stats.json`, `$W/results/e0_parts/`. The
template equivalent of the main check:

```bash
for yr in 2024 2025; do for f in GPP NEE ER ET; do
  qsub -N e0_${f}_${yr} -v AUSEF_ENV=step12,NEW=$REFP/$f/AusEFlux_${f}_500m_quantiles_${yr}_v2.1.nc,REF=$PUBP/$f/AusEFlux_${f}_500m_quantiles_${yr}_v2.1.nc,FLUX=$f,OUT=$W/e0/xc0_vs_ub8_${f}_${yr} $P/compare.pbs
done; done
```

Pass: `max abs Δ = 0`, `% exact = 100`, Jaccard 1 for every band.

### E1: combine isolation (`$W/results/e1/`)

```bash
M25=$XC/results/predictions/annual_update/v2.1/2025
for f in GPP NEE ER ET; do
  python $G/recombine_members.py $M25/$f --flux $f --method nearest \
    --check $XC/results/AusEFlux/v2.1/$f/AusEFlux_${f}_500m_quantiles_2025_v2.1.nc \
    --json $W/results/e1/${f}_2025_nearest_check.json
  python $G/recombine_members.py $M25/$f --flux $f --method linear \
    --out $W/results/e1/AusEFlux_${f}_500m_quantiles_2025_v2.1_linear.nc
done
python $G/recombine_members.py $M25/GPP --flux GPP --method nearest --n_models 20 \
  --out $W/results/e1/AusEFlux_GPP_500m_quantiles_2025_v2.1_nearest_n20.nc   # and --n_models 10
```

2024 uses the historical member files (one 60 GB file per member, 2003 to
2024) with `--year 2024`, for example
`recombine_members.py $XC/results/predictions/historical/v2.1/GPP --flux GPP --method nearest --year 2024 --check <2024 product>`
(checks in `$W/results/e1/{flux}_2024_nearest_check.json`). Run with
`recombine.pbs`; the linear products are scored with `compare.pbs`
(`AUSEF_ENV=step12`) into `$W/results/e1/{flux}_{year}_linear/`.

### E2a: prediction-engine parity (`$W/results/e2a/`)

Continental, 2025, months 1, 6 and 12, GPP and NEE, 24 cpus per job:

```bash
for f in GPP NEE; do for m in 1 6 12; do
python $C/step3_ensemble_prediction.py --fluxes $f --year 2025 --months $m --threads 24 \
  --features_path $XC/data/500m --static_path $XC/data --variables_file $W/results/variables.txt \
  --models_path $W/results/models/ensemble/v2.1 --output_path $W/results/e2a/ref_mask \
  --quantile_method nearest --mask_vars LAI_anom,VegH,NDWI,rain_anom,Tavg --valid_only \
  --dump_members $W/results/e2a/members_ref_mask
done; done
```

Ablations (GPP, month 6, same flags except): `--mask_vars VegH,NDWI,rain_anom,Tavg`
into `abl_a_nolai`; `--all_pixels` into `abl_b_allpix`; `--models_path
$W/results/models_compiled` (tl2cgen) into `abl_c_tl2cgen` with
`--dump_members $W/results/e2a/members_tl2cgen`; each with and without
`--mask_test nan`. Analysis (the helper job files are in the experiment
scratch directory, not in the repo):

```bash
M25=$XC/results/predictions/annual_update/v2.1/2025; R=$W/results/e2a
python $G/member_parity.py $R/members_ref_mask/GPP $M25/GPP --flux GPP --months 1,6,12 --out_md parity_GPP.md --json parity_GPP.json
python $G/compare_products.py $R/ref_mask/GPP/parts/AusEFlux_GPP_500m_quantiles_2025_v2.1_m06.nc \
  $XC/results/AusEFlux/v2.1/GPP/AusEFlux_GPP_500m_quantiles_2025_v2.1.nc --flux GPP --out cmp_GPP_m06
python $G/member_parity.py $R/members_tl2cgen/GPP $M25/GPP --flux GPP --months 6 --members GPP_lgbm_1,...,GPP_lgbm_15 --out_md tl2cgen_vs_ref.md
```

Product parts are also checked with `recombine_members.py --check` on the
dumped members (symlinked per month under `$R/recomb/{flux}/mMM/`).

### E2b: reference drift floor (`$W/results/e2b/`)

2024 on the three windows from the reference's current features, GPP and NEE
(`$W/results/e2b/pbs/e2b_step3.pbs`, `qsub -v FLUX=GPP,ROWS=600:1000`):

```bash
python $C/step3_ensemble_prediction.py --year 2024 --fluxes $FLUX \
  --features_path $XC/data/500m --static_path $XC/data --variables_file $W/results/variables.txt \
  --models_path $W/results/models/ensemble/v2.1 --output_path $W/results/e2b/product \
  --quantile_method nearest --mask_vars LAI_anom,VegH,NDWI,rain_anom,Tavg --mask_test nan --valid_only \
  --dump_members $W/results/e2b/members --lat_rows $ROWS --threads $NCPUS
```

Analysis (`e2b_analysis.pbs`, `qsub -v FLUX=GPP,ROWS=600:1000,WIN=north`):
`member_parity.py` of the windowed member dumps against
`$XC/results/predictions/historical/v2.1/$FLUX --months 1-12`,
`recombine_members.py --year 2024 --check` on the product, and
`compare_products.py` against the 2024 product, into `$W/results/e2b/analysis/`.

### E3 / E3w: feature-stage parity (`$W/results/e3/`, `$W/results/e3w/`)

E3 (continental VPD gapfill and timing) used `step2_stage.pbs` with
`--interim_path $XC/data/interim_500m/ --years_out 2024-2025` and
`--tile_rows 25..100`. E3w ran three variants on the three windows
(`$W/results/e3w/e3w_step2.pbs`, `qsub -v VARIANT=refsem,WIN=north,ROWS=600:1000,FLAGS="--no_bare_clip --anom_guard none"`):

```bash
python $W/results/e3w/code/step2_feature_datasets.py --interim_path $XC/data/interim_500m \
  --results_path $W/results/e3w/$VARIANT/$WIN --static_path $XC/data --lat_rows $ROWS \
  --years_out 2024-2025 --threads 8 --verbose $FLAGS
# VARIANT default:      FLAGS=""                                   (old defaults)
# VARIANT refsem:       FLAGS="--no_bare_clip --anom_guard none"   (now the default)
# VARIANT refsem_fast:  FLAGS="--no_bare_clip --anom_guard none --bottleneck --flox"
```

`$W/results/e3w/code/` holds the step2 copy with the chunked-read fix that was
then merged into the repo. Parity per feature: `e3w_parity.py` and
`e3w_extra.py` (`e3w_compare.pbs`), results in `$W/results/e3w/parity/`.

### E4: harmonisation parity (`$W/results/e4/`)

```bash
for yr in 2024 2025; do
  python $C/step1_harmonisation.py --year $yr --threads 12 --vars climate,SILO,rain_prior --lst_qc ref --orientation_check
  python $C/step1_harmonisation.py --year $yr --threads 8  --vars NDWI,kNDVI --lst_qc ref --orientation_check
  python $C/step1_harmonisation.py --year $yr --threads 8  --vars NDVI,LAI   --lst_qc ref --orientation_check
  python $C/step1_harmonisation.py --year $yr --threads 8  --vars LST        --lst_qc ref --orientation_check
  python $C/step1_harmonisation.py --year $yr --threads 4  --vars VegH       --lst_qc ref --orientation_check
done   # one job each (step1_year.pbs); output $W/interim_500m/
# ablations, 2025:
python $C/step1_harmonisation.py --year 2025 --threads 8 --vars LST --lst_qc new --orientation_check \
  --results_path $W/results/e4/lst_qc_new
python $C/step1_harmonisation.py --year 2025 --threads 12 --vars NDWI,kNDVI,climate,SILO,rain_prior \
  --lst_qc ref --orientation_check --engine odc --results_path $W/results/e4/engine_odc
# parity (e4_parity.pbs):
python $G/feature_parity.py $W/interim_500m $XC/data/interim_500m --interim --years 2024-2025 \
  --out_md $W/results/e4/parity_main.md --json $W/results/e4/parity_main.json
```

The odc run predates the `--engine odc` fix (source GeoBox from float32
coordinates).

### E6w: step3 efficiency (`$W/results/e6w/`)

`$W/results/e6w/scripts/e6w_run.pbs`, `qsub -l ncpus=N -v TAG=tN,THREADS=N,WINDOWS="2800:3200 5000:5400",MODELS=joblib|compiled,QM=nearest|linear`:

```bash
python $C/step3_ensemble_prediction.py --fluxes GPP --year 2025 --months 6 \
  --threads $THREADS --lat_rows $ROWS \
  --features_path $XC/data/500m --static_path $XC/data --variables_file $W/results/variables.txt \
  --models_path <$W/results/models/ensemble/v2.1 | $W/results/models_compiled> \
  --output_path $W/results/e6w/$TAG/w<START-END> \
  --quantile_method $QM --mask_vars LAI_anom,VegH,NDWI,rain_anom,Tavg --mask_test nan --valid_only
```

Tags: `t1 t4 t8 t12 t24 t48` (threads), `c12 c24` (compiled), `sr24`
(normalsr), `lin24` (linear quantiles), `*r` repeats. Exactness:
`scripts/cmp.py` against `t24` and `t1`. RF alternatives: `scripts/rf_alt.pbs`.

### E7: distilled quantile path (`$W/results/e7/`)

Targets are nearest-rank quantiles of the 30 reference member files at sampled
pixels (test windows excluded):

```bash
cd $W/results/e7
python scripts/e7_sample.py --year 2025 --n_rows 300 --n_samples 600000 --workers 8
python scripts/e7_sample.py --year 2024 --n_rows 150 --n_samples 300000 --workers 8     # held out
python scripts/e7_sample.py --year 2025 --n_rows 1200 --n_samples 3000000 --seed 1 --fluxes GPP,NEE --tag _big --workers 8
python scripts/e7_train.py --flux GPP --threads 12 [--families ... --train_tag _big]
# windows, both years, per family (models in $W/results/models_quantile_<fam>):
python $C/step3_ensemble_prediction_quantile.py --year $yr --mode predict \
  --features_path $XC/data/500m --models_path $W/results/models_quantile_$fam \
  --output_path $W/results/e7/win_$fam --fluxes GPP NEE --valid_only \
  --mask_vars LAI_anom,VegH,NDWI,rain_anom,Tavg --lat_rows $ROWS --threads 12
# continental 2025 (24 cpus):
python $C/step3_ensemble_prediction_quantile.py --year 2025 --mode predict \
  --features_path $XC/data/500m --models_path $W/results/models_quantile_$FAM \
  --output_path $W/results/e7/product_continental --fluxes $FLUX --valid_only \
  --mask_vars LAI_anom,VegH,NDWI,rain_anom,Tavg --threads 24
qsub -N e7_cc_$f -v AUSEF_ENV=step12,NEW=<continental product>,REF=$REFP/$f/AusEFlux_${f}_500m_quantiles_2025_v2.1.nc,FLUX=$f,OUT=$W/results/e7/cmp_cont/$f,EXTRA="--plots" $P/compare.pbs
```

The continental run used the 3 M-sample L2 models (500 trees × 31 leaves);
NEE was scored with `--thresholds` using the mean-|ref| monthly denominator.
Scoring of the windows: `scripts/e7_score.py` and `compare_products.py` per
window into `$W/results/e7/cmp/`.

## Local test (synthetic fixture)

```bash
cd benchmark/gadi
FX=/tmp/fx
uv run make_fixture.py --out $FX
uv run compare_products.py $FX/new/AusEFlux_GPP_500m_quantiles_2025_v2.1.nc \
    $FX/ref/AusEFlux_GPP_500m_quantiles_2025_v2.1.nc --flux GPP --mvg $FX/mvg.tif \
    --out $FX/cmp --plots --border 1
uv run recombine_members.py $FX/members_ref --flux GPP --method nearest \
    --check $FX/ref/AusEFlux_GPP_500m_quantiles_2025_v2.1.nc        # EXACT: YES
uv run recombine_members.py $FX/members_ref --flux GPP --method nearest --out $FX/rn.nc
uv run recombine_members.py $FX/members_ref --flux GPP --method nearest --check $FX/rn.nc  # EXACT: YES
uv run merge_months.py $FX/parts --flux GPP --year 2025 --out $FX/merged.nc
uv run member_parity.py $FX/members_new/GPP $FX/members_ref --flux GPP --months 1-6
uv run feature_parity.py $FX/features_new $FX/features_ref --years 2024-2025
uv run feature_parity.py $FX/interim_new $FX/interim_ref --interim --years 2024-2025
```

The fixture is built with deliberate faults, so some of these runs report
failures on purpose:

- the new product has flipped latitude, 2 extra longitude columns, a different time encoding, about 1 % noise and mask flips;
- one new member (`GPP_lgbm_2`) is perturbed and has one NaN flip, and `GPP_rf_1` is stored as month parts;
- new `ΔT` has an Inf and a NaN in 2024;
- the new interim `rain` is flipped vertically.

Expected results: `member_parity` exits 1 because of `GPP_lgbm_2`, and
`feature_parity --interim` flags `rain` as `FLIPPED?`.
