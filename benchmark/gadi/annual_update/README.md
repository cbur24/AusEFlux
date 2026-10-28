# Annual update on Gadi (e.g. the 2026 product)

One command submits the whole year as a PBS dependency chain, with every
default validated by V25 (full 2025, four fluxes, at float noise from the
published v2.1 product, published file layout):

```bash
# 0. code on Gadi (from the workstation)
rsync -av --exclude .git --exclude .venv --exclude __pycache__ \
      /home/prl900/github/AusEFlux/ gadi:/scratch/ub8/pl5189/auseflux/code/AusEFlux/

# 1. on Gadi
AU=/scratch/ub8/pl5189/auseflux/code/AusEFlux/benchmark/gadi/annual_update
$AU/check_inputs.sh 2026          # lists every raw file step1 needs; exit 1 while any is missing
DRY_RUN=1 $AU/run_year.sh 2026    # prints the ~80 qsub commands, submits nothing
$AU/run_year.sh 2026              # submits step1 -> step2 -> step3 (4 fluxes) -> post-check
```

Everything for the year lands in `/scratch/ub8/pl5189/auseflux/annual/2026/`:

| Path | Content |
|---|---|
| `interim_500m/{var}/{var}_500m_2026.nc` | step1 output, 13 variables |
| `interim_hybrid/{var}/` | symlinks: history 2003–2025 (rain 2002–) from `INTERIM_SOURCES` + the 2026 files; what step2 reads |
| `features_500m/` | 28 feature stacks for 2026 (`--years_out 2026-2026`) + `climatology/` (links to the stored 2003–2022 climatologies) |
| `product/{FLUX}/AusEFlux_{FLUX}_500m_quantiles_2026_v2.1.nc` | the four products, published layout; `parts/` holds the month files |
| `cmp/summary_2026.md`, `cmp/format_*` | post-check: `check_format.py --any-year` vs the latest published file of each flux and a month-by-month table (continental means of the three bands, valid fraction) beside the published year |
| `logs/`, `RUNLOG.md` | every PBS log (step3 args files are in `$WORK/logs/args/`) and a timestamped log of what `run_year.sh` did |

## What the chain does

1. `check_inputs.sh YEAR`: the 14 raw files (MCD43A4 bands 1/2/5, MYD11A1 LST + QC,
   OzWALD NDVI/LAI/Tmin/Tmax/kTavg, SILO radiation/rain/vp, previous-year SILO rain),
   their time-step counts (a complete year = 365/366 daily, 46 eight-day, 12 monthly),
   static rasters, models (and whether they are compiled), the two venvs, the history
   interim coverage and the stored climatologies. `FORCE=1` lets `run_year.sh` go on
   with missing items (a partial year will produce partial features; step3 then writes
   the months it can and lists the rest in `auseflux_partial_months`).
2. `make_interim.sh YEAR`: hybrid interim from symlinks. Sources searched in order
   (`INTERIM_SOURCES`, ':'-separated): the reference `/g/data/xc0/project/AusEFlux/data/interim_500m`
   (2003–2025), earlier annual updates `annual/*/interim_500m`, the V25 2025 interim.
   So for 2026 the history is the reference's own interim and only 2026 is new; for
   2027 the 2026 files come from `annual/2026`. Links to the new year may dangle
   until step1 has run.
3. step1: 5 jobs (`pbs/step1_year.pbs`, defaults `--lst_qc ref --modis_nodata poison`,
   `--orientation_check`), 12 SU, 25 min.
4. step2: 19 jobs (`step2_year.pbs`, 8 cpus / 32 GB each): 5 gap-fill pairs → vegfrac,
   cmlrain, deltat → 10 anomalies → c4, all `afterok` on the step1 jobs. The stored
   climatologies (2003–2022, written by V25, `CLIM_SRC`) are linked into
   `features_500m/climatology/` and reused (`--reuse_clim` is the step2 default), so the
   anomaly and gap-fill stages only read the new year: est. 15–25 SU instead of 129.
   `CLIM_SRC=` (empty) computes them fresh (+100 SU, +2 h) — only needed if the
   2003–2022 interim changes.
5. compile: for each flux without `$WORK/results/models_compiled/{FLUX}/compiled/`
   a `compile_models.pbs` job (tl2cgen, 0.4 SU). All four are compiled already.
6. step3: `pbs/submit_step3_year.sh` per flux with `THREADS=4 MONTHS_PER_JOB=1
   WALLTIME=04:00:00`, compiled models, `DEPEND=` all step2 (+ compile) jobs; each
   flux gets 12 month jobs and an `afterok` merge (`merge.pbs` → `merge_months.py`,
   bit-exact read-back check). 72–90 SU per flux, 43–55 min per month on a normal node.
7. post-check (`postcheck.pbs` → `postcheck.sh`): format check and monthly summary
   (above). Exit 1 and `FAIL` in `RUNLOG.md` if a product is missing or the layout
   differs.

Total ≈ 350–370 SU for four fluxes (≈ 455 with fresh climatologies); wall time
≈ 4–6 h depending on the queue.

## After it finishes

- Read `cmp/summary_2026.md`: every month should have a plausible continental mean
  (compare with the published-year column) and a valid fraction of about 48 %; a
  month far off or at 0 % means a step1 input was incomplete.
- Copy the products and the summary to the durable location (scratch is purged after
  100 days): `rsync -a annual/2026/product annual/2026/cmp annual/2026/logs
  /g/data/ub8/au/blobs/auseflux_refactor/annual_2026/` (adjust to the publication
  path once agreed with the data owner; nothing is ever written under xc0).
- Optional scoring against another product (e.g. the reference's own 2026 run when
  it exists): `qsub -v AUSEF_ENV=step12,NEW=...,REF=...,FLUX=GPP,OUT=...,EXTRA="--mvg
  /g/data/xc0/project/AusEFlux/data/aus6_0e_mvg.tif --plots" pbs/compare.pbs`
  (NEE: add `--thresholds benchmark/gadi/e5/th_nee.json`).

## When something fails

- **A month job killed at walltime** (slow node; V25 saw 45-minute months take
  > 2 h 30 min on `clx-1542`/`clx-2272`): its `afterok` merge is deleted by PBS.
  Resubmit the month with the args file printed by `run_year.sh`, then the merge,
  then `run_year.sh YEAR check` (exact commands are printed at the end of every run).
- **A step2 job fails**: fix and rerun `run_year.sh YEAR "step2 step3 check"`; outputs
  are skip-if-exists, so only the missing stacks are recomputed (month jobs refuse
  to run until all 28 features exist).
- **Rerunning stages later**: `STEP1_DEP`/`STEP2_DEP` (printed at the end) chain a
  later call onto jobs submitted earlier; without them the stage runs immediately.
- **Only some fluxes**: `FLUXES="GPP NEE" run_year.sh 2026 "step3 check"`.
- Never use `PROJECT` as a variable (Gadi exports `PROJECT=xc0` in the login shell);
  the scripts use `PBS_PROJECT` (default `ea89`).

## Files

| File | Role |
|---|---|
| `common.sh` | paths and defaults (`WORK`, `CODE`, `XC`, `PUB`, `ANNUAL`, `INTERIM_SOURCES`, `CLIM_SRC`, `MODELS_COMPILED`, `PBS_PROJECT`), `qsub_` with `DRY_RUN` |
| `check_inputs.sh` | input completeness report |
| `make_interim.sh` | hybrid interim from symlinks |
| `step2_year.pbs` | step2 stage job on the hybrid interim |
| `compile_models.pbs` | tl2cgen compilation of one flux |
| `run_year.sh` | the orchestrator (stages `inputs interim step1 step2 step3 check`) |
| `postcheck.pbs`, `postcheck.sh` | format check + monthly summary |

The scripts used for a year are copied into its tree (`annual/YEAR/*.sh`, `*.pbs`)
so the run is reproducible after the repo moves on. The V25 run that validated this
recipe is documented in `../../GADI_REPORT.md` (V25 notes) and
`/scratch/ub8/pl5189/auseflux/v25/RUNLOG.md`.
