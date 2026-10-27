# AusEFlux Efficiency Benchmarks — equivalent results with less compute

> **Update 2026-10-27.** The bbox measurements below (8-core workstation, 2020) are kept
> as measured. The Gadi measurements at continental scale are in
> [§ Gadi measurements](#gadi-measurements-2026-10-27) at the end, and change two
> conclusions: ensemble subsampling (N = 20 "effectively equivalent") **fails** the
> continental tolerance, and the quantile-distillation path does **not** reproduce the
> ensemble within the per-pixel tolerance. All bbox equivalence numbers were measured
> against a baseline that used linear quantiles; the current default is nearest-rank.

Measured variants of the ensemble prediction step, each scored on **speed** *and*
**equivalence to the current 30-model ensemble** (the "must-match" target). All
numbers are **GPP, bbox 2020** (640×640 px, 12 months), same machine (8 cores),
run idle, timed with `/usr/bin/time -v`. Equivalence is vs the preserved baseline
product (`AusEFlux_baseline_ref/`).

## Headline table

| Variant | Members | Wall | Speedup | Peak RSS | Intermediate I/O | median R² vs baseline | median bias |
|---|---|---|---|---|---|---|---|
| Baseline (old per-model loop) | 30 | 23m34s | 1.0× | 1.39 GB | 30 files, 389 MB | — (target) | — |
| **Fused, load-once** (now in `step3_ensemble_prediction.py`) | 30 | 12m30s | 1.9× | 2.87 GB | none | **1.0000 (exact)** | 0.000 |
| **Fused + treelite-compiled LightGBM** (default now) | 30 | **11m20s** | **2.1×** | 2.85 GB | none | **1.0000 (exact)** | 0.000 |
| Fused, N=20 | 20 | 9m29s | 2.5× | 2.26 GB | none | 0.9993 | −0.06 |
| Fused, N=15 | 15 | 8m25s | 2.8× | 2.05 GB | none | 0.9977 | −0.70 |
| Fused, N=10 | 10 | 7m08s | 3.3× | 1.85 GB | none | 0.9960 | −0.76 |
| Fused, N=5 | 5 | 5m53s | 4.0× | 1.52 GB | none | 0.9883 | −1.32 |
| **Quantile distillation** (`step3_..._quantile.py`) | 3 | **~1.9 min** | **~12×** | low | none | 0.9941 | −0.12 |

(Quantile-path 25th/75th bands are over-dispersed: bias −3.16 / +2.55 — see §4.)

## 1. Fused predict+combine (exact, 1.9×)
The shipped path reads every feature **30×** (once per model), writes **30
prediction files**, then re-reads them to combine. `step3_fused.py` reads each
feature **once per tile**, predicts all 30 models, computes the 25/50/75
percentiles on the tile, and writes the final product directly. Output is
**bit-identical** (max|Δ| = 0.000, mask Jaccard = 1.00000) because the no-data and
urban masks are the same for all members, so the per-tile quantile over the 30
predictions equals the shipped write-then-quantile. ~11 min of the baseline's
23.5 min was pure I/O overhead, now removed. Cost: ~2× peak RSS (holds the
30-member stack per tile).

## 2. Ensemble subsampling (tunable accuracy/speed)
Using the first N of 30 members (interleaved LightGBM/RandomForest). **N=20 is the
knee** (R²=0.9993, bias −0.06 — effectively equivalent) at 2.5×; N=10 holds
R²≈0.996 at 3.3×. The speedup tapers because at bbox scale the load-once feature
I/O is a **fixed ~272 s overhead** with predict only ~15.6 s/member on top — see §5
for why this flips at continental scale.

## 3. LightGBM vs RandomForest (no inference-speed difference)
Measured (15 models each, single-thread, 500k rows): LightGBM 221.5 s vs
RandomForest 228.1 s — **within ~3%**, ~400 trees/model each. So the ensemble cost
is driven by **model count and I/O, not model type**; swapping RF→LightGBM for speed
would not help. The compute levers are fewer members and less I/O.

## 4. Quantile distillation (max throughput, ~12×)
3 quantile-loss LightGBM models distilled from the 30-member ensemble. The
**median matches the ensemble at R²=0.994** (bias −0.12) — comparable to a ~10-member
subsample but ~4–6× faster. Trade-off: the **25th/75th bands are over-dispersed**
(25th biased low ≈ −3, 75th high ≈ +2.5), so the uncertainty envelope is wider than
the ensemble's. Best when median accuracy + throughput matter and exact band widths
do not.

## 5. Tree compilation (treelite/tl2cgen) — LightGBM only
Measured (GPP member, 300k rows):
- **LightGBM → tl2cgen compiled: 1.5× faster, bit-identical** (max|Δ|=0), ~6 s to compile.
- **RandomForest → treelite GTIL: 8× *slower*** (13.4 s vs 1.7 s), and C-compilation of
  the deep RF trees (depth ≤35) blows up (codegen explosion, times out).

So treelite helps the **LightGBM** half only; RandomForest members stay on joblib.
Implemented as `benchmark/compile_models.py` (compiles LightGBM members in a folder to
`compiled/*.so`, skips RF), and `step3_ensemble_prediction.py._load_predictors` uses a
compiled `.so` when present, else joblib. With the 15 LightGBM GPP members compiled, the
fused path drops 12m30s → **11m20s** (2.1× over baseline), still bit-identical.
(`lleaves` was tried first but is broken by this env's `llvmlite`.)

## How the ranking changes at continental scale
At bbox scale the fixed feature-I/O (~272 s) masks the per-member predict cost, so
subsampling tapers. At continental scale (≈140× the pixels) **predict dominates** and
the picture flips:
- Fused **load-once** matters far more — the shipped path would read ~68 GB × 30 ≈
  **2 TB** of features per flux (plus 30× ~7 GB intermediate write/read); fused reads
  ~68 GB once.
- Subsampling becomes ~**linear** in N (predict-bound), so N=20 ≈ 1.5× and N=10 ≈ 3×
  on top of the load-once win.
- Quantile distillation stays the throughput champion (~10–13×).

## Recommendations & implementation status
1. **Fused predict+combine — IMPLEMENTED** as the default in
   `step3_ensemble_prediction.py` (all members, no intermediate files). Verified
   bit-identical to the old path (max|Δ|=0), 1.9× faster (2.1× with compiled LightGBM),
   and far more at continental scale (reads features once vs ~2 TB before). The old
   `step3_fused.py` benchmark prototype is kept for reference.
2. *(Not exposed by request)* — the production path runs all members; the `--n-models`
   trade-off is documented above (N=20 ≈ R²0.999) for anyone who wants to dial it.
3. **Quantile distillation** — used for continental/operational throughput (~12×).
4. **treelite compilation — IMPLEMENTED** for LightGBM (`benchmark/compile_models.py`;
   `step3` auto-uses `compiled/*.so`). Exact, ~1.5× on LightGBM members; RandomForest is
   deliberately left on joblib (treelite is impractical/slower for its deep trees).

## Reproduce
```bash
# Production path (fused; auto-uses compiled LightGBM if present):
uv run --with tl2cgen ../step3_ensemble_prediction.py --year 2020 --fluxes GPP

# Compile the LightGBM members once (optional, ~1.5x on the LightGBM half):
uv run compile_models.py --models-folder .../results/models/ensemble/v2.1/GPP

# Benchmark prototypes (subsampling / per-variant timing):
uv run step3_fused.py --flux GPP --year 2020 --n-models 20 --out-subdir AusEFlux_fused_N20
```

---

## Gadi measurements (2026-10-27)

Measured on NCI Gadi (`normal` queue, 2 SU per core-hour) during the validation
experiments; full ledger in [`GADI_REPORT.md`](GADI_REPORT.md). step3 numbers are for
GPP 2025 month 6 on two 400-row windows (rows 2800:3200 and 5000:5400) unless stated.
"Exact" means `max|Δ| = 0` and identical NaN masks against the reference configuration.

### step3: what pays

| Optimisation | Effect | Exact? |
|---|---|---|
| `--valid_only` (predict only valid, non-urban pixels) | predicts 47.9 % of rows; predict 1385 s → 775 s per continental month (**1.8×**, 24 cpus) | yes |
| tl2cgen-compiled LightGBM (`benchmark/compile_models.py`) | LightGBM half 320 s → 173 s per continental month (**1.85×**); 1.4 to 2× in the windows | yes, including the 2 dart members per flux |
| nearest-rank quantiles vs `np.nanquantile` | 0.9 s vs 185 s on a window (≈ 10 s vs ≈ 30 min per continental month, **≈ 200×**); `np.nanquantile` is a per-pixel Python loop in numpy 1.26 | nearest is the reference method; linear is biased |
| `normalsr` queue instead of `normal` | about +20 % throughput per SU (LightGBM members) | yes |

Thread scaling (predict, LightGBM + RandomForest, centre window; every configuration
gave a bit-identical product):

| threads | 1 | 4 | 8 | 12 | 24 | 48 |
|---|---|---|---|---|---|---|
| speedup | 1.0 | 4.2 | 6.9 | 10.6 | 16.9 | 21.9 |
| parallel efficiency | 1.0 | ≈ 1.0 | 0.86 | 0.88 | 0.70 | 0.45 |

RandomForest gains nothing beyond 24 threads. The lowest SU per product is at 4 threads.

### step3: what does not pay

| Option | Result |
|---|---|
| RandomForest via ONNX (float32) | inexact (≈ 6e-5, split thresholds rounded) and not faster |
| RandomForest via treelite GTIL | exact but 4 to 9× slower than scikit-learn |
| RandomForest via hummingbird | about 100× slower and wrong |
| Member subsampling (first N of 30, from the reference member files, E1) | N = 20: R² 0.994, RMSE 7 % of std; N = 10: R² 0.984. Both **fail** the tolerance, and the first-N mix is LightGBM-heavy (biased). Rejected. |
| Quantile distillation (E7, L2 loss on nearest-rank targets, 500 trees × 31 leaves, 3 M samples) | 21 SU per flux-year (8× cheaper). Continental 2025: GPP annual within 0.02 %, monthly within 0.3 %, but per-pixel RMSE 6.4 % (GPP) and 12 % (NEE) of std, above the 5 % threshold. Larger models (3000 × 255) reach 5.0 to 5.6 % but cost ≈ 220 SU per flux-year, more than the ensemble. Not a reproduction. |

### Cost per stage (new vs reference)

| Stage | Reference (2025 run) | New (measured) |
|---|---|---|
| step1, per year | ≈ 3.5 h (dask cluster, hugemem/normalsr) | **21 min wall, ≈ 13 SU** sharded by variable group (climate + SILO 21 min on 12 cpus, 8.4 SU; NDWI + kNDVI 10 min, 2.6 SU; NDVI + LAI 0.9 SU; LST 1.5 SU; VegH 0.1 SU). RSS < 4 GB. Bound by the 13 to 15 GB OzWALD daily files; effective CPU use 1.2 to 2.1 cores, so 4 to 6 cpus per job would suffice. |
| step2, annual update | ≈ 6.5 h wall for the full 2003 to 2025 stacks | `--years_out 2024-2025`, 8 cpus: 0.29 to 0.56 h per 400-row window, so ≈ 5 to 9.5 h and ≈ 80 to 150 SU continental (extrapolated). I/O-bound (145 to 490 % CPU of 800 %). Output ≈ 45 GB + 15 GB climatologies instead of 1.7 TB. |
| step3, per flux-year (30 members) | 30 jobs × ≤ 1 h × 26 cpus ≈ **1,560 SU upper bound**, plus ≈ 17 min combine per flux | 24 threads, `--valid_only`: ≈ 150 SU (6 to 9 core-hours per month, RandomForest 55 to 60 % of predict, RSS ≈ 11 GB). **4 threads + compiled LightGBM: ≈ 60 to 80 SU** (extrapolated from windows, ±30 %). |

The step2 read-path fix mattered more than any compute option: opening the contiguous
interim files with `chunks={}` made every latitude tile re-read whole 2.7 GB files
(68 s per tile); chunking by tile brings it to 0.55 s. `--bottleneck` makes rolling
windows about 8× faster but the stage is I/O-bound, and both `--bottleneck` and `--flox`
move results away from the reference, so they stay off.

### Production recipe (step3)

```
--threads 4  (PBS: ncpus=4, mem=16GB)   3 months per job (≈ 3 h)
compiled LightGBM members (compile_models.py once per flux)
--quantile_method nearest --valid_only   (defaults)
normalsr queue if available
```

≈ 60 to 80 SU per flux-year, against ≈ 1,560 SU for the reference's job layout (an
upper bound: 30 jobs × 26 cores × 1 h walltime). See `gadi/README.md` for the commands.
