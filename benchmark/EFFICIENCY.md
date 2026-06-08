# AusEFlux Efficiency Benchmarks — equivalent results with less compute

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
