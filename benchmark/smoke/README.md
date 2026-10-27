# Local smoke harnesses (synthetic data, no Gadi needed)

Copied from the 2026-10-27 review session. Each harness builds its own small
synthetic inputs, runs the step script under several flag combinations and checks
that (a) the run succeeds, (b) ablation flags reproduce saved baselines bit-for-bit,
and (c) new defaults equal the explicit reference-semantics flags. Baseline output
directories referenced by the shell scripts (`out_default`, `out_qcref`, `rv2/poison`,
`smoke_step2/ref_*`, ...) are **not** in the repo: regenerate them once with the
generators (`make_inputs.py`, `make_nan_inputs.py`, `make_fake.py`, `make_mixed.py`,
`make_smoke.py`, ...) and the current script, then treat that run as the baseline for
later edits. The scripts find the repo via `REPO=` (default: three directories up).

| dir | step | entry point | what it checks |
|---|---|---|---|
| `step1/` | `step1_harmonisation.py` | `run_all.sh <odc-geo req> <out prefix>` | default vs `--lst_qc`, `--engine odc`, `--modis_nodata`; NaN-poisoning on an offset grid; atomic `.part` writes and concurrent shards; wide-grid warps; orientation |
| `step2/` | `step2_feature_datasets.py` | `run_tests.sh` | 38 cases: stages, `--vars`, `--years_out`, `--reuse_clim`, `--threads`/`--pool`, `--lat_rows`, `--no_bare_clip`/`--anom_guard`, mixed interim with/without `spatial_ref`; `compare.py` is the bit-comparer |
| `step3/` | `step3_ensemble_prediction.py` | `run_smoke.sh` then `check_smoke.py` | fused product vs an independent recombination (`recombine.py`), `--months`/`--merge`, `--lat_rows`, `--valid_only`, `--mask_test`, ±Inf features, `--dump_members`, config attributes |

All runs use `uv run` with the scripts' PEP 723 headers; see each script's header.
