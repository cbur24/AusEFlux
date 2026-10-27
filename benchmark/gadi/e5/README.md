# E5 scripts (end-to-end acceptance on Gadi, 2026-10-27/28)

Exact scripts that produced the E5 results in `../../GADI_REPORT.md`. Paths are
hard-wired to `/scratch/ub8/pl5189/auseflux` and project `ea89`; edit before reuse.

- `submit_step2.sh`, `e5_step2.pbs`: continental step2 `--years_out 2025-2025` on the
  hybrid interim (reference 2003–2023 + new 2024/2025), sharded over 22 jobs.
- `submit_win.sh`, `e5_step3_win*.pbs`: 2024 latitude windows, full chain per window
  (`winp` = step1 `--modis_nodata poison` variants).
- `e5_compile.pbs`: tl2cgen compilation of the NEE members.
- `step3_cont.args`: argument file for the continental step3 jobs (used with
  `../pbs/submit_step3_year.sh`).
- `cmp.sh`, `e5_crop_cmp.pbs`, `crop.py`: scorecards with `../compare_products.py`
  (continental and window crops of the continental product).
- `e5_parity.pbs`, `e5_parity.py`, `par_summary.py`: per-feature parity on the windows
  (attribution of the north-window residual to NDWI/kNDVI).
- `chk_chunks.py`: interim chunk-layout check.

Outputs: `/scratch/ub8/pl5189/auseflux/results/e5/` (100-day purge) and the durable
copy `/g/data/ub8/au/blobs/auseflux_refactor/e5/`.
