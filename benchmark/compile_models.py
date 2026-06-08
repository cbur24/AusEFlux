"""
Compile the LightGBM ensemble members in a models folder to native shared
libraries with treelite/tl2cgen, for an exact (~1.5x) predict speedup that
step3_ensemble_prediction.py picks up automatically.

Only LightGBM members are compiled. RandomForest members are deliberately
skipped: treelite's C-codegen explodes on their deep trees (very slow / huge
.so) and the GTIL interpreter is ~8x slower than scikit-learn for them — so RF
members keep loading from joblib.

The compiled libraries are written to `<models_folder>/compiled/<member>.so`;
`_load_predictors` in step3 uses them when present and falls back to joblib
otherwise. Compilation needs scikit-learn 1.3.2 to unpickle the members and a C
compiler (gcc) for tl2cgen.

    uv run --with treelite --with tl2cgen compile_models.py \
        --models-folder /path/results/models/ensemble/v2.1/GPP
"""
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["numpy<2", "scikit-learn==1.3.2", "lightgbm", "joblib",
#                 "treelite", "tl2cgen"]
# ///
import argparse
import os
import time

import joblib
import treelite
import tl2cgen


def compile_folder(models_folder, toolchain="gcc", parallel=8, verbose=True):
    out_dir = os.path.join(models_folder, "compiled")
    os.makedirs(out_dir, exist_ok=True)
    members = sorted(f for f in os.listdir(models_folder) if f.endswith(".joblib"))
    n_done = n_skip = 0
    for mf in members:
        name = mf[:-len(".joblib")]
        so = os.path.join(out_dir, name + ".so")
        if "lgbm" not in name.lower():
            if verbose:
                print(f"  skip {name} (not LightGBM; RandomForest stays joblib)")
            n_skip += 1
            continue
        if os.path.exists(so):
            if verbose:
                print(f"  {name}: already compiled")
            n_done += 1
            continue
        m = joblib.load(os.path.join(models_folder, mf))
        booster = getattr(m, "booster_", None)
        if booster is None:
            print(f"  skip {name}: no LightGBM booster_")
            n_skip += 1
            continue
        tlm = treelite.frontend.from_lightgbm(booster)
        t = time.time()
        tl2cgen.export_lib(tlm, toolchain=toolchain, libpath=so,
                           params={"parallel_comp": parallel}, verbose=False)
        if verbose:
            print(f"  {name}: compiled in {time.time()-t:.1f}s -> {so}")
        n_done += 1
    print(f"done: {n_done} compiled, {n_skip} skipped (RandomForest/other)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-folder", required=True,
                    help="e.g. .../results/models/ensemble/v2.1/GPP "
                         "or .../models/quantile/v2.1/GPP")
    ap.add_argument("--toolchain", default="gcc")
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()
    compile_folder(args.models_folder, args.toolchain, args.parallel)


if __name__ == "__main__":
    main()
