import os
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["numpy<2", "pandas", "xarray>=2025.1.0", "netCDF4", "joblib",
#                 "lightgbm", "scikit-learn==1.3.2"]
# ///
"""Run step3_ensemble_prediction.main() with TILE_ROWS=7 (multi-tile test)."""
import sys
sys.path.insert(0, os.environ.get('REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
import step3_ensemble_prediction as s3
s3.TILE_ROWS = 7
s3.main()
