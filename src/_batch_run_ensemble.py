import os
import sys
sys.path.append('/g/data/xc0/project/AusEFlux/src/')
from _ensemble_prediction import predict_ensemble

#get variables for script
model_var = os.getenv("model_var")
year_start = os.getenv("year_start")
year_end = os.getenv("year_end")
target_grid = os.getenv("target_grid")
model_path = os.getenv("model_path")
base = os.getenv("base")

#set up paths
prediction_data = f'/g/data/xc0/project/AusEFlux/data/{target_grid}/'
results_path = f'{base}results/predictions/ensemble/historical/{model_var}/'
features_list = f'{base}results/variables.txt'

#run prediction for a given model
predict_ensemble(
   base=base,
   prediction_data=prediction_data,
   model_path=model_path,
   model_var=model_var,
   models_folder=models_folder,
   features_list=features_list,
   results_path=results_path,
   year_start=year_start,
   year_end=year_end,
   target_grid=target_grid,
   compute_early=False,
   verbose=True
)