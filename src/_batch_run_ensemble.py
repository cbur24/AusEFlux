import os
import sys
sys.path.append('/g/data/xc0/project/AusEFlux/src/')
from _ensemble_prediction import predict_ensemble
from _utils import start_local_dask

#get variables for python script
model_var = os.getenv("model_var")
year_start = os.getenv("year_start")
year_end = os.getenv("year_end")
target_grid = os.getenv("target_grid")
model_path = os.getenv("model_path")
results_path = os.getenv("results_path")
prediction_data = os.getenv("prediction_data")
features_list = os.getenv("features_list")

## variables for dask
n_workers = os.getenv("n_workers")
memory_limit = os.getenv("memory_limit")


#run function
if __name__ == '__main__':
    
    #start a dask client
    start_local_dask(
        n_workers=int(n_workers),
        threads_per_worker=1,
        memory_limit=memory_limit
                    )

    #run prediction for a given model
    predict_ensemble(
           prediction_data=prediction_data,
           model_path=model_path,
           model_var=model_var,
           features_list=features_list,
           results_path=results_path,
           year_start=year_start,
           year_end=year_end,
           target_grid=target_grid,
           masking_vars=['LAI_anom', 'VegH','NDWI', 'rain_anom', 'Tavg'],
           # masking_vars=['VegH','NDVI', 'rain_anom', 'tavg'],
           compute_early=False, #keep prediction data lazy
           verbose=True
        )