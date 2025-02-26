import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from joblib import load
from odc.geo.xr import assign_crs

import warnings
warnings.filterwarnings("ignore")

sys.path.append('/g/data/xc0/project/AusEFlux/src/')
from _prediction import collect_prediction_data, predict_xr, HiddenPrints
from _utils import round_coords

def predict_ensemble(prediction_data,
                     model_path,
                     model_var,
                     results_path,
                     features_list,
                     year_start,
                     year_end,
                     target_grid='500m',
                     masking_vars=['LAI_anom', 'VegH','NDWI', 'rain_anom', 'Tavg'],
                     # masking_vars=['VegH','NDVI', 'rain_anom', 'tavg'],
                     dask_chunks=dict(time=1),
                     compute_early=True,
                     verbose=True
):
    """
    Predict a time-series of a carbon or water flux variable given
    a path to a model. 
  
    """
    #model name
    name = model_path.split('/')[-1].split('.')[0]

    #check if its already been  predicted
    if os.path.exists(f'{results_path}{name}.nc'):
        if verbose:
            print('skipping model '+name+' prediction')
        
        return None

    else:
        ## open data for prediction
        data = collect_prediction_data(data_path=f'{prediction_data}/',
                                     time_range=(f'{year_start}',f'{year_end}'),
                                     verbose=False,
                                     export=False,
                                     chunks=dask_chunks
                                     )
        if compute_early:
            data = data.compute()
    
        # nodata masks and urban+water masks
        if verbose:
            print('Creating no-data mask')
        mask = data[masking_vars].to_array().isnull().any('variable').compute()
        urban = xr.open_dataset(f'/g/data/xc0/project/AusEFlux/data/urban_water_mask_{target_grid}.nc')['urban_water_mask']
        urban = urban.rename({'latitude':'y', 'longitude':'x'})
    
        # Index by variables and check variable order
        train_vars = list(pd.read_csv(features_list))[0:-1]
        train_vars.remove('site')
        train_vars=[i[:-3] for i in train_vars]
        
        try:
            data = data[train_vars]
            if verbose:
                print('Variables match, n:', len(data.data_vars))
        except:
            raise ValueError("Variables don't match")
    
        # Predict
        if verbose:
            print('Model: ', name)
    
        #open model
        warnings.filterwarnings("ignore")
        model = load(model_path).set_params(n_jobs=1)
        
        results = []
        i=0
        #loop through the time-steps
        for i in range(0, len(data.time)): 
            if verbose:
                print("  {:03}/{:03}\r".format(i + 1, len(range(0, len(data.time)))), end="")
    
            with HiddenPrints():
                warnings.filterwarnings("ignore")
                predicted = predict_xr(model,
                                    data.isel(time=i),
                                    proba=False,
                                    clean=True,
                                    chunk_size=1000000, #this number is optimized to maximise pred speed at 1km 875000
                                      ).compute()
    
                #mask no-data areas
                predicted = predicted.Predictions.where(~mask.isel(time=i))
            
                #add back time dim
                predicted['time'] = data.isel(time=i).time.values
            
                #append to list
                results.append(predicted.astype('float32'))
                i+=1 
    
        #join together into a Dataset
        ds = xr.concat(results, dim='time').sortby('time').rename(model_var).astype('float32')
        
        #mask urban and water areas
        ds = ds.where(urban!=1).astype('float32')

        #save results
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        
        ds.to_netcdf(f'{results_path}{name}.nc')

























   