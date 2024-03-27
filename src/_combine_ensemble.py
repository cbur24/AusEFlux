import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from odc.geo.xr import assign_crs
import warnings
warnings.filterwarnings("ignore")

sys.path.append('/g/data/os22/chad_tmp/AusEFlux/src/')
from _percentile import xr_quantile

def combine_ensemble(base,
                     model_var,
                     results_path,
                     predictions_folder,
                     year_start,
                     year_end,
                     attrs,
                     quantiles=[0.05,0.5,0.95],
                     dask_chunks=dict(x=1000, y=1000, time=1),
                     verbose=True
):
    """
    Find the median and uncertainty envelope from the
    ensemble predictions. Export a compliant netcdf with
    metadata stored in the attributes.
    
    """
    #list of predictions stored as netcdfs
    files = os.listdir(predictions_folder)
    pred_filepaths = [predictions_folder+i for i in files if i.endswith('.nc')]
    
    if len(pred_filepaths) == 0:
        ValueError('No predictions found, maybe your "predictions_folder" path is incorrect?') 
    
    if verbose:
        n = len(pred_filepaths)
        print(f'{n} ensemble members to combine')

    arrs=[]
    for pred in pred_filepaths:
        ds=xr.open_dataarray(pred, chunks=dask_chunks)
        arrs.append(ds.rename(pred[-8:-3]))
        
    ds = xr.merge(arrs)
    ds = ds.to_array()

    #find median and uncertainty envelope
    if verbose:
        print(f'Compute quantiles: {quantiles}')
    ds = xr_quantile(ds, quantiles=quantiles, nodata=np.nan)
    ds = ds.rename({'band':model_var+'_quantiles'}).to_array().squeeze().drop('variable')
    ds.attrs['nodata']=np.nan
    ds = ds.compute()

    #set up for export with metadata
    if verbose:
        print('Exporting netcdf files')
        
    ds = ds.rename(attrs['long_name'])
    ds = ds.to_dataset(dim='quantile').rename({0.05:model_var+'_5th_percentile',
                                               0.5:model_var+'_median',
                                               0.95:model_var+'_95th_percentile',
                                               'x':'longitude', 'y':'latitude'})
    ds = assign_crs(ds, crs='EPSG:4326')
    ds.attrs = attrs

    #list of years and export
    years = [str(i) for i in range(year_start, year_end+1)]
    version=attrs['version']
    for year in years:
        xx = ds.sel(time=year)
        xx.to_netcdf(f'{results_path}/AusEFlux_{model_var}_5km_quantiles_{year}_{version}.nc')
    

   