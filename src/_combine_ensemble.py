import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from odc.geo.xr import assign_crs
import warnings
warnings.filterwarnings("ignore")

sys.path.append('/g/data/xc0/project/AusEFlux/src/')
from _percentile import xr_quantile

def combine_ensemble(model_var,
                     results_path,
                     predictions_folder,
                     year_start,
                     year_end,
                     attrs,
                     target_grid='1km',
                     quantiles=[0.25,0.5,0.75],
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
    ds = ds.to_dataset(dim='quantile').rename({0.25:model_var+'_25th_percentile',
                                               0.5:model_var+'_median',
                                               0.75:model_var+'_75th_percentile',
                                               'x':'longitude', 'y':'latitude'})
    ds = assign_crs(ds, crs='EPSG:4326')
    ds.attrs = attrs

    #list of years and export
    years = [str(i) for i in range(year_start, year_end+1)]
    version=attrs['version']

    for year in years:

        xx = ds.sel(time=year)
        
        #annual summaries, need to remask after sum()
        xx_mean = xx.resample(time='YE').mean()
        xx_sum = xx.resample(time='YE').sum()
    
        mask = ~np.isnan(xx_mean[var+'_median'])
        xx_sum = xx_sum.where(mask).astype(np.float32)
    
        #update units for annual sums
        if var =='ET':
            units = 'mm/year'
        else:
            units = 'gC/m\N{SUPERSCRIPT TWO}/year'
        
        xx_sum.attrs['units'] = units
        
        # hack to make time dim work with OpenDAP which doesn't like datetime64
        start_time = xx.time.values[0].astype('datetime64[D]')###first date
        # set time as the duration between actual and first date
        coords_time = np.array(xx.time, dtype='datetime64[D]') - np.array(xx.time, dtype='datetime64[D]')[0]        
        xx['time'] = coords_time.astype('int32')
        xx.time.attrs = {'units': f'days since {start_time}'} #make sure attrs explain int32 time
    
        annual_time = xx_mean.time.values[0].astype('datetime64[D]')
        xx_mean['time'] = np.array([0], dtype='timedelta64[D]').astype('int32') #zero days since 'annual_time'
        xx_sum['time'] = np.array([0], dtype='timedelta64[D]').astype('int32') #zero days since 'annual_time'
        xx_mean.time.attrs = {'units': f'days since {annual_time}'}
        xx_sum.time.attrs = {'units': f'days since {annual_time}'}
        
        xx.to_netcdf(f'{results_path}/monthly/{model_var}/AusEFlux_{model_var}_1km_quantiles_{year}_{version}.nc')
        xx_mean.to_netcdf(f'{results_path}/annual/AnnualMean/{model_var}/AusEFlux_{model_var}_1km_AnnualMean_{year}_{version}.nc')
        xx_sum.to_netcdf(f'{results_path}/annual/AnnualSum/{model_var}/AusEFlux_{model_var}_1km_AnnualSum_{year}_{version}.nc')

   