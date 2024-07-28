import os
import pickle
import pandas as pd
import xarray as xr
import rioxarray as rxr
import numpy as np
from odc.geo.xr import assign_crs

import sys
sys.path.append('/g/data/os22/chad_tmp/AusEFlux/src/')
from _utils import round_coords

import warnings
warnings.simplefilter(action='ignore')

def create_feature_datasets(base,
                            results_path,
                            exclude,
                            target_grid='5km',
                            dask_chunks=dict(time=-1, latitude=250, longitude=250),
                            verbose=False
):
    """
    Combine results of the spatiotemporal harmonisation into stacked
    netcdf files, and create new features/variables based on the climate
    (e.g. anomalies) and remote sesning (e.g veg fractions) datasets 

    This is the top-level function that runs all the other
    functions in this file.
    """

    #---step 1 Combine data in the interim folder---------------
    if verbose:
        print('Combining datasets:')
    folders = [i for i in os.listdir(base) if i not in exclude]
    folders.sort()
    
    for f in folders:
        
        # if os.path.exists(results_path+f+'_5km.nc'):
        #     if verbose:
        #         print(' ',f+' exists, skipping')
        #     continue
        # else:
        #     if verbose:
        print(' ',f)
        
        files = [f'{base}{f}/{i}' for i in os.listdir(base+f) if i.endswith(".nc")]
        files.sort()
    
        #combine annual files into one file
        ds = xr.open_mfdataset(files)
        ds = ds.chunk(dask_chunks)
                      
        # Gapfill NDWI & kNDVI differently (has real gaps)
        if f in ['NDWI', 'kNDVI']:
            # seperate into climatologies and anomalies
            ds_monthly = ds.groupby('time.month').mean()
            ds_anom = ds.groupby('time.month') - ds_monthly  
            
            # fill linearly by max 2 steps
            ds_anom = ds_anom.interpolate_na(dim='time', method='linear', limit=2)
            
            #recombine anomalies and climatology
            ds = ds_anom.groupby('time.month') + ds_monthly
            ds = ds.drop('month')
            
            #fill remaining gaps with climatology
            ds = ds.groupby("time.month").fillna(ds_monthly)
    
        # ensure no gaps in other datasets (there shouldn't be any)
        # this is just to be cautious
        else:
            ds_monthly = ds.groupby('time.month').mean()
            ds = ds.groupby("time.month").fillna(ds_monthly)
        
        ds = ds.drop('month').compute()
    
        ds.to_netcdf(f'{results_path}{f}_{target_grid}.nc')

    #---step 2 Create new features--------------------
    #veg fraction
    if verbose:
        print('Vegetation fractions')
    _vegetation_fractions(results=results_path, target_grid=target_grid)
    
    if verbose:
        print('Cumulative rainfall')
    _cumulative_rainfall(f'{results_path}rain_{target_grid}.nc', target_grid=target_grid, results=results_path)
    
    if verbose:
        print('Fractional anomalies')
    _fractional_anomalies(results=results_path,  target_grid=target_grid, verbose=verbose)

    if verbose:
        print('LST minus Tair')
    tair = xr.open_dataarray(f'{results_path}Tavg_{target_grid}.nc')
    lst = xr.open_dataarray(f'{results_path}LST_{target_grid}.nc')
    deltaT = lst - tair
    deltaT.name = u'ΔT'
    deltaT.to_netcdf(results_path+u'ΔT_'+target_grid+'.nc')

    if verbose:
        print('C4 grass fraction')
    _c4_grass_fraction(results=results_path, target_grid=target_grid)


def _vegetation_fractions(results,
                          target_grid='5km',
                          ndvi_max=0.91,
                          dask_chunks=dict(latitude=250, longitude=250, time=-1)
):
    """
    Calculate per-pixel fraction of trees, grass, bare using the methods defined by
    Donohue et al. (2009).

    Requires NDVI (not any other vegetation index).

    `ndvi_min` is the minimum NDVI that a pixel can achieve, this was computed
    for Australia and supplied by Dr Luigi Renzullo.
    
    """
    ndvi_path=f'/g/data/os22/chad_tmp/AusEFlux/data/5km/NDVI_{target_grid}.nc'
    ndvi_min_path =f'/g/data/os22/chad_tmp/AusEFlux/data/ndvi_of_baresoil_{target_grid}.nc'
    
    # NDVI value of bare soil (supplied by Luigi Renzullo)
    ndvi_min = xr.open_dataarray(ndvi_min_path,
                                chunks=dict(latitude=dask_chunks['latitude'],
                                longitude=dask_chunks['longitude'])
                                )
    #ndvi data is here
    ds = xr.open_dataarray(ndvi_path, chunks=dask_chunks)

    #calculate f-total
    ft = (ds - ndvi_min) / (ndvi_max - ndvi_min)
    ft = xr.where(ft<0, 0, ft)
    ft = xr.where(ft>1, 1, ft)
    
    #calculate initial persistent fraction (equation 1 & 2 in Donohue 2009)
    persist = ft.rolling(time=7, min_periods=1).min()
    persist = persist.rolling(time=9, min_periods=1).mean()
    
    #calculate initial recurrent fraction (equation 3 in Donohue 2009)
    recurrent = ft - persist
    
    ###------- equations 4 & 5 in Donohue 2009----------------
    persist = xr.where(recurrent<0, persist - np.abs(recurrent), persist) #eq4
    recurrent = ft - persist # eq 5
    ## ---------------------------------------------------------
    
    #ensure values are between 0 and 1
    persist = xr.where(persist<0, 0, persist)
    recurrent = xr.where(recurrent<0, 0, recurrent)
    
    #assign variable names
    recurrent.name='grass'
    persist.name='trees'
    
    # Aggregate to annual layers
    # Use the maximum fraction of trees and grass to create annual layers.
    # Bare soil is the residual
    persist_annual = persist.resample(time='1Y').max().compute()
    recurrent_annual = recurrent.resample(time='1Y').max().compute()
    bare_annual = 1-(persist_annual+recurrent_annual)
    bare_annual.name='bare'

    #create a monthly timeseries (same vale for each month within a year)
    dss_trees=[]
    dss_grass=[]
    dss_bare=[]
    for y in bare_annual.time.dt.year.values:
        # print(y)
        y = str(y)
        time = pd.date_range(y+"-01", y+"-12", freq='MS') 
        time = [t+pd.Timedelta(14, 'd') for t in time]
    
        #trees
        ds_persist = persist_annual.sel(time=y).squeeze().drop('time')
        ds_persist = ds_persist.expand_dims(time=time)
        dss_trees.append(ds_persist)
    
        #grass
        ds_recurrent = recurrent_annual.sel(time=y).squeeze().drop('time')
        ds_recurrent = ds_recurrent.expand_dims(time=time)
        dss_grass.append(ds_recurrent)
    
        ds_bare = bare_annual.sel(time=y).squeeze().drop('time')
        ds_bare = ds_bare.expand_dims(time=time)
        dss_bare.append(ds_bare)
    
    # join all the datasets back together
    trees = xr.concat(dss_trees, dim='time').sortby('time')
    grass = xr.concat(dss_grass, dim='time').sortby('time')
    bare = xr.concat(dss_bare, dim='time').sortby('time')
    
    # add right metadata
    trees.attrs['nodata'] = np.nan
    grass.attrs['nodata'] = np.nan
    bare.attrs['nodata'] = np.nan
    trees = assign_crs(trees, crs='EPSG:4326')
    grass = assign_crs(grass, crs='EPSG:4326')
    bare = assign_crs(bare, crs='EPSG:4326')

    #export
    trees.to_netcdf(f'{results}trees_{target_grid}.nc')
    grass.to_netcdf(f'{results}grass_{target_grid}.nc')
    bare.to_netcdf(f'{results}bare_{target_grid}.nc')

def _cumulative_rainfall(rain_path,
                         results,
                         target_grid='5km',
                         dask_chunks=dict(latitude=300, longitude=300)
):
    
    rain = xr.open_dataarray(rain_path,chunks=dask_chunks)
    rain_cml_3 = rain.rolling(time=3, min_periods=3).sum()
    rain_cml_3 = rain_cml_3.rename('rain_cml3').sel(time=slice('2003','2052'))
    
    rain_cml_6 = rain.rolling(time=6, min_periods=6).sum()
    rain_cml_6 = rain_cml_6.rename('rain_cml6').sel(time=slice('2003','2052'))
    
    rain_cml_12 = rain.rolling(time=12, min_periods=12).sum()
    rain_cml_12 = rain_cml_12.rename('rain_cml12').sel(time=slice('2003','2052'))

    rain_cml_3.compute().to_netcdf(f'{results}rain_cml3_{target_grid}.nc')
    rain_cml_6.compute().to_netcdf(f'{results}rain_cml6_{target_grid}.nc')
    rain_cml_12.compute().to_netcdf(f'{results}rain_cml12_{target_grid}.nc')

def _fractional_anomalies(results,
                          target_grid='5km',
                          vars=['NDWI', 'kNDVI','LAI','rain', 'rain_cml3','rain_cml6',
                               'rain_cml12', 'SRAD','Tavg', 'VPD'],
                          verbose=True
):
    
    for v in vars:
        if verbose:
            print(' ', v)
        ds = assign_crs(xr.open_dataset(f'{results}{v}_{target_grid}.nc'), crs='EPSG:4326')
        mean = ds.groupby("time.month").mean("time")
        frac = ds.groupby("time.month") / mean
        frac.drop('month').rename({v:v+'_anom'}).to_netcdf(f'{results}{v}_anom_{target_grid}.nc')


def _c4_grass_fraction(results,
                       target_grid='5km',
                       c4_path='/g/data/os22/chad_tmp/AusEFlux/data/Aust_C4_grass_cover_percentage.tif',
                       dask_chunks=dict(x=1500, y=1500)
    
):
    ds = rxr.open_rasterio(c4_path,chunks=dask_chunks).squeeze().drop('band')
    ds = assign_crs(ds, crs='epsg:4326')

    # Grab a common grid to reproject too
    gbox_path = f'/g/data/os22/chad_tmp/AusEFlux/data/grid_{target_grid}'
    with open(gbox_path, 'rb') as f:
        gbox = pickle.load(f)

    # Open a mask of aus extent at target resolution
    p = f'/g/data/os22/chad_tmp/AusEFlux/data/land_sea_mask_{target_grid}.nc'
    mask = xr.open_dataarray(p)

    #reproject
    ds = ds.where(ds>=0).odc.reproject(gbox, resampling='average').compute()
    ds = round_coords(ds)
    ds = ds.where(mask)
    ds = ds/100 #convert to fraction

    grass = xr.open_dataset(f'{results}grass_{target_grid}.nc')['grass']
    c4_grass = grass * ds #fraction of grass that is C4
    c4_grass = c4_grass.rename('C4_grass')
    c4_grass.attrs['nodata']=np.nan

    c4_grass.to_netcdf(f'{results}C4_grass_{target_grid}.nc')

