import os
import pickle
import pandas as pd
import xarray as xr
import numpy as np
from odc.geo.xr import assign_crs

import warnings
warnings.simplefilter(action='ignore')

import sys
sys.path.append('/g/data/xc0/project/AusEFlux/src/')
from _utils import round_coords

def spatiotemporal_harmonisation(year_start,
                                 year_end,
                                 target_grid = '5km',
                                 base_path='/g/data/ub8/au/',
                                 results_path='/g/data/xc0/project/AusEFlux/data/interim/',
                                 verbose=False
                                ):
    """
    Top level fuction to run all the functions associated with
    harmonising the input datasets.
    """

    #list of years to run
    years = [str(i) for i in range(year_start, year_end+1)]

    # Grab a common grid to reproject all datasets too 
    gbox_path = f'/g/data/os22/chad_tmp/AusEFlux/data/grid_{target_grid}'
    with open(gbox_path, 'rb') as f:
        gbox = pickle.load(f)

    # Create a land mask
    if verbose:
        print('load land/sea mask')
    
    # Open a mask of aus extent as target resolution
    p = f'/g/data/os22/chad_tmp/AusEFlux/data/land_sea_mask_{target_grid}.nc'
    mask = xr.open_dataarray(p)
    
    #run NDWI
    if verbose:
        print('Process NDWI, estimated time 10 mins/year')
    _modis_indices(years, 'NDWI', base_path, results_path, gbox, mask,target_grid=target_grid, verbose=verbose)

    #run kNDVI
    if verbose:
        print('Process kNDVI, estimated time 10 mins/year')
    _modis_indices(years, 'kNDVI', base_path, results_path, gbox, mask, target_grid=target_grid,verbose=verbose)

    #run NDVI & LAI
    if verbose:
        print('Process NDVI & LAI, estimated time 1 min/year')
    _ozwald_indices(years, 'NDVI', base_path, results_path, gbox, mask, target_grid=target_grid, verbose=verbose)
    _ozwald_indices(years, 'LAI', base_path, results_path, gbox, mask, target_grid=target_grid, verbose=verbose)

    #run LST
    if verbose:
        print('Process LST, estimated time 5 mins/year')
    _modis_LST(years, 'LST', base_path, results_path, gbox, mask, target_grid=target_grid, verbose=verbose)

    #run VegH
    if verbose:
        print('Process Veg Height, estimated time 1 mins/year')
    _veg_height(years, 'VegH', base_path, results_path, gbox, mask, target_grid=target_grid,verbose=verbose)
    
    #run temperature-average from ozwald
    if verbose:
        print('Process Tavg, estimated time 80 mins/year')
    _ozwald_climate(years, 'Tavg', base_path, results_path, gbox, mask, target_grid=target_grid, verbose=verbose)

    #run SILO climate grids
    if verbose:
        print('Process SILO Climate, estimated time 5 seconds/year/variable')
    
    # update the list of years to run by adding an extra year at the start. This is
    # because later we will calculate cumulative rainfall and the first year of the
    # cumulative rainfll timeseries will have NaN values, so process a year earlier than
    # desired and later drop that extra year.
    years = [str(i) for i in range(year_start-1, year_end+1)]
    _SILO_climate(years, None, base_path, results_path, gbox, mask, target_grid=target_grid,verbose=verbose)

def _modis_indices(years,
                var,
                base,
                results,
                geobox,
                mask,
                target_grid='5km',
                dask_chunks=dict(latitude=1000, longitude=1000, time=1),
                verbose=False
                ):

    """
    Process MODIS surface reflectance bands into NDWI and kNDVI based on MODIS data
    from /g/data/ub8/au/MODIS/mosaic/MCD43A4.06/. Reproject data to "geobox" and
    resample to monthly means

    The orginal Gao (1996) paper says to use the 1230_1250nm band (band 5 in MODIS),
    but other sources suggest band 6. Sticking with Gao for now.
    
    """
    for year in years:
    
        if os.path.exists(f'{results}{var}/{var}_{target_grid}_{year}.nc'):
                continue
        else:
            if verbose:
                print(' ', year)

        if var=='NDWI':
            modis_sr_inputs = {
                'SR_B2': 'MODIS/mosaic/MCD43A4.006/MCD43A4.006.b02.500m_0841_0876nm_nbar.'+year+'.nc',
                # 'SR_B6': 'MODIS/mosaic/MCD43A4.006/MCD43A4.006.b06.500m_1628_1652nm_nbar.'+year+'.nc',
                'SR_B5': 'MODIS/mosaic/MCD43A4.006/MCD43A4.006.b05.500m_1230_1250nm_nbar.'+year+'.nc',
                 }
        
        if var=='kNDVI':
             modis_sr_inputs = {
                'SR_B1': 'MODIS/mosaic/MCD43A4.006/MCD43A4.006.b01.500m_0620_0670nm_nbar.'+year+'.nc',
                'SR_B2': 'MODIS/mosaic/MCD43A4.006/MCD43A4.006.b02.500m_0841_0876nm_nbar.'+year+'.nc',
                 }
    
        d = {}
        for k,i in modis_sr_inputs.items():
            
            #open and do some prelim processing
            ds = xr.open_dataset(base+i, chunks=dask_chunks)
            ds = assign_crs(ds, crs='epsg:4326')
            ds = ds.to_array()
            ds = ds.squeeze().drop_vars('variable')
            ds.attrs['nodata'] = np.nan
            ds = ds.rename(k)        
            d[k] = ds #add to dict
        
        if var=='NDWI':
            ds = (d['SR_B2'] - d['SR_B5']) / (d['SR_B2'] + d['SR_B5'])

        if var=='kNDVI':
            ds = np.tanh(((d['SR_B2'] - d['SR_B1']) / (d['SR_B2'] + d['SR_B1'])) ** 2)
        
        #resample time
        ds = ds.resample(time='MS', loffset=pd.Timedelta(14, 'd')).mean().persist()
    
        # resample spatial
        ds = ds.odc.reproject(geobox, resampling='average').compute()  # bring into memory
        
        #tidy up
        ds = round_coords(ds)
        ds.attrs['nodata'] = np.nan
        ds = ds.rename(var)
    
        #mask to aus land extent
        ds = ds.where(mask)
        
        #export result
        folder = f'{results}{var}'
        if not os.path.exists(folder):
            os.makedirs(folder)
    
        ds.astype('float32').to_netcdf(f'{results}{var}/{var}_{target_grid}_{year}.nc')


def _ozwald_indices(years,
                var,
                base,
                results,
                geobox,
                mask,
                target_grid='5km',
                dask_chunks=dict(latitude=1000, longitude=1000, time=-1),
                verbose=False
                   ):

    """
    Process OzWaLD spectral indices from /g/data/ub8/au/OzWALD/8day/
    Reproject data to "geobox" and resample to monthly means
    
    """
    for year in years:
        
        ozwald_vars = {
            var :f'OzWALD/8day/{var}/OzWALD.{var}.{year}.nc'
         }
    
        for k,i in ozwald_vars.items():
             
            if os.path.exists(f'{results}{k}/{k}_{target_grid}_{year}.nc'):
                continue
            else:
                if verbose:
                    print(' ', year)
    
            ds = xr.open_dataset(base+i,chunks=dask_chunks)
            ds = ds.transpose('time', 'latitude', 'longitude')
            
            #tidy up
            ds = assign_crs(ds, crs='epsg:4326')
            ds = ds.to_array()
            ds = ds.squeeze().drop('variable')
            ds.attrs['nodata'] = np.nan
            
            #resample time
            ds = ds.resample(time='MS', loffset=pd.Timedelta(14, 'd')).mean().persist()
        
            # resample spatial
            ds = ds.odc.reproject(geobox, resampling='average').compute()
            
            #tidy up
            ds = round_coords(ds)
            ds.attrs['nodata'] = np.nan
            ds = ds.rename(k)
        
            #mask to aus land extent
            ds = ds.where(mask)
            
            #export result
            folder = f'{results}{k}'
            if not os.path.exists(folder):
                os.makedirs(folder)
        
            ds.astype('float32').to_netcdf(f'{results}{var}/{k}_{target_grid}_{year}.nc')

def _modis_LST(years,
                var,
                base,
                results,
                geobox,
                mask,
                target_grid='5km',
                dask_chunks=dict(latitude=1000, longitude=1000, time=-1),
                verbose=False
                ):

    """
    Process MODIS land surface temperature from /g/data/ub8/au/MODIS/mosaic/MYD11A1.006/
    Reproject data to "geobox" and resample to monthly means

    QC masking MODIS, help: https://spatialthoughts.com/2021/08/19/qa-bands-bitmasks-gee/
    decimal to binary converter: https://www.rapidtables.com/convert/number/decimal-to-binary.html

    - 0  (decimal): 0 0 0 0 0 0 0 0; LST produced, good quality, good data, emis err < 0.01, LST error <1K
    - 5  (decimal): 0 0 0 0 0 1 0 1; LST produced, other quality, other quality, emis err < 0.01, LST error <= 1K
    - 17 (decimal): 0 0 0 1 0 0 0 1; LST produced, other quality, good data, emis err < 0.02, LST error <= 1K
    - 21 (decimal): 0 0 0 1 0 1 0 1; LST produced, other quality, other quality, emis err < 0.02, LST error <= 1K
    - 64 (decimal): 0 1 0 0 0 0 0 0; LST produced, good quality, good data, emis err < 0.01, LST error <= 2K
    - 65 (decimal): 0 1 0 0 0 0 0 1; LST produced, other quality, good data, emis err < 0.01, LST error <= 2K
    - 81 (decimal): 0 1 0 1 0 0 0 1; LST produced, other quality, other quality, emis err < 0.02, LST error <= 2K
    
    """
    for year in years:
    
        if os.path.exists(f'{results}{var}/{var}_{target_grid}_{year}.nc'):
                continue
        else:
            if verbose:
                print(' ', year)
        
        modis_sr_inputs = {
            'LST' :'MODIS/mosaic/MYD11A1.006/MYD11A1.006.LST_Day_1km.'+year+'.nc'
             }
    
        for k,i in modis_sr_inputs.items():

            ds = xr.open_dataset(base+i,chunks=dask_chunks)
    
            #deal with messed up QC in 2022 (temporary hopefully)
            if year not in ['2022','2023']:
                qc = xr.open_dataset(base+'MODIS/mosaic/MYD11A1.006/MYD11A1.006.QC_Day.'+year+'.nc',
                         chunks=dask_chunks)
                #data is high quality <2k error see above.
                m = xr.where((qc.QC_Day==0) | (qc.QC_Day==5) | (qc.QC_Day==17) | (qc.QC_Day==21) |
                             (qc.QC_Day==64) | (qc.QC_Day==65) | (qc.QC_Day==81), 1, 0)
                
                ds = ds.where(m)
    
            #tidy up
            ds = assign_crs(ds, crs='epsg:4326')
            ds = ds.to_array()
            ds = ds.squeeze().drop('variable')
            ds.attrs['nodata'] = np.nan
            
             #resample time
            ds = ds.resample(time='MS', loffset=pd.Timedelta(14, 'd')).mean().persist()
        
             # resample spatial
            if round(geobox.resolution.x, 3) == 0.01:
                resampling='nearest'
            else:
                resampling='average'
            ds = ds.odc.reproject(geobox, resampling=resampling).compute()
            
            #tidy up
            ds = round_coords(ds)
            ds.attrs['nodata'] = np.nan
            ds = ds.rename(var)
        
            #mask to aus land extent
            ds = ds.where(mask)
    
            #convert to celsius
            ds = ds-273.15
            
            #export result
            folder = f'{results}{var}'
            if not os.path.exists(folder):
                os.makedirs(folder)
        
            ds.astype('float32').to_netcdf(f'{results}{var}/{var}_{target_grid}_{year}.nc')

def _veg_height(years,
                var,
                base,
                results,
                geobox,
                mask,
                target_grid='5km',
                dask_chunks=dict(latitude=250, longitude=250),
                verbose=False
                ):

    """
    Process vegetation height data from /g/data/xc0/project/AusEFlux/data/
    This dataset is originally from g/data/ub8/LandCover/OzWALD_LC/ and 
    was reprojected from 25m to 1 km previously. It is a static dataset that
    is replicated at every time-step.
    """
                    
    p='/g/data/xc0/project/AusEFlux/data/VegH_1km_2007_2010.nc'
    ds = xr.open_dataset(p, chunks=dask_chunks)
    ds = assign_crs(ds, crs='epsg:4326')
    ds.attrs['nodata'] = np.nan
    ds = ds['VegH']
    if target_grid=='5km':
        resampling='average'
    else:
        resampling='nearest'
    ds = ds.odc.reproject(geobox, resampling='average').compute()
    ds = round_coords(ds)
    
    # convert to time-series (same values for each time-step)
    # open another dataset so we can grab the time dim
    for year in years:
        
        if os.path.exists(f'{results}{var}/{var}_{target_grid}_{year}.nc'):
                continue
        else:
            if verbose:
                print(' ', year)

        #open time dim from anotther dataset
        da_time = xr.open_dataarray(f'{results}NDWI/NDWI_{target_grid}_{year}.nc').time
        
        #expand time dim using other dataset's time.
        dss = ds.expand_dims(time=da_time)
        
        #mask to aus land extent
        dss = dss.where(mask)
        dss = dss.rename(var)
        
        #export result
        folder = f'{results}{var}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # export
        dss.to_netcdf(f'{results}{var}/{var}_{target_grid}_{year}.nc')

def _ozwald_climate(years,
                var,
                base,
                results,
                geobox,
                mask,
                target_grid='5km',
                dask_chunks=dict(latitude=5000, longitude=5000, time=1),
                verbose=False
                   ):

    """
    Process OzWaLD temperature climate data from /g/data/ub8/au/OzWALD/daily/meteo/
    This is very memory and compute intensive (and slow)
    Consider switching to SILO temperature grids.
    The process is split into two steps.
    """

    # -----------Step 1-----------------------------------------------
    for year in years:
        clim_inputs = {
            'Tmin':'OzWALD/daily/meteo/Tmin/OzWALD.Tmin.'+year+'.nc', 
            'Tmax':'OzWALD/daily/meteo/Tmax/OzWALD.Tmax.'+year+'.nc',
            'kTavg':'OzWALD/daily/meteo/kTavg/OzWALD.kTavg.'+year+'.nc'
             }
        
        for k,i in clim_inputs.items():
            
            if os.path.exists(f'{results}/{k}/{k}_{target_grid}_{year}.nc'):
                continue
            else:
                if verbose:
                    print(' ', k, year)
            
            #open and do some prelim processing
            ds = xr.open_dataset(base+i, chunks=dask_chunks) # open as one chunk per time
            ds = assign_crs(ds, crs='epsg:4326')
            ds = ds.to_array()
            ds = ds.squeeze().drop_vars('variable')
            ds.attrs['nodata'] = np.nan
            
            #resample time
            ds = ds.resample(time='MS', loffset=pd.Timedelta(14, 'd')).mean().persist()
            
            #we need to spatial resample first to reduce RAM/speed up.
            if k=='kTavg':
                #upscaling from 10km to target grid
                ds = ds.odc.reproject(geobox, resampling='nearest').compute()
                ds = round_coords(ds)
            else:
                # downsacling from 500m to target grid
                ds = ds.odc.reproject(geobox, resampling='average').compute()
                ds = round_coords(ds)

            #resample time
            ds = ds.resample(time='MS', loffset=pd.Timedelta(14, 'd')).mean()

            #tidy up
            ds = ds.transpose('time', 'latitude', 'longitude')
            ds = ds.rename(k)
            ds = assign_crs(ds, crs='epsg:4326')
            
            # #export result
            folder = results+k
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            ds.astype('float32').to_netcdf(f'{results}/{k}/{k}_{target_grid}_{year}.nc')

    # -----------Step 2-----------------------------------------------
    for year in years:
    
        clim_inputs = {
            'Tmin':f'{results}/Tmin/Tmin_{target_grid}_{year}.nc', 
            'Tmax':f'{results}/Tmax/Tmax_{target_grid}_{year}.nc',
            'kTavg':f'{results}/kTavg/kTavg_{target_grid}_{year}.nc'
             }
        
        if os.path.exists(f'{results}{var}/{var}_{target_grid}_{year}.nc'):
                continue
        else:
            if verbose:
                print('', var, year)
        
        d={}
        for k,i in clim_inputs.items():
            ds = xr.open_dataarray(i)
            d[k] = ds
        
        #calculate tavg
        ds = d['Tmin'] + d['kTavg']*(d['Tmax'] - d['Tmin'])
    
        # #resample time
        # ds = ds.resample(time='MS', loffset=pd.Timedelta(14, 'd')).mean()
        
        #tidy up
        ds.attrs['nodata'] = np.nan
        ds = ds.rename(var)
    
        #mask to aus land extent
        ds = ds.where(mask)
    
        #export result
        folder = f'{results}{var}'
        if not os.path.exists(folder):
            os.makedirs(folder)
    
        ds.astype('float32').to_netcdf(f'{results}{var}/{var}_{target_grid}_{year}.nc')

def _SILO_climate(years,
                var,
                base,
                results,
                geobox,
                mask,
                target_grid='5km',
                dask_chunks=dict(lat=250, lon=250, time=-1),
                verbose=False
                   ):
    
    """
    Process SILO climate grids from /g/data/ub8/au/SILO/
    """
    #loop through each year
    for year in years:
            
            clim_inputs = {
                'SRAD':'SILO/radiation/'+year+'.radiation.nc',
                'rain':'SILO/daily_rain/'+year+'.daily_rain.nc',
                'VPD':'SILO/vp/'+year+'.vp.nc'
                 }
            
            for k,i in clim_inputs.items():
                
                if (year=='2002') & (k!='rain'):
                    continue
                
                if os.path.exists(f'{results}/{k}/{k}_{target_grid}_{year}.nc'):
                    continue
                else:
                    if verbose:
                        print(' ', k, year)
                
                #open and do some prelim processing
                ds = xr.open_dataset(base+i).drop('crs').chunk(dask_chunks)
                ds = assign_crs(ds, crs='epsg:4326')
                ds = ds.to_array()
                ds = ds.squeeze().drop_vars('variable')
                ds.attrs['nodata'] = np.nan
        
                # resample time and space
                if k=='rain':
                    ds = ds.resample(time='MS', loffset=pd.Timedelta(14, 'd')).sum()
                else:
                    ds = ds.resample(time='MS', loffset=pd.Timedelta(14, 'd')).mean()
                
                if k in ['SRAD', 'VPD']:
                    if target_grid=='1km':
                        method='bilinear'
                    else:
                        method='nearest'
                
                if k=='rain':
                    method='bilinear'
                
                ds = ds.odc.reproject(geobox, resampling=method).compute()

                if k=='VPD':
                    #calculate VPD
                    ds = round_coords(ds)
                    ds = assign_crs(ds, crs='epsg:4326')
                    ds = ds.where(mask)
                    ta = xr.open_dataarray(f'{results}/Tavg/Tavg_{target_grid}_{year}.nc')
                    sat_vp = (6.11 * np.exp((2500000/461) * (1/273 - 1/(273 + ta))))
                    ds = sat_vp - ds
                    ds = ds.rename(k)
                    ds.attrs['nodata'] = np.nan
                    ds.attrs['units'] = 'hPa'

                else:
                    # tidy up and mask land
                    ds = round_coords(ds)
                    ds = ds.rename(k)
                    ds = assign_crs(ds, crs='epsg:4326')
                    ds = ds.where(mask)
        
                # # #export result
                folder = results+k
                if not os.path.exists(folder):
                    os.makedirs(folder)
                
                ds.astype('float32').to_netcdf(f'{results}/{k}/{k}_{target_grid}_{year}.nc')
    














