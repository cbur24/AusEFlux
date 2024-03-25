import os
import pandas as pd
import xarray as xr
import numpy as np
from odc.geo.xr import assign_crs

import warnings
warnings.simplefilter(action='ignore')

import sys
sys.path.append('/g/data/os22/chad_tmp/AusEFlux/src/')
from _utils import round_coords

def spatiotemporal_harmonisation(year,
                                 base_path='/g/data/ub8/au/',
                                 results_path='/g/data/os22/chad_tmp/AusEFlux/data/interim/',
                                 verbose=False
                                ):
    """
    Run everything
    """

    #list of years to run
    years = [str(i) for i in range(year,year+1)]

    # Grab a common grid to reproject too and a create a land mask
    p = '/g/data/os22/chad_tmp/climate-carbon-interactions/data/5km/WCF_5km_monthly_1982_2022.nc'
    gbox = xr.open_dataset(p).odc.geobox

    if verbose:
        print('Create land/sea mask')
    #create a mask of aus extent
    mask = xr.open_dataset(p)['WCF']
    mask = mask.mean('time')
    mask = xr.where(mask>-99, 1, 0)
    
    #run NDWI
    if verbose:
        print('Process NDWI, estimated time 10 mins/year')
    _modis_indices(years, 'NDWI', base_path, results_path, gbox, mask, verbose=verbose)

    #run kNDVI
    if verbose:
        print('Process kNDVI, estimated time 10 mins/year')
    _modis_indices(years, 'kNDVI', base_path, results_path, gbox, mask, verbose=verbose)

    #run NDVI
    if verbose:
        print('Process NDVI, estimated time 5 mins/year')
    _ozwald_indices(years, 'NDVI', base_path, results_path, gbox, mask, verbose=verbose)

    #run LST
    if verbose:
        print('Process LST, estimated time 5 mins/year')
    _modis_LST(years, 'LST', base_path, results_path, gbox, mask, verbose=verbose)

    #run VegH
    if verbose:
        print('Process Veg Height, estimated time 1 mins/year')
    _veg_height(years, 'VegH', base_path, results_path, gbox, mask, verbose=verbose)
    
    #run temperature-average from ozwald
    if verbose:
        print('Process Tavg, estimated time 80 mins/year')
    _ozwald_climate(year, 'Tavg', base_path, results_path, gbox, mask, verbose=verbose)

    #run SILO climate grids
    if verbose:
        print('Process Tavg, estimated time 5 seconds/year/variable')
    _SILO_climate(years, None, base_path, results_path, gbox, mask, verbose=verbose)


def _modis_indices(years,
                var,
                base,
                results,
                geobox,
                mask,
                dask_chunks=dict(latitude=1000, longitude=1000, time=1),
                verbose=False
                ):

    """
    Process MODIS surface reflectance bands into NDWI and kNDVI based on MODIS data
    from /g/data/ub8/au/MODIS/mosaic/MCD43A4.06/. Reproject data to "geobox" and resample
    to monthly means
    """
    for year in years:
    
        if os.path.exists(f'{results}{var}/{var}_5km_{year}.nc'):
                continue
        else:
            if verbose:
                print(' ', year)

        if var=='NDWI':
            modis_sr_inputs = {
                'SR_B2': 'MODIS/mosaic/MCD43A4.006/MCD43A4.006.b02.500m_0841_0876nm_nbar.'+year+'.nc',
                'SR_B6': 'MODIS/mosaic/MCD43A4.006/MCD43A4.006.b06.500m_1628_1652nm_nbar.'+year+'.nc',
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
            ds = (d['SR_B2'] - d['SR_B6']) / (d['SR_B2'] + d['SR_B6'])

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
    
        #ds.astype('float32').to_netcdf(f'{results}{var}/{var}_5km_{year}.nc')


def _ozwald_indices(years,
                var,
                base,
                results,
                geobox,
                mask,
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
             
            if os.path.exists(f'{results}{k}/{k}_5km_{year}.nc'):
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
        
            #ds.astype('float32').to_netcdf(f'{results}{var}/{k}_5km_{year}.nc')


def _modis_LST(years,
                var,
                base,
                results,
                geobox,
                mask,
                dask_chunks=dict(latitude=500, longitude=500, time=-1),
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
    
        if os.path.exists(f'{results}{var}/{var}_5km_{year}.nc'):
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
            ds = ds.odc.reproject(geobox, resampling='average').compute()  # bring into memory
            
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
        
            #ds.astype('float32').to_netcdf(f'{results}{var}/{var}_5km_{year}.nc')


def _veg_height(years,
                var,
                base,
                results,
                geobox,
                mask,
                dask_chunks=dict(latitude=250, longitude=250),
                verbose=False
                ):

    """
    Process vegetation height data from /g/data/os22/chad_tmp/AusEFlux/data/
    This dataset is originally from g/data/ub8/LandCover/OzWALD_LC/ and 
    was reprojected from 25m to 1 km previously. It is a static dataset that
    is replicated at every time-step.
    """
                    
    p='/g/data/os22/chad_tmp/AusEFlux/data/VegH_1km_2007_2010.nc'
    ds = xr.open_dataset(p, chunks=dask_chunks)
    ds = assign_crs(ds, crs='epsg:4326')
    ds.attrs['nodata'] = np.nan
    ds = ds['VegH']
    ds = ds.odc.reproject(geobox, resampling='average').compute()
    ds = round_coords(ds)
    
    # convert to time-series (same values for each time-step)
    # open another dataset so we can grab the time dim
    for year in years:
        
        if os.path.exists(f'{results}{var}/{var}_5km_{year}.nc'):
                continue
        else:
            if verbose:
                print(' ', year)

        #open time dim from anotther dataset
        da_time = xr.open_dataarray(f'{results}NDWI/NDWI_5km_{year}.nc').time
        
        #expand time dim using other dataset's time.
        dss = ds.expand_dims(time=da_time)
        
        #mask to aus land extent
        dss = dss.where(mask)
        dss = dss.rename(var)
        
        #export
        #dss.to_netcdf(f'{results}{var}/{var}_5km_{year}.nc')

def _ozwald_climate(years,
                var,
                base,
                results,
                geobox,
                mask,
                dask_chunks=dict(latitude=10000, longitude=10000, time=1),
                verbose=False
                   ):

    """
    Process OzWaLD temperature climate data from /g/data/ub8/au/OzWALD/daily/meteo/
    This is very memory and compute intensive (and slow)
    Consider switching to SILO temperature grids.
    The process is split into two steps.
    """

    # -----------Step 1-----------------------------------------------
    clim_inputs = {
        'Tmin':'OzWALD/daily/meteo/Tmin/OzWALD.Tmin.'+year+'.nc', 
        'Tmax':'OzWALD/daily/meteo/Tmax/OzWALD.Tmax.'+year+'.nc',
        'kTavg':'OzWALD/daily/meteo/kTavg/OzWALD.kTavg.'+year+'.nc'
         }
    
    for k,i in clim_inputs.items():
        
        if os.path.exists(f'{results}/{k}/{k}_5km_{year}.nc'):
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
        #ds = ds.chunk(latitude=10000, longitude=10000, time=1) # now rechunk for the reproject
        
        #we need to spatial resample first to reduce RAM/speed up.
        if k=='kTavg':
            #upscaling from 10km to 5km
            ds = ds.odc.reproject(geobox, resampling='nearest').compute()
            ds = round_coords(ds)
        else:
            # downsacling from 500m to 5km
            ds = ds.odc.reproject(geobox, resampling='average').compute()
            ds = round_coords(ds)

        #tidy up
        ds = ds.transpose('time', 'latitude', 'longitude')
        ds = ds.rename(k)
        ds = assign_crs(ds, crs='epsg:4326')
        
        # #export result
        folder = results+k
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        #ds.astype('float32').to_netcdf(f'{results}/{k}/{k}_5km_{year}.nc')

    # -----------Step 2-----------------------------------------------
    for year in years:
    
        clim_inputs = {
            'Tmin':f'{results}/Tmin/Tmin_5km_{year}.nc', 
            'Tmax':f'{results}/Tmax/Tmax_5km_{year}.nc',
            'kTavg':f'{results}/kTavg/kTavg_5km_{year}.nc'
             }
        
        if os.path.exists(f'{results}{var}/{var}_5km_{year}.nc'):
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
    
        #resample time
        ds = ds.resample(time='MS', loffset=pd.Timedelta(14, 'd')).mean()
        
        #tidy up
        ds.attrs['nodata'] = np.nan
        ds = ds.rename(var)
    
        #mask to aus land extent
        ds = ds.where(mask)
    
        #export result
        folder = f'{results}{var}'
        if not os.path.exists(folder):
            os.makedirs(folder)
    
        #ds.astype('float32').to_netcdf(f'{results}{var}/{var}_5km_{year}.nc')

def _SILO_climate(years,
                var,
                base,
                results,
                geobox,
                mask,
                dask_chunks=dict(latitude=250, longitude=250, time=-1),
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
                
                if os.path.exists(f'{results}/{k}/{k}_5km_{year}.nc'):
                    continue
                else:
                    if verbose:
                        print(' ', k, year)
                
                #open and do some prelim processing
                ds = xr.open_dataset(base+i).drop('crs').chunk(chunks)
                ds = assign_crs(ds, crs='epsg:4326')
                ds = ds.to_array()
                ds = ds.squeeze().drop_vars('variable')
                ds.attrs['nodata'] = np.nan
        
                # resample time and space
                ds = ds.resample(time='MS', loffset=pd.Timedelta(14, 'd')).mean()
                
                if k in ['SRAD', 'VPD']:
                    method='nearest'
                if k=='rain':
                    method='bilinear'
                
                ds = ds.odc.reproject(gbox, resampling=method).compute()

                if k=='VPD':
                    #calculate VPD
                    ds = round_coords(ds)
                    ds = assign_crs(ds, crs='epsg:4326')
                    ds = ds.where(mask)
                    ta = xr.open_dataarray(f'{results}/Tavg/Tavg_5km_{year}.nc')
                    sat_vp = (6.11 * np.exp((2500000/461) * (1/273 - 1/(273 + ta))))
                    ds = sat_vp - ds# tidy up
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
                
                #ds.astype('float32').to_netcdf(f'{results}/{k}/{k}_5km_{year}.nc')
    














