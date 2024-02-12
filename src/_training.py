import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from odc.geo.xr import assign_crs

def VPD(rh, ta):
    sat_vp = (6.11 * np.exp((2500000/461) * (1/273 - 1/(273 + ta))))
    vpd = (((100 - rh)/100) * sat_vp)
    return vpd

def extract_rs_vars(path, flux_time, time_start, time_end, idx, add_comparisons=False):
    if add_comparisons:
        if 'quantiles' in path:
            ds = xr.open_dataset(path).sel(quantile=0.5).drop('quantile')
        else:
            ds = xr.open_dataset(path)
        
        if 'FLUXCOM' in path:
            ds = ds*30
            
        if 'meteo_era5' in path:
            ds = ds.rename({'lat':'latitude', 'lon':'longitude'})
            ds=ds[path[-20:-17]]

        try:
            ds = ds.rename({'y':'latitude', 'x':'longitude'})
        
        except:
            pass
    else:
        ds = assign_crs(xr.open_dataset(path), crs='EPSG:4326')
        #ds = ds.to_array()

    ds = ds.sel(idx, method='nearest').sel(time=slice(time_start, time_end)) # grab pixel
    ds = ds.reindex(time=flux_time, method='nearest', tolerance='1D').compute() 

    try:
        ds = ds.to_dataframe().drop(['latitude', 'longitude', 'spatial_ref'], axis=1)
    except:
        ds = ds.to_dataframe().drop(['latitude', 'longitude'], axis=1)
    
    return ds

def extract_ozflux(version='2023_v2',
                    level='L6',
                    type='default',
                    rs_data_folder=None,
                    save_ec_data=None,
                    return_coords=True,
                    export_path=None,
                    verbose=True
                           ):
    """
    Extract OzFlux data from THREDDS, and environmental
    data from remote sensing/climate datasets over pixels at EC
    tower location.
    
    Params:
    ------
    version: str. version ID of the ozflux data to download e.g '2023_v2'
    level: str. processing level to download, 'L3 to 'L6'
    type: str. Either 'default' or 'site_pi'
    rs_data_folder: path.
    save_ec_data: path. If a path is specified, netcdf data will be saved
    return_coords : bool. If True returns the x,y coordinates of the EC tower as columns on the
            pandas dataframe / csv.
    export_path: path. If a path is provided, a .csv file is output with the ozflux data.
    verbose : bool. If true progress statements are printed
    
    
    Returns:
    -------
        Pandas.Dataframe
        
    """
    #-----Get Eddy covariance data from the OzFlux THREDDS server--------------------
    
    #get list of all the folders on the THREDDS server
    url = "https://dap.tern.org.au/thredds/catalog/ecosystem_process/ozflux/catalog.html"
    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    sites_names = []
    for link in soup.select('a[href*=".html"]'):
        sites_names.append(link["href"])
    
    #get rid of the couple of unneeded files
    sites_names = sites_names[2:-2]
    
    if verbose:
        print('Total number of sites to collect:', len(sites_names))
    
    #loop through all the sites and open the datasets specified by the version and level etc.
    for i in range(len(sites_names)):
        if verbose:
            print(sites_names[i][0:-13])
        
        partial_url = url.replace("catalog.html", "")+sites_names[i].replace("catalog.html", "")+version+'/'+level+'/'+type+'/'+'catalog.html'
        soup = BeautifulSoup(requests.get(partial_url).content, "html.parser")
        
        files = []
        for link in soup.select('a[href*="Monthly.nc"]'):
            files.append(link["href"])
        
        try:
            full_path = partial_url.replace("catalog.html", "").replace('catalog', 'dodsC')+files[0][files[0].rindex('/')+1:]
        except:
            print('', sites_names[i][0:-13] + ' does not exist for this combination of versions, level...skipping.')
            continue
        
        if os.path.exists(export_path+sites_names[i][0:-13]+'.csv'):
            print('   skipping '+sites_names[i][0:-13]+' as already exists in save location')
            continue
        
        flux = xr.open_dataset(full_path)

        if save_ec_data:
            if not os.path.exists(save_ec_data):
                os.makedirs(save_ec_data)
            
            del flux.attrs['_NCProperties'] #delete 'reserved' property name
            flux.to_netcdf(save_ec_data+sites_names[i][0:-13]+'_'+version+'_'+level+'.nc')
        
        # # Set negative GPP, ER, and ET measurements as zero
        # flux['GPP_SOLO'] = xr.where(flux.GPP_SOLO < 0, 0, flux.GPP_SOLO)
        # flux['ET'] = xr.where(flux.ET < 0, 0, flux.ET)
        # flux['ER_SOLO'] = xr.where(flux.ER_SOLO < 0, 0, flux.ER_SOLO)
        
        # offset time to better match gridded data
        flux['time'] = flux.time + np.timedelta64(14,'D') 
        
        #indexing spatiotemporal values at EC site
        lat = flux.latitude.values[0]
        lon = flux.longitude.values[0]
        time_start = str(np.datetime_as_string(flux.time.values[0], unit='D'))
        time_end = str(np.datetime_as_string(flux.time.values[-1], unit='D'))
        
        if "Longr" in full_path[63:68]: #coordinates on nc file are wrong
            lat=-23.5232
            lon=144.3104
        
        #index for grabbing pixels
        idx=dict(latitude=lat,  longitude=lon)

        #convert to dataframe
        df_ec = flux.to_dataframe().reset_index(level=[1, 2]).drop(['latitude', 'longitude'], axis=1)
        df_ec = df_ec.add_suffix('_EC')
     
        # calculate VPD on ec data
        df_ec['VPD_EC'] = VPD(df_ec.RH_EC, df_ec.Ta_EC)
        df_ec = df_ec.drop(['VP_EC'], axis=1) # drop VP

        #add canopy height from the attributes
        try:
            df_ec['VegH_EC'] = float(flux.attrs['canopy_height'][:-1])
        except:
            df_ec['VegH_EC'] = np.nan
    
        #--------Remote sensing data--------------------------------------
        
        # extract the first remote sensing variable
        if rs_data_folder:
            covariables = [rs_data_folder+i for i in os.listdir(rs_data_folder) if i.endswith('.nc')]
            covariables.sort()
        
            first_var = covariables[0]
            if verbose:
                print('   Extracting RS data...')
            
            first = extract_rs_vars(covariables[0],
                          flux.time, time_start, time_end, idx)
            
            #extract the rest of the RS variables in loop    
            dffs = []
            for var in covariables[1:]:   
                df = extract_rs_vars(var,
                       flux.time, time_start, time_end, idx)
                
                dffs.append(df)
    
            # join all the datasets EC and RS
            df_rs = first.join(dffs)
            df_rs = df_rs.add_suffix('_RS') 
            df = df_ec.join(df_rs)
            
            if return_coords:
                df['x_coord'] = lon
                df['y_coord'] = lat
            
            time = df.reset_index()['time'].dt.normalize()
            df = df.set_index(time)
         
        if export_path:
            if not os.path.exists(export_path):
                os.makedirs(export_path)
            df.to_csv(export_path+sites_names[i][0:-13]+'.csv')

    
