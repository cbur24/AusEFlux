<img align="centre" src="https://github.com/cbur24/AusEFlux/blob/master/banner_picture.png?raw=True" width="100%">

# Running AusEFlux

The `Annual_update.ipynb` and `Historical.ipynb` notebooks contain extensively documented workflows for running AusEFlux. The notebooks mostly run the functions stored in the `AusEFlux/src/` folder.

The `Annual_update.ipynb` notebook contains the workflow for annual updating of the products. It contains four main steps:
1. Spatiotemporal harmonisation of input datasets
2. The creation of feature datasets
3. Gridded ensemble predictions
4. Combining ensembles with an ensemble median and uncertainty range, and exporting production ready datasets to THREDDS

The `Historical.ipynb` notebook contains the workflow for creating historical carbon and water fluxes for Australia through the full length of the MODIS archive (i.e., 2003-2022). It has six steps:
1. Spatiotemporal harmonisation of input datasets
2. The creation of feature datasets
3. Extracting OzFlux eddy covaraince data from the TERN server and combining it with gridded remote sensing and climate datasets
4. Generating an ensemble of models
5. Gridded ensemble predictions
6. Combining ensembles with an ensemble median and uncertainty range

The final output of these notebooks are annual netcdf files for each carbon or water flux, for example: `AusEFlux_GPP_5km_quantiles_2003_v1.2.nc` where the naming convention is `"AusEFlux_[flux]_[spatial resolution]_quantiles_[year]_[version].nc`.  Results are stored in, for example, `'root_directory/results/AusEFlux/GPP/'`

---
**`analysis` folder**: Notebooks in this folder describe various analysis workflows like intercomparisons with other products, plotting etc.

---
**Data Sources:**
* OzFlux eddy covariance flux tower data:
    * https://dap.ozflux.org.au/thredds/catalog/ozflux/sites/catalog.html
* Climate data:
    * Ozwald temperature
        * /g/data/ub8/au/OzWALD/daily/meteo/Tmin/OzWALD.Tmin.<'year'>.nc
        * /g/data/ub8/au/OzWALD/daily/meteo/Tmax/OzWALD.Tmax.<'year'>.nc
        * /g/data/ub8/au/OzWALD/daily/meteo/kTavg/OzWALD.kTavg.<'year'>.nc
    * SILO: VPD, rain, & SRAD
        * /g/data/ub8/au/SILO/radiation/<'year'>.radiation.nc
        * /g/data/ub8/au/SILO/vp/<'year'>.vp.nc
        * /g/data/ub8/au/SILO/rain/<'year'>.rain.nc
* MODIS:
    * kNDVI and NDWI derived from: /g/data/ub8/au/MODIS/mosaic/MCD43A4.006/
    * LST: /g/data/ub8/au/MODIS/mosaic/MYD11A1.006/
    * NDVI: /g/data/ub8/au/OzWALD/8day/
    * LAI: /g/data/ub8/au/OzWALD/8day/
* Vegetation height: /g/data/ub8/LandCover/OzWALD_LC/
* C4 grass cover percentage: https://data.csiro.au/collection/csiro:58485
