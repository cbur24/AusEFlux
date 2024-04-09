# Jupyter Notebooks for running AusEFlux

`annual_update`: Notebooks in this folder describe the methods for creating quasi-operational annual carbon and water fluxes. Use these notebooks if your goal is to generate annual fluxes for the past year.

`historical`: Notebooks in this folder decribe the methods for creating historical carbon and water fluxes for Australia through the full length of the MODIS archive (i.e., 2003-present). These notebooks are similar to the `annual_update` folder, but optimised for the task of running 20+ years of predictions and the attendent complications that come with acquiring, harmonising, and predicting a full archive of data.

---
Data Sources:
* Climate data:
    * Ozwald temperature
        * /g/data/ub8/au/OzWALD/daily/meteo/Tmin/OzWALD.Tmin.<'year'>.nc
        * /g/data/ub8/au/OzWALD/daily/meteo/Tmax/OzWALD.Tmin.<'year'>.nc
        * /g/data/ub8/au/OzWALD/daily/meteo/kTavg/OzWALD.Tmin.<'year'>.nc
    * SILO: VPD, rain, & SRAD
        * /g/data/ub8/au/SILO/radiation/<'year'>.radiation.nc
        * /g/data/ub8/au/SILO/vp/<'year'>.vp.nc
        * /g/data/ub8/au/SILO/rain/<'year'>.vp.nc
* MODIS:
    * kNDVI and NDWI derived from: /g/data/ub8/au/MODIS/mosaic/MCD43A4.006/
    * LST: /g/data/ub8/au/MODIS/mosaic/MYD11A1.006/
    * NDVI: /g/data/ub8/au/OzWALD/8day/
* Vegetation height: /g/data/ub8/LandCover/OzWALD_LC/
* C4 grass over percentage: https://data.csiro.au/collection/csiro:58485
