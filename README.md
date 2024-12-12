
<img align="centre" src="banner_picture.png" width="100%">

# AusEFlux: Empirical upscaling of terrestrial carbon and water fluxes over Australia using the OzFlux eddy covariance network

AusEFlux (**Aus**tralian **E**mpirical **Flux**es) is a high resolution (500 metre) _operational_ gridded estimate of Gross Primary Productivity, Ecosystem Respiration, Net Ecosystem Exchange, and Evapo-transpiration over the Australian continent for the period January 2003 to present.  This new estimate of Australia’s terrestrial carbon cycle provides a benchmark for assessment against Land Surface Model simulations, and a means for monitoring of Australia’s terrestrial carbon cycle at an unprecedented high-resolution.

**The datasets are free and openly available through the NCI's high-performance THREDDS data service:** https://thredds.nci.org.au/thredds/catalog/ub8/au/AusEFlux/catalog.html

* The `notebooks/` folder contains all the methods and instructions for running the workflow.
* The `src/` folder contains the python scripts for supporting the workflows. 

The methods in this repository develop an **operational workflow for production of AusEFlux**. This repository differs from the [NEE_modelling](https://github.com/cbur24/NEE_modelling) repo which describes the research methods that inform the [EGU Biogeosciences publication](https://doi.org/10.5194/bg-20-4109-2023).

If using these methods or datasets please cite:

> Burton, C.A., Renzullo, L. J., Rifai, S. W., & Van Dijk, A. I., Empirical upscaling of OzFlux eddy covariance for high-resolution monitoring of terrestrial carbon uptake in Australia. Biogeosciences, 2023. 20(19): p. 4109-4134.

**License:** The code in this repository is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
