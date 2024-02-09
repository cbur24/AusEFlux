# Jupyter notebooks for running AusEFlux for a single year

Notebooks in this folder describe the methods for creating quasi-operational annual carbon and water fluxes for Australia. Use these notebooks if your goal is to generate annual fluxes for the past calender year. The notebooks are numbered in sequential order of their operation.

The notebook `3a_Retrain_models_(optional).ipynb` provides a means to retrain the ML models. This is an optional notebook that is provided if wishing to update the models in light of new eddy-covariance (EC) sites coming into operation, or to take advantage of longer time-series of EC data. 

**Ideal compute environment:**
- NCI's 'normal' queue
- X-large (24 cores, 95GiB)
- Python 3.10.0
- Python venv: `/g/data/os22/chad_tmp/AusEFlux/env/py310`

