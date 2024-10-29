import os
import joblib
import numpy as np
import xarray as xr
import dask.array as da
from odc.geo.xr import assign_crs
from dask_ml.wrappers import ParallelPostFit

import sys
sys.path.append('/g/data/os22/chad_tmp/AusEFlux/src/')
from _utils import round_coords


def allNaN_arg(da, dim, stat, idx=True):
    """
    Calculate da.argmax() or da.argmin() while handling
    all-NaN slices. Fills all-NaN locations with an
    float and then masks the offending cells.

    Parameters
    ----------
    da : xarray.DataArray
    dim : str
        Dimension over which to calculate argmax, argmin e.g. 'time'
    stat : str
        The statistic to calculte, either 'min' for argmin()
        or 'max' for .argmax()
    idx : bool
        If True then use da.idxmax() or da.idxmin(), otherwise
        use ds.argmax() or ds.argmin()

    Returns
    -------
    xarray.DataArray
    """
    # generate a mask where entire axis along dimension is NaN
    mask = da.isnull().all(dim)

    if stat == "max":
        y = da.fillna(float(da.min() - 1))
        if idx==True:
            y = y.idxmax(dim=dim, skipna=True).where(~mask)
        else:
            y = y.argmax(dim=dim, skipna=True).where(~mask)
        return y

    if stat == "min":
        y = da.fillna(float(da.max() + 1))
        if idx==True:
            y = y.idxmin(dim=dim, skipna=True).where(~mask)
        else:
            y = y.argmin(dim=dim, skipna=True).where(~mask)
        return y
    

def collect_prediction_data(data_path=None,
                            time_range=None,
                            chunks=dict(time=1),
                            export=False,
                            verbose=True
                           ):

    """
    Gather all gridded prediction/feature data into a
    single xrray.Dataset. Assumes all datasets in the 
    'data_path' folder are spatio-temporally harmonised.

    Parameters
    ----------
    data_path : str.
        Path to a folder where netcdf files are stored.
        All datasets in this folder are joined into an 
        xr.Dataarray.
    time_range : tuple.
        We can clip the time-range of the data using the 
        this parameters, useful if some of the datasets
        in 'data_path' have different length time dimensions.
        e.g ('1982', '2022')
    export: str
        If specifying a path, the Dataset will be exported as
        a netcdf.

    Returns
    -------
        xarray.DataAset
        
    """
    
    # Grab a list of all datasets in the folder
    covariables = [data_path+i for i in os.listdir(data_path) if i.endswith('.nc')]
    covariables.sort()

    #loop through datasets and append
    dss=[]
    for var in covariables:
        if verbose:
            print('Extracting', var.replace(base, ''))

        ds = assign_crs(xr.open_dataset(var, chunks=chunks), crs='EPSG:4326')
        if time_range:
            ds = ds.sel(time=slice(str(time_range[0]), time_range[1]))
        ds = round_coords(ds)
        dss.append(ds)
    
    #merge all datasets together
    if verbose:
        print('   Merge datasets')
    
    data = xr.merge(dss, compat='override')

    # format
    data = data.rename({'latitude':'y', 'longitude':'x'}) #this helps with predict_xr
    data = data.astype('float32') #make sure all data is in float32
    data = assign_crs(data, crs='epsg:4326')

    if export:
        if verbose:
            print('   Exporting netcdf')
        data.compute().to_netcdf(f'{export}/prediction_data_{time_start}_{time_end}.nc')
    
    return data

def predict_xr(
    model,
    input_xr,
    chunk_size=None,
    persist=False,
    proba=False,
    clean=False,
    return_input=False,
):
    """
    Using dask-ml ParallelPostfit(), runs  the parallel
    predict and predict_proba methods of sklearn
    estimators. Useful for running predictions
    on a larger-than-RAM datasets.

    Last modified: September 2020

    Parameters
    ----------
    model : scikit-learn model or compatible object
        Must have a .predict() method that takes numpy arrays.
    input_xr : xarray.DataArray or xarray.Dataset.
        Must have dimensions 'x' and 'y'
    chunk_size : int
        The dask chunk size to use on the flattened array. If this
        is left as None, then the chunks size is inferred from the
        .chunks method on the `input_xr`
    persist : bool
        If True, and proba=True, then 'input_xr' data will be
        loaded into distributed memory. This will ensure data
        is not loaded twice for the prediction of probabilities,
        but this will only work if the data is not larger than
        distributed RAM.
    proba : bool
        If True, predict probabilities
    clean : bool
        If True, remove Infs and NaNs from input and output arrays
    return_input : bool
        If True, then the data variables in the 'input_xr' dataset will
        be appended to the output xarray dataset.

    Returns
    ----------
    output_xr : xarray.Dataset
        An xarray.Dataset containing the prediction output from model.
        if proba=True then dataset will also contain probabilites, and
        if return_input=True then dataset will have the input feature layers.
        Has the same spatiotemporal structure as input_xr.

    """
    # if input_xr isn't dask, coerce it
    dask = True
    if not bool(input_xr.chunks):
        dask = False
        input_xr = input_xr.chunk({"x": len(input_xr.x), "y": len(input_xr.y)})

    # set chunk size if not supplied
    if chunk_size is None:
        chunk_size = int(input_xr.chunks["x"][0]) * int(input_xr.chunks["y"][0])

    def _predict_func(model, input_xr, persist, proba, clean, return_input):
        x, y, crs = input_xr.x, input_xr.y, input_xr.spatial_ref.data.item()

        input_data = []

        for var_name in input_xr.data_vars:
            input_data.append(input_xr[var_name])

        input_data_flattened = []

        for arr in input_data:
            data = arr.data.flatten().rechunk(chunk_size)
            input_data_flattened.append(data)

        # reshape for prediction
        input_data_flattened = da.array(input_data_flattened).transpose()

        if clean == True:
            input_data_flattened = da.where(
                da.isfinite(input_data_flattened), input_data_flattened, 0
            )

        if (proba == True) & (persist == True):
            # persisting data so we don't require loading all the data twice
            input_data_flattened = input_data_flattened.persist()

        # apply the classification
        print("predicting...")
        out_class = model.predict(input_data_flattened)

        # Mask out NaN or Inf values in results
        if clean == True:
            out_class = da.where(da.isfinite(out_class), out_class, 0)

        # Reshape when writing out
        out_class = out_class.reshape(len(y), len(x))

        # stack back into xarray
        output_xr = xr.DataArray(out_class, coords={"x": x, "y": y}, dims=["y", "x"])

        output_xr = output_xr.to_dataset(name="Predictions")

        if proba == True:
            print("   probabilities...")
            out_proba = model.predict_proba(input_data_flattened)

            # convert to %
            out_proba = da.max(out_proba, axis=1) * 100.0

            if clean == True:
                out_proba = da.where(da.isfinite(out_proba), out_proba, 0)

            out_proba = out_proba.reshape(len(y), len(x))

            out_proba = xr.DataArray(
                out_proba, coords={"x": x, "y": y}, dims=["y", "x"]
            )
            output_xr["Probabilities"] = out_proba

        if return_input == True:
            print("   input features...")
            # unflatten the input_data_flattened array and append
            # to the output_xr containin the predictions
            arr = input_xr.to_array()
            stacked = arr.stack(z=["y", "x"])

            # handle multivariable output
            output_px_shape = ()
            if len(input_data_flattened.shape[1:]):
                output_px_shape = input_data_flattened.shape[1:]

            output_features = input_data_flattened.reshape(
                (len(stacked.z), *output_px_shape)
            )

            # set the stacked coordinate to match the input
            output_features = xr.DataArray(
                output_features,
                coords={"z": stacked["z"]},
                dims=[
                    "z",
                    *["output_dim_" + str(idx) for idx in range(len(output_px_shape))],
                ],
            ).unstack()

            # convert to dataset and rename arrays
            output_features = output_features.to_dataset(dim="output_dim_0")
            data_vars = list(input_xr.data_vars)
            output_features = output_features.rename(
                {i: j for i, j in zip(output_features.data_vars, data_vars)}
            )

            # merge with predictions
            output_xr = xr.merge([output_xr, output_features], compat="override")

        return assign_crs(output_xr, crs='EPSG:'+str(crs))

    if dask == True:
        # convert model to dask predict
        model = ParallelPostFit(model)
        with joblib.parallel_backend("dask"):
            output_xr = _predict_func(
                model, input_xr, persist, proba, clean, return_input
            )

    else:
        output_xr = _predict_func(
            model, input_xr, persist, proba, clean, return_input
        ).compute()

    return output_xr


class HiddenPrints:
    """
    For concealing unwanted print statements called by other functions
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
