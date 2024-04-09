# Some of this is COPIED FROM: https://github.com/opendatacube/datacube-core/blob/develop/datacube/utils/dask.py






import os
from random import randint
import toolz  # type: ignore[import]
import queue
from dask.distributed import Client
import dask
import threading
import logging
from typing import Any, Iterable, Optional, Union, Tuple
import odc.geo.xr
import xarray as xr
import numpy as np
import geopandas as gpd
import rasterio.features


__all__ = (
    "start_local_dask",
    "pmap",
    "compute_tasks",
    "partition_map",
    "save_blob_to_file",
    "save_blob_to_s3",
)

_LOG = logging.getLogger(__name__)


def get_total_available_memory(check_jupyter_hub=True):
    """ Figure out how much memory is available
        1. Check MEM_LIMIT environment variable, set by jupyterhub
        2. Use hardware information if that not set
    """
    if check_jupyter_hub:
        mem_limit = os.environ.get('MEM_LIMIT', None)
        if mem_limit is not None:
            return int(mem_limit)

    from psutil import virtual_memory
    return virtual_memory().total


def compute_memory_per_worker(n_workers: int = 1,
                              mem_safety_margin: Optional[Union[str, int]] = None,
                              memory_limit: Optional[Union[str, int]] = None) -> int:
    """ Figure out how much memory to assign per worker.

        result can be passed into ``memory_limit=`` parameter of dask worker/cluster/client
    """
    from dask.utils import parse_bytes

    if isinstance(memory_limit, str):
        memory_limit = parse_bytes(memory_limit)

    if isinstance(mem_safety_margin, str):
        mem_safety_margin = parse_bytes(mem_safety_margin)

    if memory_limit is None and mem_safety_margin is None:
        total_bytes = get_total_available_memory()
        # leave 500Mb or half of all memory if RAM is less than 1 Gb
        mem_safety_margin = min(500*(1024*1024), total_bytes//2)
    elif memory_limit is None:
        total_bytes = get_total_available_memory()
    elif mem_safety_margin is None:
        total_bytes = memory_limit
        mem_safety_margin = 0
    else:
        total_bytes = memory_limit

    return (total_bytes - mem_safety_margin)//n_workers


def start_local_dask(n_workers: int = 1,
                     threads_per_worker: Optional[int] = None,
                     mem_safety_margin: Optional[Union[str, int]] = None,
                     memory_limit: Optional[Union[str, int]] = None,
                     **kw):
    """
    Wrapper around ``distributed.Client(..)`` constructor that deals with memory better.

    It also configures ``distributed.dashboard.link`` to go over proxy when operating
    from behind jupyterhub.

    :param n_workers: number of worker processes to launch
    :param threads_per_worker: number of threads per worker, default is as many as there are CPUs
    :param memory_limit: maximum memory to use across all workers
    :param mem_safety_margin: bytes to reserve for the rest of the system, only applicable
                              if ``memory_limit=`` is not supplied.

    .. note::

        if ``memory_limit=`` is supplied, it will be parsed and divided equally between workers.

    """

    # if dashboard.link set to default value and running behind hub, make dashboard link go via proxy
    if dask.config.get("distributed.dashboard.link") == '{scheme}://{host}:{port}/status':
        jup_prefix = os.environ.get('JUPYTERHUB_SERVICE_PREFIX')
        if jup_prefix is not None:
            jup_prefix = jup_prefix.rstrip('/')
            dask.config.set({"distributed.dashboard.link": f"{jup_prefix}/proxy/{{port}}/status"})

    memory_limit = compute_memory_per_worker(n_workers=n_workers,
                                             memory_limit=memory_limit,
                                             mem_safety_margin=mem_safety_margin)

    client = Client(n_workers=n_workers,
                    threads_per_worker=threads_per_worker,
                    memory_limit=memory_limit,
                    **kw)

    return client

#- other utils -----------------------------------------------------------------

def add_geobox(ds, crs=None):
    """
    Ensure that an xarray DataArray has a GeoBox and .odc.* accessor
    using `odc.geo`.

    If `ds` is missing a Coordinate Reference System (CRS), this can be
    supplied using the `crs` param.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Input xarray object that needs to be checked for spatial
        information.
    crs : str, optional
        Coordinate Reference System (CRS) information for the input `ds`
        array. If `ds` already has a CRS, then `crs` is not required.
        Default is None.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The input xarray object with added `.odc.x` attributes to access
        spatial information.

    """

    # Import the odc-geo package to add `.odc.x` attributes
    # to our input xarray object
    import odc.geo.xr

    # If a CRS is not found, use custom provided CRS
    if ds.odc.crs is None and crs is not None:
        ds = ds.odc.assign_crs(crs)
    elif ds.odc.crs is None and crs is None:
        raise ValueError(
            "Unable to determine `ds`'s coordinate "
            "reference system (CRS). Please provide a "
            "CRS using the `crs` parameter "
            "(e.g. `crs='EPSG:3577'`)."
        )

    return ds


def create_project_directories(root_dir):
    data_dir = ['5km', 'interim', 'ozflux_netcdf', 'training_data']
    results_dir = ['AusEFlux', 'cross_val', 'figs', 'models', 'predictions']
    main_dir = [data_dir, results_dir]
    main_dir_names = ['data', 'results']
    
    #create main directories
    for i in range(0, len(main_dir)):
        for j in range(0,len(main_dir[i])):
            dirName = f'{root_dir}/{main_dir_names[i]}/{main_dir[i][j]}'
            if os.path.exists(dirName):
                print(f'Directory {dirName} already exists')   
                 
            else:
                os.makedirs(dirName)
                print(f'Directory {dirName} created')
    
    #add some further subdirectories in results/
    subs=[
        f'{root_dir}/results/predictions/ensemble/historical/',
        f'{root_dir}/results/models/ensemble/',
        f'{root_dir}/results/AusEFlux/',
        f'{root_dir}/results/cross_val/ensemble/'
    ]
    
    for var in ['GPP','ER','NEE', 'ET']:
        for s in subs:
            if os.path.exists(s+var):
                print(f'Directory {s+var} already exists')   
                 
            else:
                os.makedirs(s+var)
                print(f'Directory {s+var} created')


def xr_rasterize(
    gdf,
    da,
    attribute_col=None,
    crs=None,
    name=None,
    output_path=None,
    verbose=True,
    **rasterio_kwargs,
):
    """
    Rasterizes a vector ``geopandas.GeoDataFrame`` into a
    raster ``xarray.DataArray``.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        A ``geopandas.GeoDataFrame`` object containing the vector
        data you want to rasterise.
    da : xarray.DataArray or xarray.Dataset
        The shape, coordinates, dimensions, and transform of this object
        are used to define the array that ``gdf`` is rasterized into.
        It effectively provides a spatial template.
    attribute_col : string, optional
        Name of the attribute column in ``gdf`` containing values for
        each vector feature that will be rasterized. If None, the
        output will be a boolean array of 1's and 0's.
    crs : str or CRS object, optional
        If ``da``'s coordinate reference system (CRS) cannot be
        determined, provide a CRS using this parameter.
        (e.g. 'EPSG:3577').
    name : str, optional
        An optional name used for the output ``xarray.DataArray`.
    output_path : string, optional
        Provide an optional string file path to export the rasterized
        data as a GeoTIFF file.
    verbose : bool, optional
        Print debugging messages. Default True.
    **rasterio_kwargs :
        A set of keyword arguments to ``rasterio.features.rasterize``.
        Can include: 'all_touched', 'merge_alg', 'dtype'.

    Returns
    -------
    da_rasterized : xarray.DataArray
        The rasterized vector data.
    """

    # Add GeoBox and odc.* accessor to array using `odc-geo`
    da = add_geobox(da, crs)

    # Reproject vector data to raster's CRS
    gdf_reproj = gdf.to_crs(crs=da.odc.crs)

    # If an attribute column is specified, rasterise using vector
    # attribute values. Otherwise, rasterise into a boolean array
    if attribute_col is not None:
        # Use the geometry and attributes from `gdf` to create an iterable
        shapes = zip(gdf_reproj.geometry, gdf_reproj[attribute_col])
    else:
        # Use geometry directly (will produce a boolean numpy array)
        shapes = gdf_reproj.geometry

    # Rasterise shapes into a numpy array
    im = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=da.odc.geobox.shape,
        transform=da.odc.geobox.transform,
        **rasterio_kwargs,
    )

    # Convert numpy array to a full xarray.DataArray
    # and set array name if supplied
    da_rasterized = odc.geo.xr.wrap_xr(im=im, gbox=da.odc.geobox)
    da_rasterized = da_rasterized.rename(name)

    # If a file path is supplied, export to file
    if output_path is not None:
        if verbose:
            print(f"Exporting raster data to {output_path}")
        write_cog(da_rasterized, output_path, overwrite=True)

    return da_rasterized

def round_coords(ds):
    """
    Due to precision of float64 on coordinates, coordinates
    don't quite match after reprojection, resulting in adding spurious
    pixels after merge. Converting to float32 rounds coords so they match.
    """
    ds['latitude'] = ds.latitude.astype('float32')
    ds['longitude'] = ds.longitude.astype('float32')
    
    ds['latitude'] = np.array([round(i,4) for i in ds.latitude.values])
    ds['longitude'] = np.array([round(i,4) for i in ds.longitude.values])
    
    return ds