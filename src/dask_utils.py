# This file is part of the Open Data Cube, see https://opendatacube.org for more information
# COPIED THIS FROM: https://github.com/opendatacube/datacube-core/blob/develop/datacube/utils/dask.py

""" Dask Distributed Tools

"""
from typing import Any, Iterable, Optional, Union, Tuple
from random import randint
import toolz  # type: ignore[import]
import queue
from dask.distributed import Client
import dask
import threading
import logging
import os


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

