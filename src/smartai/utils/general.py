"""General Utility functions
"""
import sys
from functools import partial
import pandas as pd
from tqdm import tqdm, tqdm_notebook
from IPython import get_ipython


# noinspection PyBroadException
def auto_tqdm(unit='iter'):
    notebook = False
    if not unit.startswith(' '):
        unit = ' ' + unit
    try:
        ip = get_ipython()
        if ip.has_trait('kernel'):
            notebook = True
    except:
        pass
    if notebook is True:
        return partial(tqdm_notebook, unit=unit)
    else:
        return partial(tqdm, unit=unit, ascii=True)


def get_real_size(obj, visited=None):
    """Recursively caculate the size in bytes of a python object

    :param obj: object, any python object
    :param visited: set, only used by recursive calls
    :return: int, object size in bytes
    """
    size = sys.getsizeof(obj)
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return 0
    # important to mark as visited *before* entering recursion
    # to handle self-referential objects
    visited.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_real_size(v, visited) for v in obj.values()])
        size += sum([get_real_size(k, visited) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_real_size(obj.__dict__, visited)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_real_size(i, visited) for i in obj])
    return size


def approximate_size(size, use_1024_bytes=False):
    """Convert a file size to human-readable form.

    Copy from Dive into Python 3 examples.

    Args:
        - size -- file size in bytes
        - use_1024_bytes -- if True (default), use multiples of 1024, if False, 1000

    Returns:
        a string of the file size in a human readable form
    """
    if size < 0:
        raise ValueError('number must be non-negative')

    suffixes = {
        1000: ['KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'],
        1024: ['KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB']
    }

    multiple = 1024 if use_1024_bytes else 1000
    for suffix in suffixes[multiple]:
        size /= multiple
        if size < multiple:
            return '{0:.1f} {1}'.format(size, suffix)

    raise ValueError('number too large')


def memory_usage(obj, deep=False, use_1024_bytes=False):
    """Return memory usage of a pandas DataFrame or Series in human-readable form.

    Args:
        - obj -- a Python object
        - deep -- args pass into df.memory_usage() method
        - use_1024_bytes -- if True (default), use multiples of 1024, if False, 1000

    Returns:
        a string of the DataFrame size in a human readable form
    """
    if isinstance(obj, pd.DataFrame):
        size = obj.memory_usage(deep=deep).sum()
    elif isinstance(obj, pd.Series):
        size = obj.memory_usage(deep=deep)
    else:
        size = get_real_size(obj) if deep else sys.getsizeof(obj)
    return approximate_size(size, use_1024_bytes=use_1024_bytes)
