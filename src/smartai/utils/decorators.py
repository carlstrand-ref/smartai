"""Useful custom decorators
"""
from datetime import datetime
from functools import wraps

def print_now(prefix='current time:'):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(prefix, now)


def format_time_delta(time_delta):
    return str(time_delta).split('.')[0]


def timelogger(func):
    """Decorator to log the start and end time for a function call
    """
    @wraps(func)
    def wrapper(*args, **kwds):
        ctime = datetime.now()
        print_now(prefix='start time:')
        result = func(*args, **kwds)
        print_now(prefix='end time:')
        used_time = datetime.now() - ctime
        print('Time used:', format_time_delta(used_time))
        return result
    return wrapper
