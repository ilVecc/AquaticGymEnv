import datetime as dt
import os
from time import perf_counter


def assure_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_time(start):
    return str(dt.timedelta(seconds=int(perf_counter() - start)))
