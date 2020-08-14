import logging
import time
from numbers import Integral
from typing import Optional


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        interval = time.time() - self.start
        logging.info("[%s] Spent time: %.4f seconds.", self.name, interval)


class Singleton(type):
    """
    Metaclass for creating singletons.
    """
    _INSTANCES = dict()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._INSTANCES:
            cls._INSTANCES[cls] = super(Singleton, cls).__call__(*args,
                                                                 **kwargs)
        return cls._INSTANCES[cls]


class CommandLineArguments(metaclass=Singleton):
    """
    Command line arguments used at different places within the project.
    """
    def __init__(self):
        self.num_concurrent_jobs: Optional[Integral] = None
