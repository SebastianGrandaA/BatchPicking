from functools import wraps
from logging import debug
from time import time
from typing import Any

from pydantic import BaseModel

from domain.models.instances import Warehouse


def measure_time(func: Any) -> Any:
    """Decorator to measure the execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        elapsed = round(end - start, 2)
        debug(f"Function {func.__name__} took {elapsed} seconds.")

        return result, elapsed

    return wrapper


class Method(BaseModel):
    warehouse: Warehouse
    timeout: int = 100  # seconds

    @measure_time
    def solve(self):
        raise NotImplementedError


class Callbacks(BaseModel):
    distance: Any = None
    demand: Any = None
    volume: Any = None
