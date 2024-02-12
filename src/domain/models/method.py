from functools import wraps
from logging import info
from time import time
from typing import Any

from memory_profiler import memory_usage
from pydantic import BaseModel

from domain.models.instances import Warehouse


def measure_consumption(func: Any) -> Any:
    """Decorator to measure the consumption of time and memory of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        start_memory = memory_usage(-1, interval=0.1, timeout=1, include_children=True)

        result = func(*args, **kwargs)

        end_time = time()
        end_memory = memory_usage(-1, interval=0.1, timeout=1, include_children=True)

        elapsed_time = round(end_time - start_time, 2)
        memory_peak = round(max(end_memory) - max(start_memory), 2)
        info(
            f"{func.__name__} | Time (sec) {elapsed_time} | Memory peak (MB) {memory_peak}"
        )

        return result, elapsed_time

    return wrapper


class Method(BaseModel):
    warehouse: Warehouse
    timeout: int = 100  # seconds

    @measure_consumption
    def solve(self):
        raise NotImplementedError


class Callbacks(BaseModel):
    distance: Any = None
    demand: Any = None
    volume: Any = None
