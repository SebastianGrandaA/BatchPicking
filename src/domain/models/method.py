from typing import Any

from pydantic import BaseModel

from domain.models.instances import Warehouse


class Method(BaseModel):
    warehouse: Warehouse
    timeout: int = 100  # seconds

    def solve(self):
        raise NotImplementedError


class Callbacks(BaseModel):
    distance: Any = None
    demand: Any = None
    volume: Any = None
