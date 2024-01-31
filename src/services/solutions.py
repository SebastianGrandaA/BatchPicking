from logging import warning

import pyomo.environ as pyo
from pydantic import BaseModel

from .instances import Item, Order, Vehicle, Warehouse


class Problem(BaseModel):
    warehouse: Warehouse

    @property
    def minimum_batches(self) -> int:
        return self.warehouse.minimum_batches

    def is_valid(self, result) -> bool:
        if result.solver.termination_condition == pyo.TerminationCondition.infeasible:
            warning(f'GraphPartition | Infeasible model')
            return False

        return (
            result.solver.status == pyo.SolverStatus.ok
            and result.solver.termination_condition == pyo.TerminationCondition.optimal
        )
    
    def build_model(self, **kwargs):
        raise NotImplementedError

    def solve(self, **kwargs):
        model = self.build_model(**kwargs)
        solver = pyo.SolverFactory('gurobi')
        result = solver.solve(model, tee=True)

        if self.is_valid(result):
            return self.build_solution(model)

class Route(BaseModel):
    sequence: list[Item] = []
    total_distance: int = 0

    @property
    def nb_positions(self) -> int:
        return len(self.sequence)

class Batch(BaseModel):
    orders: list[Order]
    route: Route = Route()

    @property
    def id(self) -> str:
        return '-'.join([str(order.id) for order in self.orders])

    @property
    def depot(self) -> tuple[int, int]:
        sample_items = self.orders[1].items

        return (sample_items[0].id, sample_items[-1].id)
    
    @property
    def items(self) -> list[Item]:
        """
        Returns all the items in the batch excluding the depot.
        """
        return [item for order in self.orders for item in order.items if item.id not in self.depot]
    
    @property
    def volume(self) -> int:
        return sum(order.volume for order in self.orders)
    
    @property
    def nb_items(self) -> int:
        return sum(order.nb_items for order in self.orders)

    def is_valid(self, vehicle: Vehicle) -> bool:
        return (
            self.volume <= vehicle.max_volume
            and self.nb_items <= vehicle.max_nb_items
        )
    
    def __str__(self) -> str:
        order_ids = [order.id for order in self.orders]
        position_ids = [item.id for item in self.route.sequence]

        return f'Batch(order_ids={order_ids}, route={position_ids})'
    
class Solution(BaseModel):
    batches: list[Batch]

    def save(self, path: str = 'solution.txt'):
        with open(path, 'w') as file:
            file.write(str(self))