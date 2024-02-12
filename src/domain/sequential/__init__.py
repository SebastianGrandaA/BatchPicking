from logging import info

from domain.models.method import Method, measure_consumption
from domain.models.solutions import Batch, Problem
from domain.sequential.construction.partition import Clustering
from domain.sequential.construction.tsp import TSPBase, TSPMultiCommodityFlow

BASE_METHOD = "TSPBase"


class Sequential(Method):
    """Cluster-first, route-second approach."""

    @measure_consumption
    def solve(self, routing_method: str = BASE_METHOD) -> list[Batch]:
        initial_solution = Construction(**self.__dict__).solve(routing_method)
        improved_solution = LocalSearch(**self.__dict__).solve(initial_solution)

        return improved_solution


class Construction(Problem):
    """Construction heuristic interface to build an initial solution."""

    def solve(self, routing_method: str) -> list[Batch]:
        batches = Clustering(**self.__dict__).solve()
        info(f"Construction | Batches | {[str(batch) for batch in batches]}")

        if routing_method == "TSPMultiCommodityFlow":
            routing_model = TSPMultiCommodityFlow(**self.__dict__)
        elif routing_method == "TSPBase":
            routing_model = TSPBase(**self.__dict__)
        else:
            raise ValueError(f"Unknown routing method {routing_method}")

        routes = routing_model.solve(batches=batches)
        info(f"Construction | Routes | {[str(route) for route in routes]}")

        return routes


class LocalSearch(Problem):
    """Local search interface to improve the initial solution."""

    def solve(self, batches: list[Batch]) -> list[Batch]:
        return batches
