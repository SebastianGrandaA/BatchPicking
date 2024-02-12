from logging import info

from domain.models.method import Method, measure_time
from domain.models.solutions import Batch, Problem
from domain.sequential.construction.partition import Clustering
from domain.sequential.construction.tsp import TSPBase, TSPMultiCommodityFlow


class Sequential(Method):
    """
    Cluster-first, route-second approach.
    """

    def build_initial_solution(self) -> list[Batch]:
        """
        Construction heuristic to build an initial solution.
        """
        batches = Clustering(**self.__dict__).solve()
        info(f"{[str(batch) for batch in batches]}")

        routing_method = TSPBase(**self.__dict__)
        # TODO probar con TSPMultiCommodityFlow

        routes = routing_method.solve(batches=batches)
        info(f"{[str(route) for route in routes]}")

        return routes

    @measure_time
    def solve(self) -> list[Batch]:
        initial = self.build_initial_solution()
        # TODO implement local search
        return initial


class Construction(Problem):
    pass


class LocalSearch(Problem):
    pass
