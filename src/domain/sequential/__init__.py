from domain.models.solutions import Problem
from domain.sequential.construction.tsp import TSPBase, TSPMultiCommodityFlow
from domain.models.solutions import Batch
from domain.models.method import Method
from domain.sequential.construction.partition import Clustering

from logging import info

class Sequential(Method):
    """
    Cluster-first, route-second approach.
    """
    def build_initial_solution(self) -> list[Batch]:
        """
        Construction heuristic to build an initial solution.
        """
        batches = Clustering(**self.__dict__).solve()
        info(f'{[str(batch) for batch in batches]}')

        routing_method = TSPBase(**self.__dict__)
        # TODO probar con TSPMultiCommodityFlow

        routes = routing_method.solve(batches=batches)
        info(f'{[str(route) for route in routes]}')

        return routes

    def solve(self) -> list[Batch]:
        initial = self.build_initial_solution()
        # TODO implement local search
        return initial


class Construction(Problem):
    pass


class LocalSearch(Problem):
    pass