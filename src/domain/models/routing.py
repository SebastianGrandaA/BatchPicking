from concurrent.futures import ProcessPoolExecutor
from typing import Any
from domain.models.instances import Item
from domain.models.solutions import Batch, Problem

class Routing(Problem):
    """
    Interface for routing problems.
    """
    graph: dict[int, Item] = {}

    @property
    def nodes(self) -> list[Item]:
        return list(self.graph.values())
    
    @property
    def node_ids(self) -> list[int]:
        return list(self.graph.keys())

    @property
    def status(self) -> str:
        raise NotImplementedError

    @property
    def is_valid(self) -> bool:
        return self.status in ['Optimal', 'Feasible']

    def route(self, *args, **kwargs) -> Any:
        raise NotImplementedError
        
    def build_graph(self) -> Any:
        raise NotImplementedError

    def build_matrix(self) -> Any:
        raise NotImplementedError

    def build_model(self) -> Any:
        raise NotImplementedError

    def solve(self, batches: list[Batch]) -> list[Batch]:
        """
        Solve multiple TSP instances in parallel, one for each batch (CPU-bound operation).

        Parameters
        ----------
        method : str
            Method to solve the TSP. Options: 'TSPBase', 'TSPMultiCommodityFlow'.

        batches : list[Batch]
            Batches of items to be routed independently.
        """
        routes = []

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self.route(batch=batch))
                for batch in batches
            ]

            for future in futures:
                routes.append(future.result())

        return routes

    def solve_sequential(self, method: str, batches: list[Batch]):
        """
        Solve multiple TSP instances sequentially.

        Parameters
        ----------
        method : str
            Method to solve the TSP. Options: 'TSPBase', 'TSPMultiCommodityFlow'.

        batches : list[Batch]
            Batches of items to be routed independently.
        """
        routes = []

        for batch in batches:
            routes.append(self.route(batch=batch))

        return routes
    
    def build_model(self, batches: list[Batch]):
        return self.solve(batches=batches)
