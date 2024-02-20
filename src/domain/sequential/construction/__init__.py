from logging import info

from domain.joint.vrp import VRP
from domain.models.solutions import Batch, Problem
from domain.sequential.construction.batching import Clustering, GraphPartition, PMedian
from domain.sequential.construction.tsp import TSPBase, TSPMultiCommodityFlow

CONSTRUCTION_BATCHING_METHOD_DEFAULT = "PMedian"
CONSTRUCTION_BATCHING_METHODS = {
    "PMedian": PMedian,
    "GraphPartition": GraphPartition,
    "Clustering": Clustering,
}
CONSTRUCTION_ROUTING_METHOD_DEFAULT = "VRP"
CONSTRUCTION_ROUTING_METHODS = {
    "TSPMultiCommodityFlow": TSPMultiCommodityFlow,
    "TSPBase": TSPBase,
    "VRP": VRP,
}
CONSTRUCTION_ROUTING_DEFAULT_PARAMS = {"is_warehouse_complete": False}


class Construction(Problem):
    """
    # Construction heuristic

    The construction heuristic is a sequential approach that given the batches are formed, the routing problem is solved independently as a TSP for each batch.
    The main motivation for this approach is to employ a distance metric that does not require to enumerate all the possible routes to measure the convenience of grouping orders into batches.
    The Hausdorff distance is employed to measure the geographical closeness between each pair of orders.
    """

    def batch(self, batching_method: str) -> list[Batch]:
        if batching_method not in CONSTRUCTION_BATCHING_METHODS:
            raise ValueError(f"Unknown batching method {batching_method}")

        batching_model = CONSTRUCTION_BATCHING_METHODS[batching_method](**self.__dict__)

        return batching_model.solve()

    def route(self, routing_method: str, batches: list[Batch]) -> list[Batch]:
        if routing_method not in CONSTRUCTION_ROUTING_METHODS:
            raise ValueError(f"Unknown routing method {routing_method}")

        routing_model = CONSTRUCTION_ROUTING_METHODS[routing_method]
        data = self.__dict__

        if routing_method == "VRP":
            data = {**self.__dict__, **CONSTRUCTION_ROUTING_DEFAULT_PARAMS}

        return routing_model(**data).solve_sequential(batches=batches)

    def solve(self, **kwargs) -> list[Batch]:
        batching_method = kwargs.get("batching_method", "PMedian")
        batches = self.batch(batching_method)
        info(
            f"Construction | Batching {batching_method} | {[str(batch) for batch in batches]}"
        )

        routing_method = kwargs.get("routing_method", "VRP")
        routes = self.route(routing_method, batches)
        info(
            f"Construction | Routing {routing_method} | {[str(route) for route in routes]}"
        )

        return routes
