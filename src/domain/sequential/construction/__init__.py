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
    #### Construction heuristic

    The construction heuristic is a sequential approach that given the batches are formed, the routing problem is solved independently as a TSP for each batch.
    The main motivation for this approach is to employ a distance metric that does not require to enumerate all the possible routes to measure the convenience of grouping orders into batches.

    The Hausdorff distance measures the geographical closeness between the items of two orders.
    The intra-order closeness is 0 since the items are already grouped in the same order.
    This leads to inconveniences when modeling the batching problem as a set partitioning problem, as it would always create single-order batches because there is no incentive to group orders together.
    Classical clustering algorithms can not be easily applied because most of them rely on the Euclidean distance, which is not suitable for our desired metric. Also, it became difficult to control the number and the capacity of the clusters.
    To overcome these issues, a location-allocation problem is proposed to exploit the Hausdorff distance as a measure of the convenience for clustering orders.
    Specifically, the p-median problem determines the subset of orders that are closer to other orders based on the unitary and the volume capacity constraints.
    The set partitioning problem and the clustering algorithms were also implemented as a proof of concept, in the classes `GraphPartition` and `Clustering`, respectively.
    More details about this metric can be found at `src/services/distances.py`.

    Once the batches are formed, the routing problem can be decomposed for each batch and solved in parallel using a TSP solver.
    From an experimentation point of view, the TSP solver can be a simple TSP (TSPBase), a multi-commodity flow TSP (TSPMultiCommodityFlow), or a simplified implementation of [OR-Tools](https://developers.google.com/optimization/routing/vrp) for the VRP problem.
    The last option is the default routing method as it shows the best performance in terms of solution quality and computational time.
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
