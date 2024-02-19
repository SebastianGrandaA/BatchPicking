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
    Construction heuristic interface to build an initial solution.

    #### 2.1.1. Construction heuristic

        Heursitic: cluster-first, route-second heuristic initially proposed buy Bodin and Sexton [4]

        [chapter 9 book]
        Clustering algorithms use customer proximity to guide and possibly simplify the routing aspect. Geographical closeness among customers is used either a priori or in parallel with
        the routing process to cluster them. An early approach was that of Cullen, Jarvis, and Ratliff [8], who proposed an interactive optimization approach for the multiple vehicle dial-
        a-ride problem where customers are serviced by a homogeneous fleet. For the same context,
        Bodin and Sexton [4] developed a traditional cluster-first, route-second approach. Single vehicle cases are solved using the method of Sexton and Bodin [44


        cluster-first-route-second heuristic

        Sequential approach:
            Batching
                Most of the literature relates this problem with the set partitioning problem, in which the objective function requires to compute the best picking tour for each combination of orders (all the positions in those orders).
                !! However, this approach would require a column-generation schema to avoid enumerating all the possible routes. (opportunity to continue the research)
                Therefore, we seek to evaluate the closeness and apply a clustering algorithm

                    [2] Order Batching in Order Picking Warehouses: A Survey of Solution Approaches.pdf

                Clustering
                by closeness
            2.1.1.1.TSP (only one formulation)

    ## Implementation details

    N versions of the TSP is proposed.

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
