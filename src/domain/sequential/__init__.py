from domain.models.method import Method, measure_consumption
from domain.models.solutions import Batch
from domain.sequential.construction import Construction
from domain.sequential.local_search import LocalSearch


class Sequential(Method):
    """
    ### Sequential approach: batch-first route-second heuristic

    This approach consists of a construction heuristic and a local search algorithm to improve the initial solution.
    The construction method is based on a batch-first route-second heuristic, where once the batches are formed, the routing problem is solved independently for each batch.

    ## Implementation details
    The routing problem is solved in parallel with the TSP problem. Since it is a CPU-bound problem, the parallelization is done by using the multi-processing technique.
    Three versions of the batching problem are proposed: `PMedian`, `Clustering`, and `GraphPartitioning`.
    Similarly, three versions of the TSP are proposed: `TSPBase`, `TSPMultiCommodityFlow`, and `VRP`.
    """

    @measure_consumption
    def solve(self, **kwargs) -> list[Batch]:
        construction = Construction(**self.__dict__)
        initial_solution = construction.solve(**kwargs)
        local_search_params = {
            **self.__dict__,
            **{
                "routing_method": construction.route,
                "current_solution": initial_solution,
            },
        }
        improved_solution = LocalSearch(**local_search_params).solve()

        return improved_solution
