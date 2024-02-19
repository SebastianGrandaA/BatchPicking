from domain.models.method import Method, measure_consumption
from domain.models.solutions import Batch
from domain.sequential.construction import Construction
from domain.sequential.local_search import LocalSearch


class Sequential(Method):
    """
    ## Implementation details
    The routing problem is solved in parallel with the TSP problem. This implements multiprocessor because it is an ...-bound problem.
    N version of the partitioning problem.
    N versions of the TSP is proposed.


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
