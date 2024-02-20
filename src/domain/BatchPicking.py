from logging import debug, error, info

from domain.joint import Joint
from domain.models.method import Method
from domain.models.solutions import Solution
from domain.sequential import Sequential
from services.io import Reader

MAX_ITERATIONS = 1
APPROACHES = {
    "sequential": Sequential,
    "joint": Joint,
}


class BatchPicking:
    """
    # The Batch-Picking problem

    The Batch-Picking problem, formally known in the literature as the Joint Order Batching and Picker Routing Problem (JOBPRP), is a well-known problem in the context of warehouse logistics.
    As the name suggests, the problem combines two optimization problems: the order batching problem and the picker routing problem.
    The order batching problem consists of grouping subsets of orders to pick their items together, if that leads to a reduced total distance traveled.
    On the other hand, the picker routing problem consists of determining the sequence of storage locations to pick all orders in a batch.

    This problem has been studied in the literature under different approaches.
    A Clustered Vehicle Routing Problem (CluVRP) has been proposed by [1] to model the joint problem. In this problem, the customers are grouped into clusters, and the vehicles can enter and leave the clusters multiple times.
    In our case, the customers are the items to be picked, and the clusters represent the order integrality condition to pick all items of an order in the same tour.
    An adapted formulation for this problem as a CluVRP is described in the Annexes section at the end of the [report](https://www.overleaf.com/read/xfgcnzwccnqj#8fe7b9).

    This problem has also been addressed sequentially, where the order batching problem is solved first, and then the routes are obtained for each batch.
    The advantage of this approach is the evident reduction in the complexity of the problem, as the routing problem can be solved as a Traveling Salesman Problem (TSP) for each batch.
    The drawback is the lack of coordination between the two problems, which can lead to suboptimal solutions because the batching decisions are made without considering the routing problem.
    Related ideas can be found at [2], where the authors discuss the batch-first route-second approaches against the joint approach, and the benefits of solving the problems simultaneously.
    These approaches usually require a computationally expensive calculation of the distance metric, such as calculating the shortest path between each combination of orders for a set-partitioning problem (i.e. the best sequence to pick all items in a batch).
    We consider a relevant research challenge to find a metric that best approximates the shortest sequence of orders to pick in a batch, without the need for an exhaustive search.

    In this work, we study the Joint Order Batching and Picker Routing Problem (JOBPRP) as a variant of the Pickup and Delivery Problem (PDP), and we propose a batch-first route-second heuristic to solve large instances of the problem.
    The proposed heuristic is based on the Hausdorff distance, which is a measure the closeness of two sets of points, and it is used to determine the best way to group orders into batches.
    The initial solution is obtained by solving the p-median problem, and the best sequence of items to pick in a batch is determined by solving a set of independent TSPs.
    Once the initial solution is obtained, a local-search algorithm is applied to improve the solution by swapping orders between batches and re-optimizing the routes.

    ## References
    [1] Aerts, B., Cornelissens, T., & Sörensen, K. (2021). The joint order batching and picker routing problem: Modelled and solved as a clustered vehicle routing problem. Computers & Operations Research, 129, 105168.
    [2] Henn, S., Koch, S., & Wäscher, G. (2012). Order batching in order picking warehouses: a survey of solution approaches (pp. 105-137). Springer London.
    """

    @staticmethod
    def dispatch(method: str, **kwargs) -> Method:
        """Dispatch to the optimization approach."""
        name = method.lower()

        if name in APPROACHES:
            return APPROACHES[name](**kwargs)
        else:
            raise ValueError(f"Invalid method: {name}")

    @classmethod
    def optimize(cls, method: str, instance_name: str, timeout: int) -> None:
        """
        Orchestrates the optimization process.
        This process executes the optimization method and save the best solution found in a maximum number of iterations.
        """
        has_improved, should_continue, count = False, True, 0
        warehouse = Reader(instance_name=instance_name).load_instance()

        try:
            while should_continue:
                debug(
                    f"BatchPicking | Start optimization | Method: {method.upper()} | Warehouse: {str(warehouse)}"
                )

                if not warehouse.is_valid:
                    raise ValueError("Invalid instance")

                solver = cls.dispatch(method, warehouse=warehouse, timeout=timeout)
                routes, time = solver.solve()
                solution = Solution(
                    instance_name=instance_name,
                    warehouse=warehouse,
                    batches=routes,
                )
                info(
                    f"BatchPicking | Finished in {time} seconds | Method: {method.upper()} | Solution: {str(solution)}"
                )
                has_improved = solution.save(time, method)
                count += 1

                should_continue = not has_improved and count < MAX_ITERATIONS

        except Exception as err:
            error(f"BatchPicking | Error optimization | {str(err)}")
