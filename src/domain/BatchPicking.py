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

        combining a set of orders into batches, and for each batch determining the sequence of storage locations to pickup all orders.

        The joint problem has been explored desde diferentes perspectivas.
        [1] Joint order batching and order picking in warehouse operations
            bin packing + TSP

        [1] The joint order batching and picker routing problem: Modelled and solved as a clustered vehicle routing problem
            Clustered VRP
        [2] Joint order batching and picker routing in single and multiple-cross-aisle warehouses using cluster-based tabu search algorithms
            bin packing + TSP

        Sin embargo, nuestro interes no es minimizar la cantidad de rutas, sino estricatemnte minimizar la distancia total recorrida, asumiendo suficiente capacidad de picking.
        The problem is defined as follows. The warehouse receives a set of items grouped in orders...




    The Batch-Picking problem, formally known in the literature as ...
        This problem is known as The joint order batching and picker routing problem (JOBPRP).

        The Batch-Picking problem, formally known in the literature as the Joint Order Batching and Picker Routing Problem (JOBPRP), is a well-known problem within warehouse environments.
        The problem consists of determining the best way to pick a set of orders in a warehouse, where each order is composed of a set of items. The objective is to minimize the total distance traveled by the pickers.
        ... Order integrality condition: all items of a customer order must be picked on the same tour
            Let the set of items be \(\mathcal{I}\) in a warehouse, where multiple items can share the same location.
            These items are grouped into a set of orders \(\mathcal{O}\), where each items belonging to an order is represented by \(I(o)\) for \(o \in \mathcal{O}\).

        ... Also capacity constraints
        ... multiple depots

    ...aims to solve the batching and picking problem as one single optimization problem.
    ...Oriented to the warehouse context.


        Two optimization problems
            how to combine orders
            into batches, and how to sequence the pick operations for each
            batch. Since large savings on the walking distance can be realised
            by solving the batching and routing problem simultaneously.


            Joint order batching and picker routing problem (JOBPRP)

            An adapted formulation
                for this problem as a CluVRP in the [document annexes]().

            TODO As an Annexe, show the VRP




    This problem is often approached as a Clustered Vehicle Routing, however, in the literature research I have not found a modeling approach as a pickup and delivery problem.

            the Hausdorff distance as batching criterion

            !! Adapted Hausdorff batching heuristic

            Defryn and Sörensen (2017) describe the following Hausdorff based constructive heuristic,

            !! Prior to the assignment, orders are sorted
            in decreasing order according to size (number of requested items
            (see ﬁg. 4b)), without further distinction between equally sized
            orders

            the number of available batches
            is known, which is the minimal number necessary to have all
            orders assigned to a batch (discussed in more detail in section
            5.1).

            TODO como un knapskack::: creamos una cantidad de mochiles, y por cada elemento llenamos en aquel que tenga el mejor hauf... distance (for which the last added order in the knspacks is closest)

            HOWEVER, even though this heuristic is better than the original (2017), this heuristic still fails in the look-ahead phase, where the .... (GREEDY)

        VRP with pick-up and delivery problem (VRPPD)



    In this work, we study The Joint Order Batching and Picker Routing Problem (JOBPRP) as a variant of the Pickup and Delivery Problem (PDP), and we propose a two-phase heuristic algorithm to solve large instances of the problem.






    ## Related work

    ## References
    [1] Aerts, B., Cornelissens, T., & Sörensen, K. (2021). The joint order batching and picker routing problem: Modelled and solved as a clustered vehicle routing problem. Computers & Operations Research, 129, 105168.


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
        Entry point for the optimization process.
        Execute the optimization method and save the best improving solution in MAX_ITERATIONS.

        Multi-start optimization method.
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
