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

    The problem combines two optimization problems: the order batching problem and the picker routing problem.
    The batching problem consists of grouping orders to pick their items together if it leads to a reduced total distance traveled.
    The routing problem consists of determining the sequence of storage locations to pick all orders in a batch.
    """

    @staticmethod
    def dispatch(method: str, **kwargs) -> Method:
        """Dispatches to the optimization approach."""
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
