from logging import debug, error, info

from domain.joint import Joint
from domain.models.method import Method
from domain.models.solutions import Solution
from domain.sequential import Sequential
from services.io import Reader


def dispatch(method: str, **kwargs) -> Method:
    """Dispatch to the optimization method."""
    name = method.lower()

    if name == "sequential":
        return Sequential(**kwargs)

    elif name == "joint":
        return Joint(**kwargs)
    else:
        raise ValueError(f"Invalid method: {name}")


def optimize(method: str, instance_name: str, timeout: int) -> None:
    """Entry point for the optimization process."""
    warehouse = Reader(instance_name=instance_name).load_instance()

    try:
        if not warehouse.is_valid:
            raise ValueError("Invalid instance")

        debug(f"BatchPicking | Start optimization | {str(warehouse)}")
        solver = dispatch(method, warehouse=warehouse, timeout=timeout)
        routes, time = solver.solve()
        solution = Solution(
            instance_name=instance_name,
            batches=routes,
        )
        info(f"BatchPicking | Finished in {time} seconds | {str(solution)}")
        solution.save(time)

    except Exception as err:
        error(f"BatchPicking | Error optimization | {str(err)}")
