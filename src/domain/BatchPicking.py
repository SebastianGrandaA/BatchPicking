from logging import info
from domain.joint import Joint
from domain.models.method import Method
from domain.sequential import Sequential
from services.io import Reader
from domain.models.solutions import Solution

def dispatch(method: str, **kwargs) -> Method:
    """Dispatch to the optimization method."""
    name = method.lower()

    if name == 'sequential':
        return Sequential(**kwargs)
    
    elif name == 'joint':
        return Joint(**kwargs)
    else:
        raise ValueError(f"Invalid method: {name}")

def optimize(method: str, instance_name: str, timeout: int) -> None:
    """Entry point for the optimization process."""
    warehouse = Reader(instance_name=instance_name).load_instance()
    if not warehouse.is_valid:
        raise ValueError("Invalid instance")
    
    info(f'BatchPicking | Start optimization | {str(warehouse)}')
    routes = dispatch(method, warehouse=warehouse, timeout=timeout).solve()
    solution = Solution(
        instance_name=instance_name,
        batches=routes,
    )
    info(f'BatchPicking | Finish optimization | {str(solution)}')
    solution.save()

def run(method: str, instance_name: str, timeout: int) -> None:
    """Interface for the optimization process."""
    optimize(method, instance_name, timeout)
