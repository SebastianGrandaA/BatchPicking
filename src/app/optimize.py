from domain.BatchPicking import optimize


def run_optimize(method: str, instance_name: str, timeout: int) -> None:
    """Interface to execute the optimization use case."""
    optimize(method, instance_name, timeout)
