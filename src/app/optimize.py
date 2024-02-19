from domain.BatchPicking import BatchPicking


def run_optimize(method: str, instance_name: str, timeout: int) -> None:
    """Interface to execute the optimization use case."""
    BatchPicking.optimize(method, instance_name, timeout)
