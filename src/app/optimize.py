from domain.BatchPicking import BatchPicking


def run_optimize(method: str, instance_name: str, timeout: int) -> None:
    """
    # Optimization use case.

    Execute the optimization process using the given method and instance name.
    """
    BatchPicking.optimize(method, instance_name, timeout)
