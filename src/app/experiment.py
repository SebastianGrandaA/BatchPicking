from logging import info

from services.benchmark import Benchmark


def run_experiment(method: str, instance_names: list[str], timeout: int) -> None:
    """
    # Experiment use case.

    Execute a set of instances to benchmark different methods.
    """
    benchmark = Benchmark(
        instance_names=instance_names,
        method=method,
        timeout=timeout,
    )
    benchmark.execute()
    info(
        f"Benchmark | Instances {instance_names} | Method {method} | Timeout {timeout}"
    )
