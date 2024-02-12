from logging import info

from services.benchmark import Benchmark


def run_experiment(method: str, instance_names: list[str], timeout: int) -> None:
    """Interface to execute the experiment use case."""
    benchmark = Benchmark(
        instance_names=instance_names,
        method=method,
        timeout=timeout,
    )
    benchmark.execute()
    benchmark.analyze()
    info(
        f"Benchmark completed | Instances: {instance_names} | Method: {method} | Timeout: {timeout}"
    )
