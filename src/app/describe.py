from logging import info

from services.benchmark import Benchmark


def run_describe() -> None:
    """Interface to execute the describe use case."""
    benchmark = Benchmark(instance_names=[], method="", timeout=0)
    benchmark.analyze()
    info(f"Describe completed")
