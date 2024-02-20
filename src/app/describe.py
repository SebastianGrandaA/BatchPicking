from logging import info

from services.benchmark import Benchmark


def run_describe() -> None:
    """
    # Describe use case.

    Provide an analysis of the results.
    """
    benchmark = Benchmark(instance_names=[], method="", timeout=0)
    benchmark.analyze()
    info(f"Describe completed")
