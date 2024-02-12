from domain.BatchPicking import run

def run_optimize(method: str, instance_name: str, timeout: int) -> None:
    run(method, instance_name, timeout)
