from logging import INFO, basicConfig, info

from domain.sequential.clustering import Clustering
from domain.sequential.routing import Routing
from services.instances import Reader
from services.solutions import Solution


def optimize():
    instance_name = 'toy_instance'
    warehouse = Reader(instance_name=instance_name).load_instance()
    info(f'BatchPicking | Warehouse: {str(warehouse)}')

    batches = Clustering(warehouse=warehouse).solve()
    info(f'{[str(batch) for batch in batches]}')

    routes = Routing(warehouse=warehouse).solve(batches=batches)
    info(f'{[str(route) for route in routes]}')

    solution = Solution(
        instance_name=instance_name,
        warehouse=warehouse,
        batches=batches,
        routes=routes
    )
    solution.save()

def initialize():
    basicConfig(level=INFO)

if __name__ == "__main__":
    initialize()
    optimize()