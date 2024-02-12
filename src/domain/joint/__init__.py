from domain.joint.vrp import VRP
from domain.models.method import Method, measure_time
from domain.models.solutions import Batch


class Joint(Method):
    """
    Joint approach consists of solving the Capacitated Vehicle Routing Problem (VRP) with pickup and delivery.
    Multiple depots are considered, one for the start and the end of each route.
    Return a list of batches, each one with a route.
    """

    @measure_time
    def solve(self) -> list[Batch]:
        model = VRP(**self.__dict__)

        return model.route()
