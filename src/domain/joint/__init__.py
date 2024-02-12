from domain.joint.vrp import VRP
from domain.models.method import Method
from domain.models.solutions import Batch


class Joint(Method):
    """
    Joint approach: Capacitated Vehicle Routing Problem (VRP) with pickup and delivery.
    Furthermore, there are multiple depots, one for start and one for end.

    Return a list of batches, each one with a route.
    """
    def solve(self) -> list[Batch]:
        model = VRP(**self.__dict__)

        return model.route()
