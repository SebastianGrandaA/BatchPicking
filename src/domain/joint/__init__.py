from domain.joint.vrp import VRP, VRPFormulation
from domain.models.method import Method, measure_consumption
from domain.models.solutions import Batch

JOINT_ROUTING_METHOD_DEFAULT = "VRP"
JOINT_ROUTING_METHODS = {
    "VRP": VRP,
    "VRPFormulation": VRPFormulation,
}


class Joint(Method):
    """
    Joint approach consists of solving the Capacitated Vehicle Routing Problem (VRP) with pickup and delivery.
    Multiple depots are considered, one for the start and the end of each route.
    Return a list of batches, each one with a route.
    """

    @measure_consumption
    def solve(self, **kwargs) -> list[Batch]:
        routing_method = kwargs.get("routing_method", JOINT_ROUTING_METHOD_DEFAULT)

        if routing_method not in JOINT_ROUTING_METHODS:
            raise ValueError(f"Unknown routing method {routing_method}")

        vrp_model = JOINT_ROUTING_METHODS[routing_method](**self.__dict__)

        return vrp_model.solve()
