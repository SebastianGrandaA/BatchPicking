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
    # Joint approach: Pickup and Delivery Problem (PDP)

    Our joint approach consists of modeling the Batch-Picking problem as the Multi-depot Vehicle Routing Problem with Mixed Pickup and Delivery (MDVRPMPD).
    Two options are available to solve the joint problem: an heuristic method and a mathematical programming formulation.
    The problem statement and the solution approaches can be found at the [report](https://www.overleaf.com/read/xfgcnzwccnqj#8fe7b9).
    """

    @measure_consumption
    def solve(self, **kwargs) -> list[Batch]:
        routing_method = kwargs.get("routing_method", JOINT_ROUTING_METHOD_DEFAULT)

        if routing_method not in JOINT_ROUTING_METHODS:
            raise ValueError(f"Unknown routing method {routing_method}")

        vrp_model = JOINT_ROUTING_METHODS[routing_method](**self.__dict__)

        return vrp_model.solve()
