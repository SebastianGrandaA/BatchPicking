from domain.joint.vrp import VRP, VRPFormulation
from domain.models.method import Method, measure_consumption
from domain.models.solutions import Batch

JOINT_ROUTING_METHOD_DEFAULT = "VRP"
JOINT_ROUTING_METHODS = {
    "VRP": VRP,
    "VRPFormulation": VRPFormulation,
}


class Joint(Method):
    # """
    # ## Joint approach: Pickup and Delivery Problem (PDP)

    # Our joint approach consists of modeling the Batch-Picking problem as a variant of the Pickup and Delivery Problem (PDP).
    # More specifically, we model the problem as the Multi-depot Vehicle Routing Problem with Mixed Pickup and Delivery (MDVRPMPD).
    # According to [1], this problem is classified as the multi-vehicle one-to-many-to-one PDP with single demands and mixed solutions [1-M-1|P/D|m].

    # ### Problem statement

    # In this problem, we aim to minimize the total distance traveled by the pickers to satisfy all the orders in the warehouse.
    # Therefore, we model the items instead of the positions in such a way that items belonging to different orders can share the same position.
    # We are given a set of items \(I\) grouped by disjoint orders \(O\), where \(i(o)\) refers to the items of order \(o \in O\).
    # Let \(G = (V, A)\) be the directed graph representing the warehouse layout, where \(V\) is the set of nodes and \(A\) is the set of arcs.

    # The set of nodes \(V\) is composed of the item nodes \(I\), the artificial nodes \(J\), and the depot nodes \(D\). These sets are disjoint and satisfy \(V = I \cup J \cup D\).
    # The depot nodes \(D = \{0, n\}\), where \(0\) is the origin and \(n = |I| + 1\) is the destination of each route.
    # The artificial nodes \(J = \{|I| + 2, \dots, |I| + |O| + 1\}\) are the consolidation points for each order to ensure the order integrity condition.
    # Each node \(i \in V\) is associated with a demand and a volume, denoted by \(p_i \geq 0\) and \(v_i \geq 0\), respectively.
    # However, only artificial nodes have positive demand and volume, which are equal to 1 and to the total volume of the order, respectively.
    # The demand and the volume of the item and depot nodes are 0 \(p_i = v_i = 0\) for \(i \in I \cup D\).
    # Furthermore, the physical position in which each node is located is represented by \(pos(i) \forall i \in V\). It is possible that multiple nodes share the same position.

    # The arc set \(A = \{(i, j) \in V \times V : i \neq j\}\) are all feasible paths between the nodes, each with a distance \(d_{ij} \geq 0\) obtained from the positions of nodes \(pos(i)\) and \(pos(j)\) in the warehouse.
    # The distance to or from the depot nodes is 0 \(d_{ij} = d_{ji} = 0 \forall i \in D, j \in V\). The distance matrix is asymmetric and satisfies the triangle inequality.

    # The nodes are visited by a set of identical pickers \(k \in K\), each with an unitary and volume capacity.
    # The unitary capacity constraint refers to the maximum number of orders that a picker can transport in a route, whereas the volume capacity constraint, to the maximum volume.
    # The upper bounds are denoted by \(C_{unit}\) and \(C_{volume}\), respectively.
    # We assume that there are enough pickers to cover all the orders in the warehouse, which is given by \(\underline{B} > 0\), the minimal number of batches necessary to have all orders assigned to a batch with a \(\alpha\%\) of slack \ref{eq:min_batches}.

    # \underline{B} = \max\left(\frac{\sum_{i \in I} v_i}{C_{volume}}, \frac{|O|}{C_{unit}}\right) \times (1 + \alpha)\) \label{eq:min_batches}

    # The artificial nodes act as consolidation points for each order, ensuring that all the items of an order are picked-up in the same route.
    # This is achieved by forcing the pickers to visit the item nodes and delivering them to the artificial nodes.
    # Let \(d(i) \in J\) be the delivery - artificial - node of the item node \(i\).

    # ### Joint formulation

    # We present a three-index commodity-flow formulation that involves two types of variables: selection and flow variables.
    # Let \(x_{ijk} \in \{0, 1\}\) be the binary decision variables indicating whether the arc \((i, j) \in A\) is selected by the picker \(k \in K\).
    # The commodity-flow variables \(l_{ik} \geq 0\) indicate the cumulative load of picker \(k \in K\) after visiting node \(i \in V\). This load represents the number of orders picked up (i.e. quantity artificial nodes).
    # The complete model is available at the [report](https://www.overleaf.com/read/xfgcnzwccnqj#8fe7b9).
    # The MDVRPMPD is NP-Hard because it is an special case of the Capacitated Vehicle Routing Problem (CVRP) and the Pickup and Delivery Problem (PDP).

    # ### Implementation details

    # Two options are available to solve the joint problem: an heuristic method and a mathematical programming formulation.
    # The MIP formulation is implemented using the [Pyomo](http://www.pyomo.org/) library for validation purposes.
    # However, due to the complexity of this problem, a generic local-search algorithm is implemented using the [OR-Tools](https://developers.google.com/optimization) library.
    # This tool implements a Tabu Search algorithm to improve the initial solution, which is obtained from the S-shaped path of serving the orders individually (`supportList.txt`).
    # The capacity and precedence constraints are handled by the constraint programming solver `pywrapcp` provided by the package.
    # For large instances, a timeout is imposed to avoid long computations. Therefore, this solution can also be used as an initial solution in the customized local-search heuristic (see `LocalSearch` class).

    # Further improvements can be achieved by using a more specialized solver, such as the [VRP solver](https://vrpsolver.math.u-bordeaux.fr) developed at [INRIA](https://www.inria.fr/fr).
    # However, currently the [Python](https://github.com/inria-UFF/VRPSolverEasy) implementation does not cover the pick-up and delivery feature.
    # A workaround is to implement the solver in [Julia](https://github.com/inria-UFF/BaPCodVRPSolver.jl) and integrated it with python using [PyJulia](https://github.com/JuliaPy/pyjulia?tab=readme-ov-file).

    # ## References
    # [1] Berbeglia, G., Cordeau, J. F., Gribkovskaia, I., & Laporte, G. (2007). Static pickup and delivery problems: a classification scheme and survey. Top, 15, 1-31.
    # [2] Desaulniers, G., Desrosiers, J., Erdmann, A., Solomon, M. M., & Soumis, F. (2002). VRP with Pickup and Delivery. The vehicle routing problem, 9, 225-242.
    # [3] Battarra, M., Cordeau, J. F., & Iori, M. (2014). Chapter 6: pickup-and-delivery problems for goods transportation. In Vehicle Routing: Problems, Methods, and Applications, Second Edition (pp. 161-191). Society for Industrial and Applied Mathematics.
    # [4] Parragh, S. N., Doerner, K. F., & Hartl, R. F. (2008). A survey on pickup and delivery problems: Part I: Transportation between customers and depot. Journal für Betriebswirtschaft, 58, 21-51.
    # """
    @measure_consumption
    def solve(self, **kwargs) -> list[Batch]:
        routing_method = kwargs.get("routing_method", JOINT_ROUTING_METHOD_DEFAULT)

        if routing_method not in JOINT_ROUTING_METHODS:
            raise ValueError(f"Unknown routing method {routing_method}")

        vrp_model = JOINT_ROUTING_METHODS[routing_method](**self.__dict__)

        return vrp_model.solve()
