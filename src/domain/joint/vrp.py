from logging import debug, error, info
from typing import Any

import numpy as np
import pyomo.environ as pyo
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from domain.models.instances import Item, Order
from domain.models.method import Callbacks
from domain.models.routing import Routing
from domain.models.solutions import Batch, Metrics, Route


class VRP(Routing):
    """OR-Tools implementation of the Vehicle Routing Problem (VRP)."""

    manager: Any = None
    routing: Any = None
    parameters: Any = None
    callbacks: Callbacks = Callbacks()
    is_warehouse_complete: bool = True

    @property
    def demands(self) -> list[int]:
        """
        The demands of the nodes are used to set the unitary capacity constraints of a group of nodes (orders).
        Dummy nodes have a demand of 1, whereas the item nodes have a demand of 0.
        """
        return [int(node.is_dummy) for node in self.sorted_nodes]

    @property
    def volumes(self) -> list[int]:
        """
        The volumes of the nodes are used to set the volume capacity constraints of a group of nodes (orders).
        Dummy nodes have a volume of the whole order, whereas the item nodes have a volume of 0.
        """
        return [
            self.get_order(node).volume if node.is_dummy else 0
            for node in self.sorted_nodes
        ]

    @property
    def groups(self) -> list[list[int]]:
        """
        Group nodes by orders, where the last item in each group is the dummy node. The depots are single-item groups.
        For example, if there are 4 orders and 10 items, [0, 11] represent the depots, and [12, 13, 14, 15] represent the dummy nodes,
        then: groups = [[0], [1, 2, 3, 12], [4, 5, 13], [6, 7, 14], [8, 9, 15], [11]]. The depots are excluded from the groups.
        """
        groups, group = [], []

        for idx, node in zip(self.sorted_indices, self.sorted_nodes):
            group.append(idx)

            if not node.is_pickup:
                groups.append(group)
                group = []

        return groups

    @property
    def status(self) -> str:
        status = self.routing.status()
        map = {
            0: "Not solved",
            1: "Optimal",
            2: "Feasible",
            3: "No solution found",
            4: "Timeout",
            5: "Invalid",
            6: "Infeasible",
        }

        return map.get(status, "Unknown")

    def build_matrix(self) -> np.ndarray:
        """Return the distance matrix between all nodes."""
        return np.array(
            [
                [
                    self.warehouse.distance(i, j)
                    if (not i.is_dummy and not j.is_dummy)
                    else 0
                    for j in self.graph.values()
                ]
                for i in self.graph.values()
            ]
        )

    def set_parameters(self) -> None:
        """
        Set the solver parameters.
        The construction and local search parameters have been selected based on experimentation.
        [Reference](https://developers.google.com/optimization/routing/routing_options)
        """
        # TODO probar con diferentes parameteros y a pesar de no implementar el algoritmo, explicarlo!
        self.parameters = pywrapcp.DefaultRoutingSearchParameters()
        self.parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            # SWEEP, SAVINGS, PARALLEL_CHEAPEST_INSERTION, etc..
        )
        self.parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
            # GUIDED_LOCAL_SEARCH
        )
        self.parameters.time_limit.FromSeconds(self.timeout)
        self.parameters.log_search = self.verbose
        self.parameters.solution_limit = 500
        self.routing.CloseModelWithParameters(self.parameters)

    def get_initial_solution(self) -> Any:
        """
        Retrieve the first solution, either from the file (previous) or the provided S-shaped path.
        List of items to be picked up in each batch. (S-shaped or previous)
        Retrieve the S-shaped path to visit the items of each order individually.
        This solution is encoded by the node indices of the positions belong to.

        [Reference](https://developers.google.com/optimization/routing/routing_tasks#setting_initial_routes_for_a_search).
        """
        if self.warehouse.current_solution:
            solution = self.warehouse.current_solution
        else:
            solution = self.warehouse.base_solution

        grouped_nodes = [
            [self.get_node_idx(item) for item in batch] for batch in solution
        ]

        initial_solution = self.routing.ReadAssignmentFromRoutes(grouped_nodes, True)
        debug(f"VRP | Initial solution | Node indices {grouped_nodes}")

        return initial_solution

    def update_current_solution(self, batches: list[Batch]) -> None:
        """
        Update the current solution with the new solution.
        The current solution does not include the depots and the dummy nodes.
        """
        self.warehouse.current_solution = [
            [item for item in batch.route.sequence if item.is_pickup]
            for batch in batches
        ]
        debug("Initial solution updated.")

    def build_solution(self, solution: Any) -> list[Batch]:
        """
        Build a list of batches, each one with a route and the orders to be picked.
        Note that the position id is not unique because there is one node per item.
        """
        batches = []
        demands, volumes = self.demands, self.volumes
        is_unique = lambda item, sequence: item not in sequence and not item.is_dummy

        for vehicle_id in range(self.nb_vehicles):
            index = self.routing.Start(vehicle_id)
            sequence = []
            distance, unit_load, volume_load = 0, 0, 0

            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                item = self.graph[node_index]
                previous_index = index
                index = solution.Value(self.routing.NextVar(index))

                if is_unique(item, sequence):
                    sequence.append(item)
                    unit_load += demands[node_index]
                    volume_load += volumes[node_index]
                    distance += self.routing.GetArcCostForVehicle(
                        previous_index, index, vehicle_id
                    )

            item = self.graph[self.manager.IndexToNode(index)]
            if is_unique(item, sequence):
                sequence.append(item)

            route = Route(sequence=sequence)
            orders = list(
                set(self.get_order(node) for node in route.sequence if node.is_pickup)
            )
            if not orders:
                continue

            metrics = Metrics(distance=distance, units=unit_load, volume=volume_load)
            batches.append(Batch(orders=orders, route=route, metrics=metrics))

        self.update_current_solution(batches)

        return batches

    # Objective functions
    # -------------------

    def minimize_total_distance(self) -> None:
        """Minimize the total distance traveled by the pickers."""
        distances = self.build_matrix()

        def distance_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)

            return distances[from_node][to_node]

        self.callbacks.distance = self.routing.RegisterTransitCallback(
            distance_callback
        )
        self.routing.SetArcCostEvaluatorOfAllVehicles(self.callbacks.distance)

        debug(f"Minimize total distance | Distance sample: {distances[0]}")

    # Constraints
    # -----------

    def unit_capacity_constraints(self) -> None:
        """
        Set the capacity constraints for the vehicles.
        The capacity is the quantity of orders the vehicle can pick in a route.
        A dummy node for each order is introduced with a demand of 1, whereas the item nodes have a demand of 0.

        [Reference](https://developers.google.com/optimization/routing/cvrp).
        """

        def demand_callback(from_index):
            """Returns the demand of the node."""
            from_node = self.manager.IndexToNode(from_index)

            return self.demands[from_node]

        self.callbacks.demand = self.routing.RegisterUnaryTransitCallback(
            demand_callback
        )
        self.routing.AddDimensionWithVehicleCapacity(
            self.callbacks.demand,
            0,  # no capacity slack
            [self.warehouse.vehicle.max_nb_orders]
            * self.nb_vehicles,  # vehicle maximum capacities
            True,  # start cumul to zero
            "UnitCapacity",
        )

        debug(f"Unit capacity constraints | Demands: {self.demands}")

    def volume_capacity_constraints(self) -> None:
        """
        Set the volume capacity constraints for the vehicles.
        Similar to the unit capacity constraints, but the volume of an order is the total volume of all the items.
        """

        def volume_callback(from_index):
            """Returns the volume of the node."""
            from_node = self.manager.IndexToNode(from_index)

            return self.volumes[from_node]

        self.callbacks.volume = self.routing.RegisterUnaryTransitCallback(
            volume_callback
        )
        self.routing.AddDimensionWithVehicleCapacity(
            self.callbacks.volume,
            0,  # no capacity slack
            [self.warehouse.vehicle.max_volume]
            * self.nb_vehicles,  # vehicle maximum capacities
            True,  # start cumul to zero
            "VolumeCapacity",
        )

        debug(f"Volume capacity constraints | Volumes: {self.volumes}")

    def pickup_delivery_constraints(self) -> None:
        """
        Set the pick-up and delivery constraints for the vehicles.
        All the positions (items) for an order must be visited in the same batch route.
        Therefore, we consider that the items are pick-up positions and the dummy nodes are delivery positions.

        [Reference](https://developers.google.com/optimization/routing/pickup_delivery).
        """
        grouped_items = self.groups[1:-1]  # exclude the depots
        assert len(grouped_items) == self.nb_vehicles

        for group in grouped_items:
            delivery_idx = group[-1]
            pickup_indices = group[:-1]

            for pickup_idx in pickup_indices:
                pickup_idx = self.manager.NodeToIndex(pickup_idx)
                delivery_idx = self.manager.NodeToIndex(delivery_idx)

                if pickup_idx == delivery_idx or pickup_idx < 0 or delivery_idx < 0:
                    continue

                self.routing.AddPickupAndDelivery(pickup_idx, delivery_idx)
                self.routing.solver().Add(
                    self.routing.VehicleVar(pickup_idx)
                    == self.routing.VehicleVar(delivery_idx)
                )

        debug(f"Pickup and delivery constraints | Items: {grouped_items}")

    # Model
    # -----

    def build_model(self) -> None:
        """
        Build the model for the Capacitated VRP with pick-up and delivery.
        Multiple depots are considered to model the start and end of the routes.

        [Reference](https://developers.google.com/optimization/routing/routing_tasks#setting-start-and-end-locations-for-routes)
        """
        self.build_graph()
        self.manager = pywrapcp.RoutingIndexManager(
            len(self.graph),  # number of nodes
            self.nb_vehicles,  # number of vehicles
            [self.start_node_idx] * self.nb_vehicles,  # start nodes
            [self.end_node_idx] * self.nb_vehicles,  # end nodes
        )
        self.routing = pywrapcp.RoutingModel(self.manager)

        self.minimize_total_distance()

        if self.is_warehouse_complete:
            self.unit_capacity_constraints()
            self.volume_capacity_constraints()
            self.pickup_delivery_constraints()

    def route(self) -> list[Batch]:
        """
        Main method to solve the VRP.
        Returns a list of batches, each one with a route and the orders to be picked.
        """
        self.build_model()
        self.set_parameters()
        info(
            f"VRP | Warehouse {self.warehouse.name} | Vehicles {self.nb_vehicles} | Nodes {len(self.graph)}"
        )

        if self.is_warehouse_complete:
            initial_solution = self.get_initial_solution()
            solution = self.routing.SolveFromAssignmentWithParameters(
                initial_solution, self.parameters
            )

        else:
            solution = self.routing.SolveWithParameters(self.parameters)

        if solution and self.is_valid:
            info(
                f"VRP | Warehouse {self.warehouse.name} | Solution obtained | Status: {self.status}"
            )

            return self.build_solution(solution)

        else:
            error("No solution found")

    def solve(self, **kwargs) -> list[Batch]:
        """
        Solve an instance of the VRP.
        """
        return self.route()

    def route_batch(self, batch: Batch) -> Batch:
        """
        Route a single batch.
        """
        self.warehouse.orders = batch.orders

        routes = self.route()
        assert len(routes) == 1

        return routes[0]


class VRPFormulation(Routing):
    """Pyomo implementation of the Vehicle Routing Problem (VRP)."""

    @property
    def demands(self) -> dict[int, int]:
        """
        The demands of the nodes are used to set the unitary capacity constraints of a group of nodes (orders).
        Dummy nodes have a demand of 1, whereas the item nodes have a demand of 0.
        """
        return {idx: int(node.is_dummy) for idx, node in self.node_items}

    @property
    def volumes(self) -> dict[int, int]:
        """
        The volumes of the nodes are used to set the volume capacity constraints of a group of nodes (orders).
        Dummy nodes have a volume of the whole order, whereas the item nodes have a volume of 0.
        """
        return {
            idx: self.get_order(node).volume if node.is_dummy else 0
            for idx, node in self.node_items
        }

    @property
    def pickups(self) -> list[int]:
        """Return the indices of the pick-up nodes."""
        return [i for i, node in self.node_items if node.is_pickup]

    @property
    def deliveries(self) -> list[int]:
        """Return the indices of the delivery nodes."""
        return [i for i, node in self.node_items if node.is_dummy]

    def is_valid(self, result) -> bool:
        if result.solver.termination_condition == pyo.TerminationCondition.infeasible:
            error(f"Problem | Infeasible model")
            return False

        return (
            result.solver.status == pyo.SolverStatus.ok
            and result.solver.termination_condition == pyo.TerminationCondition.optimal
        )

    def get_delivery_node_idx(self, item: Item) -> int:
        """Get the corresponding artificial for a given pickup item."""
        order = self.get_order(item)
        last_item = order.pickups[-1]
        artificial = self.graph[self.get_node_idx(last_item) + 1]
        assert (
            artificial.is_dummy
        ), f"Artificial node {artificial} is not a delivery node"

        return self.get_node_idx(artificial)

    def minimize_total_distance(self, model: pyo.ConcreteModel) -> float:
        """Return the total distance traveled by the pickers."""
        # TODO probar quitando los dummy
        # warning(f'x: {[i for i in model.x.keys()]}')
        # warning(f'x: {[model.x[i, j, k] for (i, n_i) in self.node_items for (j, n_j) in self.node_items for k in range(self.nb_vehicles)]}')
        # warning(f'distances {[self.warehouse.distance(node_i, node_j) for (i, node_i) in self.node_items for (j, node_j) in self.node_items]}')
        return sum(
            self.warehouse.distance(node_i, node_j) * model.x[i, j, k]
            if (i != j and not node_i.is_dummy and not node_j.is_dummy)
            else 0
            for (i, node_i) in self.node_items
            for (j, node_j) in self.node_items
            for k in range(self.nb_vehicles)
        )

    def service_constraints(self, model: pyo.ConcreteModel, i: int) -> float:
        """
        Return the service constraints for the pick-up nodes.

        \sum_{k \in K} \sum_{j \in V \setminus \{0\}} x_{ijk} = 1, \quad \forall i \in I, \label{eq:service_constraints_vrp}
        """
        # NEW: \[ \sum_{k \in K} \sum_{j \in V} x_{ijk} = 1, \quad \forall i \in I \]
        if not self.graph[i].is_pickup:
            return pyo.Constraint.Skip

        return (
            sum(
                model.x[i, j, k] for j in self.node_ids for k in range(self.nb_vehicles)
            )
            == 1
        )
        # return sum(model.x[i, j, k] for j in self.graph.keys() if j != self.start_node_idx for k in range(self.nb_vehicles)) == 1

    def pickup_delivery_constraints(
        self, model: pyo.ConcreteModel, i: int, k: int
    ) -> float:
        """
        Return the pick-up and delivery constraints for the pick-up nodes.

        \sum_{j \in V} x_{ijk} = \sum_{j \in V} x_{j,d(i),k}, \quad \forall k \in K, \forall i \in I, \label{eq:pickup_delivery}
        TODO ojo prece que hacer pickup y delivery consectivo, y por lo tanto, todos los items de una orden estan jjuntos y luego va l sgtw order (no es mixed)?
        """
        # \[ \sum_{j \in V} x_{ijk} = \sum_{j \in V} x_{jd(i)k}, \quad \forall i \in J, \forall k \in K \]
        item = self.graph[i]

        if not item.is_pickup:
            return pyo.Constraint.Skip

        delivery = self.get_delivery_node_idx(item)

        return sum(model.x[i, j, k] for j in self.node_ids) == sum(
            model.x[j, delivery, k] for j in self.node_ids
        )
        # return sum(model.x[i, j, k] for j in self.graph.keys()) == sum(model.x[j, delivery, k] for j in self.graph.keys())

    def start_end_route_constraints(
        self, model: pyo.ConcreteModel, i: int, k: int
    ) -> float:
        """
        Return the start and end route constraints for the pick-up nodes.

        \sum_{j \in I \cup \{n\}} x_{0jk} = 1, \quad \forall k \in K, \label{eq:start_end_route}
        """
        # NEW: \[ \sum_{j \in V \setminus D} x_{djk} = 1, \quad \forall d \in D, \forall k \in K \]
        if not self.graph[i].is_depot:
            return pyo.Constraint.Skip

        return (
            sum(model.x[i, j, k] for j, item in self.node_items if not item.is_depot)
            == 1
        )

        # return sum(model.x[self.start_node_idx, j, k] for j, item in self.node_items if (item.is_pickup or j == self.end_node_idx)) == 1

    # == 1 TODO no funciona con == 1???

    def finish_at_depot_constraints(
        self, model: pyo.ConcreteModel, i: int, k: int
    ) -> float:
        # \[ \sum_{i \in V \setminus D} x_{idk} = 1, \quad \forall d \in D, \forall k \in K \]
        if not self.graph[i].is_depot:
            return pyo.Constraint.Skip

        return (
            sum(model.x[j, i, k] for j, item in self.node_items if not item.is_depot)
            == 1
        )

    def flow_conservation_constraints(
        self, model: pyo.ConcreteModel, j: int, k: int
    ) -> float:
        """
        Return the flow conservation constraints for the pick-up nodes.

        \sum_{i \in V \setminus \{n\}} x_{ijk} = \sum_{i \in V \setminus \{0\}} x_{jik}, \quad \forall k \in K, \forall j \in V, \label{eq:flow_conservation}
        """
        # NEW : \[ \sum_{i \in V} x_{ijk} = \sum_{i \in V} x_{jik}, \quad \forall j \in V, \forall k \in K \]
        return sum(model.x[i, j, k] for i in self.node_ids) == sum(
            model.x[j, i, k] for i in self.node_ids
        )

        # return sum(model.x[i, j, k] for i in self.graph.keys() if i != self.end_node_idx) == sum(model.x[j, i, k] for i in self.graph.keys() if i != self.start_node_idx)

    # def delivery_end_route_constraints(self, model: pyo.ConcreteModel, k: int) -> float:
    # TODO este es el de arriba
    # """
    # Return the delivery end route constraints for the pick-up nodes.

    # \sum_{i \in J \cup \{0\}} x_{ink} = 1, \quad \forall k \in K, \label{eq:delivery_end_route}
    # """

    # return sum(model.x[i, self.end_node_idx, k] for i, item in self.node_items if (item.is_dummy or i == self.start_node_idx)) == 1
    # TODO probar item.is_pickup instead

    def flow_definition_constraints(
        self, model: pyo.ConcreteModel, i: int, j: int, k: int
    ) -> float:
        """
        Return the flow definition constraints for the pick-up nodes.

        x_{ijk} \times (l_{ik} + p_{j} - l_{jk}) = 0, \quad \forall k \in K, \forall (i, j) \in A, \label{eq:flow_definition}

        TODO linearizar:
            u_ik + ]sum_{j \in V} p_j x_{ijk} - u_jk \leq C_{unit} - p_j, \quad \forall k \in K, \forall (i, j) \in A
        """
        # NEW: \[ l_{ik} + \sum_{j \in V} p_{j} x_{ijk} \leq l_{jk} + C_{unit}(1 - x_{ijk}), \quad \forall (i, j) \in A, \forall k \in K \]
        return model.u[i, k] + sum(
            model.x[i, j, k] * self.demands[j] for j in self.node_ids
        ) <= model.u[j, k] + self.warehouse.vehicle.max_nb_orders * (
            1 - model.x[i, j, k]
        )

        # return model.x[i, j, k] * (model.u[i, k] + self.demands[j] - model.u[j, k]) == 0

    # TODO probar nueva constraint

    # def unit_capacity_constraints(self, model: pyo.ConcreteModel, k: int) -> float:
    #     """
    #     Return the unit capacity constraints for the pick-up nodes.

    #     \sum_{i \in V} p_{i} l_{ik} \leq C_{unit}, \quad \forall k \in K, \label{eq:unit_capacity}
    #     """
    #     return sum(self.demands[i] * model.u[i, k] for i in self.graph.keys()) <= self.warehouse.vehicle.max_nb_orders
    #     # return sum(self.demands[i-1] * model.l[i, k] for i in model.V) <= self.warehouse.vehicle.max_nb_orders

    def volume_capacity_constraints(self, model: pyo.ConcreteModel, k: int) -> float:
        """
        Return the volume capacity constraints for the pick-up nodes.

        \sum_{i \in V} v_{i} l_{ik} \leq C_{volume}, \quad \forall k \in K, \label{eq:volume_capacity}
        """
        # # NEW: \[ \sum_{i \in I} v_{i} x_{ijk} \leq C_{volume}, \quad \forall k \in K \]

        return (
            sum(self.volumes[i] * model.u[i, k] for i in self.node_ids)
            <= self.warehouse.vehicle.max_volume
        )
        # return sum(self.volumes[i] * model.u[i, k] for i in self.graph.keys()) <= self.warehouse.vehicle.max_volume
        # return sum(self.volumes[i-1] * model.l[i, k] for i in model.V) <= self.warehouse.vehicle.max_volume

    def initial_load_constraints(
        self, model: pyo.ConcreteModel, i: int, k: int
    ) -> float:
        """
        Return the initial load constraints for the pick-up nodes.

        l_{0k} = 0, \quad \forall k \in K, \label{eq:initial_load}
        """
        # NEW: \[ l_{dk} = 0, \quad \forall d \in D, \forall k \in K \]
        if not self.graph[i].is_depot:
            return pyo.Constraint.Skip

        return model.u[i, k] == 0

        # return model.l[self.start_node_idx, k] == 0

    def build_model(self) -> pyo.ConcreteModel:
        """
        Build the mathematical model for the VRP using pyomo.
        """
        self.build_graph()
        model = pyo.ConcreteModel()

        # Variables
        nodes = self.sorted_indices
        pickers = list(range(self.nb_vehicles))
        model.x = pyo.Var(nodes, nodes, pickers, within=pyo.Binary)
        model.u = pyo.Var(nodes, pickers, within=pyo.NonNegativeReals)

        # Objective function
        model.objective = pyo.Objective(
            rule=self.minimize_total_distance, sense=pyo.minimize
        )

        # Constraints
        model.service_constraints = pyo.Constraint(nodes, rule=self.service_constraints)
        model.pickup_delivery_constraints = pyo.Constraint(
            nodes, pickers, rule=self.pickup_delivery_constraints
        )
        model.start_end_route_constraints = pyo.Constraint(
            nodes, pickers, rule=self.start_end_route_constraints
        )
        model.finish_at_depot_constraints = pyo.Constraint(
            nodes, pickers, rule=self.finish_at_depot_constraints
        )
        model.flow_conservation_constraints = pyo.Constraint(
            nodes, pickers, rule=self.flow_conservation_constraints
        )
        model.flow_definition_constraints = pyo.Constraint(
            nodes, nodes, pickers, rule=self.flow_definition_constraints
        )
        model.volume_capacity_constraints = pyo.Constraint(
            pickers, rule=self.volume_capacity_constraints
        )
        model.initial_load_constraints = pyo.Constraint(
            nodes, pickers, rule=self.initial_load_constraints
        )

        return model

    def build_solution(self, model: pyo.ConcreteModel) -> list[Batch]:
        """
        Build a list of batches, each one with a route and the orders to be picked.
        """
        batches = []
        demands, volumes = self.demands, self.volumes

        for k in range(self.nb_vehicles):
            sequence = [self.graph[self.start_node_idx]]
            distance, unit_load, volume_load = 0, 0, 0

            for i, item_i in self.node_items:
                for j, item_j in self.node_items:
                    if model.x[i, j, k].value > 0.5:
                        if item_j.is_pickup:
                            sequence.append(item_j)
                            unit_load += demands[j]
                            volume_load += volumes[j]
                            distance += self.warehouse.distance(item_i, item_j)

            sequence.append(self.graph[self.end_node_idx])

            route = Route(sequence=sequence)
            orders = list(
                set(self.get_order(node) for node in route.sequence if node.is_pickup)
            )
            if not orders:
                continue

            metrics = Metrics(distance=distance, units=unit_load, volume=volume_load)
            batches.append(Batch(orders=orders, route=route, metrics=metrics))

        return batches

    def solve(self) -> list[Batch]:
        """
        Main method to solve the VRP.
        Returns a list of batches, each one with a route and the orders to be picked.
        """
        model = self.build_model()
        solver = pyo.SolverFactory("gurobi")
        result = solver.solve(
            model, tee=self.verbose, options={"TimeLimit": self.timeout}
        )

        if self.is_valid(result):
            return self.build_solution(model)

        else:
            error("No solution found")
