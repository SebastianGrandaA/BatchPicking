from logging import debug, error, info
from typing import Any
from domain.models.method import Callbacks
from domain.models.routing import Routing
from domain.models.instances import Item, Order
from domain.models.solutions import Batch, Metrics, Route
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import numpy as np

class VRP(Routing):
    """
    # Capacitated Vehicle Routing Problem with pick-up and deliveries.

    Originally, the problem is defined as a Clustered VRP, in which the items are grouped by orders.
    However, the problem can be also modeled as an Pick-up and Delivery VRP by introducing dummy nodes to serve as delivery points all the items of a given order.

    The graph is composed of the depots (start and end), the pick-up items, and the dummy nodes for each order.
    Therefore, the positions are not unique because multiple items from different orders might share the same position.

    ## Constraints

    ### Capacity
        Pickers have a capacity constraint in terms of the quantity and the volume of the orders they can pick in a route.
        To model the unitary capacity constraint, we set the demand of the dummy nodes to 1, whereas the demand of the item nodes is 0.
        ... formula ...

        Similarly, the volume capacity constraint is set by the volume of the whole order for the dummy nodes, and 0 for the item nodes.

    ### Pick-up and delivery
        To model the cluster of items by orders, we state that all the items of an order must be picked-up during the route and delivered to the dummy node.
        ... formula ...

    ### Multiple depots
        The problem is modeled with multiple depots to represent the start and end of the route.
        ... formula ...

    ## Objective
    The objective is to minimize the total distance traveled by the pickers.
    ... formula ...

    ## Assumptions
    There is enough pickers to cover all the orders in the warehouse (i.e. the number of pickers is equal to the number of orders). The pickers without routes are ignored.

    ## Additional notes
    Aiming to improve the efficiency of the local search procedure, the S-shaped path (a.k.a. base solution) for each order is used as the initial solution.
    This implementation is a proof of concept and does not focus on performance.
    """
    manager: Any = None # TODO dar un tipo
    routing: Any = None
    parameters: Any = None
    callbacks: Callbacks = Callbacks()
    node_to_order: dict[int, int] = {}
    
    @property
    def sorted_indices(self) -> list[int]:
        return sorted(self.graph.keys())
    
    @property
    def sorted_nodes(self) -> list[Item]:
        return [self.graph[idx] for idx in self.sorted_indices]

    @property
    def nb_vehicles(self) -> int:
        """Number of available vehicles - or pickers - equals the number of orders. Assumes enough resources."""
        return len(self.warehouse.orders)

    @property
    def demands(self) -> list[int]:
        """
        The demands of the nodes are used to set the unitary capacity constraints of a group of nodes (orders).
        Dummy nodes have a demand of 1, whereas the item nodes have a demand of 0.
        """
        return [
            int(node.is_dummy)
            for node in self.sorted_nodes
        ]
    
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
    def start_node_idx(self) -> int:
        """Start node is the first node in the graph."""
        return 0
    
    @property
    def end_node_idx(self) -> int:
        """End node is the last node in the graph."""
        return len(self.graph) - 1
    
    @property
    def start_node_id(self) -> int:
        return self.graph[self.start_node_idx].position_id
    
    @property
    def end_node_id(self) -> int:
        return self.graph[self.end_node_idx].position_id

    @property
    def groups(self) -> list[list[int]]:
        """
        Group the items (nodes) by orders. The last item in each group is the dummy node. The depots are single-item groups.
        For example, if there are 4 orders and 10 items, [0, 11] represent the depots, and [12, 13, 14, 15] represent the dummy nodes, then:
            groups = [[0], [1, 2, 3, 12], [4, 5, 13], [6, 7, 14], [8, 9, 15], [11]]
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
            0: 'Not solved',
            1: 'Optimal',
            2: 'Feasible',
            3: 'No solution found',
            4: 'Timeout',
            5: 'Invalid',
            6: 'Infeasible',
        }

        return map.get(status, 'Unknown')

    def get_order(self, node: Item) -> Order:
        """Get the order of the node."""
        order_id = self.node_to_order[node.id]
        matched = [order for order in self.warehouse.orders if order.id == order_id]
        assert len(matched) == 1

        return matched[0]

    def get_node_idx(self, item: Item) -> int:
        """Get the node index from the item."""
        return next(key for key, value in self.graph.items() if value == item)

    def build_graph(self) -> None:
        """
        Build the graph with the items and dummy nodes for each order in the warehouse, and the depots (start and end).
        The dummy nodes allow the pick-up and delivery operation.
        The depots are the first and last nodes in the graph.
        """
        depots = self.warehouse.depots
        nodes = [depots[0]]
        dummy_idx = self.warehouse.nb_positions + 1

        for order in self.warehouse.orders:
            dummy = Item(id=dummy_idx, is_dummy=True)
            vertices = order.pickups + [dummy]
            nodes.extend(vertices)

            for i in vertices:
                if i.id in self.node_to_order:
                    raise ValueError(f'Order {order.id} | Node {i.id} is already in the graph')
                
                self.node_to_order[i.id] = order.id

            dummy_idx += 1

        nodes.append(depots[1])

        self.graph = {idx: item for idx, item in enumerate(nodes)}

    def build_matrix(self) -> np.ndarray:
        """
        Return the distance matrix between all nodes.
        # TODO no deberiamos hacer que la distancia del dummy a los items del mismo pedido sea 0 y que a otros pedidos sea inf??
        """
        return np.array([
            [
                self.warehouse.distance(i, j) if (not i.is_dummy and not j.is_dummy) else 0
                for j in self.graph.values()
            ]
            for i in self.graph.values()
        ])

    def set_parameters(self) -> None:
        """
        Set the solver parameters.
        The construction and local search parameters have been selected based on experimentation.
        """
        self.parameters = pywrapcp.DefaultRoutingSearchParameters()
        self.parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            # routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        )
        self.parameters.local_search_metaheuristic = (
            # routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
        )
        self.parameters.time_limit.FromSeconds(self.timeout)
        self.routing.CloseModelWithParameters(self.parameters) # due to the initial solution

    def get_initial_solution(self) -> Any:
        """
        Get the S-shaped path to visit the items of each order individually.
        The solution is encoded by the node indices the positions belong to.
        Reference: https://developers.google.com/optimization/routing/routing_tasks#setting_initial_routes_for_a_search
        """
        grouped_nodes = [
            [self.get_node_idx(item) for item in group]
            for group in self.warehouse.base_solution
        ]
        initial_solution = self.routing.ReadAssignmentFromRoutes(grouped_nodes, True)
        debug(f'VRP | Initial solution | Node indices {grouped_nodes}')

        return initial_solution

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
            orders = list(set(
                self.get_order(node)
                for node in route.sequence
                if node.is_pickup
            ))
            if not orders:
                continue

            metrics = Metrics(
                distance=distance,
                units=unit_load,
                volume=volume_load
            )
            batches.append(Batch(orders=orders, route=route, metrics=metrics))

        return batches

    # Objective functions
    # -------------------
    
    def minimize_total_distance(self) -> None:
        """
        The objective function is to minimize the total distance traveled by the pickers.
        """
        distances = self.build_matrix()

        def distance_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)

            return distances[from_node][to_node]
        
        self.callbacks.distance = self.routing.RegisterTransitCallback(distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(self.callbacks.distance)

        debug(f'Minimize total distance | Distance sample: {distances[0]}')

    # Constraints
    # -----------

    def unit_capacity_constraints(self) -> None:
        """
        Set the capacity constraints for the vehicles. The capacity is the number of orders.
            To limit the quantity of orders, we create a dummy node for each order and set the demand to 1.
            Therefore, the capacity of the vehicle is the number of orders.
            For the volume, it is similar since the we know the volume of the whole order and not each item individually.

        Reference: https://developers.google.com/optimization/routing/cvrp
        """
        def demand_callback(from_index):
            """Returns the demand of the node."""
            from_node = self.manager.IndexToNode(from_index)

            return self.demands[from_node]
        
        self.callbacks.demand = self.routing.RegisterUnaryTransitCallback(demand_callback)
        self.routing.AddDimensionWithVehicleCapacity(
            self.callbacks.demand,
            0,  # no capacity slack
            [self.warehouse.vehicle.max_nb_orders] * self.nb_vehicles,  # vehicle maximum capacities
            True,  # start cumul to zero
            'UnitCapacity'
        )

        debug(f'Unit capacity constraints | Demands: {self.demands}')

    def volume_capacity_constraints(self) -> None:
        """
        Set the volume capacity constraints for the vehicles. The capacity is the volume of the orders.
        """
        def volume_callback(from_index):
            """Returns the volume of the node."""
            from_node = self.manager.IndexToNode(from_index)

            return self.volumes[from_node]
        
        self.callbacks.volume = self.routing.RegisterUnaryTransitCallback(volume_callback)
        self.routing.AddDimensionWithVehicleCapacity(
            self.callbacks.volume,
            0,  # no capacity slack
            [self.warehouse.vehicle.max_volume] * self.nb_vehicles,  # vehicle maximum capacities
            True,  # start cumul to zero
            'VolumeCapacity'
        )

        debug(f'Volume capacity constraints | Volumes: {self.volumes}')

    def pickup_delivery_constraints(self) -> None:
        """
        Set the pick-up and delivery constraints for the vehicles.
            All the positions (items) for an order must be visited in the same batch route.
            Therefore, we consider that the items are pick-up positions and the dummy nodes are delivery positions.
            TODO esto permitiria unloading en medio de la ruta? Es decir que puede que exceda la capacidad al final pero no en cada punto.

        Reference: https://developers.google.com/optimization/routing/pickup_delivery
        """
        grouped_items = self.groups[1:-1] # exclude the depots
        assert len(grouped_items) == self.nb_vehicles

        for group in grouped_items:
            delivery_idx = group[-1]
            pickup_indices = group[:-1]

            for pickup_idx in pickup_indices:
                self.routing.AddPickupAndDelivery(pickup_idx, delivery_idx)
                self.routing.solver().Add(
                    self.routing.VehicleVar(pickup_idx) == self.routing.VehicleVar(delivery_idx)
                )

        debug(f'Pickup and delivery constraints | Items: {grouped_items}')

    # Model
    # -----
        
    def build_model(self) -> None:
        """
        Build the model for the Capacitated VRP with pick-up and delivery.
        Mutiple depot feature to model the start and end of the route.
        Source: https://developers.google.com/optimization/routing/routing_tasks#setting-start-and-end-locations-for-routes
        """
        self.build_graph()
        self.manager = pywrapcp.RoutingIndexManager(
            len(self.graph), # number of nodes
            self.nb_vehicles, # number of vehicles
            [self.start_node_idx] * self.nb_vehicles, # start nodes
            [self.end_node_idx] * self.nb_vehicles # end nodes
        )
        self.routing = pywrapcp.RoutingModel(self.manager)

        self.minimize_total_distance()

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
        debug(f'VRP | Start node (idx-id): {(self.start_node_idx, self.start_node_id)} | End node (idx-id): {(self.end_node_idx, self.end_node_id)} | Vehicles: {self.nb_vehicles} | Graph: {len(self.graph)}')

        initial_solution = self.get_initial_solution()
        solution = self.routing.SolveFromAssignmentWithParameters(
            initial_solution, self.parameters
        )

        if solution and self.is_valid:
            info(f'VRP | Solution obtained | Status: {self.status}')

            return self.build_solution(solution)
        
        else:
            error("No solution found")
