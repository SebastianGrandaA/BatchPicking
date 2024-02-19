from concurrent.futures import ProcessPoolExecutor
from typing import Any

from domain.models.instances import Item, Order
from domain.models.solutions import Batch, Problem


class Routing(Problem):
    """
    Interface for routing problems.
    """

    graph: dict[int, Item] = {}
    node_to_order: dict[int, int] = {}

    @property
    def nodes(self) -> list[Item]:
        return list(self.graph.values())

    @property
    def node_ids(self) -> list[int]:
        return list(self.graph.keys())

    @property
    def is_valid(self) -> bool:
        return self.status in ["Optimal", "Feasible"]

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
    def artificial_idx(self) -> int:
        return self.warehouse.nb_positions + 1

    @property
    def node_items(self) -> list[tuple[int, Item]]:
        return list(self.graph.items())

    def route(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def route_batch(self, batch: Batch) -> Any:
        raise NotImplementedError

    def build_graph(self) -> None:
        """
        Build the graph with the items and dummy nodes for each order in the warehouse, and the depots (start and end).
        The dummy nodes allow the pick-up and delivery operation.
        The depots are the first and last nodes in the graph.
        """
        depots = self.warehouse.depots
        nodes = [depots[0]]
        dummy_idx = self.artificial_idx

        for order in self.warehouse.orders:
            dummy = Item(id=dummy_idx, is_dummy=True)
            vertices = order.pickups + [dummy]
            nodes.extend(vertices)

            for i in vertices:
                if i.id in self.node_to_order and self.is_warehouse_complete:
                    raise ValueError(
                        f"Order {order.id} | Node {i.id} is already in the graph"
                    )

                self.node_to_order[i.id] = order.id

            dummy_idx += 1

        nodes.append(depots[1])

        self.graph = {idx: item for idx, item in enumerate(nodes)}

    def get_order(self, node: Item) -> Order:
        """Get the order of the node."""
        order_id = self.node_to_order[node.id]
        matched = [order for order in self.warehouse.orders if order.id == order_id]
        assert len(matched) == 1

        return matched[0]

    def get_node_idx(self, item: Item) -> int:
        """Get the node index from the item."""
        occurrences = [key for key, value in self.node_items if value == item]
        assert len(occurrences) == 1, f"Item {item} has multiple indices {occurrences}"

        return occurrences[0]
        # idxs = next(key for key, value in self.self.node_items if value == item)
        # assert len(idxs) == 1, f"Item {item} has multiple indices {idxs}"

    def build_matrix(self) -> Any:
        raise NotImplementedError

    def build_model(self) -> Any:
        raise NotImplementedError

    def solve_parallel(self, batches: list[Batch]) -> list[Batch]:
        """
        Solve multiple TSP instances in parallel, one for each batch (CPU-bound operation).

        Parameters
        ----------
        method : str
            Method to solve the TSP. Options: 'TSPBase', 'TSPMultiCommodityFlow'.

        batches : list[Batch]
            Batches of items to be routed independently.
        """
        routes = []

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self.route_batch(batch=batch)) for batch in batches
            ]

            for future in futures:
                routes.append(future.result())

        return routes

    def solve_sequential(self, batches: list[Batch]) -> list[Batch]:
        """
        Solve multiple TSP instances sequentially.

        Parameters
        ----------
        method : str
            Method to solve the TSP. Options: 'TSPBase', 'TSPMultiCommodityFlow'.

        batches : list[Batch]
            Batches of items to be routed independently.
        """
        routes = []

        for batch in batches:
            routes.append(self.route_batch(batch=batch))

        return routes

    def build_model(self, batches: list[Batch]) -> Any:
        return self.solve(batches=batches)
