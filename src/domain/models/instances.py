from logging import warning
from typing import Any

import numpy as np
from pydantic import BaseModel, validator


class Position(BaseModel):
    """Pick-up position in the warehouse."""

    id: int = np.nan  # Position ID is NOT unique between orders.
    x: float = np.nan
    y: float = np.nan

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __eq__(self, other: Any) -> bool:
        return self.x == other.x and self.y == other.y

    def __str__(self) -> str:
        return f"Position(id={self.id}, x={self.x}, y={self.y})"


class Item(BaseModel):
    """Item to be picked up within an order."""

    id: int
    position: Position = Position()
    is_depot: bool = False
    is_dummy: bool = False

    @property
    def is_pickup(self) -> bool:
        return not self.is_depot and not self.is_dummy

    @property
    def position_id(self) -> int:
        return self.position.id

    @property
    def coordinates(self) -> tuple[float, float]:
        return self.position.x, self.position.y

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        return self.id == other.id

    def __str__(self) -> str:
        return f"Item(id={self.id}, position={self.position}, is_depot={self.is_depot}, is_dummy={self.is_dummy})"


class Distances(BaseModel):
    """Interface to calculate the distance between two items based on their positions in the warehouse."""

    matrix: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    @validator("matrix")
    def validate_matrix(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError("Not a matrix")

        if (v < 0).any():
            raise ValueError("Matrix must be positive")

        return v

    def distance(self, i: Item, j: Item) -> int:
        if i == j or i.position == j.position:
            return 0

        value = self.matrix[i.position_id, j.position_id]

        if np.isnan(value) or value < 0:
            warning(f"Invalid distance ({str(i)}, {str(j)}: {value}). Setting to 0.")

            return 0

        return value


class Order(BaseModel):
    """Set of items to be picked up together in a support (physical box)."""

    id: int
    volume: int
    items: list[Item]

    @property
    def nb_items(self) -> int:
        return len(self.items)

    @property
    def item_ids(self) -> list[int]:
        return [item.id for item in self.items]

    @property
    def position_ids(self) -> list[int]:
        return [item.position_id for item in self.items]

    @property
    def pickups(self) -> list[Item]:
        return [item for item in self.items if item.is_pickup]

    @property
    def depots(self) -> list[Item]:
        depots = [item for item in self.items if item.is_depot]
        assert len(depots) == 2, f"Depots {depots} are not two"

        return depots

    @property
    def depot_ids(self) -> list[int]:
        return [depot.position_id for depot in self.depots]

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        return self.id == other.id

    def __str__(self) -> str:
        return f"Order(id={self.id}, volume={self.volume}, items={self.item_ids}, positions={self.position_ids}, depots={self.depot_ids})"


class Capacity(BaseModel):
    volume: int
    nb_orders: int

    def __str__(self):
        return f"Capacity(volume={self.volume}, nb_orders={self.nb_orders})"


class Vehicle(BaseModel):
    capacity: Capacity

    @property
    def max_volume(self) -> int:
        return self.capacity.volume

    @property
    def max_nb_orders(self) -> int:
        return self.capacity.nb_orders

    def __str__(self):
        return f"Vehicle(capacity={self.capacity})"


class Instance(BaseModel):
    orders: list[Order]

    @property
    def order_ids(self) -> list[int]:
        return [order.id for order in self.orders]

    @property
    def id(self) -> str:
        return "-".join(str(id) for id in self.order_ids)

    @property
    def total_volume(self) -> int:
        return sum(order.volume for order in self.orders)

    @property
    def nb_orders(self) -> int:
        return len(self.orders)

    @property
    def items(self) -> list[Item]:
        """Items in the warehouse excluding the depots."""
        return [item for order in self.orders for item in order.items if item.is_pickup]

    @property
    def item_ids(self) -> list[int]:
        return [item.id for item in self.items]

    @property
    def positions(self) -> list[Position]:
        """Includes the depots and fake items."""
        return [item.position for order in self.orders for item in order.items]

    @property
    def position_ids(self) -> list[int]:
        return [position.id for position in self.positions]

    @property
    def coordinates(self) -> list[tuple[float, float]]:
        return [item.coordinates for item in self.items]

    @property
    def nb_positions(self) -> int:
        return len(self.positions)

    @property
    def nb_items(self) -> int:
        return len(self.items)

    @property
    def depots(self) -> list[Item]:
        return self.orders[0].depots

    @property
    def depot_ids(self) -> list[int]:
        return [depot.position_id for depot in self.depots]

    @property
    def is_valid(self) -> bool:
        """Checks if the instance is valid based on the unique IDs and sequences."""
        are_ids_unique = len(set(order for order in self.orders)) == len(self.orders)
        is_sequence = all(order.id == idx for idx, order in enumerate(self.orders))

        item_ids_unique = len(set(item for item in self.item_ids)) == len(self.item_ids)

        return are_ids_unique and is_sequence and item_ids_unique


class Warehouse(Instance):
    instance_name: str
    distances: Distances
    vehicle: Vehicle
    current_solution: list[list[Item]] = []

    @property
    def name(self) -> str:
        return self.instance_name

    @property
    def minimum_batches(self) -> int:
        """
        The minimum number of batches required to fulfill the orders based on the capacity.
        In the worst case, we consider the maximum between the volume and the number of orders capacity.
        """
        nb_batches = (
            max(
                [
                    self.total_volume / self.vehicle.max_volume,
                    self.nb_orders / self.vehicle.max_nb_orders,
                ]
            )
            * 1.1
        )

        assert (
            nb_batches < self.nb_orders
        ), f"Too many batches {nb_batches} for {self.nb_orders} orders"

        return np.ceil(nb_batches).astype(int)

    @property
    def base_solution(self) -> list[list[Item]]:
        """
        The base solution is the S-shaped path to visit each order individually excluding the depots.
        The path for each order is the sequence of pickups (excluding the depots).
        """
        return [order.pickups for order in self.orders]

    def __str__(self) -> str:
        orders = ",\n".join([str(order) for order in self.orders])

        return f"Warehouse(name={self.name}, volume={self.total_volume}, vehicle={self.vehicle}, items={self.nb_items}, orders=[\n{orders}\n])"

    def distance(self, i: Item, j: Item) -> int:
        return self.distances.distance(i, j)
