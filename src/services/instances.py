from os import path

import numpy as np
from pydantic import BaseModel, validator

"""
Pick-up position in the warehouse.
"""
class Position(BaseModel):
    x: float
    y: float

"""
Item of an order which is located in a position.
"""
class Item(BaseModel):
    id: int
    position: Position

    @property
    def position_id(self) -> int:
        return self.id

    @property
    def coordinates(self) -> tuple[float, float]:
        return self.position.x, self.position.y

"""
Distance matrix between each position in the warehouse.
"""
class Distances(BaseModel):
    matrix: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    @validator('matrix')
    def validate_matrix(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError("Not a matrix")
        
        if (v < 0).any():
            raise ValueError('Matrix must be positive')
        
        return v
    
    @property
    def distance(self, i: Item, j: Item) -> int:
        return self.matrix[i.id, j.id]

"""
Set of items that are grouped together into a support (physical box).
"""
class Order(BaseModel):
    id: int
    volume: int
    items: list[Item]

    @property
    def nb_items(self) -> int:
        return len(self.items)


class Capacity(BaseModel):
    volume: int
    nb_orders: int

    def __str__(self):
            return f"Max volume: {self.volume} | Max nb items: {self.nb_orders}"

class Vehicle(BaseModel):
    capacity: Capacity

    @property
    def max_volume(self) -> int:
        return self.capacity.volume
    
    @property
    def max_nb_orders(self) -> int:
        return self.capacity.nb_orders
    
class Warehouse(BaseModel):
    instance_name: str
    distances: Distances
    orders: list[Order]
    vehicle: Vehicle

    @property
    def total_volume(self) -> int:
        return sum(order.volume for order in self.orders)
    
    @property
    def total_nb_orders(self) -> int:
        return len(self.orders)
    
    def __str__(self) -> str:
        return f'Warehouse(name={self.instance_name}, orders={self.total_nb_orders}, volume={self.total_volume})'

    @property
    def minimum_batches(self) -> int:
        """
        Calculate the minimum number of batches required to fulfill the orders based on the capacity.
        We consider the maximum between the volume and the number of orders capacity.
        """
        return max([
            self.total_volume // self.vehicle.max_volume,
            self.total_nb_orders // self.vehicle.max_nb_orders
        ])

class Reader(BaseModel):
    instance_name: str

    @property
    def base_directory(self) -> str:
        return path.abspath(path.join(path.dirname(__file__), '..', '..'))

    def split(self, string: str) -> str:
        return string.strip().split('\n')[1:]
    
    def read(self, filename: str) -> str:
        directory = path.join(
            self.base_directory,
            'data',
            self.instance_name,
            f'{filename}.txt'
        )

        if path.exists(directory):
            with open(directory, 'r') as file:
                return file.read()
            
        else:
            raise FileNotFoundError(f"File {directory} not found.")

    def build_matrix(self, string: str) -> Distances:
        matrix = np.array([
            [item for item in line.split(' ') if item != '']
            for line in self.split(string) if line != ''
        ], dtype=int)

        return Distances(matrix=matrix)

    def build_orders(self, supports: str, positions: str) -> list[Order]:
        support_list = self.split(supports)
        position_list = self.split(positions)
        orders = []

        for batch in range(0, len(support_list), 2):
            order_id, volume, _ = support_list[batch].split(' ')
            items = [
                Item(id=int(item), position=Position(id=int(item), x=float(x), y=float(y)))
                for item, (x, y) in enumerate([position.split(' ') for position in position_list])
                if str(item) in support_list[batch + 1].split(' ')
            ]
            orders.append(Order(id=int(order_id), volume=int(volume), items=items))

        return orders
    
    def build_vehicle(self, string: str) -> Vehicle:
        max_nb_orders, max_volume = string.strip().split(' ')

        return Vehicle(capacity=Capacity(
            volume=int(max_volume), nb_orders=int(max_nb_orders)
        ))

    def load_instance(self) -> Warehouse:
        distance_matrix = self.build_matrix(
            self.read('adjacencyMatrix')
        )
        orders = self.build_orders(
            self.read('supportList'),
            self.read('positionList')
        )
        vehicle = self.read('constraints')

        return Warehouse(
            instance_name=self.instance_name,
            distances=distance_matrix,
            orders=orders,
            vehicle=self.build_vehicle(vehicle)
        )
