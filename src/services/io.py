from os import path
from pydantic import BaseModel
import numpy as np
from domain.models.instances import Capacity, Distances, Item, Order, Position, Vehicle, Warehouse

class IO(BaseModel):
    instance_name: str = ''

    @property
    def directory(self) -> str:
        return path.abspath(path.join(path.dirname(__file__), '..', '..'))

class Reader(IO):
    def split(self, string: str) -> str:
        return string.strip().split('\n')[1:]
    
    def read(self, filename: str) -> str:
        """Read the content of the file."""
        dir = path.join(
            self.directory,
            'data.nosync',
            self.instance_name,
            f'{filename}.txt'
        )

        if path.exists(dir):
            with open(dir, 'r') as file:
                return file.read()
            
        else:
            raise FileNotFoundError(f"File {dir} not found.")

    def build_matrix(self, string: str) -> Distances:
        """Parse the distance matrix from the string to a Distances object."""
        matrix = np.array([
            [item for item in line.split(' ') if item != '']
            for line in self.split(string) if line != ''
        ], dtype=int)

        return Distances(matrix=matrix)

    def build_orders(self, supports: str, positions: str) -> list[Order]:
        """
        Parse the support list and the position list to a list of orders.
        Assumes that the depots are in the first and last position of the support list.
        """
        support_list = self.split(supports)
        position_list = self.split(positions)
        orders, order_idx, item_idx, is_depot = [], 0, 0, False

        for batch in range(0, len(support_list), 2):
            order_id, volume, _ = support_list[batch].split(' ')
            coordinates = [position.split(' ') for position in position_list]
            items = []

            for item, (x, y) in enumerate(coordinates):
                if str(item) in support_list[batch + 1].split(' '):
                    is_depot = item in [0, len(coordinates) - 1]
                    item = Item(id=item_idx, position=Position(id=int(item), x=float(x), y=float(y)), is_depot=is_depot)
                    items.append(item)
                    item_idx += 1

            order = Order(id=int(order_id), volume=int(volume), items=items)
            assert order.id == order_idx, f"Order id {order.id} does not match the index {order_idx}"
            order_idx += 1
            orders.append(order)

        return orders
    
    def build_vehicle(self, string: str) -> Vehicle:
        """Parse the vehicle constraints from the string to a Vehicle object."""
        max_nb_orders, max_volume = string.strip().split(' ')

        return Vehicle(capacity=Capacity(
            volume=int(max_volume), nb_orders=int(max_nb_orders)
        ))

    def load_instance(self) -> Warehouse:
        """Load the instance from the input files."""
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
