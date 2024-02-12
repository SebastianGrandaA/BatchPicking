from domain.models.instances import Order
import numpy as np
from scipy.spatial.distance import directed_hausdorff


class Hausdorff:
    def get_coordinates(self, order: Order) -> list[tuple[float, float]]:
        return [item.coordinates for item in order.items]
    
    def closeness(self, order_1: Order, order_2: Order) -> float:
        """
        Symmetric Hausdorff distance between two orders, which is the maximum of the directed distances.
        The directed Hausdorff distance between two orders, which is the maximum distance between any item in order 1 and its nearest item in order 2.

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html
        """
        if order_1 == order_2:
            return 0
        
        positions_1 = self.get_coordinates(order_1)
        positions_2 = self.get_coordinates(order_2)

        distance_1_2 = directed_hausdorff(positions_1, positions_2)[0]
        distance_2_1 = directed_hausdorff(positions_2, positions_1)[0]

        return max(distance_1_2, distance_2_1)

    def build_matrix(self, orders: list[Order]) -> np.ndarray:
        """
        Return the distance matrix between all orders.
        """
        matrix = np.zeros((len(orders), len(orders)))

        for i, order_i in enumerate(orders):
            for j, order_j in enumerate(orders):
                if i != j:
                    matrix[i, j] = self.closeness(order_i, order_j)
        
        return matrix
