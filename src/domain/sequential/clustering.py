import numpy as np
from k_means_constrained import KMeansConstrained
from services.distances import Hausdorff
from services.solutions import Batch, Problem
from sklearn.cluster import DBSCAN


class Clustering(Problem):
    def build_model(self):
        """
        Source: https://joshlk.github.io/k-means-constrained/
        """
        return KMeansConstrained(
            n_clusters=self.minimum_batches,
            size_min=1,
            size_max=self.warehouse.total_nb_orders,
            random_state=0
        )
        
    def build_solution(self, solution: list[int]) -> list[Batch]:
        clusters = [
            Batch(
                orders=[
                    self.warehouse.orders[i]
                    for i, cluster in enumerate(solution)
                    if cluster == k
                ]
            )
            for k in range(self.minimum_batches)
        ]

        return clusters
    
    def solve(self):
        matrix = Hausdorff().build_matrix(orders=self.warehouse.orders)
        model = self.build_model()
        solution = model.fit_predict(matrix)

        return self.build_solution(solution=solution)
