import pyomo.environ as pyo
from services.distances import Hausdorff
from domain.models.solutions import Batch, Problem
from k_means_constrained import KMeansConstrained


class Clustering(Problem):
    def build_model(self):
        """
        Source: https://joshlk.github.io/k-means-constrained/
        """
        return KMeansConstrained(
            n_clusters=self.minimum_batches,
            size_min=1,
            size_max=self.warehouse.total_nb_orders,
            # TODO anadir capacidad de volumen !
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

class GraphPartition(Problem):
    def closeness_objective(self, model: pyo.ConcreteModel) -> float:
        closeness = Hausdorff(self.warehouse.distances.matrix).distance

        return sum(
            closeness(od_i, od_j) * model.x[i+1, j+1]
            for i, od_i in enumerate(self.warehouse.orders)
            for j, od_j in enumerate(self.warehouse.orders)
        )

    def unique_assignment_constraint(self, model: pyo.ConcreteModel, i: int):
        return sum(model.y[i, k] for k in model.K) == 1
    
    def intra_batch_constraint(self, model: pyo.ConcreteModel, i: int, j: int, k: int):
        if i == j:
            return pyo.Constraint.Skip
        
        return model.y[i, k] + model.y[j, k] <= 1 + model.x[i, j]
    
    def capacity_volume_constraint(self, model: pyo.ConcreteModel, k: int):
        return sum(self.warehouse.orders[i-1].volume * model.y[i, k] for i in model.R) <= self.warehouse.vehicle.max_volume
    
    def capacity_quantity_constraint(self, model: pyo.ConcreteModel, k: int):
        return sum(model.y[i, k] for i in model.R) <= self.warehouse.vehicle.max_nb_orders
    
    def symmetry_constraint(self, model: pyo.ConcreteModel, i: int, j: int):
        if i < j:
            return model.x[i, j] == model.x[j, i]
        
        elif i == j:
            return model.x[i, j] == 0
        
        return pyo.Constraint.Skip
        
    def build_model(self):
        model = pyo.ConcreteModel()

        # Parameters
        model.R = pyo.RangeSet(len(self.warehouse.orders))
        model.K = pyo.RangeSet(self.minimum_batches)

        # Variables
        model.x = pyo.Var(model.R, model.R, domain=pyo.Binary)
        model.y = pyo.Var(model.R, model.K, domain=pyo.Binary)
        
        # Objective
        model.objective = pyo.Objective(rule=self.closeness_objective, sense=pyo.maximize)

        # Constraints
        model.unique_assignment = pyo.Constraint(model.R, rule=self.unique_assignment_constraint)
        model.intra_batch = pyo.Constraint(model.R, model.R, model.K, rule=self.intra_batch_constraint)
        model.capacity_volume = pyo.Constraint(model.K, rule=self.capacity_volume_constraint)
        model.capacity_quantity = pyo.Constraint(model.K, rule=self.capacity_quantity_constraint)
        model.symmetry = pyo.Constraint(model.R, model.R, rule=self.symmetry_constraint)
        
        return model

    def build_solution(self, model: pyo.ConcreteModel) -> list[Batch]:
        batches = [
            Batch(
                orders=[
                    self.warehouse.orders[i-1]
                    for i in model.R
                    if pyo.value(model.y[i, k]) > 0.5
                ]
            )
            for k in model.K
        ]

        return [batch for batch in batches if len(batch.orders) > 0]
    