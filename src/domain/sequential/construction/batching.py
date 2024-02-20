from logging import error

import pyomo.environ as pyo
from k_means_constrained import KMeansConstrained

from domain.models.solutions import Batch, Problem
from services.distances import Hausdorff


class PMedian(Problem):
    # """
    # ## The p-median problem

    # The p-median problem consists of selecting a subset of facilities, among a set of candidates, to be used to serve a set of demand points [1].
    # The objective is to minimize the total travel distance between the demand points and the facilities.

    # In our context, we seek to group the orders into batches based on capacity constraints and a custom distance metric.
    # Therefore, the concept of "batch" can be interpreted as a consolidation facility for a set of orders.
    # Let \(\mathcal{I}\) be the set of potential orders to select \(p\) batches from, and \(\mathcal{J}\) be the set of orders to be served.
    # The closeness between orders \(i \in \mathcal{I}\) and \(j \in \mathcal{J}\) is given by \(c_{ij}\).
    # As before, let \(C_{unit}\) and \(C_{volume}\) be the maximum number of orders and the maximum volume that a batch can serve, respectively.
    # Also, let \(v_i\) be the volume of order \(i \in \mathcal{I}\ and \(\underline{B}\) be the minimum number of batches to be formed \ref{eq:min_batches}.

    # Let \(x_{ij} \in \{0, 1\}\) be the allocation variable, where \(x_{ij} = 1\) if order \(j\) is assigned to batch \(i\), and \(x_{ij} = 0\) otherwise.
    # As mentioned in [1], when \(\mathcal{I} = \mathcal{J}\) and \(c_{ii} = 0 \forall i \in \mathcal{I}\), the traditional location variables \(y_i\) can be replaced by the allocation variables \(x_{ii} \forall i \in \mathcal{I}\).
    # The objective is to maximize the total closeness between the orders in the same batch, and can be formulated as follows:
    # A complete formulation of the p-median problem is available at the [report](https://www.overleaf.com/read/xfgcnzwccnqj#8fe7b9).
    # The p-median problem is NP-hard. However, since we the locations represent the orders (the number of locations is small) we can use exact methods to solve it.

    # ## References
    # [1] Laporte, G., Nickel, S., & Saldanha-da-Gama, F. (2019). Introduction to location science (pp. 1-21). Springer International Publishing.
    # """

    def closeness_objective(self, model: pyo.ConcreteModel) -> float:
        closeness = Hausdorff().closeness

        return sum(
            closeness(od_i, od_j) * model.x[i + 1, j + 1]
            for i, od_i in enumerate(self.warehouse.orders)
            for j, od_j in enumerate(self.warehouse.orders)
        )

    def unique_assignment_constraint(self, model: pyo.ConcreteModel, j: int):
        return sum(model.x[i, j] for i in model.I) == 1

    def selected_batches_constraint(self, model: pyo.ConcreteModel, i: int):
        return (
            sum(model.x[i, j] for j in model.J if j != i)
            <= (len(model.J) - model.p) * model.x[i, i]
        )

    def maximum_batches_constraint(self, model: pyo.ConcreteModel):
        return sum(model.x[i, i] for i in model.I) <= model.p

    def unitary_capacity_constraint(self, model: pyo.ConcreteModel, i: int):
        return sum(model.x[i, j] for j in model.J) <= model.C_unit

    def volume_capacity_constraint(self, model: pyo.ConcreteModel, i: int):
        return (
            sum(model.x[i, j] * self.warehouse.orders[j - 1].volume for j in model.J)
            <= model.C_volume
        )

    def symmetry_constraint(self, model: pyo.ConcreteModel, i: int, j: int):
        if i < j:
            return model.x[i, j] == model.x[j, i]

        return pyo.Constraint.Skip

    def build_model(self) -> pyo.ConcreteModel:
        model = pyo.ConcreteModel()

        # Parameters
        model.I = pyo.RangeSet(len(self.warehouse.orders))
        model.J = pyo.RangeSet(len(self.warehouse.orders))
        model.p = self.minimum_batches
        model.C_unit = self.warehouse.vehicle.max_nb_orders
        model.C_volume = self.warehouse.vehicle.max_volume

        # Variables
        model.x = pyo.Var(model.I, model.J, domain=pyo.Binary)

        # Objective
        model.objective = pyo.Objective(
            rule=self.closeness_objective, sense=pyo.maximize
        )

        # Constraints
        model.unique_assignment = pyo.Constraint(
            model.J, rule=self.unique_assignment_constraint
        )
        model.selected_batches = pyo.Constraint(
            model.I, rule=self.selected_batches_constraint
        )
        model.maximum_batches = pyo.Constraint(rule=self.maximum_batches_constraint)
        model.unitary_capacity = pyo.Constraint(
            model.I, rule=self.unitary_capacity_constraint
        )
        model.volume_capacity = pyo.Constraint(
            model.I, rule=self.volume_capacity_constraint
        )

        return model

    def build_solution(self, model: pyo.ConcreteModel) -> list[Batch]:
        batches = [
            Batch(
                orders=[
                    self.warehouse.orders[j - 1]
                    for j in model.J
                    if pyo.value(model.x[i, j]) > 0.5
                ]
            )
            for i in model.I
        ]

        return [batch for batch in batches if len(batch.orders) > 0]

    def single_orders_fallback(self) -> list[Batch]:
        """Fallback to return batches of single orders."""
        return [Batch(orders=[order]) for order in self.warehouse.orders]

    def solve(self) -> list[Batch]:
        """
        Entry point to optimize the p-median problem.

        A fallback to single orders is implemented in case the optimization fails.
        """
        try:
            model = self.build_model()
            solver = pyo.SolverFactory("gurobi")
            result = solver.solve(
                model,
                tee=self.verbose,
                options={"TimeLimit": self.timeout, "OutputFlag": int(self.verbose)},
            )

            if self.is_valid(result):
                return self.build_solution(model)

            raise Exception("Invalid batching solution")

        except Exception as err:
            error(f"Batching | Optimization failed: {err} | Fallback to single orders")

            return self.single_orders_fallback()


class Clustering(Problem):
    """K-means constrained clustering."""

    def build_model(self):
        """
        Source: https://joshlk.github.io/k-means-constrained/
        """
        return KMeansConstrained(
            n_clusters=self.minimum_batches,
            size_min=1,
            size_max=self.warehouse.nb_orders,
            random_state=0,
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
    """Graph partitioning problem."""

    def closeness_objective(self, model: pyo.ConcreteModel) -> float:
        closeness = Hausdorff().closeness

        return sum(
            closeness(od_i, od_j) * model.x[i + 1, j + 1]
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
        return (
            sum(self.warehouse.orders[i - 1].volume * model.y[i, k] for i in model.R)
            <= self.warehouse.vehicle.max_volume
        )

    def capacity_quantity_constraint(self, model: pyo.ConcreteModel, k: int):
        return (
            sum(model.y[i, k] for i in model.R) <= self.warehouse.vehicle.max_nb_orders
        )

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
        model.objective = pyo.Objective(
            rule=self.closeness_objective, sense=pyo.maximize
        )

        # Constraints
        model.unique_assignment = pyo.Constraint(
            model.R, rule=self.unique_assignment_constraint
        )
        model.intra_batch = pyo.Constraint(
            model.R, model.R, model.K, rule=self.intra_batch_constraint
        )
        model.capacity_volume = pyo.Constraint(
            model.K, rule=self.capacity_volume_constraint
        )
        model.capacity_quantity = pyo.Constraint(
            model.K, rule=self.capacity_quantity_constraint
        )
        model.symmetry = pyo.Constraint(model.R, model.R, rule=self.symmetry_constraint)

        return model

    def build_solution(self, model: pyo.ConcreteModel) -> list[Batch]:
        batches = [
            Batch(
                orders=[
                    self.warehouse.orders[i - 1]
                    for i in model.R
                    if pyo.value(model.y[i, k]) > 0.5
                ]
            )
            for k in model.K
        ]

        return [batch for batch in batches if len(batch.orders) > 0]
