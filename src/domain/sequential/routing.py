from concurrent.futures import ProcessPoolExecutor

import pyomo.environ as pyo
from services.solutions import Batch, Problem


class TSP(Problem):
    batch: Batch

    def total_distance(self, model: pyo.ConcreteModel):
        matrix = self.warehouse.distances.matrix

        return sum(
            matrix[self.batch.items[i].id, self.batch.items[j].id]
                if i <= self.batch.depot[1] and j <= self.batch.depot[1]
                else 0
            * model.x[i, j]
            for (i, j) in model.edges
        )
    
    def entry_constraint(self, model: pyo.ConcreteModel, i: int):
        return sum(model.x[i, j] for j in model.nodes if i != j) == 1
    
    def exit_constraint(self, model: pyo.ConcreteModel, j: int):
        return sum(model.x[i, j] for i in model.nodes if i != j) == 1
    
    def flow_constraint(self, model: pyo.ConcreteModel, l: int):
        return (
            sum(model.z[0, j, l] for j in model.positions if j not in self.batch.depot)
            - sum(model.z[j, 0, l] for j in model.positions if j not in self.batch.depot) == -1
        )
    
    
    def one_commodity_per_vertex_constraint(self, model: pyo.ConcreteModel, i: int):
        if i >= self.batch.depot[1]:
            return pyo.Constraint.Skip
        
        return (
            sum(model.z[i, j, i] for j in model.nodes if j not in self.batch.depot)
            - sum(model.z[j, i, i] for j in model.nodes if j not in self.batch.depot) == 1
        )
    
    def flow_conservation_constraint(self, model: pyo.ConcreteModel, i: int, l: int):
        if i == l:
            return pyo.Constraint.Skip

        return (
            sum(model.z[i, j, l] for j in model.nodes)
            - sum(model.z[j, i, l] for j in model.nodes) == 0
        )
    
    def flow_arc_constraint(self, model: pyo.ConcreteModel, i: int, j: int, l: int):
        return model.z[i, j, l] <= model.x[i, j]
    
    def build_model(self):
        model = pyo.ConcreteModel()

        # Parameters
        model.positions = pyo.Set(initialize=range(len(self.batch.items)))
        model.nodes = pyo.Set(initialize=range(len(self.batch.items) + 2))
        model.edges = pyo.Set(initialize=[(i, j) for i in model.nodes for j in model.nodes if i != j])

        # Variables
        model.x = pyo.Var(model.edges, domain=pyo.Binary)
        model.z = pyo.Var(model.edges, model.positions, domain=pyo.NonNegativeIntegers)

        # Objective
        model.objective = pyo.Objective(rule=self.total_distance, sense=pyo.minimize)

        # Constraints
        model.entry = pyo.Constraint(model.nodes, rule=self.entry_constraint)
        model.exit = pyo.Constraint(model.nodes, rule=self.exit_constraint)
        model.flow = pyo.Constraint(model.positions, rule=self.flow_constraint)
        model.one_commodity_per_vertex = pyo.Constraint(model.positions, rule=self.one_commodity_per_vertex_constraint)
        model.flow_conservation = pyo.Constraint(model.positions, model.positions, rule=self.flow_conservation_constraint)
        model.flow_arc = pyo.Constraint(model.edges, model.positions, rule=self.flow_arc_constraint)

        return model
    
    def build_solution(self, solution):
        return solution
    

class Routing(Problem):
    def solve(self, batches: list[Batch]):
        """
        Solve multiple TSP instances in parallel. CPU-bound operation.
        """
        routes = []

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self.solve, batch=batch)
                for batch in batches
            ]

            for future in futures:
                routes.append(future.result())

        return routes

    def build_model(self, batches: list[Batch]):
        return self.solve(batches=batches)

