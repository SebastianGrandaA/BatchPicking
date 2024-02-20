from collections import defaultdict
from itertools import combinations
from logging import error
from typing import Any

import pyomo.environ as pyo
from gurobipy import GRB, Model, quicksum

from domain.models.routing import Routing
from domain.models.solutions import Batch, Route


class TSPMultiCommodityFlow(Routing):
    """
    # TSP multi-commodity flow

    To eliminate sub-tours, the formulation proposed by Claus (1984) uses multi-commodity flows.
    """

    def total_distance(self, batch: Batch, model: pyo.ConcreteModel):
        matrix = self.warehouse.distances.matrix

        return sum(
            matrix[batch.items[i].id, batch.items[j].id]
            if i <= batch.depot_ids[1] and j <= batch.depot_ids[1]
            else 0 * model.x[i, j]
            for (i, j) in model.edges
        )

    def entry_constraint(self, model: pyo.ConcreteModel, i: int):
        return sum(model.x[i, j] for j in model.nodes if i != j) == 1

    def exit_constraint(self, model: pyo.ConcreteModel, j: int):
        return sum(model.x[i, j] for i in model.nodes if i != j) == 1

    def flow_constraint(self, batch: Batch, model: pyo.ConcreteModel, l: int):
        return (
            sum(model.z[0, j, l] for j in model.positions if j not in batch.depot_ids)
            - sum(model.z[j, 0, l] for j in model.positions if j not in batch.depot_ids)
            == -1
        )

    def one_commodity_per_vertex_constraint(
        self, batch: Batch, model: pyo.ConcreteModel, i: int
    ):
        if i >= batch.depot_ids[1]:
            return pyo.Constraint.Skip

        return (
            sum(model.z[i, j, i] for j in model.nodes if j not in batch.depot_ids)
            - sum(model.z[j, i, i] for j in model.nodes if j not in batch.depot_ids)
            == 1
        )

    def flow_conservation_constraint(self, model: pyo.ConcreteModel, i: int, l: int):
        if i == l:
            return pyo.Constraint.Skip

        return (
            sum(model.z[i, j, l] for j in model.nodes)
            - sum(model.z[j, i, l] for j in model.nodes)
            == 0
        )

    def flow_arc_constraint(self, model: pyo.ConcreteModel, i: int, j: int, l: int):
        return model.z[i, j, l] <= model.x[i, j]

    def build_model(self, batch: Batch):
        model = pyo.ConcreteModel()

        # Parameters
        model.positions = pyo.Set(initialize=range(len(batch.items)))
        model.nodes = pyo.Set(initialize=range(len(batch.items) + 2))
        model.edges = pyo.Set(
            initialize=[(i, j) for i in model.nodes for j in model.nodes if i != j]
        )

        # Variables
        model.x = pyo.Var(model.edges, domain=pyo.Binary)
        model.z = pyo.Var(model.edges, model.positions, domain=pyo.NonNegativeIntegers)

        # Objective
        model.objective = pyo.Objective(rule=self.total_distance, sense=pyo.minimize)

        # Constraints
        model.entry = pyo.Constraint(model.nodes, rule=self.entry_constraint)
        model.exit = pyo.Constraint(model.nodes, rule=self.exit_constraint)
        model.flow = pyo.Constraint(model.positions, rule=self.flow_constraint)
        model.one_commodity_per_vertex = pyo.Constraint(
            model.positions, rule=self.one_commodity_per_vertex_constraint
        )
        model.flow_conservation = pyo.Constraint(
            model.positions, model.positions, rule=self.flow_conservation_constraint
        )
        model.flow_arc = pyo.Constraint(
            model.edges, model.positions, rule=self.flow_arc_constraint
        )

        return model

    def build_solution(self, solution):
        return solution


class TSPBase(Routing):
    """[Reference](https://www.gurobi.com/jupyter_models/traveling-salesman/)."""

    def build_graph(self, batch: Batch) -> None:
        self.graph = {idx: item for idx, item in enumerate(batch.items)}

    def build_matrix(self) -> dict[tuple[int, int], float]:
        all_distances = {
            (id_i, id_j): self.warehouse.distance(i, j)
            for id_i, i in self.node_items
            for id_j, j in self.node_items
        }

        return {
            (i, j): all_distances[(i, j)] for i, j in combinations(self.node_ids, 2)
        }

    def shortest_tour(self, edges: list[tuple[int, int]]) -> list[int]:
        node_neighbors = defaultdict(list)
        for i, j in edges:
            node_neighbors[i].append(j)

        assert all(len(neighbors) == 2 for neighbors in node_neighbors.values())

        unvisited = set(node_neighbors)
        shortest = None

        while unvisited:
            cycle = []
            neighbors = list(unvisited)

            while neighbors:
                current = neighbors.pop()
                cycle.append(current)
                unvisited.remove(current)
                neighbors = [j for j in node_neighbors[current] if j in unvisited]

            if shortest is None or len(cycle) < len(shortest):
                shortest = cycle

        assert shortest is not None

        return shortest

    def selected_edges(self, solution: Any) -> list[tuple[int, int]]:
        return [(i, j) for (i, j), v in solution.items() if v > 0.5]

    def subtour_elimination(self, model: Model, where: Any):
        if where == GRB.Callback.MIPSOL:
            try:
                solution = model.cbGetSolution(model._vars)
                edges = self.selected_edges(solution)
                tour = self.shortest_tour(edges)

                if len(tour) < len(self.node_ids):
                    model.cbLazy(
                        quicksum(model._vars[i, j] for i, j in combinations(tour, 2))
                        <= len(tour) - 1
                    )

            except Exception as e:
                error(f"SubtourEliminationCallback: {e}")
                model.terminate()

    def build_solution(self, batch: Batch, solution: Any, value: float) -> Batch:
        edges = [(i, j) for (i, j), v in solution.items() if v.X > 0.5]
        tour = self.shortest_tour(edges)

        assert len(tour) == len(self.node_ids)

        route = Route(sequence=[self.graph[i] for i in tour], distance=value)

        return Batch(orders=batch.orders, route=route)

    def route_batch(self, batch: Batch) -> Batch:
        model = Model()
        self.build_graph(batch)
        matrix = self.build_matrix()

        # Variables
        is_edge = model.addVars(
            matrix.keys(), obj=matrix, vtype=GRB.BINARY, name="is_edge"
        )
        is_edge.update({(j, i): v for (i, j), v in is_edge.items()})

        # Constraints
        for i in self.node_ids:
            model.addConstr(
                quicksum(is_edge[i, j] for j in self.node_ids if i != j) == 2
            )

        model._vars = is_edge
        model.Params.lazyConstraints = 1

        model.optimize(lambda model, where: self.subtour_elimination(model, where))
        value = 0

        return self.build_solution(batch, is_edge, value)
