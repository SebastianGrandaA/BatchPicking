from logging import error, info
from os import makedirs, path

import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
from pandas import DataFrame, concat, read_csv
from pydantic import BaseModel

from services.io import IO
from services.scripts.solution_checker import evaluate

from .instances import Instance, Item, Vehicle, Warehouse

DEFAULT_METRICS = {
    "total_distance": 0,
}


class Problem(BaseModel):
    warehouse: Warehouse
    timeout: int = 100  # seconds

    @property
    def minimum_batches(self) -> int:
        return self.warehouse.minimum_batches

    def is_valid(self, result) -> bool:
        if result.solver.termination_condition == pyo.TerminationCondition.infeasible:
            error(f"GraphPartition | Infeasible model")
            return False

        return (
            result.solver.status == pyo.SolverStatus.ok
            and result.solver.termination_condition == pyo.TerminationCondition.optimal
        )

    def build_model(self, **kwargs):
        raise NotImplementedError

    def solve(self, **kwargs):
        model = self.build_model(**kwargs)
        solver = pyo.SolverFactory("gurobi")
        result = solver.solve(model, tee=True)

        if self.is_valid(result):
            return self.build_solution(model)


class Metrics(BaseModel):
    distance: float = np.nan
    units: int = np.nan
    volume: float = np.nan

    def __str__(self) -> str:
        return f"Metrics(distance={self.distance}, units={self.units}, volume={self.volume})"


class Route(BaseModel):
    sequence: list[Item] = []

    @property
    def nb_positions(self) -> int:
        return len(self.sequence)

    @property
    def position_ids(self) -> list[int]:
        return [item.position_id for item in self.sequence]

    def __str__(self) -> str:
        return f"Route(positions={self.nb_positions}, sequence={self.position_ids})"


class Load(BaseModel):
    """Load of a vehicle."""

    volume: int = np.nan
    nb_items: int = np.nan

    def __str__(self) -> str:
        return f"Load(volume={self.volume}, nb_items={self.nb_items})"


class Batch(Instance):
    route: Route = Route()
    load: Load = Load()
    metrics: Metrics = Metrics()

    @property
    def position_ids(self) -> list[int]:
        return self.route.position_ids

    def is_feasible(self, vehicle: Vehicle) -> bool:
        return (
            self.total_volume <= vehicle.max_volume
            and self.total_nb_orders <= vehicle.max_nb_orders
        )

    def __str__(self) -> str:
        return f"Batch(order_ids={self.order_ids}, position_ids={self.position_ids})"

    def to_txt(self, id: str) -> str:
        """
        Parse the batch in three lines of text.
        The first line contains the number of supports and positions.
        The second line contains the supports in any order.
        The third line contains the positions in the order of visitation.
        """
        order_ids = [str(order_id) for order_id in self.order_ids]
        position_ids = []
        for position_id in self.position_ids:
            if str(position_id) not in position_ids:
                position_ids.append(str(position_id))

        return (
            f"{id} {len(self.orders)} {len(position_ids)}\n"
            + " ".join(order_ids)
            + "\n"
            + " ".join(position_ids)
            + "\n"
        )

    def save_map(self, path: str) -> None:
        """Plot the route in a 2D plane and draw the route sequence."""
        x = [item.position.x for item in self.route.sequence]
        y = [item.position.y for item in self.route.sequence]

        plt.plot(x, y, "bo-")
        plt.plot(x[0], y[0], "go")
        plt.plot(x[-1], y[-1], "ro")
        plt.legend(["Route", "Start", "End"])
        plt.savefig(path)
        plt.close()


class Solution(IO):
    batches: list[Batch]

    def __str__(self) -> str:
        return f"Solution(routes={[str(batch) for batch in self.batches]})"

    def to_txt(self) -> str:
        return f"{len(self.batches)}\n" + "".join(
            batch.to_txt(id=str(idx)) for idx, batch in enumerate(self.batches)
        )

    def get_stats(self) -> DataFrame:
        """Return a DataFrame with the stats of the solution."""
        input_path = path.join(
            self.directory,
            "data",
            self.instance_name,
        )
        solution_file = path.join(
            self.directory,
            "outputs",
            self.instance_name,
            "solution.txt",
        )

        stats = evaluate(input_path, solution_file)

        return DataFrame(stats, index=[0])

    def save(self):
        """Save the solution in a text file and the map of each batch."""
        dir = path.join(
            self.directory,
            "outputs",
            self.instance_name,
        )

        if not path.exists(dir):
            makedirs(dir)

        # Save the solution in a text file
        with open(path.join(dir, f"solution.txt"), "w") as file:
            file.write(self.to_txt())

        # Save the map of each batch
        for idx, batch in enumerate(self.batches):
            batch.save_map(path.join(dir, f"batch_{idx}.png"))

        # Save the stats of the solution
        benchmark_file = path.join(dir, "..", "..", "benchmark.csv")

        if path.exists(benchmark_file):
            file = read_csv(benchmark_file)
            file = concat([file, self.get_stats()], ignore_index=True)

        else:
            file = self.get_stats()

        file.to_csv(benchmark_file, index=False)

        info(f"Solution saved | Instance {self.instance_name}")
