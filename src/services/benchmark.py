from os import path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import read_csv

from domain.BatchPicking import optimize
from services.io import IO


class Benchmark(IO):
    instance_names: list[str]
    method: str
    timeout: int
    results: Any = None

    @property
    def output_dir(self) -> str:
        return path.join(self.directory, "outputs")

    def execute(self) -> None:
        """Execute the optimization process for each instance."""
        for instance_name in self.instance_names:
            optimize(self.method, instance_name, self.timeout)

    def analyze(self) -> None:
        """Analyze the results of the benchmark."""
        self.results = read_csv(path.join(self.output_dir, "benchmark.csv"))
        self.save_stats()
        self.save_boxplot()

    def save_stats(self) -> None:
        """Save the stats of the benchmark."""
        stats = (
            self.results["improvement"]
            .agg(["mean", "median", "std", "min", "max"])
            .round(2)
            .reset_index()
        )
        stats.columns = ["Statistic", "Value"]
        filename = path.join(self.output_dir, "improvement_stats.csv")
        stats.to_csv(filename, index=False)

    def save_boxplot(self) -> None:
        """Save the boxplot of the benchmark."""
        self.results["warehouse_name"] = (
            self.results["instance_name"].str.split("/").str[-2]
        )

        plt.figure(figsize=(10, 6))
        sns.boxplot(x="warehouse_name", y="improvement", data=self.results)
        plt.title("Improvement by Warehouse")
        plt.xlabel("Warehouse Name")
        plt.ylabel("Improvement")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(path.join(self.output_dir, "boxplot.png"))
