from os import path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import read_csv

from domain.BatchPicking import BatchPicking
from services.io import IO

INVALID_INSTANCES = [
    "warehouse_D/data_2023-01-31_20",
]


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
            if instance_name in INVALID_INSTANCES:
                continue

            BatchPicking.optimize(self.method, instance_name, self.timeout)

    def preprocess(self) -> None:
        """If there are repeated instance names, only take the best improvement (others are discarded). Also, if there is no improvement, discard the row."""
        self.results = self.results.sort_values("improvement", ascending=False)
        self.results = self.results.drop_duplicates(
            subset=["instance_name", "method"], keep="first"
        )
        self.results = self.results[self.results["improvement"] > 0]

    def analyze(self) -> None:
        """Analyze the results of the benchmark."""
        self.results = read_csv(path.join(self.output_dir, "benchmark.csv"))
        self.preprocess()
        self.save_stats()
        self.save_boxplot()

        self.results.to_csv(
            path.join(self.output_dir, "benchmark_preprocessed.csv"), index=False
        )

    def save_stats(self) -> None:
        """Save the stats of the benchmark."""
        stats = (
            self.results["improvement"]
            .agg(["mean", "median", "std", "min", "max"])
            .round(2)
            .reset_index()
        )
        stats.columns = ["Statistic", "Value"]
        batching_rate = (
            self.results["nb_orders"] / self.results["nb_batches"] * 100
        ).round(2)
        metrics = {
            "mean batching rate (%)": batching_rate.mean().round(2),
            "p90 batching rate (%)": batching_rate.quantile(0.9).round(2),
            "mean improvement (%)": stats.loc[0, "Value"],
            "p90 improvement (%)": self.results["improvement"].quantile(0.9).round(2),
            "mean execution time (s)": self.results["execution_time"].mean().round(2),
            "p90 execution time (s)": self.results["execution_time"]
            .quantile(0.9)
            .round(2),
        }
        filename = path.join(self.output_dir, "metrics.csv")

        with open(filename, "w") as file:
            file.write("Metric,Value\n")
            for metric, value in metrics.items():
                file.write(f"{metric},{value}\n")

    def save_boxplot(self) -> None:
        """Save the boxplot of the benchmark."""
        data = {
            "y": "improvement",
            "data": self.results,
            "palette": "Set2",
            "fliersize": 5,
            "linewidth": 2.5,
            "boxprops": {"facecolor": "None"},
        }

        # Boxplot of the improvement by warehouse
        self.results["warehouse_name"] = (
            self.results["instance_name"].str.split("/").str[-2]
        )

        plt.figure(figsize=(10, 6), dpi=300)
        sns.boxplot(
            x="warehouse_name",
            **data,
        )
        plt.title("Improvement by Warehouse", fontsize=14, fontweight="bold")
        plt.xlabel("Warehouse", fontsize=12)
        plt.ylabel("Improvement (%)", fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")
        plt.tight_layout()
        plt.savefig(path.join(self.output_dir, "boxplot_warehouse.png"))

        # Boxplot of the improvement by method
        plt.figure(figsize=(10, 6), dpi=300)
        sns.boxplot(
            x="method",
            **data,
        )
        plt.title("Improvement by Method", fontsize=14, fontweight="bold")
        plt.xlabel("Method", fontsize=12)
        plt.ylabel("Improvement (%)", fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")
        plt.tight_layout()
        plt.savefig(path.join(self.output_dir, "boxplot_method.png"))

        # Boxplot of the execution time by method
        plt.figure(figsize=(10, 6), dpi=300)
        sns.boxplot(
            **{
                **data,
                "y": "execution_time",
                "x": "method",
            }
        )
        plt.title("Execution Time by Method", fontsize=14, fontweight="bold")
        plt.xlabel("Method", fontsize=12)
        plt.ylabel("Execution Time (s)", fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")
        plt.tight_layout()
        plt.savefig(path.join(self.output_dir, "boxplot_execution_time.png"))

        plt.close("all")
