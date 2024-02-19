from os import path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import read_csv

from services.io import IO


class Benchmark(IO):
    """
    ## Numerical analysis
        Benchmark: export table with `feasible, base_cost, objective_cost, improvement` columns for each instance
            (base) SG @ poip-2024 $ python solution_checker.py -p data/A_data_2023-05-27 -s data/A_data_2023-05-27/solution.txt
                Solution is feasible!
                Base cost = 2261211.0
                Objective cost = 1451245.0
                Improvement amount = 35.82%

            (base) SG @ poip-2024 $ python solution_checker.py -p data/A_data_2023-05-22 -s data/A_data_2023-05-22/solution.txt
                Solution is feasible!
                Base cost = 2334211.0
                Objective cost = 1441272.0
                Improvement amount = 38.25%

            !!(base) SG @ poip-2024 $ python3 solution_checker.py -p data/A_data_2023-05-25 -s data/A_data_2023-05-25/solution.txt
                Solution is feasible!
                Base cost = 2024208.0
                Objective cost = 1379935.0
                Improvement amount = 31.83%

            ---

            Their best solution:
                Solution is feasible!
                Base cost = 1667.0
                Objective cost = 928.0
                Improvement amount = 44.33%

            My best solution:
                Solution is feasible!
                Base cost = 1667.0
                Objective cost = 716.0
                Improvement amount = 57.05%

            TODO mencionar en el reporte que mi solicion es mejor que la solution reportada por ellos (ademas que mejor que la solucion base)





        Tests were performed on N instances ...
        Explicacion de metricas generales, improvement promedio por warehouse, etc...
        comparacion de algoritmos, este es n veces mas rapido y da mejor soljucon en el n% de instancias, ....

        Descritive analysis.... of each row!!

        Time benchmark...

        The numerical results were obtained on a Intel Core i9...

    """

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

    def preprocess(self) -> None:
        """If there are repeated instance names, only take the best improvement (others are discarded). Also, if there is no improvement, discard the row."""
        self.results = self.results.sort_values("improvement", ascending=False)
        self.results = self.results.drop_duplicates("instance_name")
        self.results = self.results[self.results["improvement"] > 0]

    def analyze(self) -> None:
        """Analyze the results of the benchmark."""
        self.results = read_csv(path.join(self.output_dir, "benchmark.csv"))
        self.preprocess()
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

        plt.figure(figsize=(10, 6), dpi=300)
        sns.boxplot(
            x="warehouse_name",
            y="improvement",
            data=self.results,
            palette="Set2",
            fliersize=5,
            linewidth=2.5,
            boxprops={"facecolor": "None"},
        )
        plt.title("Improvement by Warehouse", fontsize=14, fontweight="bold")
        plt.xlabel("Warehouse Name", fontsize=12)
        plt.ylabel("Improvement (%)", fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")
        plt.tight_layout()
        plt.savefig(path.join(self.output_dir, "boxplot.png"))
