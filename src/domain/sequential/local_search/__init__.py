from logging import debug, info
from random import choice
from typing import Any

from domain.models.solutions import Batch, Problem
from domain.sequential.local_search.operators import Move, Relocate, Swap
from domain.sequential.local_search.search import SimmulatedAnnealing, TabuSearch

LS_MAX_ITERATIONS = 10


class LocalSearch(Problem):
    """
    ## Local search

    The main motivation for the local search is to exploit the local structure of the problem to find better solutions.
    Particularly, due that the set of items for each batch remains fixed in the initial solution, natural neighborhoods are to swap and to relocate orders between batches.
    At each iteration, until the stopping criterion is met, a move operator is randomly selected, between the swap and the relocation operators, and the first-improving solution is selected from the neighborhood based on Tabu Search and Simulated Annealing principles.
    To avoid cycling through the same solutions, the Tabu Search memory is used to store properties of the solutions that are forbidden to be selected again.
    This memory is adjusted during the search process to force the algorithm to explore different regions of the search space (diversification).
    Furthermore, non-improving solutions might be accepted to escape from local optima using the Metropolis criterion.
    The best solution found is returned as the final solution after a maximum number of iterations.
    """

    current_solution: list[Batch]
    routing_method: Any
    operators: list[Move] = []
    strategies: dict[str, Any] = {}

    @property
    def tabu_search(self) -> TabuSearch:
        return self.strategies["tabu_search"]

    @property
    def simulated_annealing(self) -> SimmulatedAnnealing:
        return self.strategies["simulated_annealing"]

    def to_diversify(self, count: int) -> bool:
        """Diversify the search at the half of the iterations."""
        return count > LS_MAX_ITERATIONS // 2

    def initialize(self):
        operator_params = {
            "routing_method": self.routing_method,
        }
        self.operators = [Relocate(**operator_params), Swap(**operator_params)]
        self.strategies = {
            "simulated_annealing": SimmulatedAnnealing(),
            "tabu_search": TabuSearch(),
        }

    def should_continue(self, count: int) -> bool:
        return count < LS_MAX_ITERATIONS

    def compute_distance(self, solution: list[Batch]) -> float:
        return sum(batch.metrics.distance for batch in solution)

    def should_accept(self, new_solution: list[Batch], count: int) -> bool:
        """
        Acceptance criterion

        Integrate tabu search memory and the simulated annealing strategies to accept or reject the new solution.
        The solution is accepted if it is not tabu and it is better than the current solution (based on the Metropolis criterion).
        """
        if self.tabu_search.is_tabu(new_solution):
            return False

        self.tabu_search.update_memory(
            new_solution, to_diversify=self.to_diversify(count)
        )

        improvement = self.compute_distance(
            self.current_solution
        ) - self.compute_distance(new_solution)
        is_better = self.simulated_annealing.metropolis_criterion(improvement)

        self.simulated_annealing.update_temperature()

        return is_better

    def select_operator(self) -> Move:
        return choice(self.operators)

    def solve(self) -> list[Batch]:
        """Local search algorithm to improve the initial solution."""
        return self.current_solution  # TODO implement local search
        count = 0
        self.initialize()
        info(f"Starting local search with {LS_MAX_ITERATIONS} iterations.")
        best_solution = self.current_solution

        while self.should_continue(count):
            count += 1
            operator = self.select_operator()
            new_solution = operator.apply(self.current_solution)

            if self.should_accept(new_solution, count):
                debug(
                    f"Local search | Iteration {count} | Accepted new solution with distance {self.compute_distance(new_solution)}."
                )
                self.current_solution = new_solution

        info(f"Local search finished after {count} iterations.")

        final_solution = (
            self.current_solution
            if self.compute_distance(self.current_solution)
            < self.compute_distance(best_solution)
            else best_solution
        )

        return final_solution
