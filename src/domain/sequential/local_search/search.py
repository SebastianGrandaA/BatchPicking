import numpy as np
from pydantic import BaseModel

from domain.models.solutions import Batch

SA_INITAL_TEMPERATURE = 100.0
SA_COOLING_RATE = 0.003
TS_MEMORY_SIZE = 10


class SimmulatedAnnealing(BaseModel):
    temperature: float = SA_INITAL_TEMPERATURE
    cooling_rate: float = SA_COOLING_RATE

    def update_temperature(self) -> None:
        self.temperature *= 1 - self.cooling_rate

    def acceptance_probability(self, improvement: float) -> float:
        return np.exp(-improvement / self.temperature)

    def metropolis_criterion(self, improvement: float) -> bool:
        """Accept the new solution if it is better than the current solution or using the Metropolis criterion."""
        return improvement > 0 or np.random.rand() < self.acceptance_probability(
            improvement
        )


class TabuSearch(BaseModel):
    memory_size: int = TS_MEMORY_SIZE
    memory: list[dict] = []

    def update_memory(self, solution: list[Batch], to_diversify: bool):
        """
        Update the tabu memory with the representation of the new solution.
        A solution is represented by a dictionary that maps the batch id to the list of order ids.
        An adaptative memory is implemented to control the size of the memory when the search is stuck in a local minimum.
        When diversifying, the memory is reduced to the half of its size.
        """
        self.memory.append({batch.id: batch.orders for batch in solution})

        if to_diversify:
            self.memory = self.memory[-self.memory_size // 2 :]

    def is_tabu(self, solution: list[Batch]) -> bool:
        """A movement is tabu if all batches have the same order ids."""
        return any(
            all(batch.orders == tabu[batch.id] for batch in solution)
            for tabu in self.memory
        )
