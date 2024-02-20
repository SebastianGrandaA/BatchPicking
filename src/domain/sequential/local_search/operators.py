from random import choice
from typing import Any

from pydantic import BaseModel

from domain.models.solutions import Batch


def validate_move(func):
    def wrapper(self, solution: list[Batch]) -> list[Batch]:
        nb_batches = len(solution)
        new_solution = func(self, solution)
        assert len(new_solution) == nb_batches

        return new_solution

    return wrapper


class Move(BaseModel):
    """Generic move for the local search."""

    routing_method: Any = None

    @validate_move
    def apply(self, _: list[Batch]) -> list[Batch]:
        """Apply the move to the solution and return the new solution."""
        raise NotImplementedError

    def route(self, batch: Batch) -> Batch:
        """Route the batch."""
        routes = self.routing_method("VRP", [batch])
        assert len(routes) == 1

        return routes[0]


class Swap(Move):
    """
    # Swap move

    Select two random orders from two different batches and swap them.
    Both new batches are routed again.
    """

    @validate_move
    def apply(self, solution: list[Batch]) -> list[Batch]:
        source, destination = choice(solution), choice(solution)
        order_source, order_destination = choice(source.orders), choice(
            destination.orders
        )

        source.orders.remove(order_source)
        destination.orders.remove(order_destination)

        source.orders.append(order_destination)
        destination.orders.append(order_source)

        source = self.route(source)
        destination = self.route(destination)

        return solution


class Relocate(Move):
    """
    # Relocate move

    Select a random order from the least loaded batch and relocate it to another randomly selected batch from the p50 least loaded batches.
    The new batch is routed again, whereas the previous batch is just updated without the items of the relocated order.
    Criteria: Prioritize the batches with single orders.
    """

    @validate_move
    def apply(self, solution: list[Batch]) -> list[Batch]:
        solution.sort(key=lambda batch: len(batch.orders))

        # Select the source batch and the order to relocate
        source = solution.pop(0)
        order = choice(source.orders)
        source.orders.remove(order)

        # Select the destination batch and route the new batch
        candidates = solution[: len(solution) // 2]
        destination = choice(candidates)
        solution.remove(destination)
        destination.orders.append(order)
        destination = self.route(destination)

        # Update the solution with both modified batches
        solution.extend([source, destination])

        return solution
