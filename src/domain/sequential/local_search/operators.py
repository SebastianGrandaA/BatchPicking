from random import choice
from typing import Any

from pydantic import BaseModel

from domain.models.solutions import Batch

"""
TODO due that the clustered are formed based on the , 
    a criteria to insert an order into another batch is checking if it fits...

    TODO movement:
        Destroy-and-relocate
            Destroy a batch (starting fro single)
            Insert the orders into the p50 least loaded batches

Perturbate the sequence intra batch is not necesary because the subprolems are mostly solved to optimality
"""


def validate_move(func):
    def wrapper(self, solution: list[Batch]) -> list[Batch]:
        nb_batches = len(solution)
        new_solution = func(self, solution)
        assert len(new_solution) == nb_batches

        return new_solution

    return wrapper


class Move(BaseModel):
    routing_method: Any = None

    """Generic move for the local search."""

    @validate_move
    def apply(self, _: list[Batch]) -> list[Batch]:
        """Apply the move to the solution and return the new solution."""
        raise NotImplementedError

    def route(self, batch: Batch) -> Batch:
        """Route the batch."""
        routes = self.routing_method("VRP", [batch])
        assert len(routes) == 1

        return routes[0]


class Relocate(Move):
    """
    Select a random order from the least loaded batch and relocate it to another randomly selected batch from the p50 least loaded batches.
    The new batch is routed again, whereas the previous batch is just updated without the items of the relocated order.

    Criteria: Prioritize the batches with single orders.
    """

    @validate_move
    def apply(self, solution: list[Batch]) -> list[Batch]:
        """
        First, sort the batches by the number of orders and select the source batch, which is the least loaded batch.
        From this batch, select a random order and relocate it to another batch.
        The destination batch is selected from the 50% least loaded batches.
        The new batch is routed again, whereas the previous batch is just updated without the items of the relocated order.
        """
        solution.sort(key=lambda batch: len(batch.orders))

        # Select the source batch and the order to relocate
        source = solution.pop(0)
        order = choice(source.orders)
        # TODO is this sufficient? in the routing problem it updates the items, then we only care about orders? (remove)
        # TODO also the routing proces update batch metrics??
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
