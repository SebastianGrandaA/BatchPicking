import argparse
import os
import sys
from logging import info, warning

from .checkProblems import checkProblemCoherence, computeBaseSupportCosts
from .checkSolutions import checkSolutionCoherence, computeBatchCosts
from .openInstances import (
    openAdjacencyMatrix,
    openConstraints,
    openSolution,
    openSupportList,
)


def evaluate(
    problemFolder: str,
    solutionFile: str,
    stopReadAtFirstError: bool = True,
    stopCheckAtFirstError: bool = False,
) -> dict:
    adjMatrixFile = os.path.join(problemFolder, "adjacencyMatrix.txt")
    supportListFile = os.path.join(problemFolder, "supportList.txt")
    constraintsFile = os.path.join(problemFolder, "constraints.txt")

    supports = openSupportList(supportListFile, stopReadAtFirstError)
    adjMatrix = openAdjacencyMatrix(adjMatrixFile, stopReadAtFirstError)
    maxBatchNbSupports, maxBatchVolume = openConstraints(
        constraintsFile, stopReadAtFirstError
    )

    if not (supports and adjMatrix and maxBatchNbSupports and maxBatchVolume):
        warning("Problem files contain incoherences! Cannot verify problem.")
        sys.exit(1)
    elif not checkProblemCoherence(
        supports, adjMatrix, maxBatchNbSupports, maxBatchVolume, stopCheckAtFirstError
    ):
        warning("Problem is infeasible! Cannot verify solution.")
        sys.exit(1)

    batches = openSolution(solutionFile, stopReadAtFirstError)

    if not (batches):
        warning("Solution file contains incoherences! Cannot verify solution.")
        sys.exit(1)
    elif not checkSolutionCoherence(
        batches,
        supports,
        adjMatrix,
        maxBatchNbSupports,
        maxBatchVolume,
        stopCheckAtFirstError,
    ):
        warning("Solution is infeasible!")
        sys.exit(1)

    baseCost = computeBaseSupportCosts(supports, adjMatrix)
    solutionCost = computeBatchCosts(batches, adjMatrix)
    gain = round(100 * (baseCost - solutionCost) / baseCost, 2)
    instance_name = problemFolder.split("/")[-1]
    info(
        f"Instance {instance_name} | Solution is feasible | Base cost {baseCost} | Objective cost {solutionCost} | Improvement {gain}%"
    )

    return {
        "instance_name": instance_name,
        "is_feasible": True,
        "base_cost": baseCost,
        "objective_cost": solutionCost,
        "improvement": gain,
    }


if __name__ == "__main__":
    """
    Run as:
        python solution_checker.py -p data/A_data_2023-05-27 -s data/A_data_2023-05-27/solution.txt
    """
    parser = argparse.ArgumentParser(
        description="Solution Checker and evaluator for POIP 2024. Problem folder and solution file are mandatory."
    )

    parser.add_argument(
        "-p",
        "--problem_folder",
        type=str,
        help="Path to the problem folder",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--solution_file",
        type=str,
        help="Path to the solution file",
        required=True,
    )
    read_options = parser.add_mutually_exclusive_group()
    read_options.add_argument(
        "-ffr",
        "--fail_fast_read",
        action="store_true",
        help="Stop program at first file reading error found -> DEFAULT",
    )
    read_options.add_argument(
        "-flr",
        "--fail_last_read",
        action="store_true",
        help="Stop program after all file reading errors are found",
    )
    check_options = parser.add_mutually_exclusive_group()
    check_options.add_argument(
        "-ffc",
        "--fail_fast_check",
        action="store_true",
        help="Stop program at first solution-checking error found",
    )
    check_options.add_argument(
        "-flc",
        "--fail_last_check",
        action="store_true",
        help="Stop program after all soluction-checking errors are found -> DEFAULT",
    )

    args = parser.parse_args()

    stopReadAtFirstError = False if args.fail_last_read else True
    stopCheckAtFirstError = True if args.fail_fast_check else False

    problemFolder = args.problem_folder
    solutionFile = args.solution_file

    evaluate(problemFolder, solutionFile, stopReadAtFirstError, stopCheckAtFirstError)
