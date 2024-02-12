import sys, os, argparse

from .openInstances import openSupportList, openConstraints, openAdjacencyMatrix
from .checkProblems import checkProblemCoherence, computeBaseSupportCosts


def evaluate(problemFolder: str, stopReadAtFirstError: bool = True, stopCheckAtFirstError: bool = False) -> float:
    adjMatrixFile = os.path.join(problemFolder, "adjacencyMatrix.txt")
    supportListFile = os.path.join(problemFolder, "supportList.txt")
    constraintsFile = os.path.join(problemFolder, "constraints.txt")

    supports = openSupportList(supportListFile, stopReadAtFirstError)
    adjMatrix = openAdjacencyMatrix(adjMatrixFile, stopReadAtFirstError)
    maxBatchNbSupports, maxBatchVolume = openConstraints(constraintsFile, stopReadAtFirstError)
    
    if not (supports and adjMatrix and maxBatchNbSupports and maxBatchVolume):
        print("Problem files contain incoherences! Cannot verify problem")
        sys.exit(1)
    elif not checkProblemCoherence(supports, adjMatrix, maxBatchNbSupports, maxBatchVolume, stopCheckAtFirstError):
        print("Problem is infeasible!")
        sys.exit(1)

    baseCost = computeBaseSupportCosts(supports, adjMatrix)
    print("Problem is feasible!")
    print(f"Base cost = {baseCost}")

    return baseCost

if __name__ == '__main__':
    """
    Run as:
        python problem_checker.py -p data/A_data_2023-05-27
    """
    parser = argparse.ArgumentParser(description='Problem Checker and evaluator for POIP 2024. Problem folder is mandatory.')

    parser.add_argument('-p', '--problem_folder', type=str, help='Path to the problem folder', required=True)
    read_options = parser.add_mutually_exclusive_group()
    read_options.add_argument('-ffr', '--fail_fast_read', action='store_true', help='Stop program at first file reading error found -> DEFAULT')
    read_options.add_argument('-flr', '--fail_last_read', action='store_true', help='Stop program after all file reading errors are found')
    check_options = parser.add_mutually_exclusive_group()
    check_options.add_argument('-ffc', '--fail_fast_check', action='store_true', help='Stop program at first problem-checking error found')
    check_options.add_argument('-flc', '--fail_last_check', action='store_true', help='Stop program after all problem-checking errors are found -> DEFAULT')

    args = parser.parse_args()

    stopReadAtFirstError = False if args.fail_last_read else True
    stopCheckAtFirstError = True if args.fail_fast_check else False

    problemFolder = args.problem_folder

    evaluate(problemFolder, stopReadAtFirstError, stopCheckAtFirstError)
