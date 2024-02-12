import sys

def handleProblemError(fail_fast:bool, errorMessage:str):
    print("ProblemError:", errorMessage)
    if fail_fast:
        print("Stopping program!")
        sys.exit(1)


def checkCoherenceOfSupportListWithAdjMatrix(supports:list[dict], adjMatrix:list[list], fail_fast:bool=True) -> bool:
    flag = True
    nbPositions = len(adjMatrix)
    for support in supports:
        supportPositions = support['positions']
        if supportPositions[0] != 0:
            handleProblemError(fail_fast, f"support {support['id']} does not start at position 0")
            flag = False
        if supportPositions[-1] != nbPositions-1:
            handleProblemError(fail_fast, f"support {support['id']} does not end at the last position {nbPositions-1}")
            flag = False
        if max(supportPositions) > nbPositions-1 or min(supportPositions) < 0:
            handleProblemError(fail_fast, f"support {support['id']} has invalid positions")
            flag = False
        if len(set(supportPositions)) != len(supportPositions):
            handleProblemError(fail_fast, f"WARNING: support {support['id']} has redundant positions")
    return flag

def checkCoherenceOfSupportListWithConstraints(supports:list[dict], maxBatchVolume:int, fail_fast:bool=True) -> bool:
    flag = True
    for support in supports:
        if support['volume'] > maxBatchVolume:
            handleProblemError(fail_fast, f"support {support['id']} has a volume of {support['volume']} which exceeds the maximum volume {maxBatchVolume}")
            flag = False

    return flag

def checkProblemCoherence(supports:list[dict], adjMatrix:list[list], maxBatchNbSupports:int, maxBatchVolume:int, fail_fast:bool=True) -> bool:
    flag = checkCoherenceOfSupportListWithConstraints(supports, maxBatchVolume, fail_fast)
    flag = checkCoherenceOfSupportListWithAdjMatrix(supports, adjMatrix, fail_fast) and flag

    return flag


def computeBaseSupportCosts(supports:list[dict], adjMatrix:list[list]) -> float:
    totalCost = 0
    for support in supports:
        batchCost = 0
        positions = support['positions']
        for i in range(len(positions)-1):
            batchCost += adjMatrix[positions[i]][positions[i+1]]
        totalCost += batchCost
    return totalCost
