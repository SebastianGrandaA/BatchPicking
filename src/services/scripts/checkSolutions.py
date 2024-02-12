import sys


def handleSolutionError(fail_fast: bool, errorMessage: str):
    print("SolutionError: ", errorMessage)
    if fail_fast:
        print("Stopping program!")
        sys.exit(1)


def checkIdsCoherenceOfSolutionWithSupportList(
    batches: list[dict], supports: list[dict], fail_fast: bool = True
) -> bool:
    supportIds = sorted([s["id"] for s in supports])

    batchSupportIds = []
    for batch in batches:
        batchSupportIds += batch["supportIds"]
    batchSupportIds = sorted(batchSupportIds)

    if supportIds != batchSupportIds:
        handleSolutionError(
            fail_fast,
            f"some support ids in batches are redundant, invalid, or missing in comparison to supportList",
        )
        return False
    return True


def checkPositionCoherenceOfSolutionWithSupportList(
    batches: list[dict], supports: list[dict], fail_fast: bool = True
) -> bool:
    flag = True
    for batch in batches:
        batchPositions = set(batch["positions"])
        batchSupportIds = batch["supportIds"]

        supportPositions = []
        for support in filter(lambda x: (x["id"] in batchSupportIds), supports):
            supportPositions.extend(support["positions"])
        supportPositions = set(supportPositions)

        if batchPositions != supportPositions:
            handleSolutionError(
                fail_fast,
                f"batch {batch['id']}'s positions {batchPositions} do not match its support's positions {supportPositions}",
            )
            flag = False
    return flag


def checkCoherenceOfSolutionWithConstraints(
    batches: list[dict],
    supports: list[dict],
    maxBatchNbSupports: int,
    maxBatchVolume: int,
    fail_fast: bool = True,
) -> bool:
    flag = True
    for batch in batches:
        batchSupportIds = batch["supportIds"]

        if len(batchSupportIds) > maxBatchNbSupports:
            handleSolutionError(
                fail_fast,
                f"batch {batch['id']} has {len(batchSupportIds)} supports which exceeds the maximum amount {maxBatchNbSupports}",
            )
            flag = False

        batchVolume = 0
        for support in filter(lambda x: (x["id"] in batchSupportIds), supports):
            batchVolume += support["volume"]

        if batchVolume > maxBatchVolume:
            handleSolutionError(
                fail_fast,
                f"batch {batch['id']} has a total volume of {batchVolume} which exceeds the maximum volume {maxBatchVolume}",
            )
            flag = False

    return flag


def checkCoherenceOfSolutionWithAdjMatrix(
    batches: list[dict], adjMatrix: list[list], fail_fast: bool = True
) -> bool:
    flag = True
    nbPositions = len(adjMatrix)
    for batch in batches:
        batchPositions = batch["positions"]
        if batchPositions[0] != 0:
            handleSolutionError(
                fail_fast, f"batch {batch['id']} does not start at position 0"
            )
            flag = False
        if batchPositions[-1] != nbPositions - 1:
            handleSolutionError(
                fail_fast,
                f"batch {batch['id']} does not end at the last position {nbPositions-1}",
            )
            flag = False
        if max(batchPositions) > nbPositions - 1 or min(batchPositions) < 0:
            handleSolutionError(fail_fast, f"batch {batch['id']} has invalid positions")
            flag = False
        if len(set(batchPositions)) != len(batchPositions):
            print(f"WARNING: batch {batch['id']} has redundant positions")
    return flag


def checkSolutionCoherence(
    batches: list[dict],
    supports: list[dict],
    adjMatrix: list[list],
    maxBatchNbSupports: int,
    maxBatchVolume: int,
    fail_fast: bool = True,
) -> bool:
    flag = checkIdsCoherenceOfSolutionWithSupportList(batches, supports, fail_fast)
    flag = (
        checkPositionCoherenceOfSolutionWithSupportList(batches, supports, fail_fast)
        and flag
    )
    flag = (
        checkCoherenceOfSolutionWithConstraints(
            batches, supports, maxBatchNbSupports, maxBatchVolume, fail_fast
        )
        and flag
    )
    flag = checkCoherenceOfSolutionWithAdjMatrix(batches, adjMatrix, fail_fast) and flag

    return flag


def computeBatchCosts(batches: list[dict], adjMatrix: list[list[float]]) -> float:
    totalCost = 0
    for batch in batches:
        batchCost = 0
        positions = batch["positions"]
        for i in range(len(positions) - 1):
            batchCost += adjMatrix[positions[i]][positions[i + 1]]
        totalCost += batchCost
    return totalCost
