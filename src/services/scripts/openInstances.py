import os
import sys


def handleReadError(fail_fast: bool, file: str, line: int, errorMessage: str):
    line_str = ":" + str(line) if line != -1 else ""
    print("ReadError", "'" + file + line_str + "'", ":", errorMessage)
    if fail_fast:
        print("Stopping program!")
        sys.exit(1)


def openSolution(solutionFile: str, fail_fast: bool = True):
    flag = True
    line = -1
    batches = []
    try:
        with open(solutionFile, "r") as f:
            line = 1
            nbBatches = int(f.readline().strip())
            for b in range(nbBatches):
                line = 1 + b * 3 + 1
                infos = [int(i) for i in f.readline().strip().split(" ")]
                if len(infos) != 3:
                    handleReadError(
                        fail_fast,
                        solutionFile,
                        line,
                        f"syntax does not fit '<id_{infos[0]}> <nS_{infos[0]}> <nP_{infos[0]}>\\n' paradigm",
                    )
                    flag = False

                line = 1 + b * 3 + 2
                supportIds = [int(i) for i in f.readline().strip().split(" ")]
                if len(supportIds) != infos[1]:
                    handleReadError(
                        fail_fast,
                        solutionFile,
                        line,
                        f"nb supports={len(supportIds)} is different from nS_{infos[0]}={infos[1]} announced at line {1+b*3+1}",
                    )
                    flag = False

                line = 1 + b * 3 + 3
                positions = [int(i) for i in f.readline().strip().split(" ")]
                if len(positions) != infos[2]:
                    handleReadError(
                        fail_fast,
                        solutionFile,
                        line,
                        f"nb positions={len(positions)} is different from nP_{infos[0]}={infos[2]} announced at line {1+b*3+1}",
                    )
                    flag = False

                batches.append(
                    {"id": infos[0], "supportIds": supportIds, "positions": positions}
                )
    except Exception as e:
        handleReadError(fail_fast, solutionFile, line, f"{e}")
        flag = False

    if not flag:
        return False
    return batches


def openSupportList(supportListFile: str, fail_fast: bool = True):
    flag = True
    line = -1
    supportList = []
    try:
        with open(supportListFile, "r") as f:
            line = 1
            nbSupport = int(f.readline().strip())
            for s in range(nbSupport):
                line = 1 + s * 2 + 1
                infos = [int(i) for i in f.readline().strip().split(" ")]
                if len(infos) != 3:
                    handleReadError(
                        fail_fast,
                        supportListFile,
                        line,
                        f"syntax does not fit '<id_{infos[0]}> <vol_{infos[0]}> <nP_{infos[0]}>\\n' paradigm",
                    )
                    flag = False

                line = 1 + s * 2 + 2
                positions = [int(i) for i in f.readline().strip().split(" ")]
                if len(positions) != infos[2]:
                    handleReadError(
                        fail_fast,
                        supportListFile,
                        line,
                        f"nb positions={len(positions)} is different from nP_{infos[0]}={infos[2]} announced at line {1+s*2+1}",
                    )
                    flag = False

                supportList.append(
                    {"id": infos[0], "volume": infos[1], "positions": positions}
                )
    except Exception as e:
        handleReadError(fail_fast, supportListFile, line, f"{e}")
        flag = False

    if not flag:
        return False
    return supportList


def openAdjacencyMatrix(adjacencyMatrixFile: str, fail_fast: bool = True):
    flag = True
    line = -1
    adjMatrix = []
    try:
        with open(adjacencyMatrixFile, "r") as f:
            line = 1
            nbPositions = int(f.readline().strip())
            for i in range(nbPositions):
                line = 1 + i + 1
                distances = [float(i) for i in f.readline().strip().split(" ")]
                if len(distances) != nbPositions:
                    handleReadError(
                        fail_fast,
                        adjacencyMatrixFile,
                        line,
                        f"{len(distances)} values given when {nbPositions} were expected",
                    )
                    flag = False

                adjMatrix.append(distances)
    except Exception as e:
        handleReadError(fail_fast, adjacencyMatrixFile, line, f"{e}")
        flag = False

    if not flag:
        return False
    return adjMatrix


def openConstraints(constraintsFile: str, fail_fast: bool = True):
    flag = True
    line = -1
    maxBatchVolume = 0
    maxBatchNbSupports = 0
    try:
        with open(constraintsFile, "r") as f:
            line = 1
            infos = [int(i) for i in f.readline().strip().split(" ")]
            maxBatchNbSupports, maxBatchVolume = infos[0], infos[1]

            if maxBatchNbSupports <= 0:
                handleReadError(
                    fail_fast,
                    constraintsFile,
                    line,
                    f"maximum nb supports in a batch is {maxBatchNbSupports}, which is infeasible",
                )
                flag = False

            if maxBatchVolume <= 0:
                handleReadError(
                    fail_fast,
                    constraintsFile,
                    line,
                    f"maximum volume of a batch is {maxBatchVolume}, which is infeasible",
                )
                flag = False

    except Exception as e:
        handleReadError(fail_fast, constraintsFile, line, f"{e}")
        flag = False

    if not flag:
        return False, False
    return maxBatchNbSupports, maxBatchVolume


if __name__ == "__main__":
    problemFolder = "toy_instance"
    solutionFile = "batchList.txt"
    adjMatrixFile = os.path.join(problemFolder, "adjacencyMatrix.txt")
    supportListFile = os.path.join(problemFolder, "supportList.txt")
    constraintsFile = os.path.join(problemFolder, "constraints.txt")

    sl = openSupportList(supportListFile)
    adj = openAdjacencyMatrix(adjMatrixFile)
    sMax, vMax = openConstraints(constraintsFile)
    solution = openSolution(solutionFile)
    if not (sl and adj and sMax and vMax and solution):
        print("Error: Files contain incoherences!")
        sys.exit(1)
