# BatchPicking

This project aims to solve the BatchPicking problem and to compare the **joint** and the **sequential** solutions.
The complete description of the problem and the proposed solution can be found in the [documentation](https://www.overleaf.com/read/xfgcnzwccnqj#8fe7b9). A rendered version is available at `docs/` folder.

To set up this project, first create a virtual environment and install the dependencies using the following command:

```bash
$ make init
```

## Usage

For regular usage, activate the virtual environment:
    
```bash
$ make start
```

There are three use cases: optimize, experiment, and describe.

The **optimize use case** is used to solve a single instance:

```bash
$ make optimize
```

The **experiment use case** is used to solve multiple instances:

```bash
$ make experiment
```

To run all instances, use the following command:

```bash
$ make experiment_all
```

The **describe use case** is used to analyze the results of the optimization process:

```bash
$ make describe
```

Finally, to run them all, use:

```bash
$ make test
```

## Develop

Before pushing a commit, ensure to format the code and export the dependencies:

```bash
$ make format
$ make freeze
```

## Notes

The provided data contains three files outside to their corresponding instance folders (`'adjacencyMatrix.txt', 'aisleSubdivision.txt', 'positionList.txt'`). Therefore, a pre-process is required to add those files to the instance folders. For **new instances**, edit the `src/services/scripts/duplicate_files.py` script and run:

```bash
$ make pre-process
```
