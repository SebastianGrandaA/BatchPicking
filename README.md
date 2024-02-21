# Batch-Picking

The Batch-Picking problem consists of grouping orders and determining the sequence of storage locations to pick all items in a batch.
The items are partitioned into a set of orders, in such way that all items of an order must be picked in the same route, by the same picker.
The objective is to minimize the total distance traveled by the pickers to satisfy all the orders in the warehouse.

This project implements two main optimization approaches: the **sequential** and the **joint** approaches.
A detailed description of the problem, the implemented methods, and the results can be found in the [report](https://www.overleaf.com/read/xfgcnzwccnqj#8fe7b9). A rendered version is available at `docs/` folder.

## Installation and usage

To set up this project, first create a virtual environment and install the dependencies using the following command:

```bash
$ make init
```

For regular usage, activate the virtual environment:
    
```bash
$ make start
```

The project is designed to be used from the command line by directly running the `__main__.py` file with the desired arguments. The available arguments are the following:

```bash
-u, --use_case (str): Use case
-m, --method (str): Optimization method
-n, --instance_name (str): Instance name
-ns, --instance_names (str): List of instance names separated by comma
-t, --timeout (int): Timeout
-l, --log_level (str): Log level
```

The `Makefile` contains the main commands to interact with the application.
For example, to run the `toy_instance` with the `sequential` method, the following command can be used:
    
```bash
$ make optimize-sequential
```

Similarly for the `make optimize-joint` command runs the `toy_instance` with the `joint` method.

To run the experiments, the following command will run a set of instances with both methods:
    
```bash
$ make experiment
```

To run all the instances, use `make experiment-all` instead.
Finally, the `make describe` command will provide a summary of the results of the experiments.

## Implementation details

The implementation of this project focuses on intuitive and modular design, rather than performance, as it is a proof of concept.
Therefore, it is adequate for offline applications, without any execution time constraints.

### Project architecture

This project consists of three main components: the domain, the app, and the services.
The domain (`src/domain/`) contains the business logic of the application, including the optimization models and procedures.
The app (`src/app/`) implements three use cases to interact with the domain: `optimize`, `experiment`, and `describe`.
The `optimize` use case is responsible for solving a single instance of the problem using a specific method; the `experiment` use case, for executing a set of instances to benchmark different methods; and the `describe` use case, for providing an analysis of the results.
The services (`src/services/`) contain the input/output procedures, including the reader and writer classes, distance calculators, and other utilities that are external to the domain.

The `src/__main__.py` file is the entry point for the application. It initializes the application and dispatches the use case to the corresponding function.
This project expects the instances to be located in the `data/` directory, and the results will be saved in the `results/` directory.

With respect to the domain, the `src/domain/BatchPicking.py` file contains the main class, `BatchPicking`, which orchestrates the optimization process.
This class is responsible for reading the instances, solving the problem, and saving the best solution found in a maximum number of iterations. Introductory information about the domain problem, the Batch-Picking problem, is provided in that file as well.

There are two main optimization approaches implemented in this project: the sequential and the joint approaches.
The `src/domain/joint.py` file contains the implementation of the joint approach, which solves the problem by considering the order batching and picker routing problems simultaneously.
There are two versions of the joint approach implementation: the first uses the OR-Tools library, whereas the second, a commercial solver.

On the other side, the `src/domain/sequential.py` file contains the implementation of the sequential approach, which solves the problem by decomposing it into two subproblems: the order batching problem and the picker routing problem.
As can be seen, the project is organized in a modular way and implements several design patterns (inheritance, dependency injection, and others) to facilitate the extension and maintenance of the code.
Generally, the `solve` and `optimize` methods are the interfaces for generic optimization methods, while more specific methods are implemented under the corresponding submodules, such as `route`.

Finally, the `src/services/benchmark.py` file contains the benchmarking procedures, which are responsible for executing the experiments and analyzing the results.
The validation of the results is performed by comparing the solutions with the S-shaped path of serving the orders individually, and it is taken from the [UE repository](https://gitlab.com/LinHirwaShema/poip-2024).

### Language and dependencies

This project is implemented in Python 3.10.
This decision was mainly motivated by the requirements of the project.
Although Python is a good choice for prototyping due to its simplicity and versatility, it is not the best choice for performance-critical applications.
This problem is known as the Two-Language Problem, and Julia is a good alternative to replace Python in this context.
Julia offers a good balance between performance and productivity because it is a compiled language with a syntax similar to Python, and particularly suitable for scientific computing and optimization problems.

All the dependencies of this project are listed in the `requirements.txt` file. A list of the main ones is provided below:
* [Pyomo](http://www.pyomo.org/): A Python-based open-source optimization modeling language.
* [Gurobi](https://www.gurobi.com/): A commercial solver for mathematical programming problems.
* [OR-Tools](https://developers.google.com/optimization): A set of libraries for combinatorial optimization problems.

## Contribute

Before pushing a commit, ensure to format the code and export the dependencies:

```bash
$ make pre-commit
```

## Notes

The provided data contains three files outside to their corresponding instance folders (`'adjacencyMatrix.txt', 'aisleSubdivision.txt', 'positionList.txt'`). Therefore, a pre-process is required to add those files to the instance folders. For **new instances**, edit the `src/services/scripts/duplicate_files.py` script and run:

```bash
$ make pre-process
```
