""" TODO this is a draft - always gide from the french version as it is the last updated

\title{The Batch-Picking problem}

\section{Abstract}

The Batch-Picking problem consists of grouping orders and determining the sequence of storage locations to pick all items in a batch.
The items are partitioned into a set of orders, in such way that all items of an order must be picked in the same route, by the same picker.
The objective is to minimize the total distance traveled by the pickers to satisfy all the orders in the warehouse.
This project implements two main optimization approaches: the **sequential** and the **joint** approaches.

\section{Introduction}

The Batch-Picking problem, formally known in the literature as the Joint Order Batching and Picker Routing Problem (JOBPRP), is a well-known problem in the context of warehouse logistics.
As the name suggests, the problem combines two optimization problems: the order batching problem and the picker routing problem.
The order batching problem consists of grouping subsets of orders to pick their items together, if that leads to a reduced total distance traveled.
On the other hand, the picker routing problem consists of determining the sequence of storage locations to pick all orders in a batch.
This problem has been studied in the literature under both the joint and the sequential approaches.

A Clustered Vehicle Routing Problem (CluVRP) has been proposed by \cite{cvrp_jopbrp} to model the joint problem.
In this problem, the customers are grouped into clusters, which correspond to the order integrality condition to pick all items of an order in the same tour.
Their model allows the vehicles to enter and leave the clusters multiple times using soft cluster constraints.
Our case can be adapted to this model by considering the customers as the items to be picked. The CluVRP formulation is described in the Annexes section.

This problem has also been addressed sequentially, where the order batching problem is solved first, and then the routes are obtained.
The advantage of this approach is the evident reduction in the complexity of the problem, as the routing problem can be solved for each batch independently.
The drawback is the lack of coordination between the two problems, which can lead to suboptimal solutions because the batching decisions are made without considering the routing problem.    
Related ideas can be found at \cite{survey_order_batching}, where the authors discuss the batch-first route-second approaches against the joint approach, and the benefits of solving the problems simultaneously.
However, these approaches usually require a computationally expensive calculation of the distance metric, such as calculating the shortest path between each combination of orders for a set-partitioning problem (i.e. the best sequence to pick all items in a batch).
Therefore, we consider a relevant research challenge to find a metric that best approximates the shortest sequence of orders to pick in a batch, without the need for an exhaustive search.

In this work, we study the Batch-Picking problem as a variant of the Pickup and Delivery Problem (PDP), and we propose a batch-first route-second heuristic to solve large instances of the problem.
The proposed heuristic is based on the Hausdorff distance, which is a measure the closeness of two sets of items (i.e. two orders), and it is used to determine the best way to group orders into batches.
The initial solution is obtained by solving the p-median problem, and the sequence of items to pick in a batch is determined by solving a set of independent TSPs.
Once the initial solution is obtained, a local-search algorithm is applied to improve the solution by applying a set of move operators and re-optimizing the routes.

The remainder of this report is organized as follows. In the next section, we describe and formulate the problem. Then, we present the proposed heuristic, the implementation details and the numerical experiments. Finally, we provide a discussion of the results and the conclusions of this work.

\section{Problem statement}

This section aims to model the Batch-Picking problem as a variant of the Pickup and Delivery Problem (PDP).
More specifically, we model the problem as the Multi-depot Vehicle Routing Problem with Mixed Pickup and Delivery (MDVRPMPD).
According to \cite{survey_static_pdp}, this problem is classified as the multi-vehicle one-to-many-to-one PDP with single demands and mixed solutions [1-M-1|P/D|m].

In this problem, we aim to minimize the total distance traveled by the pickers to satisfy all the orders in the warehouse.
Therefore, we model the items instead of the positions in such a way that items belonging to different orders can share the same position.
We are given a set of items \(I\) grouped by disjoint orders \(O\), where \(i(o)\) refers to the items of order \(o \in O\).
Let \(G = (V, A)\) be the directed graph representing the warehouse layout, where \(V\) is the set of nodes and \(A\) is the set of arcs.

The set of nodes \(V\) is composed of the item nodes \(I\), the artificial nodes \(J\), and the depot nodes \(D\). These sets are disjoint and satisfy \(V = I \cup J \cup D\).
The depot nodes \(D = \{0, n\}\), where \(0\) is the origin and \(n = |I| + 1\) is the destination of each route.
The artificial nodes \(J = \{|I| + 2, \dots, |I| + |O| + 1\}\) are the consolidation points for each order to ensure the order integrity condition.
Each node \(i \in V\) is associated with a demand and a volume, denoted by \(p_i \geq 0\) and \(v_i \geq 0\), respectively.
However, only artificial nodes have positive demand and volume, which are equal to 1 and to the total volume of the order, respectively.
The demand and the volume of the item and depot nodes are 0 \(p_i = v_i = 0\) for \(i \in I \cup D\).
Furthermore, the physical position in which each node is located is represented by \(pos(i) \forall i \in V\). It is possible that multiple nodes share the same position.

The arc set \(A = \{(i, j) \in V \times V : i \neq j\}\) are all feasible paths between the nodes, each with a distance \(d_{ij} \geq 0\) obtained from the positions of nodes \(pos(i)\) and \(pos(j)\) in the warehouse.
The distance to or from the dummy nodes is 0 \(d_{ij} = d_{ji} = 0 \forall i \in V, j \in J\). The distance matrix is asymmetric and satisfies the triangle inequality.

The nodes are visited by a set of identical pickers \(k \in K\), each with an unitary and volume capacity.
The unitary capacity constraint refers to the maximum number of orders that a picker can transport in a route, whereas the volume capacity constraint, to the maximum volume.
The upper bounds are denoted by \(C_{unit}\) and \(C_{volume}\), respectively.
We assume that there are enough pickers to cover all the orders in the warehouse, which is given by \(\underline{B} > 0\), the minimal number of batches necessary to have all orders assigned to a batch with a \(\alpha\%\) of slack \ref{eq:min_batches}.

\(\underline{B} = \max\left(\frac{\sum_{i \in I} v_i}{C_{volume}}, \frac{|O|}{C_{unit}}\right) \times (1 + \alpha)\) \label{eq:min_batches}

Let \(d(i) \in J\) be the artificial node, or delivery point, of the item node \(i\).
The artificial nodes act as consolidation points for each order, ensuring that all the items of an order are picked-up in the same route.
This is achieved by forcing the pickers to visit the item nodes and delivering them to the artificial nodes.

Three-index commodity-flow formulations for the MDVRPMPD can be found at \cite{book_tooth1_chap9} and \cite{book_tooth2_chap6}.
These formulations involve two types of variables: selection and flow.
Let \(x_{ijk} \in \{0, 1\}\) indicate whether the arc \((i, j) \in A\) is selected by the picker \(k \in K\).
Also, let the flow variables \(u_{ik} \geq 0\) indicate the cumulative load of picker \(k \in K\) after visiting node \(i \in V\).
This load represents the number of orders picked up (i.e. quantity artificial nodes).

This problem is NP-Hard because it is an special case of the Capacitated Vehicle Routing Problem (CVRP) and the Pickup and Delivery Problem (PDP) \cite{survey_static_pdp}.
Therefore, in the next section a heuristic method is proposed to solve big instances of the problem.

\section{Heuristic method}

The proposed heuristic method is composed of two stages: a construction heuristic based on a sequential approach, and a local search meta-heuristic to improve the initial solution.

\subsection{Construction heuristic}

As a decomposition technique, the construction heuristic solves the problem by fixing the set of batches and solving the routing problem independently as a TSP for each batch.
The main motivation for this approach is to employ a distance metric that does not require to enumerate all the possible routes to measure the convenience of grouping orders into batches.

\subsubsection{Hausdorff distance}

[check French version]

\subsubsection{Batching problem}

The Hausdorff distance measures the geographical closeness between the items of two distinct orders (i.e. the intra-order closeness is 0).
This leads to inconveniences when modeling the batching problem as a set partitioning problem, as it would always create single-order batches because of the lack of incentive to group orders together.
Neither classical clustering algorithms, such as K-means constraint or DBSCAN, can be easily adapted to control the capacity of the batches.
Furthermore, most of them rely on the Euclidean distance and our desired metric cannot be applied.
To overcome these issues, a location-allocation problem is proposed to exploit the Hausdorff distance as a measure of the convenience for clustering orders.
Specifically, the p-median problem determines the subset of orders that are closer to other orders based on the unitary and the volume capacity constraints.

The p-median problem consists of selecting a subset of facilities, among a set of candidates, to be used to serve a set of demand points \cite{book_facility_location}.
The objective is to minimize the total travel distance between the demand points and the facilities.

In our context, the concept of "batch" can be interpreted as an installed facility that "serve" a subset of orders.
Let \(I\) be the set of potential orders to select \(\underline{B}\) batches from, and \(J\) be the set of orders to be served.
Similar to previous sections, \(C_{unit}\) and \(C_{volume}\) is the maximum number of orders and the maximum volume that a batch can serve, respectively; \(v_i\) is the volume of order \(i \in I\); and \(\underline{B}\) is calculated using the equation \eqref{eq:min_batches}.

Let \(x_{ij} \in \{0, 1\}\) be the allocation variable, where \(x_{ij} = 1\) if order \(j\) is assigned to batch \(i\), and \(x_{ij} = 0\) otherwise.
The book \cite{book_location_science} explain that, when \(I = J\) and \(c_{ii} = 0 \forall i \in I\), the traditional location variables \(y_i\) can be replaced by the allocation variables \(x_{ii} \forall i \in I\).
The objective is to maximize the total closeness between the orders in the same batch, and can be formulated as follows:

\begin{equation}
\max \sum_{i \in I} \sum_{j \in J: j \neq i} c_{ij} x_{ij} \label{eq:objective_pmedian}
\text{s.t.} \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J \label{eq:unique_assignment_pmedian}
\sum_{j \in J: j \neq i} x_{ij} \leq (|J| - p) x_{ii} \quad \forall i \in I \label{eq:selected_batches_pmedian}
\sum_{i \in I} x_{ii} \leq p \label{eq:maximum_batches_pmedian}
\sum_{j \in J} x_{ij} \leq C_{unit} \quad \forall i \in I \label{eq:unitary_capacity_pmedian}
\sum_{j \in J} v_j x_{ij} \leq C_{volume} \quad \forall i \in I \label{eq:volume_capacity_pmedian}
x_{ij} = x_{ji} \quad \forall i \in I, j \in J \label{eq:symmetry_pmedian}
x_{ii} \in \{0, 1\} \quad \forall i \in I \label{eq:binary_pmedian}
x_{ij} \in \{0, 1\} \quad \forall i \in I, j \in J \label{eq:binary_pmedian_2}
\end{equation}

The objective function \eqref{eq:objective_pmedian} maximizes the total closeness between the orders in the same batch.
Constraints \eqref{eq:unique_assignment_pmedian} ensure that each order is assigned to exactly one batch.
Constraints \eqref{selected_batches_pmedian} prohibit the assignment of orders to un-selected batches.
Constraints \eqref{eq:maximum_batches_pmedian} ensure that exactly \(p\) batches are selected.
Constraints \eqref{eq:unitary_capacity_pmedian} and \eqref{eq:volume_capacity_pmedian} enforce the capacity constraints.
Constraints \eqref{eq:symmetry_pmedian} ensure that the allocation variables are symmetric.
Constraints \eqref{eq:binary_pmedian} and \eqref{eq:binary_pmedian_2} define the domain of the allocation variables.

The p-median problem is NP-hard. However, since we the locations represent the orders (the number of locations is small) we can use exact methods to solve it.

\subsubsection{Routing problem}

Given a solution to the batching problem, the routing problem is decomposed for each batch and solved in parallel using a TSP solver.

[check French version]

\subsection{Local search}

Due that the batch structure is un-mutable in the routing problem, two move operators applied to the incumbent solution: the swap and the relocate operators.
At each iteration, until the stopping criterion is met, a move operator is randomly selected and the first-improving solution is obtained.
To avoid cycling through the same solutions, the Tabu Search memory \cite{tabu_search} is used to store properties of the solutions that are forbidden to be selected again.
This memory is adjusted during the search process to force the algorithm to explore different regions of the search space (diversification).
Furthermore, non-improving solutions might be accepted to escape from local optima using the Metropolis criterion \cite{simulated_annealing}.
The best solution found is returned as the final solution after a maximum number of iterations.

\subsubsection{Move operators}

The proposed operators seek to evaluate new paths between the orders that are in the different batches resulting from the p-median problem.
We consider two move operators to explore the neighborhood of the current solution: the swap and the relocate operators.
The swap operator exchanges two orders between two different batches, whereas the relocate operator moves an order from one batch to another.

The swap operator is defined as follows: given two batches \(b_1\) and \(b_2\), and two orders \(o_1 \in b_1\) and \(o_2 \in b_2\), the operator swaps the orders between the batches.
The origin batch \(b_1\) is chosen prioritizing the least filled batches (i.e. the batches with the highest capacity residual), and the destination batch \(b_2\) is chosen randomly among the batches that can accommodate the order.
To evaluate the solution, the TSP solver is used to re-compute the paths for the affected batches, and the improvement is the distance difference between the new and the old paths.
However, this operator is not sufficient since it maintains the same number of batches in the solution.

The relocate operator is given by two batches \(b_1\) and \(b_2\), and an order \(o_1 \in b_1\), the operator relocates the order to the batch \(b_2\).
The same selection criteria are used to choose the origin and destination batches.
Only the destination batch is re-routed, whereas the origin batch is just updated without the items of the relocated order.
The combination of these two operators allows the algorithm to explore all the possible solutions in the search space, since we can obtain solutions with different number of batches and different orders in each batch.
Thus, we confirm the conexity of the neighborhood, which is a desirable property for local search algorithms.

\subsubsubsection{Tabu search}

The Tabu Search memory is used to store the properties of the solutions that are forbidden to be selected again.
A solution is represented as a mapping from the batch to the set of orders in the batch. No solution is allowed to be selected again if all the batches have the same set of orders.
This memory is adjusted during the search process to force the algorithm to explore different regions of the search space (diversification).
At the end of the iterations, the search is diversified to escape from local optima and the memory is reduced to half of its size.

\subsubsubsection{Simulated annealing}

The simulated annealing strategy is used to accept non-improving solutions to escape from local optimas based on a probability.
The probability of accepting a non-improving solution is given by the Metropolis criterion, which is defined as the exponential of the negative difference between the new and the current solution divided by the temperature \ref{eq:metropolis_criterion}.

p_{accept} = \exp\left(-\frac{d_{new} - d_{current}}{T}\right) \label{eq:metropolis_criterion}

The temperature is reduced at each iteration by a cooling rate, which is a decreasing function of the iteration count.
This process avoids taking worse solutions at the end of the search process (intensification).

---

\section{Implementation}

The complete implementation of this project can be found at [this repository](https://github.com/SebastianGrandaA/BatchPicking/).
It focuses on intuitive and modular design, rather than performance, as it is a proof of concept.
Therefore, it is adequate for offline applications, without any execution time constraints.

\subsection{Project architecture}

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
In addition to the `PMedian` batching method, the set partitioning problem and the clustering algorithms were also implemented as a proof of concept, in the classes `GraphPartition` and `Clustering`, respectively.
From an experimentation point of view, the TSP solver can be a simple TSP (TSPBase), a multi-commodity flow TSP (TSPMultiCommodityFlow), or a simplified implementation of [OR-Tools](https://developers.google.com/optimization/routing/vrp) for the VRP problem.
The last option is the default routing method as it shows the best performance in terms of solution quality and computational time.

As can be seen, the project is organized in a modular way and implements several design patterns (inheritance, dependency injection, and others) to facilitate the extension and maintenance of the code.
Generally, the `solve` and `optimize` methods are the interfaces for generic optimization methods, while more specific methods are implemented under the corresponding submodules, such as `route`.

Finally, the `src/services/benchmark.py` file contains the benchmarking procedures, which are responsible for executing the experiments and analyzing the results.
The validation of the results is performed by comparing the solutions with the S-shaped path of serving the orders individually, and it is taken from the [UE repository](https://gitlab.com/LinHirwaShema/poip-2024).
Installation and usage instructions can be found in the `README.md` file of the repository.

Further improvements can be achieved by using a more specialized solver, such as the [VRP solver](https://vrpsolver.math.u-bordeaux.fr) developed at [INRIA](https://www.inria.fr/fr).
However, currently the [Python](https://github.com/inria-UFF/VRPSolverEasy) implementation does not cover the pick-up and delivery feature.
A workaround is to implement the solver in [Julia](https://github.com/inria-UFF/BaPCodVRPSolver.jl) and integrated it with python using [PyJulia](https://github.com/JuliaPy/pyjulia?tab=readme-ov-file).

\subsection{Language and dependencies}

This project is implemented in Python 3.10.
This decision was mainly motivated by the requirements of the project.
Although Python is a good choice for prototyping due to its simplicity and versatility, it is not the best choice for performance-critical applications.
This problem is known as the Two-Language Problem, and Julia is a good alternative to replace Python in this context.
Julia offers a good balance between performance and productivity because it is a compiled language with a syntax similar to Python, and particularly suitable for scientific computing and optimization problems.

All the dependencies of this project are listed in the `requirements.txt` file. A list of the main ones is provided below:
* [Pyomo](http://www.pyomo.org/): A Python-based open-source optimization modeling language.
* [Gurobi](https://www.gurobi.com/): A commercial solver for mathematical programming problems.
* [OR-Tools](https://developers.google.com/optimization): A set of libraries for combinatorial optimization problems.

\section{Numerical experiments}

The numerical experiments are performed on all available instances at the `data/` directory using a 2.3 GHz Intel Core i9 8 cœurs processor. 
These instances are classified by warehouse: `A`, `B`, `C`, `D` and `toy`.

\subsection{Caractérisation des instances}

The instances can be classified by the average size of the orders (i.e. number of items per order).
We observe that the instances `A` and `D` have the smallest average size, whereas the instances `B` and `C` have the largest average size.
  ...

This is a relevant aspect for our solution approach because orders with large quantity of items require more optimization effort to solve the routing problem.
On the contrary, if the instance contains many orders with few items each, there are more opportunities to group orders together, and the solution quality is expected to be better.

An example of the warehouse layout is shown in the following figure \ref{fig:warehouse_layout}.
  .... TODO warehouse layout (x1)...

\subsection{Résultats}

The `toy` instance, in addition to the S-shaped path solution, contains an improved solution which is used as an additional reference to measure the quality of the solutions.
This provided solution improves 44\% with respect to the baseline solution.
However, we observe that our solution, in both the sequential and the joint approaches, outperforms this improvement by 20\% and 13\%, respectively, as shown in the table below.

\begin{center}
\begin{tabular}{|c|c|c|c|}
\hline
Instance & Provided & Sequential & Joint \\
\hline
toy & 44\% & 65\% & 57\% \\
\hline
\end{tabular}
\end{center}

The distribution of the relative improvement against the baseline solution is shown in the following box-plots \ref{fig:boxplot_improvement_warehouse} and \ref{fig:boxplot_improvement_method}.
  ... TODO img boxplot improvement (x2)...

The most significant improvement are the instances of warehouse `A`, with an ...

Overall, global the improvement percentage of all the methods is ...
We also measure the batching percentage, which is the quantity of orders in the batch divided by the total quantity of orders.
  ...explicar todas las metricas ...
  ... TODO table metrics ...

Finally, the distribution of the computational time by method is shown in the following box-plot \ref{fig:boxplot_time_warehouse_method}.

Examples of the routes are shown in the following figures \ref{fig:routes_warehouse}.
  ... TODO nice route examples ...

A completed table of the results can be found at the Annexes section.
  ... TODO Annex: benchamrk_preprocessed.csv
Note that even though we have tested all the provided instances, we only report those instance with improved solutions with respect to the S-shaped path, as it is our upper bound on the total distance traveled.
There were several instances with 0\% improvement, and any of them reported a negative improvement.

\section{Conclusions}

The Batch-Picking problem is a relevant problem in the context of warehouse logistics, and it has been addressed in the literature under both the joint and the sequential approaches.
Mainly, the related literature focus on a set partitioning based approach for the sequential approach, and a clustered vehicle routing problem for the joint approach.
In this work, we have proposed a heuristic method to solve the problem, which is based on the p-median problem and the Traveling Salesman Problem (TSP).
This algorithm outperforms the baseline solution in almost all the instances, and in those instances where it does not, the improvement is not negative.
An alternative formulation based on the Pickup and Delivery Problem (PDP) was also proposed to model the problem jointly, and it was solved using the local search meta-heuristic.
The proposed method provides a good trade-off between solution quality and computational time, and it is suitable for large instances of the problem.

"""