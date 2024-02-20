"""
----- ENGLISH -----
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

This problem has been studied in the literature under different approaches.
A Clustered Vehicle Routing Problem (CluVRP) has been proposed by \cite{cvrp_jopbrp} to model the joint problem. In this problem, the customers are grouped into clusters, and the vehicles can enter and leave the clusters multiple times.
In our case, the customers are the items to be picked, and the clusters represent the order integrality condition to pick all items of an order in the same tour.
An adapted formulation for this problem as a CluVRP is described in the Annexes section.

This problem has also been addressed sequentially, where the order batching problem is solved first, and then the routes are obtained for each batch.
The advantage of this approach is the evident reduction in the complexity of the problem, as the routing problem can be solved as a Traveling Salesman Problem (TSP) for each batch.
The drawback is the lack of coordination between the two problems, which can lead to suboptimal solutions because the batching decisions are made without considering the routing problem.    
Related ideas can be found at \cite{survey_order_batching}, where the authors discuss the batch-first route-second approaches against the joint approach, and the benefits of solving the problems simultaneously.
These approaches usually require a computationally expensive calculation of the distance metric, such as calculating the shortest path between each combination of orders for a set-partitioning problem (i.e. the best sequence to pick all items in a batch).
We consider a relevant research challenge to find a metric that best approximates the shortest sequence of orders to pick in a batch, without the need for an exhaustive search.

In this work, we study the Joint Order Batching and Picker Routing Problem (JOBPRP) as a variant of the Pickup and Delivery Problem (PDP), and we propose a batch-first route-second heuristic to solve large instances of the problem.
The proposed heuristic is based on the Hausdorff distance, which is a measure the closeness of two sets of points, and it is used to determine the best way to group orders into batches.
The initial solution is obtained by solving the p-median problem, and the best sequence of items to pick in a batch is determined by solving a set of independent TSPs.
Once the initial solution is obtained, a local-search algorithm is applied to improve the solution by swapping orders between batches and re-optimizing the routes.

The remainder of this report is organized as follows. In the next section, we describe and formulate the problem. Then, we present the proposed heuristic, the implementation details and the numerical experiments. Finally, we provide a discussion of the results and the conclusions of this work.

---

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
The distance to or from the depot nodes is 0 \(d_{ij} = d_{ji} = 0 \forall i \in D, j \in V\). The distance matrix is asymmetric and satisfies the triangle inequality.

The nodes are visited by a set of identical pickers \(k \in K\), each with an unitary and volume capacity.
The unitary capacity constraint refers to the maximum number of orders that a picker can transport in a route, whereas the volume capacity constraint, to the maximum volume.
The upper bounds are denoted by \(C_{unit}\) and \(C_{volume}\), respectively.
We assume that there are enough pickers to cover all the orders in the warehouse, which is given by \(\underline{B} > 0\), the minimal number of batches necessary to have all orders assigned to a batch with a \(\alpha\%\) of slack \ref{eq:min_batches}.

\(\underline{B} = \max\left(\frac{\sum_{i \in I} v_i}{C_{volume}}, \frac{|O|}{C_{unit}}\right) \times (1 + \alpha)\) \label{eq:min_batches}

The artificial nodes act as consolidation points for each order, ensuring that all the items of an order are picked-up in the same route.
This is achieved by forcing the pickers to visit the item nodes and delivering them to the artificial nodes.
Let \(d(i) \in J\) be the delivery - artificial - node of the item node \(i\).

We present a three-index commodity-flow formulation that involves two types of variables: selection and flow variables.
Let \(x_{ijk} \in \{0, 1\}\) be the binary decision variables indicating whether the arc \((i, j) \in A\) is selected by the picker \(k \in K\).
The commodity-flow variables \(l_{ik} \geq 0\) indicate the cumulative load of picker \(k \in K\) after visiting node \(i \in V\). This load represents the number of orders picked up (i.e. quantity artificial nodes).
The model is formulated as follows:

\min \sum_{k \in K} \sum_{(i, j) \in A} d_{ij} x_{ijk} \label{eq:objective_vrp}
\sum_{k \in K} \sum_{j \in V \setminus \{0\}} x_{ijk} = 1, \quad \forall i \in I, \label{eq:service_constraints_vrp}
\sum_{j \in V} x_{ijk} = \sum_{j \in V} x_{j,d(i),k}, \quad \forall k \in K, \forall i \in I, \label{eq:pickup_delivery}

\sum_{j \in I \cup \{n\}} x_{0jk} = 1, \quad \forall k \in K, \label{eq:start_end_route}
\sum_{i \in V \setminus \{n\}} x_{ijk} = \sum_{i \in V \setminus \{0\}} x_{jik}, \quad \forall k \in K, \forall j \in V, \label{eq:flow_conservation}
\sum_{i \in J \cup \{0\}} x_{ink} = 1, \quad \forall k \in K, \label{eq:delivery_end_route}
x_{ijk} \times (l_{ik} + p_{j} - l_{jk}) = 0, \quad \forall k \in K, \forall (i, j) \in A, \label{eq:flow_definition}
\sum_{i \in V} p_{i} l_{ik} \leq C_{unit}, \quad \forall k \in K, \label{eq:unit_capacity}
\sum_{i \in V} v_{i} l_{ik} \leq C_{volume}, \quad \forall k \in K, \label{eq:volume_capacity}
l_{0k} = 0, \quad \forall k \in K, \label{eq:initial_load}

The objective function \eqref{eq:objective_vrp} minimizes the total distance traveled by the pickers.
...
    
This model can be further extended to consider heterogeneous pickers, time windows, and node-dependent costs, among extensions.
The MDVRPMPD is NP-Hard because it is an special case of the Capacitated Vehicle Routing Problem (CVRP) and the Pickup and Delivery Problem (PDP).

---

\section{Methods}

This section describes the sequential approach to solve the Batch-Picking problem.
This approach consists of a construction heuristic and a local search algorithm to improve the initial solution.

\subsection{Construction heuristic}
Inspired by the Benders decomposition technique, the construction heuristic solves the problem by fixing the set of batches and solving the routing problem independently as a TSP for each batch.
The main motivation for this approach is to employ a distance metric that does not require to enumerate all the possible routes to measure the convenience of grouping orders into batches.

\subsubsection{Hausdorff distance}
  ...

\subsubsection{Batching problem}

The Hausdorff distance measures the geographical closeness between the items of two orders.
The intra-order closeness is 0 since the items are already grouped in the same order.
This leads to inconveniences when modeling the batching problem as a set partitioning problem, as it would always create single-order batches because there is no incentive to group orders together.
Classical clustering algorithms can not be easily applied because most of them rely on the Euclidean distance, which is not suitable for our desired metric. Also, it became difficult to control the number and the capacity of the clusters.
To overcome these issues, a location-allocation problem is proposed to exploit the Hausdorff distance as a measure of the convenience for clustering orders.
Specifically, the p-median problem determines the subset of orders that are closer to other orders based on the unitary and the volume capacity constraints.

The p-median problem consists of selecting a subset of facilities, among a set of candidates, to be used to serve a set of demand points \cite{book_facility_location}.
The objective is to minimize the total travel distance between the demand points and the facilities.

In our context, we seek to group the orders into batches based on capacity constraints and a custom distance metric.
Therefore, the concept of "batch" can be interpreted as a consolidation facility for a set of orders.
Let \(\mathcal{I}\) be the set of potential orders to select \(p\) batches from, and \(\mathcal{J}\) be the set of orders to be served.
The closeness between orders \(i \in \mathcal{I}\) and \(j \in \mathcal{J}\) is given by \(c_{ij}\).
As before, let \(C_{unit}\) and \(C_{volume}\) be the maximum number of orders and the maximum volume that a batch can serve, respectively.
Also, let \(v_i\) be the volume of order \(i \in \mathcal{I}\ and \(\underline{B}\) be the minimum number of batches to be formed \ref{eq:min_batches}.

Let \(x_{ij} \in \{0, 1\}\) be the allocation variable, where \(x_{ij} = 1\) if order \(j\) is assigned to batch \(i\), and \(x_{ij} = 0\) otherwise.
As mentioned in [1], when \(\mathcal{I} = \mathcal{J}\) and \(c_{ii} = 0 \forall i \in \mathcal{I}\), the traditional location variables \(y_i\) can be replaced by the allocation variables \(x_{ii} \forall i \in \mathcal{I}\).
The objective is to maximize the total closeness between the orders in the same batch, and can be formulated as follows:

\max \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}: j \neq i} c_{ij} x_{ij} \label{eq:objective_pmedian}
\text{s.t.} \sum_{i \in \mathcal{I}} x_{ij} = 1 \quad \forall j \in \mathcal{J} \label{eq:unique_assignment_pmedian}
\sum_{j \in \mathcal{J}: j \neq i} x_{ij} \leq (|\mathcal{J}| - p) x_{ii} \quad \forall i \in \mathcal{I} \label{eq:selected_batches_pmedian}
\sum_{i \in \mathcal{I}} x_{ii} \leq p \label{eq:maximum_batches_pmedian}
\sum_{j \in \mathcal{J}} x_{ij} \leq C_{unit} \quad \forall i \in \mathcal{I} \label{eq:unitary_capacity_pmedian}
\sum_{j \in \mathcal{J}} v_j x_{ij} \leq C_{volume} \quad \forall i \in \mathcal{I} \label{eq:volume_capacity_pmedian}
x_{ij} = x_{ji} \quad \forall i \in \mathcal{I}, j \in \mathcal{J} \label{eq:symmetry_pmedian}
x_{ii} \in \{0, 1\} \quad \forall i \in \mathcal{I} \label{eq:binary_pmedian}
x_{ij} \in \{0, 1\} \quad \forall i \in \mathcal{I}, j \in \mathcal{J} \label{eq:binary_pmedian_2}

The objective function \eqref{eq:objective_pmedian} maximizes the total closeness between the orders in the same batch.
Constraints \eqref{eq:unique_assignment_pmedian} ensure that each order is assigned to exactly one batch.
Constraints \eqref{selected_batches_pmedian} prohibit the assignment of orders to un-selected batches.
Constraints \eqref{eq:maximum_batches_pmedian} ensure that exactly \(p\) batches are selected.
Constraints \eqref{eq:unitary_capacity_pmedian} and \eqref{eq:volume_capacity_pmedian} enforce the capacity constraints.
Constraints \eqref{eq:symmetry_pmedian} ensure that the allocation variables are symmetric.
Constraints \eqref{eq:binary_pmedian} and \eqref{eq:binary_pmedian_2} define the domain of the allocation variables.

The p-median problem is NP-hard. However, since we the locations represent the orders (the number of locations is small) we can use exact methods to solve it.

\subsubsection{Routing problem}

Once the batches are formed, the routing problem can be decomposed for each batch and solved in parallel using a TSP solver.
This model can be formulated...
...

\subsection{Local search}

The main motivation for the local search is to exploit the local structure of the problem to find better solutions.
Particularly, due that the set of items for each batch remains fixed in the initial solution, natural neighborhoods are to swap and to relocate orders between batches.
At each iteration, until the stopping criterion is met, a move operator is randomly selected, between the swap and the relocation operators, and the first-improving solution is selected from the neighborhood based on Tabu Search and Simulated Annealing principles.
To avoid cycling through the same solutions, the Tabu Search memory \cite{tabu_search} is used to store properties of the solutions that are forbidden to be selected again.
This memory is adjusted during the search process to force the algorithm to explore different regions of the search space (diversification).
Furthermore, non-improving solutions might be accepted to escape from local optima using the Metropolis criterion \cite{simulated_annealing}.
The best solution found is returned as the final solution after a maximum number of iterations.

\subsubsection{Move operators}

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
A solution is represented as a map from the batch to the set of orders in the batch. No solution is allowed to be selected again if all the batches have the same set of orders.
This memory is adjusted during the search process to force the algorithm to explore different regions of the search space (diversification).
At the end of the iterations, the search is diversified to escape from local optima and the memory is reduced to half of its size.

\subsubsubsection{Simulated annealing}

The simulated annealing strategy is used to accept non-improving solutions to escape from local optimas based on a probability.
The probability of accepting a non-improving solution is given by the Metropolis criterion, which is defined as the exponential of the negative difference between the new and the current solution divided by the temperature \ref{eq:metropolis_criterion}.
The temperature is reduced at each iteration by a cooling rate, which is a decreasing function of the iteration count.
This process avoids taking worse solutions at the end of the search process (intensification).

p_{accept} = \exp\left(-\frac{d_{new} - d_{current}}{T}\right) \label{eq:metropolis_criterion}

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


----

\section{References}

cvrp_jopbrp
  Aerts, B., Cornelissens, T., & Sörensen, K. (2021). The joint order batching and picker routing problem: Modelled and solved as a clustered vehicle routing problem. Computers & Operations Research, 129, 105168.

survey_order_batching
  [2] Henn, S., Koch, S., & Wäscher, G. (2012). Order batching in order picking warehouses: a survey of solution approaches (pp. 105-137). Springer London.

survey_static_pdp
  [1] Berbeglia, G., Cordeau, J. F., Gribkovskaia, I., & Laporte, G. (2007). Static pickup and delivery problems: a classification scheme and survey. Top, 15, 1-31.
  [2] Desaulniers, G., Desrosiers, J., Erdmann, A., Solomon, M. M., & Soumis, F. (2002). VRP with Pickup and Delivery. The vehicle routing problem, 9, 225-242.
  [3] Battarra, M., Cordeau, J. F., & Iori, M. (2014). Chapter 6: pickup-and-delivery problems for goods transportation. In Vehicle Routing: Problems, Methods, and Applications, Second Edition (pp. 161-191). Society for Industrial and Applied Mathematics.
  [4] Parragh, S. N., Doerner, K. F., & Hartl, R. F. (2008). A survey on pickup and delivery problems: Part I: Transportation between customers and depot. Journal für Betriebswirtschaft, 58, 21-51.

book_location_science
  [1] Laporte, G., Nickel, S., & Saldanha-da-Gama, F. (2019). Introduction to location science (pp. 1-21). Springer International Publishing.

---

----- FRENCH -----


----------































----

\section{Description du problème}

Le problème du Batch-Picking dans le contexte des opérations d'entrepôt consiste à regrouper un ensemble de commandes en batches, et à déterminer la séquence de positions de stockage à visiter pour chaque batch de manière à minimiser la distance totale parcourue par les préparateurs de commandes.
Nous présentons deux formulations pour résoudre ce problème: jointe et séquentielle.

Dans la formulation jointe, les problèmes de batching et de picking sont résolus simultanément comme un problème de routage de véhicules groupés (clustered vehicle routing).
Dans la formulation séquentielle, le problème de batching est résolu en premier comme un problème de partitionnement avec des contraintes de capacité.
La solution de batching est une entrée pour le problème de picking, qui est résolu comme un ensemble de problèmes de voyageur de commerce (traveling salesman problems, TSP), un pour chaque batch.

La formulation séquentielle est une approximation de l'approche jointe, puisque les optima locaux du problème de picking sont nécessaires pour évaluer la qualité de la solution de batching.
Cependant, cette perte de qualité est compensée par la réduction du temps de calcul de l'approche séquentielle.
Par conséquent, en abordant ce compromis, nous décrivons un algorithme heuristique dans lequel la formulation séquentielle est utilisée pour obtenir la solution initiale, et une procédure de recherche locale est effectuée pour améliorer la qualité de la solution.

Tout au long de ce document, l'ensemble des commandes est désigné par $R$, et l'ensemble des articles, par $P$.
Chaque commande $r \in R$ est composée d'un ensemble d'articles $P(r) \subseteq P$ et a un volume $v_{r}$ et une quantité d'articles $q_{r}$.
Un article $p \in P$ est associé à un emplacement de stockage dans l'entrepôt telle que plusieurs articles peuvent être stockés dans le même emplacement mais appartenant à des commandes différentes.
Les préparateurs de commandes utilisent un véhicule avec une capacité de $C_{\text{vol}}$ volume et $C_{\text{qty}}$ nombre d'articles.

La disposition de l'entrepôt est modélisée comme un graphe complet $G = (V, E)$, où $V = P \cup \{0, n-1\}$ contient l'ensemble des articles $P$ et le début et la fin du trajet $0$ et $n-1$, respectivement, et $E$ représente les déplacements possibles entre les positions.
La distance $d_{ij}$ entre les emplacements de stockage des articles $i, j \in V$ est le plus court chemin entre eux, en tenant compte de la disposition de l'entrepôt et des contraintes de mobilité des préparateurs de commandes.
Cette hypothèse nous permet de ne pas tenir compte des points de Steiner dans le graphe, qui sont des points intermédiaires ajoutés pour coder la disposition en bloc de l'entrepôt.
Soient $\delta^{+}(S)$ et $\delta^{-}(S)$ les arcs entrants et sortants d'un sous-ensemble de nœuds $S \subset V$, respectivement, pour les arêtes $(i, j) \in V \setminus S$.

La borne inférieure sur le nombre de batches, notée $\underline{K}$, est calculée comme le nombre minimal nécessaire pour que toutes les commandes soient affectées à un batch en fonction des capacités de volume et de quantité \eqref{eq:lower_bound_batches}.

\begin{align}
    \underline{K} = \max \left\{ \left\lceil \frac{\sum_{i \in R} v_{i}}{C_{\text{vol}}} \right\rceil, \left\lceil \frac{\sum_{i \in R} q_{i}}{C_{\text{qty}}} \right\rceil \right\} \label{eq:lower_bound_batches}
\end{align}

Les hypothèses sont résumées comme suit:

\begin{itemize}
    \item Les positions de stockage des commandes dans un batch peuvent être visitées dans n'importe quel ordre. Autrement dit, nous pouvons mélanger les positions de stockage de différentes commandes dans le même batch car elles ne nécessitent pas d'être ramassées séquentiellement. En termes pratiques, cela signifierait que le préparateur de commandes est capable de manipuler les supports pour placer les articles dans leurs boîtes respectives.
    \item Toutes les positions de stockage d'un batch doivent être visitées pendant le même batch. En d'autres termes, plusieurs préparateurs de commandes pour la même commande ne sont pas autorisés.
    \item Toutes les contraintes de mobilité sont prises en compte dans la matrice de distance. Ainsi, la distance entre deux positions est le plus court chemin entre elles, en tenant compte de la disposition de l'entrepôt et des contraintes de mobilité des véhicules.
    \item Les dates d'échéance et les coûts de stockage ne sont pas pris en compte dans le problème. Il n'est pas intéressant de collecter les commandes le plus tôt possible, mais de minimiser la distance totale de picking.
    \item La quantité de véhicules est suffisante pour couvrir tous les batches formés (c'est-à-dire qu'au moins les $\underline{K}$ véhicules sont disponibles). Il n'est pas intéressant de minimiser le nombre total de batches, mais la distance totale de picking à la place.
    \item Les préparateurs de commandes commencent et terminent le picking aux positions $0$ et $n-1$, respectivement.
    \item Les véhicules que les préparateurs de commandes utilisent ont les mêmes propriétés (flotte homogène).
    \item Seule la distance entre les positions est prise en compte. Tous les temps de service (temps de manipulation) sont négligeables.
\end{itemize}

\subsection{Formulation jointe: problème de routage de véhicules groupés}

Étant donné que les articles sont regroupés en commandes, le problème de picking peut être modélisé comme un problème de routage de véhicules groupés (clustered vehicle routing problem, CluVRP) comme le montre \cite{clustered_vrp}.
Dans le CluVRP, les articles appartenant à une commande doivent être visités par le même préparateur de commandes, ce qui permet au préparateur de visiter leurs positions dans n'importe quel ordre au sein du batch (c'est-à-dire que les positions de ramassage sont mélangées entre les commandes dans le même batch).
Ce dernier est une considération importante car la plupart de la littérature CluVRP suppose des contraintes de cluster rigides, de sorte que le préparateur doit terminer tous les articles d'une commande avant de commencer la suivante, même s'il existe un article plus proche d'une autre commande dans le même batch.

On définit la variable d'affectation $y_{ik} \in \{0, 1\}$ comme 1 si l'article $i \in V$ est sélectionné pour le batch $k \in K$, 0 sinon.
On définit la variable de séquence $x_{ijk} \in \{0, 1\}$ comme 1 si l'article $i \in V$ est visité avant l'article $j \in V$ dans le batch $k \in K$, 0 sinon.
Le modèle présenté par \cite{clustered_vrp} peut être adapté comme suit.

\begin{align}
    \min & \sum_{k \in K} \sum_{(i, j) \in E} d_{ij} x_{ijk} \label{eq:joint_objective} & \\
    & \text{s.t.} \quad \sum_{k \in K} y_{ik} = 1 \quad \forall i \in V \setminus \{0, n-1\} \label{eq:unique_batch} \\
    & \sum_{k \in K} y_{0k} = \sum_{j \in V \setminus \{0, n-1\}} \sum_{k \in K} x_{0jk} \leq \underline{K} \label{eq:starts_at_depot} \\
    & \sum_{j \in V} x_{ijk} = \sum_{j \in V} x_{jik} = y_{ik} \quad \forall i \in V, \forall k \in K \label{eq:flow_conservation} \\
    & \sum_{j \in V \setminus \{0, n-1\}} v_{j} y_{jk} \leq C_{\text{vol}} \quad \forall k \in K \label{eq:capacity_volume} \\
    & \sum_{j \in V \setminus \{0, n-1\}} q_{j} y_{jk} \leq C_{\text{qty}} \quad \forall k \in K \label{eq:capacity_quantity} \\
    & \sum_{i \in S} \sum_{j \in V \setminus S} x_{ijk} \geq y_{pk} \quad \forall S \subset V \setminus \{0, n-1\}, \forall p \in S, \forall k \in K \label{eq:subtour_elimination} \\
    & \sum_{(i, j) \in \delta^{+}(P(r))} \sum_{k \in K} x_{ijk} = \sum_{(i, j) \in \delta^{-}(P(r))} \sum_{k \in K} x_{ijk} \geq 1 \quad \forall r \in R \label{eq:cluster_constraint} \\
    & y_{ik} = y_{jk} \quad \forall i, j \in P(r), \forall r \in R, \forall k \in K \label{eq:order_integrality} \\
    & x_{ijk} \in \{0, 1\} \quad \forall (i, j) \in E, \forall k \in K \\
    & y_{ik} \in \{0, 1\} \quad \forall i \in V, \forall k \in K
\end{align}

La fonction objectif \eqref{eq:joint_objective} minimise la distance totale parcourue par les préparateurs de commandes.
Les contraintes \eqref{eq:unique_batch} garantissent que chaque article est affecté à exactement un batch.
Les contraintes \eqref{eq:starts_at_depot} assurent que tous les batches commencent au dépôt.
Les contraintes \eqref{eq:flow_conservation} garantissent que la conservation du flux est satisfaite pour l'emplacement de stockage de chaque article.
Les contraintes \eqref{eq:capacity_volume} et \eqref{eq:capacity_quantity} sont les contraintes de capacité de volume et de quantité pour chaque batch, respectivement.
Les contraintes \eqref{eq:subtour_elimination} éliminent les sous-tours en forçant le flux à être nul pour tous les sous-ensembles d'articles $S \subset V$.
Les contraintes \eqref{eq:cluster_constraint} forcent les articles du même batch à être visités par le même préparateur de commandes, ce qui permet de visiter leurs emplacements dans n'importe quel ordre.
Dernière contrainte \eqref{eq:order_integrality} se réfère à la condition d'intégrité de la commande pour s'assurer que tous les articles d'une commande sont affectés au même batch.

\subsection{Formulation séquentielle: partitionnement d'abord, routage ensuite}

La formulation séquentielle est composée d'un problème de partitionnement pour déterminer l'ensemble des batches, et d'un problème de voyageur de commerce pour chaque batch pour déterminer la séquence de positions à visiter.
Les avantages de cette approche sont résumés comme suit:

\begin{itemize}
    \item Le problème de partitionnement peut être simplifié en considérant la distance de Hausdorff entre les commandes au lieu d'évaluer la plus courte séquence de positions de ramassage entre chaque paire de commandes.
    \item Étant donné une solution pour le problème de batching, le problème de picking peut être résolu indépendamment pour chaque batch.
    \item Les solutions optimales pour les deux problèmes peuvent être améliorées en utilisant une procédure de recherche locale.
\end{itemize}

\subsubsection{Problème de batching: partitionnement avec contraintes de capacité}

Le problème de batching est formulé comme un problème de partitionnement avec contraintes de capacité, dans lequel nous cherchons à minimiser la proximité entre les commandes dans le même batch, tout en maximisant la proximité entre les commandes dans des batches différents.
La proximité entre deux commandes est définie comme la distance de Hausdorff entre leurs ensembles respectifs de positions de ramassage \cite{hausdorff_distance}.
La distance de Hausdorff dirigée $H(i, j)$ entre deux commandes $i, j \in R$ est la distance maximale entre un article de la commande $i$ et son article le plus proche de la commande $j$ \eqref{eq:hausdorff_distance}.

\begin{align}
    H(i, j) = \max_{p \in P(i)} \min_{q \in P(j)} d_{pq} \label{eq:hausdorff_distance}
\end{align}

Étant donné que cette distance est asymétrique, c'est-à-dire $H(i, j) \neq H(j, i) \forall i, j \in R$, la proximité $t_{ij}$ entre les commandes $i, j \in R$ est définie comme le maximum entre les deux distances dirigées \eqref{eq:closeness}.

\begin{align}
    t_{ij} = \max \{H(i, j), H(j, i)\} \label{eq:closeness}
\end{align}

Par exemple, supposons une seule allée avec 7 positions de stockage. La commande $i$ a des articles dans les positions $P(i) = \{1, 6\}$ et la commande $j$ a des articles dans les positions $P(j) = \{2, 3\}$ \eqref{eq:hausdorff_example}.

\begin{align}
    & H(i, j) = \max \{\min \{d_{12}, d_{13}\}, \min \{d_{62}, d_{63}\}\} = \max \{1, 3\} = 3 \\
    & H(j, i) = \max \{\min \{d_{21}, d_{26}\}, \min \{d_{31}, d_{36}\}\} = \max \{1, 2\} = 2 \\
    & t_{ij} = \max \{H(i, j), H(j, i)\} = \max \{3, 3\} = 3 \\
    \label{eq:hausdorff_example}
\end{align}

Cette métrique est beaucoup moins chronophage à calculer que le plus court chemin entre chaque paire de commandes.

Donc, pour formuler le problème, on définit la variable d'affectation $y_{ik} \in \{0, 1\}$ comme 1 si la commande $i \in R$ est affectée au batch $k \in K$, 0 sinon.
La variable $x_{ij} \in \{0, 1\}$ est 1 si les commandes $i, j \in R$ sont affectées au même batch, 0 sinon.
Le problème de batching est formulé comme suit \cite{graph_partitioning}.

\begin{align}
    \min & \sum_{(i, j) \in R \times R} t_{ij} x_{ij} \label{eq:batching_objective} & \\
    & \text{s.t.} \quad \sum_{k \in K} y_{ik} = 1 \quad \forall i \in R \label{eq:batching_unique} \\
    & y_{ik} + y_{jk} \leq 1 + x_{ij} \quad \forall (i, j) \in R \times R, \forall k \in K \label{eq:batching_intra_inter} \\
    & \sum_{i \in R} v_{i} y_{ik} \leq C_{\text{vol}} \quad \forall k \in K \label{eq:batching_capacity_volume} \\
    & \sum_{i \in R} q_{i} y_{ik} \leq C_{\text{qty}} \quad \forall k \in K \label{eq:batching_capacity_quantity} \\
    & x_{ij} = x_{ji} \quad \forall (i, j) \in R \times R \label{eq:batching_symmetry} \\
    & x_{ii} = 0 \quad \forall i \in R \label{eq:batching_diagonal} \\
    & x_{ij} \in \{0, 1\} \quad \forall (i, j) \in R \times R \\
    & y_{ik} \in \{0, 1\} \quad \forall i \in R, \forall k \in K \\
\end{align}

La fonction objectif \eqref{eq:batching_objective} minimise la proximité totale entre les commandes dans le même batch.
Les contraintes \eqref{eq:batching_unique} garantissent que chaque commande est affectée à exactement un batch.
Les contraintes \eqref{eq:batching_intra_inter} stipulent que si deux commandes sont affectées au même batch, alors l'arc entre elles est un arc intra-batch.
Les contraintes \eqref{eq:batching_capacity_volume} et \eqref{eq:batching_capacity_quantity} sont les contraintes de capacité de volume et de quantité pour chaque batch, respectivement.
Les contraintes \eqref{eq:batching_symmetry} et \eqref{eq:batching_diagonal} suppriment les arcs symétriques et les boucles, respectivement.
Ce problème de partitionnement est connu pour être NP-difficile lorsque le nombre de batches est supérieur à 2 \cite{graph_partitioning}.

\subsubsection{Problème de picking: voyageur de commerce}

Étant donné que les batches sont formés, le problème de déterminer la séquence de positions de stockage à visiter pour chaque batch est équivalent à résoudre un ensemble de problèmes de routage de préparateurs de commandes indépendants (single-picker routing problems, SPRP).
Ce problème peut être formulé comme un problème de voyageur de commerce (traveling salesman problem, TSP) comme le montre \cite{picking_tsp}, dans le but de minimiser la distance totale parcourue par le préparateur de commandes.

Nous utilisons la formulation de flux multi-marchandises proposée par \cite{tsp_claus}, dans laquelle les sous-tours sont éliminés en utilisant les contraintes de conservation du flux (c'est-à-dire que le préparateur commence le trajet avec tous les articles et "livre" une unité à chaque position visitée).

La représentation du graphe $G_{k}$ est similaire à celle présentée dans la formulation jointe, mais l'ensemble de nœuds est défini comme tous les articles du batch donné $\forall k \in K: V_{k} = \{ P(i) : i \in R, y_{ik} = 1 \}$, et l'ensemble d'arêtes est défini comme $E_{k} = \{(i, j) \in V_{k} \times V_{k} : i \neq j\}$.
On définit la variable $x_{ij} \in \{0, 1\}$ comme 1 si l'arc $(i, j) \in E_{k}$ est sélectionné, 0 sinon.
La variable $z^{l}_{ij}$ représente le nombre d'unités de l'article $l \in V_{k} \setminus \{0, n-1\}$ qui passent par l'arc $(i, j) \in E_{k}$.
Le TSP associé au batch $k \in K$ peut être adapté de \cite{tsp_claus} comme suit.

\begin{align}
    \min & \sum_{(i, j) \in E_{k}} d_{ij} x_{ij} \label{eq:picking_objective} & \\
    & \text{s.t.} \quad \sum_{j \in V_{k}} x_{ij} = 1 \quad \forall i \in V_{k} \label{eq:picking_unique} \\
    & \sum_{i \in V_{k}} x_{ij} = 1 \quad \forall j \in V_{k} \label{eq:picking_unique_reverse} \\
    & \sum_{j \in V_{k} \setminus \{0, n-1\}} z^{l}_{0j} - \sum_{j \in V_{k} \setminus \{0, n-1\}} z^{l}_{j0} = -1 \quad \forall l \in V_{k} \setminus \{0, n-1\} \label{eq:picking_depot_outflow} \\
    & \sum_{j \in V_{k}} z^{i}_{ij} - \sum_{j \in V_{k}} z^{i}_{ji} = 1 \quad \forall i \in V_{k} \setminus \{0, n-1\} \label{eq:one_commodity_per_vertex} \\
    & \sum_{j \in V_{k}} z^{l}_{ij} - \sum_{j \in V_{k}} z^{l}_{ji} = 0 \quad \forall i \in V_{k} \setminus \{0, n-1\}, \forall l \in V_{k} \setminus \{0, n-1\} : l \neq i \label{eq:picking_flow_conservation} \\
    & z^{l}_{ij} \leq x_{ij} \quad \forall (i, j) \in E_{k}, \forall l \in V_{k} \setminus \{0, n-1\} \label{eq:picking_flow_arc} \\
    & x_{ij} \in \{0, 1\} \quad \forall (i, j) \in E_{k} \\
    & z^{l}_{ij} \geq 0 \quad \forall (i, j) \in E_{k}, \forall l \in V_{k} \setminus \{0, n-1\} \\
\end{align}

La fonction objectif \eqref{eq:picking_objective} minimise la distance totale parcourue par le préparateur de commandes dans le batch $k \in K$.
Les contraintes \eqref{eq:picking_unique} et \eqref{eq:picking_unique_reverse} garantissent que chaque article est visité exactement une fois.
Les contraintes \eqref{eq:picking_depot_outflow} garantissent que tous les articles sont collectés à $0$ et livrés à un sommet de $V_{k} \setminus \{0, n-1\}$.
Les contraintes \eqref{eq:one_commodity_per_vertex} garantissent que chaque sommet de $V_{k} \setminus \{0, n-1\}$ reçoit exactement une unité d'article.
Les contraintes \eqref{eq:picking_flow_conservation} indiquent que la conservation du flux est satisfaite à chaque sommet de $V_{k} \setminus \{0, n-1\}$.
La contrainte \eqref{eq:picking_flow_arc} évite le flux d'articles à travers les arcs qui ne sont pas sélectionnés.

Comme le montre \cite{tsp_formulation_survey}, cette formulation nécessite $O(n^{3})$ variables et contraintes, et elle fournit une relaxation LP plus forte que celle obtenue par la formulation MTZ (Miller-Tucker-Zemlin), qui est la plus courante pour le TSP.


---


\section{Approche heuristique}

L'approche heuristique proposée est composée de deux phases: l'heuristique de construction et la procédure de recherche locale.
L'heuristique de construction implémente la formulation séquentielle pour obtenir une solution initiale, et la procédure de recherche locale est effectuée pour améliorer la qualité de la solution.
Ces opérateurs cherchent à évaluer les nouveaux chemins entre les commandes qui sont dans les différents batches résultant du problème de partitionnement, de sorte que la distance totale parcourue par les préparateurs de commandes est minimisée.
Pour garantir la connexité du voisinage, nous considérons trois opérateurs: swap, insert et destroy-repair.
Une recherche de voisinage variable \cite{vns} (variable neighborhood search, VNS) sélectionne le meilleur mouvement à chaque itération de l'algorithme en fonction de la stratégie de recherche, soit pour intensifier, soit pour diversifier.

La stratégie de routage combinée \cite{routing_strategies} (c'est-à-dire S-shape et Largest gap combinés) nous permet d'obtenir la nouvelle séquence de picking étant donné qu'un opérateur est appliqué à une solution.
La solution de première amélioration est prise à chaque itération de l'algorithme, qui arrête la recherche en utilisant un critère de Metropolis pour accepter les mouvements de détérioration \cite{simulated_annealing}.
Enfin, une mémoire adaptative de recherche tabou \cite{tabu_search} est implémentée pour éviter le cyclisme dans l'espace de recherche et pour diversifier la recherche en forçant l'algorithme à explorer de nouvelles solutions.

\subsection{Heuristique de construction}

\subsection{Opérateurs}

L'opérateur swap échange deux commandes entre deux batches.
L'opérateur insert supprime une commande d'un batch et l'insère dans un autre batch.
L'opérateur destroy-repair supprime toutes les commandes d'un batch et les réaffecte à d'autres batches.

\subsection{Stratégies de routage}

Pour évaluer les mouvements, nous pouvons explorer plusieurs stratégies de routage connues pour ce problème, telles que la S-shape, le plus grand écart ou la stratégie de routage combinée \cite{routing_strategies}.


----------

@article{clustered_vrp,
  title={The joint order batching and picker routing problem: Modelled and solved as a clustered vehicle routing problem},
  author={Aerts, Babiche and Cornelissens, Trijntje and S{\"o}rensen, Kenneth},
  journal={Computers \& Operations Research},
  volume={129},
  pages={105168},
  year={2021},
  publisher={Elsevier}
}

@article{graph_partitioning,
  title={The node capacitated graph partitioning problem: a computational study},
  author={Ferreira, Carlos Eduardo and Martin, Alexander and de Souza, C Carvalho and Weismantel, Robert and Wolsey, Laurence A},
  journal={Mathematical programming},
  volume={81},
  pages={229--256},
  year={1998},
  publisher={Springer}
}

@article{picking_tsp,
  title={A new mathematical programming formulation for the single-picker routing problem},
  author={Scholz, Andr{\'e} and Henn, Sebastian and Stuhlmann, Meike and W{\"a}scher, Gerhard},
  journal={European Journal of Operational Research},
  volume={253},
  number={1},
  pages={68--84},
  year={2016},
  publisher={Elsevier}
}

@article{tsp_claus,
  title={A new formulation for the travelling salesman problem},
  author={Claus, A},
  journal={SIAM Journal on Algebraic Discrete Methods},
  volume={5},
  number={1},
  pages={21--25},
  year={1984},
  publisher={SIAM}
}

@article{tsp_formulation_survey,
  title={An analytical comparison of different formulations of the travelling salesman problem},
  author={Padberg, Manfred and Sung, Ting-Yi},
  journal={Mathematical Programming},
  volume={52},
  number={1-3},
  pages={315--357},
  year={1991},
  publisher={Springer}
}

@article{routing_strategies,
  title={Efficient algorithms for travelling salesman problems arising in warehouse order picking},
  author={Charkhgard, Hadi and Savelsbergh, Martin},
  journal={The ANZIAM Journal},
  volume={57},
  number={2},
  pages={166--174},
  year={2015},
  publisher={Cambridge University Press}
}

@article{hausdorff_distance,
  title={An efficient algorithm for calculating the exact Hausdorff distance},
  author={Taha, Abdel Aziz and Hanbury, Allan},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={37},
  number={11},
  pages={2153--2163},
  year={2015},
  publisher={IEEE}
}

@book{vns,
  title={Variable neighborhood search},
  author={Hansen, Pierre and Mladenovi{\'c}, Nenad and Brimberg, Jack and P{\'e}rez, Jos{\'e} A Moreno},
  year={2019},
  publisher={Springer}
}

@article{simulated_annealing,
  title={Simulated annealing},
  author={Bertsimas, Dimitris and Tsitsiklis, John},
  journal={Statistical science},
  volume={8},
  number={1},
  pages={10--15},
  year={1993},
  publisher={Institute of Mathematical Statistics}
}

@article{tabu_search,
  title={Tabu search: A tutorial},
  author={Glover, Fred},
  journal={Interfaces},
  volume={20},
  number={4},
  pages={74--94},
  year={1990},
  publisher={INFORMS}
}


------------- LITERATURE -------------------

Recently, Cambazard & Catusse (2018) developed a dynamic programming approach which can solve
any rectilinear TSP and can therefore solve the PRP for any rectangular
warehouse with h cross-aisles.

!!! exact algorithms for the PRP is referred to the recent survey by Pansart et al. (2018).

!! The Lin-Kernighan–Helsgaun (LKH)
algorithm, considered one of the best heuristics to solve the
TSP, outperforms the warehouse dedicated routing heuristics
and has proven to solve the PRP close to optimality (Theys
et al., 2010; Van Gils et al., 2018).

Gademann and
Velde (2005)). Öncan (2015) introduced Mixed Integer Linear
Programming (MILP) formulations in three variants, where each
variant considers another routing strategy (s-shape, return and
midpoint routing)

! Kulak et al. (2012) use a tabu search algorithm based on a
similarity-regret value index (RS-RV) that deﬁnes the overlap in
travel distance if orders are merged.

Scholz and Wäscher (2017) present an iterated local
search approach in which they propose a routing heuristic derived
from the exact solution approach presented by Roodbergen and
Koster (2001).


Defryn and Sörensen (2017) describe the following Hausdorff-based constructive heuristic        
    Prior to the assignment, orders are sorted in decreasing order according to size (number of requested items
    !! number of available batches is known, which is the minimal number necessary to have all orders assigned to a batch

    A very simple herustic for the partitions: "fill the knapsack prioritizing the largest orders"
        Quantity of posible batches
        Sort orders by size (number of items)
        Assign orders to batches with sufficient capacity that is closest to the LAST-ADDED order in the batch (in terms of Hausdorff distance)

The idea to decompose the CluVRP into two subproblems is fur-
ther explored by Expósito-Izquierdo et al. (2016). They propose a
combination of the record-to-record travel algorithm using the
clusters’ Euclidean centres to solve the subproblem at cluster-
level and the LKH algorithm to solve the routing problem within
the clusters.


Both Defryn and Sörensen (2015) and Expósito-Izquierdo et al.
(2016) propose the clusters’ Euclidean center as an approximation
for the closeness between customer clusters, which restricts their
approaches to VRPs deﬁned by Euclidean distances.

As an alterna-tive, Defryn and Sörensen (2017) suggest to use the Hausdorff dis-
tance as closeness criterion. The Hausdorff distance is often used
for object matching in the ﬁeld op computer vision and object
recognition (Sim et al., 1999) to measure the similarities between
two sets (Hung and Yang, 2004). The Hausdorff distance can be
applied to problems deﬁned by Euclidean (e.g., VRP), as well as
Manhattan distances (e.g., PRP)

! Pop et al. (2018) utilize the TSP concorde solver at http://www.math.uwaterloo.ca/tsp/concorde.html) to opti-mize the intra-cluster routes
    
Defryn and Sörensen (2017)
    !!!

    Just as the picker routing problem (PRP) bears large similarities
    with the travelling salesman problem (TSP), we observe that the
    JOBPRP closely resembles the clustered VRP (CluVRP), a variant
    of the capacitated VRP

   !! In the CluVRP, introduced by Sevaux et al. (2008), customers are par-
    titioned into clusters based on a predeﬁned criterion (e.g., postal
    code), with the additional constraint that customers belonging to
    the same cluster need to be visited by the same vehicle.

    (WHCHis an extension of the capacitated VRP in which customers are grouped into clusters.)

    By replacing vehicles by batches, clusters by orders, and customers by
    pick operations, the JOBPRP can be modelled as the CluVRP
    
    Chisman (1975) who intro-duces the idea of a clustered TSP (CTSP)


    !!! READ
        Löfﬂer et al. (2018) who exploit the similarities
        between the CTSP and the PRP to plan the routing in an AGV-
        assisted order picking system.


[Joint order batching and order picking in warehouse operations]
    Consideran batching problem cmo un bin packing. Sin embargo, no es de nuestro interes minimizar el numero de batches, sino la distancia total recorrida por el picker.
    Also they consider the regular TSP

[The joint order batching and picker routing problem: Modelled and solved as a clustered vehicle routing problem]
    clustered vehicle routing problem a variant of the capacitated VRP in which customers are grouped into clusters. 
    

Won & Olafsson (2005) consider the JOBPRP sequentially by rst batching orders and second by solving a TSP for each batch
    
!!
Kulak et al. (2012) propose a tabu search heuristic in which the
initial batches are created with a clustering method and the routing is calculated using TSP-based heuristics such as savings and nearest neighbour
heuristics for routes construction and Or-opt and 2-opt heuristics for routes
improvement. New solutions are created by relocating and swapping orders
among batches
New solutions are created by relocating and swapping orders among batches
    
the only exact approach for
the JOBPRP so far has been proposed by Valle et al. (2017). They develop
a branch-and-cut algorithm, and the initial solution is computed using a
saving heuristic


!! Cambazard & Catusse (2018) developed a dynamic programming approach which can solve
any rectilinear TSP and can therefore solve the PRP for any rectangular
warehouse with h cross-aisles


!! The reader interested in exact algorithms for the PRP is referred to the recent survey by Pansart et al. (2018).


[7] efficient-algorithms-for-travelling-salesman-problems-arising-in-warehouse-order-picking
        Euclidean TSP with points on two parallel lines.

        Given that the batches are formed, the routing problem is decomposed into a set of independent routing problems, one for each batch.

        The picker routing problem (PRP) seeks to minimize the distance traveled by a (single) picker, given a set of pick locations that has to be visited.
        It is a special case of the travelling salesman problem (TSP) due to the typical rectangular layout of the storage area in a warehouse.
        
        Ratliﬀ and Rosenthal [9] have shown that the PRP can be solved in polynomial time

        Routing strategies: Return, S-Shape, Largest Gap, Combined
            S-shape
                S-shape routing strategy, an order picker enters an aisle and traverses the aisle if there exists at least one article that has to be picked from that aisle, then goes to the next aisle. The order picker returns to his starting point after traversing the last aisle which has to be visited
                In the return strategy, an order picker enters an aisle and returns after visiting the most distant pick locatio

                In the largest gap strategy, an order picker traverses the ﬁrst and last aisles from which articles have to be picked entirely, whereas the other aisles are traversed partially, in and out, from both ends, in such way that the distance that is not traversed is maximum
                    
                !! In the combined routing strategy, each aisle is either traversed entirely or entered and left from the same end [7], which usually generates a near-optimal solution

        All these strategies require the solution of two **aisle routing problem**:
            1. Passing strategy : Pick all items while traversing the entire aisle (more like zig-zag)
            2. Returning strategy : Pick all items and return to the starting point (more like a TSP)

        !! These are the problems! A special case of TSP

            two aisle routing problems, which are special cases of the TSP, can be solved eﬃciently: for the passing strategy in O(n2) time and for the returning strategy in O(n) time, where n is the number of pick locations in an aisle

            Because O(n2) time may be computationally prohibitive when solving large instances of the OBP, we show that an approximate cost for the passing strategy, derived from the minimum spanning tree for the pick locations, can be computed in O(n) time

            

------------- LITERATURE -------------------


"""