# Batch picking problem

We address the batching and picking problem in two stages: first, we determine how to group the orders into batches, and then we determine the sequence of storage locations to be visited by each picker.

We first introduce the notation used over this document. Let $P$ be the set of $(\text{lat}, \text{lon})$ storage locations, or positions, in the warehouse. Also let $I$ be the set of orders (a.k.a. commandes or supports), each one composed of a set of items $P(i) \subseteq P \quad \forall i \in I$. We represent the warehouse layout as a $G = (V, E)$ complete graph in which the node set represent the warehouse positions $V = P$ and the edge set is defined as $E = \{(i, j) \in I \times I : i \neq j\}$.

## Assumptions

* The pickup positions of the orders in a batch can be visited in any order. That is, we can mix pickups from different orders in the same batch as they do not require to be picked up sequentially. In practical terms, this would mean that the picker is able to manipulate the supports to place the items in their correct boxes.
* All storage locations in a batch must be visited during the same batch. That is, multiple pickers for the same order is not allowed.
* All mobility constraints are considered in the distance matrix. That is, the distance between two positions is the shortest path between them, taking into account the warehouse layout and the mobility constraints of the vehicles.

* Due dates and holding costs are not considered in the problem. Thus, it is not of interest to collect the orders as soon as possible, but to minimize the total picking distance.
* The quantity of vehicles is sufficient to cover all formed batches (i.e., one vehicle per order). Therefore, it is not of interest to minimize the total number of batches, but the total picking distance instead.
* Vehicles start and end picking at positions 0 and n-1, respectively.
* All vehicles have the same properties (homogeneous fleet).
* All service times (manipulation times) are negligible. Only distance between positions is considered.

## Batching problem

The batching problem is formulated as a graph partitioning problem with capacity constraints, in which we maximize the similarity between orders in the same batch and minimize the similarity between orders in different batches.
The similarity $\delta_{ij} \in [0, 1]$ between two orders $(i, j) \in I$ is calculated as the proportion of items in common aisles.
Let the parameter $a^{p}_{ij} \in \{0, 1\}$ be 1 if the item positions $(i, j) \in P$ are in the same aisle $p \in P$, 0 otherwise.

$$\delta_{ij} = \frac{1}{|P(i)| + |P(j)|} \sum_{p \in P} \sum_{(k, l) \in P(i) \times P(j)} a^{p}_{kl}$$

Thus, if two orders have their items in the same aisles, then the similarity is 1, and if two orders have no items in common, then the similarity is 0.
This will lead to partitions that avoid overlapping aisles, which is a desirable property for the picking process as it avoids concentration of pickers in the same area of the warehouse.
Note that this metric is much less time-consuming to compute than the similarity based on the distance between each pair of items in the orders.

Furthermore, we consider the capacity constraints of the vehicle, i.e. the maximum number of orders and the maximum volume of orders in a batch.
For each order $i \in I$, let $v_{i}$ and $q_{i}$ be the volume and the number of items in order $i$, respectively.
Similarly, let $C_{\text{vol}}$ and $C_{\text{qty}}$ be the maximum volume and the maximum number of orders in a batch, respectively.

Define the variable $y_{ib} \in \{0, 1\}$ as 1 if order $i \in V$ is assigned to batch $b \in B$, 0 otherwise.
Also, define $x_{ij} \in \{0, 1\}$ as 1 if edge $(i, j) \in E$ are assigned to the same batch, 0 otherwise.
Then, the problem can be formulated as follows:

$$\max \quad \sum_{(i, j) \in E} \delta_{ij} x_{ij}$$

$$\text{s.t.} \quad \sum_{b \in B} y_{ib} = 1 \quad \forall i \in V \quad \text{(1)}$$
$$y_{ib} + y_{jb} \leq 1 + x_{ij} \quad \forall (i, j) \in E, \forall b \in B \quad \text{(2)}$$
$$\sum_{i \in V} v_{i} y_{ib} \leq C_{\text{vol}} \quad \forall b \in B \quad \text{(3)}$$
$$\sum_{i \in V} q_{i} y_{ib} \leq C_{\text{qty}} \quad \forall b \in B \quad \text{(4)}$$
$$x_{ij} = x_{ji} \quad \forall (i, j) \in E \quad \text{(5)}$$
$$x_{ij} \in \{0, 1\} \quad \forall (i, j) \in E$$
$$y_{ib} \in \{0, 1\} \quad \forall i \in V, \forall b \in B$$

The objective function maximizes the similarity between orders in the same batch.
Constraint (1) ensures that each order is assigned to exactly one batch.
Constraint (2) states that if two orders are assigned to the same batch, then the edge between is an intra-batch edge.
Constraints (3) and (4) are the capacity constraints of the vehicle.
Constraint (5) removes symmetries by forcing the intra-batch edges to be undirected.

This model allows us to obtain the set of batches $B$ each one composed of a set of orders $I(b) \subseteq I$.

---

## Picking problem

Given that the batches are formed, the problem to determine the sequence of positions to visit for each batch is modeled as a set of independently single-picker routing problems (SPRP).
Due to the block layout of the warehouse, the picker can only move from one aisle to another at certain positions, called cross aisles.
To consider this characteristic, we define the set of Steiner points $S$ as the set of cross aisles.
The distance $d_{ij}$ between two positions $i, j \in V$ is the length of the shortest path between them, taking into account the warehouse layout and the mobility constraints of the vehicles. There are no distances from or to Steiner points.
This problem is formulated as a Steiner traveling salesman problem with points on two parallel lines, aiming to minimize the total distance traveled by the picker. 

We define the variable $x_{ij} \in \{0, 1\}$ as 1 if arc $(i, j) \in E$ is selected, 0 otherwise.
The variable $y_{i} \in \{0, 1\}$ takes the value of 1 if vertex $i \in V$ is selected, 0 otherwise.
In order to eliminate sub-tours, we define the variable $w^{k}_{ij} \in \mathbb{Z}^{+}$ that represents the number of units of item $k \in V \setminus \{0\}$ that pass through arc $(i, j) \in E (i.e. the picker starts the tour with all items and "delivers" one unit to each vertex visited).
The Steiner TSP can be formulated as follows:

$$\min \quad \sum_{(i, j) \in E} d_{ij} x_{ij}$$
$$\text{s.t.} \quad \sum_{j \in V : (i, j) \in E} x_{ij} \geq 1 \quad \forall i \in V \setminus S \quad \text{(1)}$$
$$\sum_{j \in V : (i, j) \in E} x_{ij} = \sum_{j \in V : (j, i) \in E} x_{ji} \quad \forall i \in V \quad \text{(2)}$$
$$\sum_{j \in V : (j, 1) \in E} w^{k}_{j1} + 1 = \sum_{j \in V : (1, j) \in E} w^{k}_{1j} \quad \forall k \in V \setminus \{0\} \quad \text{(3)}$$
$$\sum_{j \in V : (j, k) \in E} w^{k}_{jk} = 1 + \sum_{j \in V : (k, j) \in E} w^{k}_{kj} \quad \forall k \in V \setminus (S \cup \{0\}) \quad \text{(4)}$$
$$\sum_{j \in V : (i, j) \in E} w^{k}_{ij} = \sum_{j \in V : (j, i) \in E} w^{k}_{ji} \quad \forall i \in V \setminus \{0\}, \forall k \in V \setminus \{0, i\} \quad \text{(5)}$$
$$w^{k}_{ij} \leq x_{ij} \quad \forall (i, j) \in E, \forall k \in V \setminus (S \cup \{0\}) \quad \text{(6)}$$
$$x_{ij} \in \{0, 1\} \quad \forall (i, j) \in E$$
$$w^{k}_{ij} \geq 0 \quad \forall (i, j) \in E, \forall k \in V \setminus (S \cup \{0\})$$

The objective function minimizes the total distance traveled.
Constraint (1) ensures that each vertex that is not a Steiner point is visited exactly once.
Constraint (2) guarantees the flow conservation at each vertex.
Constraint (3) ensures that each item leaves the depot and is delivered to a vertex.
Constraint (4) ensures that each vertex receives exactly one unit of item.
Constraint (5) ensures that an item leaves a vertex that is not its final destination.
Constraint (6) ensures that an arc is selected only if both vertices are selected.

This formulation requires $O(|S| \cdot |E|)$ variables and constraints. Assuming a block layout, $|E| = O(n \cdot  m)$ arcs are needed to represent the Steiner TSP, where $n$ is the number of storage locations and $m$ is the number of cross aisles.

We can explore multiple known **routing strategies** for this problem, such as the S-shape, the largest gap, or the combined routing strategy.
   
