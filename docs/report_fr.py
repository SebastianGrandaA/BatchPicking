""" TODO this is a draft. Check the final version in the report [document](https://www.overleaf.com/read/xfgcnzwccnqj#8fe7b9)
\title{Le problème du Batch-Picking}

\section{Résumé}

Dans le contexte des opérations d'entrepôt, le problème du Batch-Picking consiste à regrouper des commandes et à déterminer la séquence de positions de stockage pour ramasser tous les articles d'un batch.
Les articles sont partitionnés en un ensemble de commandes, de telle manière que tous les articles d'une commande doivent être ramassés dans le même trajet, par le même préparateur de commandes.
L'objectif est de minimiser la distance totale parcourue par les préparateurs de commandes pour satisfaire toutes les commandes dans l'entrepôt.
Ce projet implémente deux principales approches d'optimisation: l'approche **séquentielle** et l'approche **jointe**.

\section{Introduction}

Le problème du Batch-Picking, connu dans la littérature comme << Joint Order Batching and Picker Routing Problem >> (JOBPRP), est un problème bien connu dans le contexte de la logistique d'entrepôt.
Comme son nom l'indique, le problème combine deux problèmes d'optimisation: le problème de regroupement de commandes et le problème de routage de préparateur de commandes.
Le problème de regroupement de commandes consiste à regrouper des sous-ensembles de commandes pour ramasser leurs articles ensemble, si cela conduit à une réduction de la distance totale parcourue.
D'autre part, le problème de routage de préparateur de commandes consiste à déterminer la séquence de positions de stockage pour ramasser toutes les commandes dans un batch.
Ce problème a été étudié dans la littérature sous les approches jointe et séquentielle.

Un problème de routage de véhicules groupés (Clustered Vehicle Routing Problem, CluVRP) a été proposé par \cite{clustered_vrp} pour modéliser le problème joint.
Dans ce problème, les clients sont regroupés en clusters, qui correspondent à la condition d'intégrité de la commande pour ramasser tous les articles d'une commande dans le même trajet.
Leur modèle permet aux véhicules d'entrer et de sortir des clusters plusieurs fois en utilisant des contraintes de cluster doux.
Notre cas peut être adapté à ce modèle en considérant les clients comme les articles à ramasser. La formulation CluVRP est décrite dans la section Annexes.

Ce problème a également été abordé séquentiellement, où le problème de regroupement de commandes est résolu en premier, puis les trajets sont obtenus.
L'avantage de cette approche est la réduction évidente de la complexité du problème, car le problème de routage peut être résolu pour chaque batch indépendamment.
L'inconvénient est le manque de coordination entre les deux problèmes, ce qui peut conduire à des solutions sous-optimales car les décisions de regroupement sont prises sans tenir compte du problème de routage.
Des idées relatives peuvent être trouvées dans \cite{survey_order_batching}, où les auteurs discutent des approches << batch-first route-second >> par rapport à l'approche jointe, et des avantages de résoudre les problèmes simultanément.
Cependant, ces approches nécessitent généralement un calcul coûteux de la métrique de distance, tel que le calcul du plus court chemin entre chaque combinaison de commandes pour un problème de partitionnement de l'ensemble (c'est-à-dire la meilleure séquence pour ramasser tous les articles dans un batch).
Par conséquent, nous considérons un défi de recherche pertinent de trouver une métrique qui approxime au mieux la séquence la plus courte des commandes à ramasser dans un batch, sans avoir besoin d'une recherche exhaustive.

Dans ce travail, nous étudions le problème du Batch-Picking en tant que variante du problème de ramassage et de livraison (PDP), et nous proposons une heuristique de type << batch-first route-second >> pour résoudre de grandes instances du problème.
L'heuristique proposée est basée sur la distance de Hausdorff, qui mesure la proximité de deux ensembles d'articles (c'est-à-dire deux commandes), et est utilisée pour déterminer la meilleure façon de regrouper les commandes en batches.
La solution initiale est obtenue en résolvant le problème de p-médian, et la séquence d'articles à ramasser dans un batch est déterminée en résolvant un ensemble de problèmes indépendants de voyageur de commerce (TSP).
Une fois la solution initiale obtenue, un algorithme de recherche locale est appliqué pour améliorer la solution en appliquant un ensemble d'opérateurs de déplacement et en ré-optimisant les trajets.

Le reste de ce document est organisé comme suit. Dans la section suivante, nous décrivons et formulons le problème. Ensuite, nous présentons l'heuristique proposée, les détails de l'implémentation et les expériences numériques. Enfin, nous fournissons une discussion des résultats et les conclusions de ce travail.

\section{Description du problème}

Cette section vise à modéliser le problème du Batch-Picking en tant que variante du problème de ramassage et de livraison (PDP).
Plus précisément, nous modélisons le problème comme le problème de routage de véhicules multi-dépôts avec ramassage et livraison mixtes (Multi-Depot Vehicle Routing Problem with Mixed Pick-up and Delivery, MDVRPMPD).
Selon \cite{survey_static_pdp}, ce problème est classé comme le problème de ramassage et de livraison multi-véhicules un-à-plusieurs-à-un avec des demandes simples et des solutions mixtes [1-M-1|P/D|m].

Dans ce problème, nous cherchons à minimiser la distance totale parcourue par les préparateurs de commandes pour satisfaire toutes les commandes dans l'entrepôt.
Donc, nous modélisons les articles au lieu des positions de telle manière que les articles appartenant à différentes commandes peuvent partager la même position.
Nous disposons d'un ensemble d'articles \(I\) regroupés par des commandes disjointes \(O\), où \(i(o)\) fait référence aux articles de la commande \(o \in O\).
Soit \(G = (V, A)\) le graphe orienté représentant la disposition de l'entrepôt, où \(V\) est l'ensemble des nœuds et \(A\) est l'ensemble des arcs.

L'ensemble des nœuds \(V\) est composé des nœuds d'articles \(I\), des nœuds artificiels \(J\) et des nœuds de dépôt \(D\). Ces ensembles sont disjoints et satisfont \(V = I \cup J \cup D\).
Les nœuds de dépôt \(D = \{0, n\}\), où \(0\) est l'origine et \(n = |I| + 1\) est la destination de chaque trajet.
Les nœuds artificiels \(J = \{|I| + 2, \dots, |I| + |O| + 1\}\) sont les points de consolidation pour chaque commande pour assurer la condition d'intégrité de la commande.
Chaque nœud \(i \in V\) est associé à une demande et un volume, notés \(p_i \geq 0\) et \(v_i \geq 0\), respectivement.
Cependant, seuls les nœuds artificiels ont une demande et un volume positifs, qui sont égaux à 1 et au volume total de la commande, respectivement.
La demande et le volume des nœuds d'articles et de dépôt sont 0 \(p_i = v_i = 0\) pour \(i \in I \cup D\).
De plus, la position physique dans laquelle se trouve chaque nœud est représentée par \(pos(i) \forall i \in V\). Il est possible que plusieurs nœuds partagent la même position.

L'ensemble des arcs \(A = \{(i, j) \in V \times V : i \neq j\}\) sont tous les trajets réalisables entre les nœuds, chacun avec une distance \(d_{ij} \geq 0\) obtenue à partir des positions des nœuds \(pos(i)\) et \(pos(j)\) dans l'entrepôt.
La distance vers ou depuis les nœuds artificiels est 0 \(d_{ij} = d_{ji} = 0 \forall i \in V, j \in J\).

Les nœuds sont visités par un ensemble de préparateurs de commandes identiques \(k \in K\), chacun avec une capacité unitaire et volumétrique.
La contrainte de capacité unitaire fait référence au nombre maximal de commandes qu'un préparateur de commandes peut transporter dans un trajet, tandis que la contrainte de capacité volumétrique fait référence au volume maximal.
Les bornes supérieures sont notées \(C_{unit}\) et \(C_{volume}\), respectivement.
Nous supposons qu'il y a suffisamment de préparateurs de commandes pour couvrir toutes les commandes dans l'entrepôt, ce qui est donné par \(\underline{B} > 0\), le nombre minimal de batches nécessaires pour avoir toutes les commandes attribuées à un batch avec un \(\alpha\%\) de marge \ref{eq:min_batches}.

\begin{align}
    \(\underline{B} = \max\left(\frac{\sum_{i \in I} v_i}{C_{volume}}, \frac{|O|}{C_{unit}}\right) \times (1 + \alpha)\) \label{eq:min_batches}
\end{align}

Soit \(d(i) \in J\) le nœud artificiel, ou point de livraison, du nœud d'article \(i\).
Les nœuds artificiels agissent comme des points de consolidation pour chaque commande, en garantissant que tous les articles d'une commande sont ramassés dans le même trajet.
Cela est réalisé en forçant les préparateurs de commandes à visiter les nœuds d'articles et à les livrer aux nœuds artificiels.

Les hypothèses du problème sont résumées comme suit:

\begin{itemize}
    \item Nous pouvons mélanger les positions de stockage de différentes commandes dans le même batch. En d'autres termes, les articles de différentes commandes peuvent être ramassés dans n'importe quel ordre.
    \item Tous les ramassages doivent être effectués pendant le même batch. Donc, plusieurs préparateurs de commandes pour la même commande ne sont pas autorisés.
    \item La quantité de préparateurs de commandes est suffisante pour couvrir tous les batches formés. C'est-à-dire qu'au moins les \(\underline{B}\) véhicules sont disponibles. Il n'est pas intéressant de minimiser le nombre total de batches, mais la distance totale de picking à la place.
    \item Les préparateurs de commandes ont les mêmes caractéristiques (flotte homogène). Tous leurs trajets commencent et terminent le picking aux dépôts.
    \item Toutes les contraintes de mobilité sont prises en compte dans la matrice de distance. Ainsi, la distance entre deux positions est le plus court chemin entre elles, en tenant compte de la disposition de l'entrepôt et des contraintes de mobilité des véhicules.
\end{itemize}

Des formulations de flux de marchandises à trois indices pour le MDVRPMPD peuvent être trouvées dans \cite{book_tooth1_chap9} et \cite{book_tooth2_chap6}.
Ces formulations impliquent deux types de variables: sélection et flux.
Soit \(x_{ijk} \in \{0, 1\}\) indique si l'arc \((i, j) \in A\) est sélectionné par le préparateur \(k \in K\).
De plus, les variables de flux \(u_{ik} \geq 0\) indiquent la charge cumulative du préparateur \(k \in K\) après avoir visité le nœud \(i \in V\).
Cette charge représente le nombre de commandes ramassées (c'est-à-dire la quantité de nœuds artificiels).

Ce problème est NP-Difficile car il est un cas spécial du Problème de Routage de Véhicules Capacités (CVRP) et du Problème de Ramassage et de Livraison (PDP) \cite{survey_static_pdp}.
Par conséquent, dans la section suivante, une méthode heuristique est proposée pour résoudre de grandes instances du problème.

\section {Méthode heuristique}

La méthode heuristique proposée est composée de deux étapes: une heuristique de construction basée sur une approche séquentielle, et une méta-heuristique de recherche locale pour améliorer la solution initiale.
L'approche séquentielle est une approximation de l'approche jointe, puisque les optima locaux du problème de picking sont nécessaires pour évaluer la qualité de la solution de batching.
Cependant, cette perte de qualité est compensée par la réduction du temps de calcul de l'approche séquentielle.

\subsection{Heuristique de construction}

En tant que technique de décomposition, l'heuristique de construction résout le problème en fixant l'ensemble des batches et en résolvant le problème de routage indépendamment comme un TSP pour chaque batch.
La principale motivation de cette approche est d'employer une métrique de distance qui ne nécessite pas d'énumérer tous les trajets possibles pour mesurer la convenance de regrouper les commandes en batches: la distance de Hausdorff.
    
\subsubsection{Distance de Hausdorff}

La proximité entre deux commandes est définie comme la distance de Hausdorff entre leurs ensembles respectifs de positions de ramassage \cite{hausdorff_distance}.
La distance de Hausdorff dirigée $H(i, j)$ entre deux commandes $i, j \in R$ est la distance maximale entre un article de la commande $i$ et son article le plus proche de la commande $j$ \eqref{eq:hausdorff_distance}.

\begin{align}
    H(i, j) = \max_{p \in P(i)} \min_{q \in P(j)} d_{pq} \label{eq:hausdorff_distance}
\end{align}

Étant donné que cette distance est asymétrique, c'est-à-dire $H(i, j) \neq H(j, i) \forall i, j \in R$, la proximité $c_{ij}$ entre les commandes $i, j \in R$ est définie comme le maximum entre les deux distances dirigées \eqref{eq:closeness}.

\begin{align}
    c_{ij} = \max \{H(i, j), H(j, i)\} \label{eq:closeness}
\end{align}

Par exemple, supposons une seule allée avec 7 positions de stockage. La commande $i$ a des articles dans les positions $P(i) = \{1, 6\}$ et la commande $j$ a des articles dans les positions $P(j) = \{2, 3\}$ \eqref{eq:hausdorff_example}.

\begin{align}
    & H(i, j) = \max \{\min \{d_{12}, d_{13}\}, \min \{d_{62}, d_{63}\}\} = \max \{1, 3\} = 3 \\
    & H(j, i) = \max \{\min \{d_{21}, d_{26}\}, \min \{d_{31}, d_{36}\}\} = \max \{1, 2\} = 2 \\
    & c_{ij} = \max \{H(i, j), H(j, i)\} = \max \{3, 3\} = 3 \\
    \label{eq:hausdorff_example}
\end{align}

Cette métrique est beaucoup moins chronophage à calculer que le plus court chemin entre chaque paire de commandes.

\subsubsection{Problème de regroupement}

La distance de Hausdorff mesure la proximité géographique entre les articles de deux commandes distinctes (c'est-à-dire que la proximité intra-commande est 0).
Cela conduit à des inconvénients lors de la modélisation du problème de regroupement comme un problème de partitionnement d'ensemble, car il créerait toujours des batches de commandes uniques en raison du manque d'incitation à regrouper les commandes ensemble.
Ni les algorithmes de regroupement classiques, tels que K-means ou DBSCAN, ne peuvent être facilement adaptés pour contrôler la capacité des batches.
De plus, la plupart d'entre eux reposent sur la distance euclidienne et notre métrique désirée ne peut pas être appliquée.
Pour surmonter ces problèmes, un problème de "localisation-allocation" est proposé pour exploiter la distance de Hausdorff comme mesure de la convenance pour regrouper les commandes.
Spécifiquement, le problème de p-médian détermine le sous-ensemble de commandes qui sont plus proches des autres commandes en fonction des contraintes de capacité unitaire et volumétrique.

Le problème de p-médian consiste à sélectionner un sous-ensemble d'installations, parmi un ensemble de candidats, pour être utilisé pour desservir un ensemble de points de demande \cite{book_facility_location}.
L'objectif est de minimiser la distance totale de déplacement entre les points de demande et les installations.

Dans notre contexte, le concept de "batch" peut être interprété comme un hub installé qui "dessert" un sous-ensemble de commandes.
Soit \(I\) l'ensemble de commandes potentielles à partir desquelles sélectionner \(\underline{B}\) batches, et \(J\) l'ensemble de commandes à desservir.
De manière similaire aux sections précédentes, \(C_{unit}\) et \(C_{volume}\) sont le nombre maximal de commandes et le volume maximal qu'un batch peut desservir, respectivement; \(v_i\) est le volume de la commande \(i \in I\); et \(\underline{B}\) est calculé en utilisant l'équation \eqref{eq:min_batches}.

Soit \(x_{ij} \in \{0, 1\}\) la variable d'allocation, où \(x_{ij} = 1\) si la commande \(j\) est attribuée au batch \(i\), et \(x_{ij} = 0\) sinon.
Le livre \cite{book_location_science} explique que, lorsque \(I = J\) et \(c_{ii} = 0 \forall i \in I\), les variables de localisation traditionnelles \(y_i\) peuvent être remplacées par les variables d'allocation \(x_{ii} \forall i \in I\).
L'objectif est de maximiser la proximité totale entre les commandes dans le même batch, et peut être formulé comme suit:

\begin{equation}
\max \sum_{i \in I} \sum_{j \in J: j \neq i} c_{ij} x_{ij} \label{eq:objective_pmedian}
\text{s.t.} \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J \label{eq:unique_assignment_pmedian}
\sum_{j \in J: j \neq i} x_{ij} \leq (|J| - \underline{B}) x_{ii} \quad \forall i \in I \label{eq:selected_batches_pmedian}
\sum_{i \in I} x_{ii} \leq \underline{B} \label{eq:maximum_batches_pmedian}
\sum_{j \in J} x_{ij} \leq C_{unit} \quad \forall i \in I \label{eq:unitary_capacity_pmedian}
\sum_{j \in J} v_j x_{ij} \leq C_{volume} \quad \forall i \in I \label{eq:volume_capacity_pmedian}
x_{ii} \in \{0, 1\} \quad \forall i \in I \label{eq:binary_pmedian}
x_{ij} \in \{0, 1\} \quad \forall i \in I, j \in J \label{eq:binary_pmedian_2}
\end{equation}

L'objectif \eqref{eq:objective_pmedian} maximise la proximité totale entre les commandes dans le même batch.
Les contraintes \eqref{eq:unique_assignment_pmedian} garantissent que chaque commande est attribuée à exactement un batch.
Les contraintes \eqref{selected_batches_pmedian} interdisent l'attribution de commandes à des batches non sélectionnés.
Les contraintes \eqref{eq:maximum_batches_pmedian} garantissent que exactement \(\underline{B}\) batches sont sélectionnés.
Les contraintes \eqref{eq:unitary_capacity_pmedian} et \eqref{eq:volume_capacity_pmedian} imposent les contraintes de capacité.
Les contraintes \eqref{eq:binary_pmedian} et \eqref{eq:binary_pmedian_2} définissent le domaine des variables d'allocation.

Le problème de p-médian est NP-difficile. Cependant, puisque la quantité de commandes est relativement petite par rapport au nombre d'articles, nous pouvons utiliser des méthodes exactes pour le résoudre.

\subsubsection{Problème de routage}

Étant donné une solution au problème de regroupement, le problème de routage est décomposé pour chaque batch et résolu en parallèle en utilisant un solveur TSP, comme le montre \cite{picking_tsp}.
La formulation des flux de marchandises proposée par \cite{tsp_claus} est utilisée, dans laquelle les sous-tours sont éliminés en utilisant les contraintes de conservation du flux (c'est-à-dire que le préparateur commence le trajet avec tous les articles et "livre" une unité à chaque position visitée).

La représentation du graphe $G_{k}$ est similaire à celle présentée dans la formulation jointe, mais l'ensemble de nœuds est défini comme tous les articles du batch donné $\forall k \in K: V_{k} = \{ P(i) : i \in R, y_{ik} = 1 \}$, et l'ensemble d'arêtes, comme $E_{k} = \{(i, j) \in V_{k} \times V_{k} : i \neq j\}$.
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

\subsection{Recherche locale}

En raison de la structure immuable des batches dans le problème de routage, deux opérateurs de déplacement sont appliqués à la solution courante: les opérateurs de permutation et de relocalisation.
À chaque itération, jusqu'à ce que le critère d'arrêt soit atteint, un opérateur de déplacement est sélectionné aléatoirement et la première solution améliorante est obtenue.
Pour éviter de répéter les mêmes solutions, une mémoire de Tabu Search adaptative \cite{tabu_search} est utilisée pour stocker les propriétés des solutions qui sont interdites d'être sélectionnées à nouveau.
Cette mémoire est ajustée pendant le processus de recherche pour forcer l'algorithme à explorer différentes régions de l'espace de recherche (diversification).
De plus, les solutions non améliorantes peuvent être acceptées pour échapper aux optima locaux en utilisant le critère de Metropolis \cite{simulated_annealing}.
La meilleure solution trouvée est retournée comme solution finale après un nombre maximal d'itérations.
The orchestration of this algorithm is mainly inspired on the Recherche de Voisinage Variable (Variable Neighborhood Search, VNS) heuristic \cite{vns}.

\subsection{Opérateurs de déplacement}

Les opérateurs de déplacement proposés cherchent à évaluer de nouveaux trajets entre les commandes qui sont dans les différents batches résultant du problème de p-médian.
Nous considérons deux opérateurs de déplacement pour explorer le voisinage de la solution courante: les opérateurs d'échange et de relocalisation.
L'opérateur d'échange échange deux commandes entre deux batches différents, tandis que l'opérateur de relocalisation déplace une commande d'un batch à un autre.

L'opérateur d'échange est défini comme suit: étant donné deux batches \(b_1\) et \(b_2\), et deux commandes \(o_1 \in b_1\) et \(o_2 \in b_2\), l'opérateur échange les commandes entre les batches.
Le batch d'origine \(b_1\) est choisi en priorisant les batches les moins remplis (c'est-à-dire les batches avec le plus grand résidu de capacité), et le batch de destination \(b_2\) est choisi aléatoirement parmi les batches qui peuvent accueillir la commande.
Pour évaluer la solution, le solveur TSP est utilisé pour recalculer les trajets des batches affectés, et l'amélioration est la différence de distance entre les nouveaux et les anciens trajets.
Cependant, cet opérateur n'est pas suffisant car il maintient le même nombre de batches dans la solution.

L'opérateur de relocalisation est donné par deux batches \(b_1\) et \(b_2\), et une commande \(o_1 \in b_1\), l'opérateur déplace la commande vers le batch \(b_2\).
Les mêmes critères de sélection sont utilisés pour choisir les batches d'origine et de destination.
Seul le batch de destination est re-routé, tandis que le batch d'origine est simplement mis à jour sans les articles de la commande déplacée.
La combinaison de ces deux opérateurs permet à l'algorithme d'explorer toutes les solutions possibles dans l'espace de recherche, car nous pouvons obtenir des solutions avec un nombre différent de batches et des commandes différentes dans chaque batch.
Ainsi, nous confirmons la connexité du voisinage, qui est une propriété souhaitable pour les algorithmes de recherche locale.

\subsubsection{Mémoire de Tabu Search}

La mémoire de Tabu Search est utilisée pour stocker les propriétés des solutions qui sont interdites d'être sélectionnées à nouveau.
Une solution est représentée comme une correspondance du batch à l'ensemble des commandes dans le batch.
Aucune solution n'est autorisée à être sélectionnée à nouveau si tous les batches ont le même ensemble de commandes.
Cette mémoire est ajustée pendant le processus de recherche pour forcer l'algorithme à explorer différentes régions de l'espace de recherche (diversification).
À la fin des itérations, la recherche est diversifiée pour échapper aux optima locaux et la mémoire est réduite de moitié.

\subsubsection{Recuit simulé}

La stratégie de recuit simulé est utilisée pour accepter des solutions non améliorantes pour échapper aux optima locaux en fonction d'une probabilité.
La probabilité \(p_{accept}\) d'accepter une solution non améliorante est donnée par le critère de Metropolis, qui est défini comme l'exponentielle de la différence négative entre la nouvelle et l'ancienne solution divisée par la température \ref{eq:metropolis_criterion}.

\begin{align}
    p_{accept} = \exp\left(-\frac{d_{new} - d_{current}}{T}\right) \label{eq:metropolis_criterion}
\end{align}

La température est réduite à chaque itération par un taux de refroidissement, qui est une fonction décroissante du nombre d'itérations.
Ce processus évite de prendre des solutions pires à la fin du processus de recherche (intensification).

\section{Implémentation}

L'implémentation complète de ce projet peut être trouvée dans [ce dépôt](https://github.com/SebastianGrandaA/BatchPicking/).
Il se concentre sur la conception intuitive et modulaire, plutôt que sur la performance, car il s'agit d'une preuve de concept.
Par conséquent, il est adéquat pour les applications hors ligne, sans aucune contrainte de temps d'exécution.

\subsection{Architecture du projet}

Ce projet se compose de trois composants principaux: le domaine, l'application et les services.
Le domaine (`src/domain/`) contient la logique métier de l'application, y compris les modèles et les procédures d'optimisation.
L'application (`src/app/`) implémente trois cas d'utilisation pour interagir avec le domaine: `optimize`, `experiment` et `describe`.
Les services (`src/services/`) contiennent les procédures d'entrée/sortie, y compris les classes de lecture et d'écriture, les calculateurs de distance et d'autres utilitaires externes au domaine.

\subsubsection{Application}

Le fichier `src/__main__.py` est le point d'entrée de l'application. Il initialise l'application et envoie le cas d'utilisation à la fonction correspondante.
Le cas d'utilisation `optimize` est responsable de résoudre une seule instance du problème en utilisant une méthode spécifique; le cas d'utilisation `experiment`, pour exécuter un ensemble d'instances pour comparer différentes méthodes; et le cas d'utilisation `describe`, pour fournir une analyse des résultats.
Ce projet s'attend à ce que les instances soient situées dans `data/`, et les résultats seront enregistrés `results/`.

\subsubsection{Domaine}

En ce qui concerne le domaine, le fichier `src/domain/BatchPicking.py` contient la classe principale, `BatchPicking`, qui orchestre le processus d'optimisation.
Cette classe est responsable de lire les instances, de résoudre le problème et de sauvegarder la meilleure solution trouvée dans un nombre maximal d'itérations.

Il existe deux approches d'optimisation principales implémentées dans ce projet: l'approche séquentielle et l'approche jointe.
Le fichier `src/domain/joint.py` contient l'implémentation de l'approche jointe, qui résout le problème en considérant simultanément les problèmes de regroupement et de routage.
Il existe deux versions de l'implémentation de l'approche jointe: la première utilise la bibliothèque [OR-Tools](https://developers.google.com/optimization) et la seconde, un solveur commercial.

D'autre part, le fichier `src/domain/sequential.py` contient l'implémentation de l'approche séquentielle, qui résout le problème en le décomposant en deux sous-problèmes: le problème de regroupement et le problème de routage.
En plus de la méthode de regroupement `PMedian`, le problème de partitionnement d'ensemble et les algorithmes de regroupement ont également été implémentés comme preuve de concept, dans les classes `GraphPartition` et `Clustering`, respectivement.
D'un point de vue expérimental, le solveur TSP peut être un simple TSP (TSPBase), un TSP multi-flux de marchandises (TSPMultiCommodityFlow), ou une implémentation simplifiée de [OR-Tools](https://developers.google.com/optimization/routing/vrp) pour le problème VRP.
La dernière option est la méthode de routage par défaut car elle montre la meilleure performance en termes de qualité de solution et de temps de calcul.

Comme on peut le voir, le projet est organisé de manière modulaire et implémente plusieurs modèles de conception (héritage, injection de dépendances, et autres) pour faciliter l'extension et la maintenance du code.
Généralement, les méthodes `solve` et `optimize` sont les interfaces pour les méthodes d'optimisation génériques, tandis que les méthodes plus spécifiques sont implémentées sous les sous-modules correspondants, tels que `route`.

\subsubsection{Services}

Enfin, le fichier `src/services/benchmark.py` contient les procédures de benchmarking, qui sont responsables d'exécuter les expériences et d'analyser les résultats.
Les instructions d'installation et d'utilisation peuvent être trouvées dans le fichier `README.md` du dépôt.
La validation des résultats est effectuée en comparant les solutions avec le << S-shaped >> chemin de traitement des commandes individuellement, et elle est extraite du [dépôt UE](https://gitlab.com/LinHirwaShema/poip-2024).

\subsection{Améliorations potentielles}

Des améliorations supplémentaires peuvent être apportées en utilisant un solveur plus spécialisé, tel que le [solveur VRP](https://vrpsolver.math.u-bordeaux.fr) développé à [INRIA](https://www.inria.fr/fr).
Cependant, actuellement l'implémentation en [Python](https://github.com/inria-UFF/VRPSolverEasy) ne couvre pas la fonction de ramassage et de livraison.
Une solution de contournement consiste à implémenter le solveur en [Julia](https://github.com/inria-UFF/BaPCodVRPSolver.jl) et à l'intégrer avec python en utilisant [PyJulia](https://github.com/JuliaPy/pyjulia?tab=readme-ov-file).

\subsection{Langage de programmation et dépendances}

Ce projet est implémenté en Python 3.10.
Cette décision a été principalement motivée par les exigences du projet.
Bien que Python soit un bon choix pour le prototypage en raison de sa simplicité et de sa polyvalence, il n'est pas le meilleur choix pour les applications critiques en termes de performances.
Ce problème est connu sous le nom de Problème des Deux Langages, et Julia est une bonne alternative pour remplacer Python dans ce contexte.
Julia offre un bon équilibre entre performances et productivité car c'est un langage compilé avec une syntaxe similaire à Python, et particulièrement adapté pour le calcul scientifique et les problèmes d'optimisation.

Toutes les dépendances de ce projet sont répertoriées dans le fichier `requirements.txt`. Une liste des principales est fournie ci-dessous:
* [Pyomo](http://www.pyomo.org/): Un langage de modélisation d'optimisation open-source basé sur Python.
* [Gurobi](https://www.gurobi.com/): Un solveur commercial pour les problèmes de programmation mathématique.
* [OR-Tools](https://developers.google.com/optimization): Un ensemble de bibliothèques pour les problèmes d'optimisation combinatoire.

Une représentation arborescente simplifiée du projet est présentée ci-dessous.

.
├── __main__.py
├── app
│   ├── __init__.py
│   ├── describe.py
│   ├── experiment.py
│   └── optimize.py
├── domain
│   ├── BatchPicking.py
│   ├── joint
│   ├── models
│   └── sequential
└── services
    ├── benchmark.py
    ├── distances.py
    ├── io.py
    └── scripts


\section{Expériences numériques}

Les expériences numériques sont effectuées sur toutes les instances disponibles dans `data/` en utilisant un processeur Intel Core i9 8 cœurs à 2,3 GHz.
Les solutions sont comparées avec la solution de référence, qui est le S-shaped chemin fourni pour chaque commande.

\subsection{Caractérisation des instances}

Ces instances sont classées par entrepôt: `A`, `B`, `C`, `D` et `toy`.
Elles peuvent être caractérisées par la taille moyenne des commandes (c'est-à-dire le nombre d'articles par commande).
Nous observons que les instances `A` et `D` ont la plus petite taille moyenne, tandis que les instances `B` et `C` ont la plus grande taille moyenne.
    ... TODO table ...

Cette caractérisation est pertinente pour notre approche de solution car les commandes avec un grand nombre d'articles nécessitent plus d'efforts d'optimisation pour résoudre le problème de routage.
Au contraire, si l'instance contient de nombreuses commandes avec peu d'articles chacune, il y a plus d'opportunités pour regrouper les commandes ensemble, et la qualité de la solution est attendue meilleure.

Un exemple de la disposition de l'entrepôt est présenté dans la figure suivante \ref{fig:warehouse_layout}.
    ... TODO img disposition de l'entrepôt (x1)...


\subsection{Résultats}

L'instance `toy`, en plus de la solution du chemin S-shaped, contient une solution supplémentaire qui s'améliore de 44\% par rapport à la solution de référence.
Cependant, nous observons que notre solution, dans les approches séquentielle et jointe, surpasse cette amélioration de 20\% et 13\%, respectivement, comme le montre le tableau ci-dessous.

\begin{center}
\begin{tabular}{|c|c|c|c|}
\hline
Instance & Provided & Sequential & Joint \\
\hline
toy & 44\% & 65\% & 57\% \\
\hline
\end{tabular}
\end{center}

Les distributions de l'amélioration relative par rapport à la solution de référence sont présentées dans les box-plots suivants \ref{fig:boxplot_improvement_warehouse} et \ref{fig:boxplot_improvement_method}.
    ... TODO img boxplot improvement (x2)...

Les résultats montrent que la plupart des améliorations significatives sont observées dans les instances de l'entrepôt `A`, avec un N\% d'amélioration.
Cependant, la plus grande amélioration (valeur aberrante) est observée dans l'entrepôt `C`, atteignant un N\% d'amélioration.
Les instances des entrepôts `B` et `D` ont une amélioration similaire, qui varie de N\% à N\%.

Globalement, l'amélioration moyenne de toutes les méthodes est de N\%, et de N\% dans le percentile 90.
Nous mesurons également le pourcentage de regroupement, qui est la quantité de commandes dans le batch divisée par la quantité totale de commandes.
    ...explicar todas las metricas ...
    ... TODO table metrics ...

En ce qui concerne le temps de calcul, l'approche séquentielle est considérablement plus rapide que l'approche jointe, comme le montre le box-plot suivant \ref{fig:boxplot_time_warehouse_method}.
    ... TODO img boxplot time (x1)...

Plusieurs instances ont été résolues uniquement par l'approche séquentielle, car l'approche jointe était trop lente et a atteint la limite de temps.
Des exemples de trajets sont présentés dans les figures suivantes \ref{fig:routes_warehouse}.
    ... TODO img routes (x3)...

Un tableau complet des résultats peut être trouvé dans la section Annexes.
    ... TODO Annex: benchamrk_preprocessed.csv ...

Il convient de noter que, bien que nous ayons testé toutes les instances fournies, nous ne rapportons que les instances avec des solutions améliorées par rapport au chemin S-shaped, car c'est notre borne supérieure sur la distance totale parcourue.
Il y avait plusieurs instances avec 0\% d'amélioration, et aucune d'entre elles n'a rapporté une amélioration négative.


\section{Conclusions}

Le Batch-Picking est un problème pertinent dans le contexte de la logistique d'entrepôt, et il a été abordé dans la littérature sous les approches jointe et séquentielle.
Principalement, la littérature connexe se concentre sur un problème de partitionnement d'ensemble pour l'approche séquentielle, et le problème de routage de véhicules groupés pour l'approche jointe.
Dans ce travail, nous avons proposé une méthode heuristique pour résoudre le problème, qui est basée sur le problème de p-médian et le problème du voyageur de commerce (TSP).
Cet algorithme surpasse la solution de référence dans presque toutes les instances, et dans celles où ce n'est pas le cas, l'amélioration n'est pas négative.
Une approche alternative basée sur le problème de ramassage et de livraison (PDP) a également été proposée pour modéliser le problème conjointement, et elle a été résolue en utilisant la méta-heuristique de recherche locale.
La méthode proposée fournit un bon compromis entre la qualité de la solution et le temps de calcul, et elle est adaptée pour les grandes instances du problème.


\section{Annexes}

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

\subsection{Tableau de résultats}

Le tableau suivant présente les résultats complets des expériences numériques.
    ... TODO table: benchmark_preprocessed ...

\subsection{Batching: problème de partitionnement}

Pour formuler le problème de partitionnement, on définit la variable d'affectation $y_{ik} \in \{0, 1\}$ comme 1 si la commande $i \in R$ est affectée au batch $k \in K$, 0 sinon.
La variable $x_{ij} \in \{0, 1\}$ est 1 si les commandes $i, j \in R$ sont affectées au même batch, 0 sinon.
Le problème de batching est formulé comme suit \cite{graph_partitioning}.

\begin{align}
    \max & \sum_{(i, j) \in R \times R} c_{ij} x_{ij} \label{eq:batching_objective} & \\
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
Ce modèle n'a pas réussi à fournir des lots avec plus d'une commande en raison de la limitation présentée dans la description de l'algorithme proposé.

\section{Références}

@article{clustered_vrp,
  title={The joint order batching and picker routing problem: Modelled and solved as a clustered vehicle routing problem},
  author={Aerts, Babiche and Cornelissens, Trijntje and S{\"o}rensen, Kenneth},
  journal={Computers \& Operations Research},
  volume={129},
  pages={105168},
  year={2021},
  publisher={Elsevier}
}

@book{survey_order_batching,
  title={Order batching in order picking warehouses: a survey of solution approaches},
  author={Henn, Sebastian and Koch, S{\"o}ren and W{\"a}scher, Gerhard},
  year={2012},
  publisher={Springer}
}

@article{book_tooth1_chap9,
  title={VRP with Pickup and Delivery.},
  author={Desaulniers, Guy and Desrosiers, Jacques and Erdmann, Andreas and Solomon, Marius M and Soumis, Fran{\c{c}}ois},
  journal={The vehicle routing problem},
  volume={9},
  pages={225--242},
  year={2002},
  publisher={Philadelphia}
}

@incollection{book_tooth2_chap6,
  title={Chapter 6: pickup-and-delivery problems for goods transportation},
  author={Battarra, Maria and Cordeau, Jean-Fran{\c{c}}ois and Iori, Manuel},
  booktitle={Vehicle Routing: Problems, Methods, and Applications, Second Edition},
  pages={161--191},
  year={2014},
  publisher={SIAM}
}

@article{survey_static_pdp,
  title={Static pickup and delivery problems: a classification scheme and survey},
  author={Berbeglia, Gerardo and Cordeau, Jean-Fran{\c{c}}ois and Gribkovskaia, Irina and Laporte, Gilbert},
  journal={Top},
  volume={15},
  pages={1--31},
  year={2007},
  publisher={Springer}
}

@book{book_location_science,
  title={Introduction to location science},
  author={Laporte, Gilbert and Nickel, Stefan and Saldanha-da-Gama, Francisco},
  year={2019},
  publisher={Springer}
}

@article{parragh2008survey,
  title={A survey on pickup and delivery problems: Part I: Transportation between customers and depot},
  author={Parragh, Sophie N and Doerner, Karl F and Hartl, Richard F},
  journal={Journal f{\"u}r Betriebswirtschaft},
  volume={58},
  pages={21--51},
  year={2008},
  publisher={Springer}
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
"""