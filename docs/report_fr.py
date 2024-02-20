"""
\title{Le problème du Batch-Picking}

\section{Résumé}

Le problème du Batch-Picking consiste à regrouper des commandes et à déterminer la séquence de positions de stockage pour ramasser tous les articles d'un batch.
Les articles sont partitionnés en un ensemble de commandes, de telle manière que tous les articles d'une commande doivent être ramassés dans le même trajet, par le même préparateur de commandes.
L'objectif est de minimiser la distance totale parcourue par les préparateurs de commandes pour satisfaire toutes les commandes dans l'entrepôt.
Ce projet implémente deux principales approches d'optimisation: l'approche **séquentielle** et l'approche **jointe**.

\section{Introduction}

Le problème du Batch-Picking, connu dans la littérature comme << Joint Order Batching and Picker Routing Problem >> (JOBPRP), est un problème bien connu dans le contexte de la logistique d'entrepôt.
Comme son nom l'indique, le problème combine deux problèmes d'optimisation: le problème de regroupement de commandes et le problème de routage de préparateur de commandes.
Le problème de regroupement de commandes consiste à regrouper des sous-ensembles de commandes pour ramasser leurs articles ensemble, si cela conduit à une réduction de la distance totale parcourue.
D'autre part, le problème de routage de préparateur de commandes consiste à déterminer la séquence de positions de stockage pour ramasser toutes les commandes dans un batch.
Ce problème a été étudié dans la littérature sous les approches jointe et séquentielle.

Un problème de routage de véhicules groupés (Clustered Vehicle Routing Problem, CluVRP) a été proposé par \cite{cvrp_jopbrp} pour modéliser le problème joint.
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

\(\underline{B} = \max\left(\frac{\sum_{i \in I} v_i}{C_{volume}}, \frac{|O|}{C_{unit}}\right) \times (1 + \alpha)\) \label{eq:min_batches}

Soit \(d(i) \in J\) le nœud artificiel, ou point de livraison, du nœud d'article \(i\).
Les nœuds artificiels agissent comme des points de consolidation pour chaque commande, en garantissant que tous les articles d'une commande sont ramassés dans le même trajet.
Cela est réalisé en forçant les préparateurs de commandes à visiter les nœuds d'articles et à les livrer aux nœuds artificiels.

Nous présentons une formulation de flux de marchandises à trois indices qui implique deux types de variables: les variables de sélection et de flux.


    Translate the following to french:


"""