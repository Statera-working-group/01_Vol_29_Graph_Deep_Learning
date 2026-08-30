**Volume 29. Graph Deep Learning**

# Chapter 01. Graph Theory Basics

## 01.00. Overview

![](images/image1.png){width="7.268055555555556in" height="7.268055555555556in"}

Graph theory provides a mathematical language for describing systems whose behavior depends not only on individual entities but also on the relationships among them. A graph represents entities as nodes and relationships as edges, allowing complex structures to be expressed independently of conventional grids or sequences. This relational viewpoint forms the conceptual foundation of graph representation learning and graph deep learning.

Unlike images, which are naturally arranged on regular two-dimensional grids, or text, which usually follows an ordered sequence, graph data has irregular connectivity. A node may have one neighbor or thousands of neighbors, and the topology itself may carry important information. Graph learning therefore requires methods that can process both attributes associated with entities and the structural patterns created by their connections.

A graph is commonly represented as G = (V, E), where V denotes a set of vertices or nodes and E denotes a set of edges connecting pairs of nodes. Nodes can represent people, documents, molecules, road intersections, robots, sensors, or abstract concepts. Edges can represent friendship, citation, chemical bonds, roads, communication links, spatial relationships, or any other meaningful interaction between entities.

The meaning of an edge depends strongly on the modeled system. In an undirected graph, an edge expresses a symmetric relationship, such as mutual connectivity between two devices. In a directed graph, the relationship has an orientation, such as one webpage linking to another. Edges may additionally carry weights that describe distance, strength, probability, cost, capacity, similarity, or another quantitative property.

Graphs may also contain multiple categories of nodes and relationships. Bipartite graphs, for example, separate nodes into two groups and permit connections primarily between the groups, making them useful for representing users and products in recommendation systems. More general heterogeneous graphs can contain many node and edge types, providing a natural representation for knowledge graphs, multimodal systems, and complex industrial environments.

Graph structure can be expressed algebraically through matrices. The adjacency matrix records which nodes are connected, while the degree matrix summarizes the number or weighted strength of connections associated with each node. Combining these structures produces the graph Laplacian, a central mathematical object that connects graph topology with linear algebra, spectral analysis, diffusion processes, and many graph learning algorithms.

Important graph properties describe how information and connectivity are distributed across a network. Connectivity determines whether nodes can reach one another through paths, centrality identifies structurally influential nodes, and clustering measures the tendency of neighboring nodes to form densely connected groups. Graph spectra reveal additional global structural characteristics through eigenvalues and eigenvectors derived from graph matrices.

Classical graph algorithms provide computational mechanisms for exploring these structures. Breadth-first search and depth-first search systematically traverse connected nodes, shortest-path algorithms identify efficient routes through networks, and ranking algorithms such as PageRank estimate the relative importance of nodes from connectivity patterns. These algorithms illustrate how structural information can directly support search, navigation, ranking, and reasoning.

Real-world graphs are often too complicated to characterize only through deterministic structures, making random graph models important theoretical tools. Random graphs provide controlled ways to study how network properties emerge from probabilistic connection rules. They help explain phenomena such as degree distributions, connectivity transitions, communities, hubs, and the difference between purely random networks and highly organized real-world systems.

The fundamental distinction between graph data and conventional feature vectors is that relationships are part of the data rather than merely auxiliary metadata. Two nodes with similar attributes may behave differently because they occupy different structural positions. Conversely, nodes with different local features may serve similar functions when their neighborhoods and connectivity patterns are structurally equivalent.

This observation becomes especially important in machine learning. Traditional neural networks typically process samples as approximately independent observations or assume regular spatial and sequential structures. Graph neural networks instead propagate and aggregate information through relationships. A node representation can therefore be updated using information from neighboring nodes, enabling the learned representation to encode both local attributes and relational context.

This mechanism is commonly understood through message passing. Each node receives information from connected neighbors, transforms or aggregates those messages, and combines them with its existing state. Repeating this operation across multiple layers expands the effective receptive field, allowing information from increasingly distant parts of the graph to influence the representation while preserving the graph\'s relational organization.

Graph learning tasks can consequently operate at several levels. Node-level prediction assigns properties or labels to individual entities, edge-level prediction estimates relationships between entities, and graph-level prediction characterizes an entire graph. These perspectives support applications ranging from document classification and recommendation to molecular property prediction, fraud detection, knowledge reasoning, and infrastructure analysis.

Graph theory also provides a bridge between mathematical structure and physical systems. Transportation networks can be modeled as locations connected by routes, communication systems as devices connected by links, and robotic environments as objects, places, sensors, and agents connected through spatial or functional relationships. Graphs therefore provide a flexible abstraction for representing interactions that cannot be adequately described by isolated feature vectors.

In robotics and spatial AI, this relational perspective can complement geometric representations. Scene graphs describe objects and their semantic or spatial relationships, map graphs describe navigable locations and connectivity, and sensor relationship graphs can represent dependencies among heterogeneous sensing sources. Such structures provide a foundation for reasoning about environments beyond raw pixels, point clouds, or coordinate measurements.

The same principle extends to large interconnected systems. Traffic networks, utility networks, social networks, biological interaction networks, knowledge graphs, and distributed AI systems all contain entities whose significance depends on their relationships. Graph representations make these dependencies explicit and allow algorithms to reason over local neighborhoods, global topology, paths, communities, and dynamically changing connections.

Graph deep learning builds upon these foundations by combining graph structure with learned representations. Graph Convolutional Networks, Graph Attention Networks, message-passing architectures, graph embeddings, graph transformers, and graph generative models can be viewed as increasingly powerful mechanisms for learning from relational structure. Understanding their behavior requires familiarity with the basic mathematical objects and properties introduced by graph theory.

The progression from graph definitions to matrix representations, structural properties, classical algorithms, random graphs, and applications therefore establishes the foundation for the remainder of graph deep learning. These concepts prepare the transition from explicitly designed graph computations toward neural architectures that learn how information should propagate, combine, and transform across complex relational structures.

## 01.01. Graph Definitions

![](images/image2.png){width="7.268055555555556in" height="7.268055555555556in"}

A graph provides a formal representation of entities and the relationships that connect them. It is commonly written as G = (V, E), where V is a set of vertices, or nodes, and E is a set of edges. This simple abstraction can describe structures ranging from social and citation networks to molecules, transportation systems, knowledge graphs, communication networks, and robotic environments.

Nodes represent the fundamental entities within a graph. Depending on the application, a node may correspond to a person, document, product, protein, road intersection, computer, sensor, robot, location, or conceptual object. Nodes can also contain attributes called node features, such as category, position, measurement values, semantic properties, or learned numerical representations used by graph-based machine learning models.

Edges represent relationships or interactions between nodes. An edge connecting nodes u and v indicates that some defined relationship exists between those entities. The relationship may describe physical connectivity, communication, similarity, citation, friendship, chemical bonding, spatial proximity, or information exchange. Consequently, graph structure explicitly represents dependencies that conventional collections of independent feature vectors often fail to capture.

The neighborhood of a node consists of other nodes connected to it by edges. This local connectivity becomes particularly important in graph deep learning because information can be exchanged across neighborhoods. Graph Neural Networks (GNNs) commonly use these connections to aggregate information from neighboring nodes, allowing each node representation to reflect not only its own attributes but also the relational context surrounding it.

Graphs can be classified according to whether their edges possess direction. In an undirected graph, an edge between nodes u and v represents a symmetric relationship, meaning that the connection from u to v is equivalent to the connection from v to u. Examples include certain friendship networks, physical links, and mutual communication structures where the relationship does not inherently distinguish a source from a destination.

A directed graph, or digraph, assigns an orientation to each edge. An edge u → v therefore describes a relationship originating at u and terminating at v, which does not automatically imply the reverse relationship v → u. Directed graphs naturally represent webpage hyperlinks, citation networks, dependency structures, information flows, command relationships, transportation directions, and many other asymmetric interactions.

Direction changes how neighborhoods and connectivity are interpreted. A node in a directed graph can have incoming edges and outgoing edges, producing corresponding in-degree and out-degree measures. This distinction enables graph algorithms and learning systems to determine whether a node primarily receives information, distributes information, or participates in both roles, providing richer structural information than an undirected connection alone.

The choice between directed and undirected representations should reflect the semantics of the modeled relationship rather than computational convenience. Converting a directed graph into an undirected graph may simplify processing but can remove information about causal, temporal, functional, or communication direction. Conversely, imposing artificial direction on inherently symmetric relationships can introduce distinctions that do not exist in the underlying system.

Edges may contain numerical values in addition to connectivity information, producing a weighted graph. A weighted edge assigns a value w(u,v) to the relationship between two nodes. The weight can represent distance, travel time, communication latency, bandwidth, interaction frequency, similarity, probability, cost, capacity, confidence, or physical strength depending on the semantics of the application being modeled.

Weights make it possible to distinguish relationships that would otherwise appear structurally identical. Two roads may both connect neighboring locations but have significantly different travel times, while two communication links may have different bandwidth or latency. Similarly, relationships in a recommendation or knowledge system may have different confidence levels, making weighted graphs capable of representing both connectivity and relationship intensity.

Graph algorithms frequently use weights when determining paths, rankings, flows, or similarities. In shortest-path problems, for example, minimizing the number of edges may produce a different solution from minimizing total distance or cost. Graph learning models can similarly incorporate edge weights into message aggregation, allowing stronger, more reliable, or more relevant connections to exert different influences on learned node representations.

Weighted graphs may be either directed or undirected, demonstrating that graph characteristics can be combined rather than treated as mutually exclusive categories. A transportation network can contain directed roads with travel-time weights, while a communication network can contain symmetric links with bandwidth weights. Real graph systems therefore often require multiple structural properties to represent their relationships accurately.

A bipartite graph introduces another important structural organization by dividing the node set into two disjoint groups, conventionally written as U and V. Edges connect nodes belonging to different groups rather than connecting nodes within the same group. This structure explicitly represents interactions between two different classes of entities and appears naturally in many machine learning and information-system applications.

Recommendation systems provide a common example of bipartite structure. One group of nodes can represent users and the other products, movies, documents, or services. An edge indicates an interaction such as a purchase, rating, click, or view. Learning patterns from these connections can reveal user preferences and item characteristics, enabling systems to estimate which previously unseen connections are likely to occur.

Bipartite graphs are also useful for representing authors and publications, customers and transactions, workers and tasks, sensors and observations, or robots and assigned resources. Their explicit separation of entity classes prevents fundamentally different object types from being treated as equivalent nodes while preserving their interactions within a unified relational representation.

Nodes, edges, direction, weights, and bipartite organization together establish the basic vocabulary needed to describe graph-structured data. These concepts determine what entities exist, how they interact, whether relationships have orientation, how strongly relationships are expressed, and whether nodes belong to distinct structural classes. They consequently define the topology on which subsequent graph analysis and learning operate.

These graph definitions also establish the foundation for mathematical representations such as adjacency matrices, degree matrices, and graph Laplacians. More importantly for graph deep learning, they determine the pathways through which information can propagate. Graph representations therefore transform relationships from informal descriptions into explicit computational structures that can be analyzed by algorithms and ultimately learned by Graph Neural Networks (GNNs).

## 01.02. Matrix Representations

![](images/image3.png){width="7.268055555555556in" height="7.268055555555556in"}

The mathematical representation of a graph converts relational structure into numerical objects that can be processed by linear algebra and machine learning algorithms. For a graph G = (V, E) containing n nodes, matrices provide a systematic way to encode connectivity, node degrees, and structural relationships. Among the most fundamental representations are the adjacency matrix, degree matrix, and graph Laplacian.

The adjacency matrix A directly represents which nodes are connected. For an unweighted graph with n nodes, A is an n × n matrix in which Aᵢⱼ = 1 when an edge exists between nodes i and j, and Aᵢⱼ = 0 otherwise. The matrix therefore transforms a collection of edges into a regular numerical structure that can be manipulated using matrix operations.

For an undirected graph, connectivity is symmetric, so Aᵢⱼ = Aⱼᵢ and the adjacency matrix is symmetric about its main diagonal. In a directed graph, however, Aᵢⱼ can represent an edge from node i to node j without requiring Aⱼᵢ to exist. The matrix can therefore preserve the orientation of relationships and distinguish incoming connectivity from outgoing connectivity.

Weighted graphs extend this representation by storing edge weights instead of simple binary indicators. If an edge between nodes i and j has weight wᵢⱼ, the corresponding matrix element can be defined as Aᵢⱼ = wᵢⱼ. Depending on the application, these values may represent distance, similarity, probability, interaction strength, bandwidth, cost, or another quantitative characteristic of the relationship.

Adjacency matrices provide an intuitive representation, but they can become inefficient for large sparse graphs. A graph containing millions of nodes may have only a small number of edges per node, while a dense n × n matrix reserves storage for every possible pair. Practical graph software therefore frequently uses sparse matrix formats or edge-list representations while retaining the same mathematical interpretation of adjacency.

The adjacency matrix also supports useful structural computations. Matrix multiplication can reveal multi-hop connectivity because powers of A encode information about walks through the graph. For example, entries of A² are related to the number of two-step walks between pairs of nodes. This algebraic interpretation establishes an important bridge between graph traversal, neighborhood aggregation, and later graph neural network operations.

In practical implementations, an adjacency matrix can be constructed from an edge list using numerical libraries such as NumPy, SciPy, or graph-processing frameworks. For a small graph, code may initialize an n × n matrix with zeros and assign values at positions corresponding to edges. For an undirected graph, both Aᵢⱼ and Aⱼᵢ are updated, while directed graphs update only the specified orientation.

The degree of a node summarizes how strongly that node is connected to the surrounding graph. In a simple unweighted undirected graph, the degree dᵢ of node i equals the number of edges incident to that node. Using the adjacency matrix, this quantity can be calculated as dᵢ = Σⱼ Aᵢⱼ, demonstrating how local connectivity can be extracted directly from the matrix representation.

The degree matrix D organizes these node degrees into an n × n diagonal matrix. Its diagonal elements satisfy Dᵢᵢ = dᵢ, while all off-diagonal elements are zero. Unlike the adjacency matrix, which describes pairwise connections, the degree matrix summarizes the total connectivity associated with each individual node and therefore captures an important local structural property.

For weighted graphs, node degree can be generalized to weighted degree, sometimes called node strength. Instead of counting neighboring edges, the weighted degree sums their weights. Directed graphs require additional distinctions between in-degree and out-degree, which can be obtained by summing appropriate columns or rows of the adjacency matrix according to the adopted matrix convention.

The degree matrix plays a central role in normalization. Nodes in real networks can have dramatically different numbers of neighbors, meaning that simple aggregation may cause high-degree nodes to accumulate much larger values than low-degree nodes. Degree-based normalization compensates for this imbalance and becomes especially important when graph information is repeatedly propagated through multiple computational layers.

The graph Laplacian combines adjacency and degree information into a matrix that captures important aspects of graph structure. The basic unnormalized Laplacian is defined as L = D − A. Its diagonal elements describe node degrees, while its off-diagonal elements encode negative adjacency relationships. This apparently simple construction provides one of the most important connections between graph theory and linear algebra.

The Laplacian can be interpreted in terms of differences between connected nodes. When a vector assigns a numerical value to every node, multiplication by L compares each node with values found in its neighborhood. Consequently, the Laplacian naturally describes smoothness, diffusion, flow, and variation across graph structure, making it useful in clustering, signal processing, optimization, and graph-based machine learning.

A fundamental expression associated with the Laplacian is xᵀLx, which measures how much a node-valued signal x varies across connected nodes. When neighboring nodes have similar values, this quantity tends to remain small; when strongly connected nodes have very different values, it becomes larger. This property provides a mathematical foundation for graph smoothness assumptions frequently used in representation learning.

Normalized versions of the Laplacian compensate for differences in node degree. A commonly used symmetric form is L_sym = I − D⁻¹ᐟ²AD⁻¹ᐟ², while another form is the random-walk Laplacian L_rw = I − D⁻¹A. These formulations modify the influence of connectivity according to node degree and provide useful interpretations for diffusion, random walks, spectral methods, and normalized graph propagation.

Eigenvalues and eigenvectors of the graph Laplacian reveal global structural properties that are difficult to observe from individual edges alone. The smallest eigenvalue is associated with fundamental connectivity structure, while additional eigenvalues and eigenvectors provide information related to connected components, partitions, smooth graph signals, and spectral organization. This forms the basis of spectral graph theory.

The connection between Laplacian matrices and spectral analysis is particularly important for graph deep learning. Early spectral graph convolution methods defined filtering operations using eigenvectors of the Laplacian, interpreting them as a graph analogue of Fourier bases. Later Graph Convolutional Networks simplified these operations, but degree normalization and adjacency-based propagation remain visible in many modern GNN formulations.

From an implementation perspective, Laplacian construction follows directly from the previous representations. Code first builds the adjacency matrix A, computes node degrees by summing the appropriate adjacency values, constructs the diagonal matrix D, and then evaluates L = D − A. Normalized variants additionally require inverse degree or inverse square-root degree factors, with special handling for isolated nodes whose degree is zero.

The adjacency matrix, degree matrix, and Laplacian therefore describe complementary aspects of the same graph. A records who is connected to whom, D summarizes how much connectivity each node possesses, and L combines these quantities to characterize relational differences and structural variation. Together they transform graph topology into algebraic forms suitable for efficient computation and theoretical analysis.

These matrix representations establish the mathematical bridge from classical graph theory to graph deep learning. Adjacency determines the pathways through which information can travel, degree information controls how neighboring contributions can be normalized, and Laplacian structure provides a foundation for diffusion and spectral reasoning. Their combination ultimately supports the neighborhood aggregation and message-passing mechanisms used throughout modern Graph Neural Networks.

## 01.03. Graph Properties

![](images/image4.png){width="7.268055555555556in" height="7.268055555555556in"}

Graph properties provide quantitative and structural descriptions of how nodes and edges are organized within a network. While an adjacency matrix records individual connections, graph properties reveal higher-level patterns such as whether information can travel across the network, which nodes occupy influential positions, whether dense local groups exist, and how global topology can be characterized through spectral structure.

Connectivity describes whether nodes can reach one another through sequences of edges called paths. In an undirected graph, two nodes are connected if at least one path exists between them, and the entire graph is connected when every pair of nodes is mutually reachable. If this condition is not satisfied, the graph separates into connected components, each representing a maximal group of internally reachable nodes.

Connectivity becomes more nuanced in directed graphs because the existence of a path from node u to node v does not guarantee a path from v to u. A directed graph is strongly connected when every node can reach every other node while respecting edge directions. Weak connectivity instead considers whether the graph becomes connected when edge directions are ignored, providing a less restrictive description of global reachability.

Paths also provide information about the distance between nodes. The shortest-path distance is the minimum number of edges, or minimum accumulated weight in a weighted graph, required to travel between two nodes. Measures such as graph diameter and average path length summarize these distances across the network and help characterize how efficiently information, resources, or influence can propagate through the graph.

Connectivity is particularly important in communication networks, transportation systems, distributed computing, and multi-agent systems. A disconnected communication graph may prevent information from reaching some agents, while redundant paths can improve robustness when individual links fail. In graph learning, connectivity determines which nodes can eventually exchange information through repeated neighborhood aggregation or message-passing operations.

Centrality measures attempt to identify nodes that occupy structurally important positions within a graph. Importance can have several meanings, so different centrality measures capture different structural roles. A node may be important because it has many direct neighbors, lies on many communication paths, remains close to the rest of the network, or is connected to other highly influential nodes.

Degree centrality is one of the simplest measures and evaluates importance using the number of direct connections associated with a node. High-degree nodes can function as local hubs because they interact directly with many neighbors. In directed graphs, in-degree centrality and out-degree centrality distinguish nodes that receive many connections from those that generate or distribute many connections.

Betweenness centrality measures how frequently a node lies on shortest paths connecting other node pairs. A node may therefore have relatively few direct neighbors while still acting as an important bridge between communities or regions of a network. Removing such a node can significantly disrupt information flow, making betweenness useful for identifying bottlenecks, gateways, and structurally critical intermediaries.

Closeness centrality evaluates how near a node is to other nodes according to shortest-path distances. Nodes with high closeness can reach the rest of the connected network through relatively short paths. Eigenvector-based centrality introduces another perspective by assigning greater importance to nodes connected to other important nodes, creating a recursive definition of influence that underlies ranking methods such as PageRank-related approaches.

Clustering describes the tendency of neighboring nodes to form densely interconnected local groups. If node u is connected to nodes v and w, and v and w are also connected, the three nodes form a triangle. The frequency of such triangles provides an intuitive measure of local cohesion and distinguishes networks whose neighborhoods contain strong internal relationships from networks with more tree-like or sparse local structure.

The local clustering coefficient measures how completely the neighbors of a particular node are connected to one another. A value near one indicates that most possible connections among its neighbors exist, while a value near zero indicates that few of those neighbors interact directly. Averaging local clustering coefficients across nodes provides one way to summarize the clustering tendency of the entire graph.

Clustering is closely related to community structure but is not identical to it. A community generally refers to a larger group of nodes having relatively dense internal connectivity and comparatively sparse external connectivity, whereas clustering coefficients primarily describe local triangular relationships. Nevertheless, high local clustering often provides evidence of cohesive groups that can contribute to broader community organization.

Real-world networks frequently exhibit combinations of short path lengths and significant clustering. Social networks, biological interaction networks, knowledge structures, and infrastructure systems can contain tightly connected local regions while remaining globally reachable through a smaller number of bridging connections. Understanding this balance helps explain how local organization and global communication coexist within complex networks.

Graph spectra provide a fundamentally different perspective by studying the eigenvalues and eigenvectors of matrices associated with a graph. Common choices include the adjacency matrix and graph Laplacian. Instead of examining individual nodes or edges directly, spectral analysis transforms graph structure into algebraic quantities that reveal global patterns related to connectivity, partitions, diffusion, smoothness, and structural organization.

The spectrum of the graph Laplacian is especially important. For an undirected graph, its eigenvalues are real and nonnegative, and the multiplicity of the zero eigenvalue corresponds to the number of connected components. Consequently, spectral information can determine whether a graph is connected without explicitly performing a conventional traversal of every possible relationship.

The second-smallest Laplacian eigenvalue, often called the algebraic connectivity or Fiedler value, provides information about how strongly a graph is connected. A value of zero indicates disconnection, while larger positive values generally correspond to stronger structural connectivity. The associated Fiedler vector can help separate nodes into groups and therefore provides a foundation for spectral partitioning and spectral clustering methods.

Graph eigenvectors can also be interpreted as graph-frequency components, extending concepts from classical signal processing to irregular domains. Smooth eigenvectors vary slowly across strongly connected nodes, whereas higher-frequency components represent more rapid variation across edges. This interpretation enables graph signals to be analyzed, filtered, smoothed, or decomposed according to the topology on which they are defined.

Spectral properties have a direct connection to graph machine learning. Spectral graph convolution originally defined filtering operations through the eigenstructure of graph Laplacians, while modern Graph Neural Networks often implement localized approximations through neighborhood aggregation and normalized adjacency operations. The underlying objective remains closely related: information is transformed according to the relational structure encoded by the graph.

Connectivity, centrality, clustering, and graph spectra therefore describe complementary dimensions of graph organization. Connectivity explains reachability, centrality identifies structurally significant nodes, clustering characterizes local cohesion, and spectra expose global algebraic structure. Together they provide a richer description than individual edges or node features can provide and reveal how local relationships produce network-level behavior.

These properties become particularly valuable when graphs represent physical or distributed systems. In robotic fleets, sensor networks, transportation systems, or multi-agent environments, connectivity can determine communication feasibility, centrality can identify critical agents or infrastructure, clustering can reveal operational groups, and spectral properties can characterize synchronization, diffusion, robustness, and collective information flow.

Understanding these graph properties establishes an important foundation for graph deep learning because learned representations are strongly influenced by topology. Message passing follows connectivity, aggregation can be biased by centrality and degree, communities create correlated neighborhoods, and spectral structure governs important forms of graph smoothing and propagation. Graph properties therefore connect classical network analysis directly to the behavior and limitations of modern Graph Neural Networks.

## 01.04. Graph Algorithms [w/Code]

![](images/image5.png){width="7.268055555555556in" height="7.268055555555556in"}

Graph algorithms provide systematic procedures for exploring, analyzing, and extracting information from graph structures. While matrices describe connectivity algebraically, algorithms operate on nodes and edges to answer practical questions such as which nodes are reachable, how two locations can be connected efficiently, or which entities are structurally important. Breadth-First Search, Depth-First Search, shortest-path methods, and PageRank are foundational examples.

Breadth-First Search (BFS) explores a graph outward from a selected source node one neighborhood level at a time. It first visits the source, then all directly connected neighbors, followed by nodes two edges away, and continues until no unexplored reachable nodes remain. A queue is normally used to preserve this level-by-level order, while a visited structure prevents nodes from being processed repeatedly.

BFS naturally constructs a breadth-first search tree describing how nodes are discovered from the source. In an unweighted graph, the first time BFS reaches a node corresponds to a path containing the minimum possible number of edges from the source. This makes BFS useful not only for traversal but also for unweighted shortest-path computation, connectivity testing, neighborhood discovery, and distance-based graph analysis.

A typical BFS implementation initializes a queue with the source node and marks that node as visited. The algorithm repeatedly removes the oldest node from the queue, examines its adjacency list, and inserts every unvisited neighbor. With an adjacency-list representation, its time complexity is commonly expressed as O(\|V\| + \|E\|), because each node and edge needs to be examined only a limited number of times.

Depth-First Search (DFS) follows a different exploration strategy. Instead of expanding all nearby nodes first, DFS selects an unexplored neighbor and continues deeper along that branch until further progress is impossible. It then backtracks to the most recent node with unexplored neighbors. This behavior can be implemented recursively through the call stack or iteratively using an explicit stack data structure.

DFS is particularly useful when the structure of paths matters more than distance from the source. It supports connected-component discovery, cycle detection, topological sorting, dependency analysis, and many other graph procedures. Like BFS, DFS typically runs in O(\|V\| + \|E\|) time with adjacency lists, but the two algorithms produce different traversal orders and reveal different structural perspectives.

The practical distinction between BFS and DFS is therefore determined by the objective of graph exploration. BFS emphasizes distance layers and is appropriate when minimum-hop paths or nearest reachable nodes are required. DFS emphasizes deep structural exploration and backtracking. Both algorithms also demonstrate a fundamental principle that later appears in graph learning: computation follows the connectivity defined by graph edges.

Shortest-path algorithms generalize the problem of finding efficient routes between nodes. A path consists of a sequence of connected edges, and its cost can be defined by the number of edges or by the sum of edge weights. The shortest path is the valid path having minimum total cost, making shortest-path computation fundamental to routing, navigation, communication, logistics, dependency analysis, and robotic planning.

For unweighted graphs, BFS directly produces shortest paths measured by edge count. Weighted graphs require algorithms that consider numerical edge costs. Dijkstra\'s algorithm is a standard solution when all edge weights are nonnegative. It repeatedly selects the currently reachable node with the smallest known distance and relaxes its outgoing edges to determine whether shorter routes to neighboring nodes have been discovered.

Relaxation is the central operation in many shortest-path algorithms. If the known distance from source s to node u is d(u), and an edge from u to v has weight w(u,v), the candidate distance to v becomes d(u) + w(u,v). When this candidate is smaller than the current d(v), the distance and predecessor information are updated. Repeated relaxation progressively constructs optimal routes from the source.

Efficient implementations of Dijkstra\'s algorithm commonly use a priority queue so that the node with the smallest tentative distance can be selected quickly. Code typically maintains a distance table, predecessor information, and a heap containing candidate nodes. The predecessor structure is important because distance values alone indicate path cost, whereas predecessor links allow the actual sequence of nodes forming the shortest path to be reconstructed.

Not every shortest-path problem can use Dijkstra\'s assumptions. Graphs containing negative edge weights require methods such as Bellman-Ford, while all-pairs shortest paths can be computed using methods such as Floyd-Warshall or repeated single-source searches. Selecting an algorithm therefore depends on graph size, sparsity, edge-weight properties, and whether distances are needed from one source or between many pairs.

PageRank addresses a different graph problem: estimating the relative importance of nodes from directed connectivity. It was originally developed for ranking webpages, where a hyperlink can be interpreted as one page transferring some importance to another. The central insight is that a node should receive a high score not simply because many nodes point to it, but because important nodes point to it.

This produces a recursive definition of importance. Conceptually, the PageRank score of node i depends on contributions from nodes linking to i, with each source node distributing its influence across its outgoing links. A node with many high-quality incoming connections can therefore achieve a larger score than another node having the same number of incoming edges from structurally unimportant sources.

PageRank is commonly expressed using a random-surfer interpretation. Imagine an agent moving through a directed graph by following outgoing edges. With probability d, called the damping factor, the agent follows a graph link, while with probability 1 − d it jumps to another node according to a teleportation distribution. This mechanism prevents the ranking process from becoming trapped in certain graph structures.

In matrix form, PageRank can be understood as finding a stationary distribution associated with a modified transition matrix. Computational implementations often use power iteration: initialize node scores, repeatedly distribute scores through outgoing links, add the teleportation contribution, and continue until changes become sufficiently small. Special treatment is required for dangling nodes that contain no outgoing edges.

PageRank illustrates how local relationships can generate a global measure of importance. Each iteration uses neighboring connectivity, yet repeated propagation allows information from distant regions of the graph to influence every node\'s final ranking. This principle resembles later graph learning mechanisms in which repeated message passing progressively expands the receptive field and integrates increasingly broad structural context.

Implementation choices strongly affect the efficiency of graph algorithms. Large real-world graphs are usually sparse, so adjacency lists, sparse matrices, queues, stacks, heaps, and iterative numerical procedures are preferred over dense representations when possible. Libraries such as NetworkX can provide convenient reference implementations, while specialized frameworks and custom data structures are often required for large-scale graph processing.

BFS, DFS, shortest-path algorithms, and PageRank collectively demonstrate several fundamental forms of graph computation. BFS organizes information by neighborhood distance, DFS reveals deep structural relationships, shortest-path methods optimize traversal costs, and PageRank propagates importance through directed links. Each algorithm transforms raw connectivity into information that can support reasoning, decision-making, ranking, or planning.

These classical algorithms also provide conceptual foundations for graph deep learning. Graph Neural Networks similarly operate by following edges, gathering information from neighborhoods, and repeatedly propagating transformed values through graph structure. Classical graph algorithms use explicitly designed rules, whereas GNNs learn many of their transformation and aggregation functions from data, creating a natural progression from algorithmic graph processing to learned relational computation.

## 01.05. Random Graphs

![](images/image6.png){width="7.268055555555556in" height="7.268055555555556in"}

Random graphs provide mathematical models for studying networks whose edges, nodes, or structural properties are generated according to probabilistic rules. Instead of defining every connection explicitly, a random graph specifies a probability distribution over possible graph structures. This approach makes it possible to investigate how complex network properties can emerge from simple stochastic mechanisms.

The central idea is to treat a graph G as a random variable rather than as one fixed structure. A random graph model defines an ensemble of possible graphs together with probabilities describing how likely each graph is to occur. Researchers can then study expected degree, connectivity, clustering, path length, component size, and other properties across the ensemble instead of analyzing only one observed network.

One of the simplest and most influential random graph models is the Erdős--Rényi model. In the G(n,p) formulation, the graph contains n nodes and every possible pair of nodes is independently connected with probability p. When p is small the resulting graph is sparse, while increasing p progressively introduces more edges and eventually produces a densely connected network.

For an undirected Erdős--Rényi graph, each node has n − 1 possible neighbors, and its expected degree is approximately (n − 1)p. The degree distribution follows a binomial distribution because every potential edge is generated through an independent Bernoulli trial. For large sparse graphs under appropriate conditions, this distribution can often be approximated by a Poisson distribution.

Random graphs reveal how global network behavior can change dramatically when a local probability parameter changes only slightly. As the connection probability increases, isolated nodes begin to join small components, components merge, and eventually a large connected structure can emerge. Such rapid structural transitions are commonly described as phase transitions and are fundamental to the study of complex networks.

A particularly important phenomenon is the emergence of a giant component. At sufficiently low edge density, most connected components remain relatively small. As the average degree passes a critical regime, however, a component containing a substantial fraction of all nodes can appear. This illustrates how large-scale connectivity can emerge without any centralized mechanism explicitly organizing the entire network.

Another important threshold concerns complete graph connectivity. The appearance of a giant component does not mean that every node belongs to the same connected component. A graph can contain one dominant component while still having isolated nodes or smaller disconnected groups. As edge probability increases further, the probability that the entire graph becomes connected can rise sharply.

Random graph models therefore provide useful baselines for determining whether observed network structures are genuinely meaningful. If a real network exhibits substantially more clustering, different path lengths, stronger communities, or a different degree distribution than a comparable random graph, these differences suggest that additional mechanisms influence its organization rather than connectivity arising purely from independent random edges.

The limitations of the basic Erdős--Rényi model are equally informative. Many real-world networks contain hubs, heterogeneous degree distributions, strong local clustering, communities, or growth processes that cannot be reproduced accurately by independent edge generation. Consequently, additional random graph models have been developed to capture particular structural characteristics observed in social, biological, technological, and information networks.

The Watts--Strogatz model was introduced to represent small-world network behavior. It begins with a regular lattice-like network in which nodes are connected primarily to nearby neighbors and then randomly rewires some edges. This construction can preserve relatively high local clustering while dramatically reducing average path length, reproducing the combination of local cohesion and global reachability found in many real networks.

Small-world behavior demonstrates that only a limited number of long-range connections may be required to make distant regions of a network accessible through short paths. Most relationships can remain local while occasional shortcut edges connect otherwise separated regions. This principle is relevant to communication networks, social interactions, distributed systems, and multi-agent structures where efficient global information exchange is desirable.

The Barabási--Albert model focuses on another characteristic of real networks: highly unequal node degrees. It generates a network through growth and preferential attachment, where newly introduced nodes are more likely to connect to nodes that already possess many connections. This mechanism produces hub-like structures and a heavy-tailed degree distribution that differs fundamentally from the relatively homogeneous degrees of simple Erdős--Rényi graphs.

Preferential attachment captures the intuitive principle that connectivity can reinforce itself. A highly connected node has more opportunities to attract additional relationships, causing structural advantages to accumulate over time. Although this mechanism does not explain every real network, it provides an important example of how dynamic formation rules can produce graph structures that differ significantly from independent random connectivity.

Other probabilistic graph models emphasize community structure. A stochastic block model divides nodes into latent groups and assigns different connection probabilities to pairs of groups. Nodes belonging to the same group may have a high probability of connection, while nodes from different groups may connect less frequently. The resulting graphs provide a mathematical foundation for studying communities and statistical network inference.

Random graph generation is also valuable for experimentation with graph algorithms and Graph Neural Networks. Synthetic graphs allow researchers to control graph size, density, degree distribution, clustering, communities, and other structural properties. Algorithms can then be evaluated under systematically varied conditions rather than relying entirely on fixed real-world datasets whose structural characteristics may be difficult to isolate.

For graph learning, random graphs can function as controlled environments for investigating how topology affects message passing. Increasing edge density changes neighborhood size, information propagation, and computational cost. Modifying community strength affects how easily node representations separate into groups, while introducing hubs changes aggregation patterns and may cause a small number of nodes to influence large portions of the graph.

Random graphs are also closely connected to robustness and failure analysis. Nodes or edges can be removed randomly to model failures, communication losses, sensor outages, or unavailable agents. Researchers can then measure whether the graph remains connected, how path lengths change, or when the network fragments. Targeted removal of high-degree nodes can additionally reveal vulnerabilities that differ from purely random failures.

In distributed robotics and multi-agent systems, random graph concepts can model uncertain communication and interaction structures. Wireless links may appear or disappear because of range, interference, obstacles, or mobility, causing the communication graph to change over time. Probabilistic graph models provide a useful abstraction for reasoning about whether collective information exchange remains possible despite uncertain connectivity.

A static random graph represents one sampled topology, whereas dynamic or temporal random graphs allow edges and potentially nodes to evolve with time. This extension is important when relationships are not permanent. Communication networks, transportation interactions, social systems, and robotic fleets frequently exhibit changing connectivity, requiring models that describe not only graph structure but also its stochastic evolution.

Random graphs therefore connect probability theory with graph theory by explaining how uncertain local interactions can produce predictable statistical properties at network scale. Erdős--Rényi graphs provide a fundamental independent-edge baseline, small-world models capture clustering and short paths, preferential-attachment models generate hubs, and stochastic block models represent community-dependent connectivity.

Understanding these models prepares the transition from deterministic graph structures toward probabilistic and learned representations. Graph deep learning frequently operates on networks that are incomplete, noisy, dynamic, or uncertain, making probabilistic thinking increasingly important. Random graph theory provides a foundation for analyzing such uncertainty and for understanding how topology itself influences information propagation, robustness, and learning behavior.

## 01.06. Applications

![](images/image7.png){width="7.268055555555556in" height="7.268055555555556in"}

Graph theory becomes especially valuable when relationships between entities are as important as the entities themselves. Many real systems can be represented naturally as networks in which nodes describe objects, agents, locations, or concepts and edges describe interactions or dependencies. This relational representation enables graph algorithms to analyze structures that are difficult to capture with independent feature vectors.

Social networks are among the most familiar graph applications. Users can be represented as nodes, while friendships, follows, messages, or other interactions become edges. Graph properties such as degree, centrality, clustering, communities, and shortest paths can then describe social organization, identify influential users, reveal groups with strong internal relationships, and characterize how information may spread through the network.

Recommendation systems also rely naturally on graph representations. Users and items can form a bipartite graph in which edges represent purchases, ratings, clicks, views, or other interactions. By examining neighborhoods and connectivity patterns, a system can estimate similarities among users and items and predict previously unobserved relationships, such as products or content that a particular user may prefer.

Web and information networks provide another important application. Webpages can be modeled as nodes and hyperlinks as directed edges, producing a massive directed graph. Algorithms such as PageRank exploit this structure to estimate the relative importance of pages based on incoming links and the importance of their sources. Citation networks apply similar ideas to relationships among scientific papers, patents, or documents.

Knowledge graphs represent entities and semantic relationships in a structured relational form. Nodes may correspond to people, organizations, locations, products, events, or abstract concepts, while typed edges express relationships such as located-in, works-for, created-by, or related-to. Such graphs support semantic search, question answering, recommendation, knowledge discovery, and reasoning over interconnected information.

Transportation systems can be represented as graphs in which intersections, stations, airports, or geographic locations become nodes and roads, railways, routes, or transportation links become edges. Edge weights may represent distance, travel time, energy consumption, congestion, or cost. Shortest-path and routing algorithms can then determine efficient routes while graph connectivity reveals whether destinations remain reachable.

Communication and computer networks provide another direct graph application. Routers, servers, devices, sensors, or computing nodes can be represented as vertices connected through wired or wireless communication links. Graph algorithms can analyze routing, network reachability, bottlenecks, redundancy, and failure tolerance. Weighted edges can additionally represent bandwidth, latency, reliability, or communication cost.

Biological systems frequently contain complex relational structures that are naturally modeled as graphs. Proteins can form protein-interaction networks, genes can be connected through regulatory relationships, and metabolites can participate in biochemical reaction networks. Graph analysis helps identify important molecules, functional modules, pathways, and structural patterns that may be difficult to detect by considering biological entities independently.

Molecules themselves can also be represented directly as graphs. Atoms become nodes and chemical bonds become edges, while node features describe properties such as element type and edge features describe bond type. This representation is particularly important in graph machine learning because molecular properties can be predicted by learning how local atomic interactions combine to determine the behavior of an entire molecular structure.

Fraud detection and cybersecurity are strongly relational problems. Accounts, transactions, devices, addresses, or identities can be represented as interconnected entities. Suspicious behavior may become visible not from one transaction alone but from unusual patterns of connections among multiple entities. Graph structures can therefore reveal coordinated fraud, anomalous transaction communities, attack paths, and relationships hidden within conventional tabular records.

Infrastructure systems such as electrical grids, water networks, logistics systems, and supply chains can also be modeled using graphs. Facilities or resources become nodes, while physical or operational dependencies become edges. Connectivity analysis can identify critical components, shortest-path methods can support routing and distribution, and centrality measures can reveal nodes whose failure could significantly affect overall system operation.

Graph representations are especially relevant to robotics because robotic systems interact with environments containing many related entities. A robot may represent landmarks, objects, rooms, navigation points, sensors, tasks, or other robots as nodes. Edges can encode spatial proximity, navigability, visibility, communication, semantic relationships, task dependencies, or physical interactions, creating structured representations beyond raw sensor measurements.

Navigation graphs provide a direct example. Locations or waypoints become nodes, while traversable connections become edges. Edge weights can represent geometric distance, traversal time, terrain difficulty, energy consumption, or estimated risk. Classical shortest-path algorithms can then search for feasible routes, while dynamic graph updates allow the representation to adapt when obstacles, costs, or accessibility conditions change.

Scene graphs provide a higher-level representation of an environment by describing objects together with their relationships. Instead of representing a scene only as pixels or point clouds, a graph can encode that an object is beside another object, located inside a room, supported by a surface, or associated with a particular function. This relational structure supports semantic reasoning and environment understanding.

Multi-robot and multi-agent systems naturally form graphs whose nodes correspond to individual agents and whose edges describe communication, sensing, coordination, or interaction relationships. Because these links may change as agents move, the resulting graph can be dynamic. Connectivity becomes essential because coordinated behavior depends on whether information can propagate between agents or throughout the collective system.

Sensor networks provide a closely related application. Sensors can be modeled as nodes connected according to communication range, spatial proximity, measurement correlation, or functional dependency. Graph-based processing allows information from multiple sensors to be combined according to these relationships. This can support distributed estimation, anomaly detection, environmental monitoring, and cooperative perception across heterogeneous sensing systems.

Graphs are also useful for representing dependencies in software and engineering systems. Functions, modules, services, tasks, or hardware components can become nodes, while dependencies and data flows become directed edges. Graph traversal can reveal dependency chains, topological ordering can determine valid execution sequences, and centrality or connectivity analysis can identify components that are structurally critical to the larger system.

Many applications require graphs that evolve over time rather than remaining static. Social interactions change, transportation conditions vary, communication links appear and disappear, transactions continuously occur, and robots move through changing environments. Temporal and dynamic graphs extend conventional graph representations by incorporating these changes, allowing algorithms to reason about both relational structure and its evolution.

The same application domains increasingly motivate graph machine learning and Graph Neural Networks. Classical graph algorithms operate according to explicitly designed rules, whereas learned graph models can infer useful representations directly from node attributes, edge information, and topology. This enables node classification, link prediction, graph classification, anomaly detection, recommendation, relational reasoning, and other data-driven tasks.

Graph applications therefore share a common principle despite their different domains: relationships contain information that cannot be fully understood by examining entities independently. Graph theory provides the mathematical framework for representing these relationships, while graph algorithms extract connectivity, paths, importance, communities, and other structural properties. These foundations prepare the transition from graph theory to learned graph representations.

From social and information networks to molecules, infrastructure, robotics, and multi-agent intelligence, graphs provide a unified abstraction for relational systems. The concepts developed throughout graph theory basics---nodes, edges, matrices, connectivity, centrality, clustering, spectra, algorithms, and random graphs---become practical tools for modeling real systems and establish the foundation on which modern graph deep learning is constructed.
