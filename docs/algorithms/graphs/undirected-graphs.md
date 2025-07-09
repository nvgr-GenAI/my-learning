# Undirected Graphs

Undirected graphs represent relationships where connections have no direction—if vertex A is connected to vertex B, then B is also connected to A. This section explores the fundamentals, algorithms, and applications specific to undirected graphs.

## Overview

=== "Definition"

    An undirected graph G = (V, E) consists of:
    
    - A set V of vertices (or nodes)
    - A set E of unordered pairs of distinct vertices called edges
    
    In an undirected graph, edges have no direction. An edge {u, v} connects vertices u and v symmetrically—there is no distinction between the "from" and "to" endpoints.
    
    ![Undirected Graph](https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Undirected.svg/440px-Undirected.svg.png)
    
    **Key properties:**
    
    - Edges are bidirectional
    - The degree of a vertex is the number of edges connected to it
    - A path is a sequence of vertices where each consecutive pair is connected by an edge
    - A cycle is a path that starts and ends at the same vertex

=== "Terminology"

    - **Edge**: An unordered pair {u, v} connecting vertices u and v
    - **Adjacent vertices**: Two vertices connected by an edge
    - **Degree**: The number of edges incident to a vertex
    - **Isolated vertex**: A vertex with degree 0
    - **Path**: A sequence of vertices where consecutive vertices are adjacent
    - **Cycle**: A path that starts and ends at the same vertex
    - **Connected component**: A maximal subgraph where any two vertices are connected by a path
    - **Tree**: A connected graph with no cycles
    - **Forest**: A graph with no cycles (may have multiple connected components)
    - **Complete graph**: A graph where every pair of vertices is connected by an edge
    - **Bipartite graph**: A graph whose vertices can be divided into two disjoint sets such that every edge connects vertices from different sets

=== "Applications"

    Undirected graphs model many real-world systems:
    
    - **Social networks**: People as vertices and friendships as edges
    - **Transportation networks**: Locations as vertices and roads as edges
    - **Communication networks**: Devices as vertices and communication links as edges
    - **Electrical circuits**: Components as vertices and connections as edges
    - **Molecular structures**: Atoms as vertices and bonds as edges
    - **Computer networks**: Computers as vertices and network links as edges
    - **Collaboration networks**: Researchers as vertices and joint papers as edges
    - **Infrastructure networks**: Facilities as vertices and physical connections as edges

## Graph Representations

=== "Adjacency Matrix"

    In an adjacency matrix representation for an undirected graph with n vertices:
    
    - A is an n × n matrix
    - A[i][j] = A[j][i] = 1 if there is an edge between vertices i and j
    - A[i][j] = A[j][i] = 0 otherwise
    
    For a weighted undirected graph, A[i][j] = A[j][i] = weight of the edge between i and j.
    
    **Advantages:**
    - O(1) time to check if two vertices are adjacent
    - Simple implementation
    
    **Disadvantages:**
    - O(V²) space, inefficient for sparse graphs
    - O(V) time to find all neighbors of a vertex

=== "Adjacency List"

    In an adjacency list representation:
    
    - Each vertex maintains a list of its adjacent vertices
    - For a vertex u, the list contains all vertices v such that there's an edge {u, v}
    
    For a weighted undirected graph, each entry in the list also stores the weight.
    
    **Advantages:**
    - O(V + E) space, efficient for sparse graphs
    - O(degree(u)) time to find all neighbors of vertex u
    
    **Disadvantages:**
    - May take O(degree(u)) time to check if two vertices are adjacent

=== "Edge List"

    In an edge list representation:
    
    - The graph is represented as a list of all edges
    - Each edge is stored as an unordered pair {u, v}
    
    For a weighted undirected graph, each edge also stores the weight.
    
    **Advantages:**
    - Simple to implement
    - Efficient for algorithms that need to process all edges
    
    **Disadvantages:**
    - Inefficient for checking if two vertices are adjacent
    - Inefficient for finding all neighbors of a vertex

## Traversal Algorithms

=== "Depth-First Search (DFS)"

    DFS explores as far as possible along each branch before backtracking:
    
    ```python
    def dfs(graph, start, visited=None):
        if visited is None:
            visited = set()
            
        visited.add(start)
        print(start, end=' ')  # Process the vertex
        
        for neighbor in graph[start]:
            if neighbor not in visited:
                dfs(graph, neighbor, visited)
        
        return visited
    ```
    
    **Applications specific to undirected graphs:**
    
    - **Connected components**: Find all connected components in the graph
    - **Cycle detection**: Detect cycles in the graph
    - **Bipartiteness check**: Determine if the graph is bipartite
    - **Bridge finding**: Identify bridges (edges whose removal increases the number of connected components)
    - **Articulation point finding**: Identify vertices whose removal increases the number of connected components

=== "Breadth-First Search (BFS)"

    BFS explores all neighbors at the present depth before moving to vertices at the next depth level:
    
    ```python
    from collections import deque
    
    def bfs(graph, start):
        visited = set([start])
        queue = deque([start])
        
        while queue:
            vertex = queue.popleft()
            print(vertex, end=' ')  # Process the vertex
            
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return visited
    ```
    
    **Applications specific to undirected graphs:**
    
    - **Shortest path (unweighted)**: Find shortest path between two vertices
    - **Connected components**: Find all connected components
    - **Testing bipartiteness**: Determine if the graph is bipartite
    - **Finding all vertices within a certain distance**: Useful in social network analysis

## Connectivity in Undirected Graphs

=== "Connected Components"

    A connected component is a maximal subset of vertices such that there is a path between any two vertices in the subset.
    
    ```python
    def find_connected_components(graph):
        """
        Find all connected components in an undirected graph.
        
        Args:
            graph: Dictionary representing adjacency list of undirected graph
            
        Returns:
            list: List of connected components (each component is a list of vertices)
        """
        visited = set()
        components = []
        
        def dfs(vertex, component):
            visited.add(vertex)
            component.append(vertex)
            
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for vertex in graph:
            if vertex not in visited:
                component = []
                dfs(vertex, component)
                components.append(component)
        
        return components
    ```

=== "Bridges and Articulation Points"

    - **Bridge**: An edge whose removal increases the number of connected components
    - **Articulation Point**: A vertex whose removal increases the number of connected components
    
    Finding bridges and articulation points is crucial for analyzing network reliability:
    
    ```python
    def find_bridges(graph):
        """
        Find all bridges in an undirected graph using Tarjan's algorithm.
        
        Args:
            graph: Dictionary representing adjacency list of undirected graph
            
        Returns:
            list: List of bridges (each bridge is a tuple of two vertices)
        """
        discovery = {}
        low = {}
        bridges = []
        time = [0]  # Using list to allow modification in nested function
        
        def dfs(u, parent):
            discovery[u] = low[u] = time[0]
            time[0] += 1
            
            for v in graph[u]:
                if v not in discovery:
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    
                    if low[v] > discovery[u]:
                        bridges.append((u, v))
                elif v != parent:
                    low[u] = min(low[u], discovery[v])
        
        for vertex in graph:
            if vertex not in discovery:
                dfs(vertex, None)
        
        return bridges
    ```

=== "Biconnected Components"

    A biconnected component is a maximal subgraph that remains connected after removing any single vertex.
    
    - Contains no articulation points
    - Cannot be disconnected by removing a single vertex
    - Used to analyze network resilience

## Cycles in Undirected Graphs

=== "Cycle Detection"

    Detecting cycles in undirected graphs:
    
    ```python
    def has_cycle(graph):
        """
        Check if an undirected graph contains a cycle.
        
        Args:
            graph: Dictionary representing adjacency list of undirected graph
            
        Returns:
            bool: True if cycle exists, False otherwise
        """
        visited = set()
        
        def dfs_cycle(u, parent):
            visited.add(u)
            
            for v in graph[u]:
                if v not in visited:
                    if dfs_cycle(v, u):
                        return True
                elif v != parent:
                    return True  # Back edge found
            
            return False
        
        for vertex in graph:
            if vertex not in visited:
                if dfs_cycle(vertex, None):
                    return True
        
        return False
    ```

=== "Cycle Enumeration"

    Finding all cycles in an undirected graph is more complex:
    
    - For simple cases, a modified DFS can enumerate all cycles
    - Several algorithms exist for systematic cycle enumeration
    - The number of cycles can be exponential in worst case

=== "Minimum Cycle Basis"

    The minimum cycle basis is a set of cycles such that:
    
    - Any cycle in the graph can be expressed as a symmetric difference of cycles in the basis
    - The total length of cycles in the basis is minimized
    
    Used in applications like electrical circuit analysis and chemical structure analysis.

## Trees and Forests

=== "Definition and Properties"

    A tree is a connected undirected graph with no cycles. A forest is a disjoint union of trees.
    
    **Properties of trees:**
    
    - |E| = |V| - 1 (number of edges = number of vertices - 1)
    - Any two vertices are connected by exactly one path
    - Adding any edge creates exactly one cycle
    - Removing any edge disconnects the graph
    
    Trees are fundamental structures in graph theory and computer science.

=== "Spanning Trees"

    A spanning tree of a connected undirected graph is a subgraph that:
    
    - Is a tree (connected and acyclic)
    - Includes all vertices of the original graph
    
    Every connected graph has at least one spanning tree.

=== "Minimum Spanning Tree (MST)"

    A minimum spanning tree is a spanning tree with minimum total edge weight.
    
    Two main algorithms for finding MST:
    
    1. **Kruskal's Algorithm**: Sort edges by weight, then add edges in ascending order if they don't create a cycle
    2. **Prim's Algorithm**: Grow a tree from a starting vertex, always adding the minimum-weight edge that connects a new vertex

    ```python
    def kruskal_mst(graph, vertices):
        """
        Find minimum spanning tree using Kruskal's algorithm.
        
        Args:
            graph: List of edges with weights [(u, v, weight), ...]
            vertices: Set of all vertices
            
        Returns:
            list: Edges in the MST
        """
        # Sort edges by weight
        edges = sorted(graph, key=lambda x: x[2])
        
        # Initialize disjoint set for each vertex
        parent = {v: v for v in vertices}
        rank = {v: 0 for v in vertices}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            
            if root_x == root_y:
                return
            
            # Union by rank
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
        
        mst = []
        for u, v, weight in edges:
            if find(u) != find(v):
                union(u, v)
                mst.append((u, v, weight))
                
        return mst
    ```

## Bipartite Graphs

=== "Definition and Detection"

    A bipartite graph is an undirected graph where vertices can be divided into two disjoint sets U and V such that every edge connects a vertex in U to a vertex in V.
    
    ```python
    def is_bipartite(graph):
        """
        Check if an undirected graph is bipartite.
        
        Args:
            graph: Dictionary representing adjacency list of undirected graph
            
        Returns:
            bool: True if bipartite, False otherwise
        """
        colors = {}
        
        def bfs(start):
            colors[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                
                for v in graph[u]:
                    if v not in colors:
                        colors[v] = 1 - colors[u]
                        queue.append(v)
                    elif colors[v] == colors[u]:
                        return False
            
            return True
        
        for vertex in graph:
            if vertex not in colors:
                if not bfs(vertex):
                    return False
        
        return True
    ```

=== "Applications"

    Bipartite graphs have many applications:
    
    - **Matching problems**: Job assignments, resource allocation
    - **Scheduling**: Tasks and processors
    - **Recommendation systems**: Users and items
    - **Chemical structures**: Alternating atom types
    - **Register allocation**: Variables and registers

## Planar Graphs

=== "Definition and Properties"

    A planar graph is an undirected graph that can be embedded in a plane such that no edges cross.
    
    **Key properties:**
    
    - For a connected planar graph with V vertices and E edges:
      - E ≤ 3V - 6 (if V ≥ 3)
    - Every planar graph is 4-colorable (Four Color Theorem)
    - Kuratowski's theorem: A graph is planar if and only if it doesn't contain a subdivision of K₅ (complete graph on 5 vertices) or K₃,₃ (complete bipartite graph on 3+3 vertices)

=== "Applications"

    Planar graphs have important applications:
    
    - **Circuit design**: Designing circuits without crossing wires
    - **Map coloring**: Coloring regions such that adjacent regions have different colors
    - **Network design**: Planning networks with no crossing links
    - **VLSI design**: Designing integrated circuits

## Special Undirected Graphs

=== "Complete Graphs"

    A complete graph Kₙ is a graph with n vertices where every pair of vertices is connected by an edge.
    
    **Properties:**
    
    - Has n(n-1)/2 edges
    - Every vertex has degree n-1
    - Diameter is 1
    - Chromatic number is n
    - Kₙ is planar if and only if n ≤ 4

=== "Regular Graphs"

    A regular graph is a graph where each vertex has the same degree.
    
    - A k-regular graph is a graph where every vertex has degree k
    - Examples include cycles, complete graphs, and the Petersen graph
    - Regular graphs are important in spectral graph theory and combinatorial designs

=== "Eulerian and Hamiltonian Graphs"

    - **Eulerian graph**: A connected graph where all vertices have even degree (has a cycle that uses each edge exactly once)
    - **Semi-Eulerian graph**: A connected graph with exactly two vertices of odd degree (has a path that uses each edge exactly once)
    - **Hamiltonian graph**: A graph that has a cycle visiting each vertex exactly once
    - Unlike Eulerian graphs, there's no simple characterization of Hamiltonian graphs

## Shortest Paths in Undirected Graphs

=== "Single-Source Shortest Paths"

    For unweighted undirected graphs, BFS finds shortest paths:
    
    ```python
    def shortest_paths_bfs(graph, start):
        """
        Find shortest paths from start to all other vertices in unweighted undirected graph.
        
        Args:
            graph: Dictionary representing adjacency list of undirected graph
            start: Starting vertex
            
        Returns:
            dict: Dictionary of distances from start to each vertex
            dict: Dictionary of parent pointers for reconstructing paths
        """
        distances = {start: 0}
        parents = {start: None}
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            
            for v in graph[u]:
                if v not in distances:
                    distances[v] = distances[u] + 1
                    parents[v] = u
                    queue.append(v)
        
        return distances, parents
    ```
    
    For weighted undirected graphs, Dijkstra's algorithm or Bellman-Ford algorithm can be used.

=== "All-Pairs Shortest Paths"

    For finding shortest paths between all pairs of vertices:
    
    1. **Floyd-Warshall Algorithm**: Efficient for dense graphs
    2. **Johnson's Algorithm**: Better for sparse graphs

## Comparison with Directed Graphs

=== "Key Differences"

    | Aspect | Undirected Graphs | Directed Graphs |
    |--------|-------------------|----------------|
    | Edge representation | Unordered pair {u, v} | Ordered pair (u, v) |
    | Connectivity | Connected vs. disconnected | Strong vs. weak connectivity |
    | Degree | Single degree concept | In-degree and out-degree |
    | Traversal | Can go in either direction | Can only follow edge directions |
    | Eulerian conditions | All vertices have even degree | In-degree = out-degree for all vertices |
    | Adjacency matrix | Symmetric | May be asymmetric |
    | Applications | Physical networks, social connections | Workflows, dependencies |

=== "Algorithm Adaptations"

    Many algorithms are simpler for undirected graphs:
    
    - **DFS/BFS**: No need to track edge directions
    - **Connectivity**: Simple connected components vs. strongly connected components
    - **Cycle detection**: Simpler in undirected graphs
    - **MST**: Specific to undirected graphs (directed version is the minimum spanning arborescence)

## Advanced Topics

=== "Graph Coloring"

    Graph coloring is assigning colors to vertices such that adjacent vertices have different colors:
    
    - The minimum number of colors needed is called the chromatic number
    - NP-hard in general, but efficient algorithms exist for special cases
    - Applications include register allocation, scheduling, and map coloring

=== "Spectral Graph Theory"

    Spectral graph theory studies properties of graphs using eigenvalues and eigenvectors of matrices associated with the graph:
    
    - Adjacency matrix
    - Laplacian matrix
    - Normalized Laplacian
    
    Used in clustering, partitioning, and understanding structural properties of graphs.

=== "Random Walks"

    A random walk on an undirected graph is a sequence of vertices where each step moves to a neighbor chosen uniformly at random:
    
    - Fundamental in studying diffusion processes on networks
    - Related to electrical networks and resistor networks
    - Used in PageRank-like algorithms and sampling methods

## References

- [Undirected Graph on Wikipedia](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics))
- [Connected Components](https://en.wikipedia.org/wiki/Connected_component_(graph_theory))
- [Minimum Spanning Tree](https://en.wikipedia.org/wiki/Minimum_spanning_tree)
- [Bipartite Graph](https://en.wikipedia.org/wiki/Bipartite_graph)
- [Planar Graph](https://en.wikipedia.org/wiki/Planar_graph)
- [Introduction to Algorithms (CLRS)](https://en.wikipedia.org/wiki/Introduction_to_Algorithms) - Chapters 22-23
- [Graph Theory by Reinhard Diestel](https://diestel-graph-theory.com/)
- [Algorithms by Robert Sedgewick and Kevin Wayne](https://algs4.cs.princeton.edu/)
