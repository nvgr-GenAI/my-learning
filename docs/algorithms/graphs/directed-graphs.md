# Directed Graphs

A directed graph (or digraph) is a graph where edges have directions, pointing from one vertex to another. This section covers the fundamentals, algorithms, and applications specific to directed graphs.

## Overview

=== "Definition"

    A directed graph G = (V, E) consists of:
    
    - A set V of vertices (or nodes)
    - A set E of ordered pairs of vertices called edges (or arcs)
    
    Unlike undirected graphs where edges are bidirectional, in directed graphs, an edge (u, v) is distinct from (v, u). The edge (u, v) represents a connection from vertex u to vertex v, but not necessarily from v to u.
    
    ![Directed Graph](https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Directed_graph.svg/440px-Directed_graph.svg.png)
    
    **Key properties:**
    
    - Edges have a specific direction
    - The in-degree of a vertex is the number of edges coming into it
    - The out-degree of a vertex is the number of edges going out from it
    - A path is a sequence of vertices where each consecutive pair is connected by a directed edge

=== "Terminology"

    - **Directed edge**: An edge with direction, usually represented as an ordered pair (u, v) or drawn as an arrow from u to v
    - **In-degree**: The number of incoming edges to a vertex
    - **Out-degree**: The number of outgoing edges from a vertex
    - **Source**: A vertex with in-degree 0
    - **Sink**: A vertex with out-degree 0
    - **Strongly Connected Component (SCC)**: A maximal subgraph where there is a directed path between any two vertices
    - **Directed Acyclic Graph (DAG)**: A directed graph with no directed cycles
    - **Topological Sort**: An ordering of vertices such that for every directed edge (u, v), vertex u comes before v in the ordering
    - **Transitive Closure**: A graph that contains an edge (u, v) if there is a directed path from u to v in the original graph

=== "Applications"

    Directed graphs are used to model many real-world systems:
    
    - **Web pages and hyperlinks**: Web pages as vertices and hyperlinks as directed edges
    - **Social networks**: People as vertices and "follows" as directed edges
    - **Citation networks**: Papers as vertices and citations as directed edges
    - **Dependency graphs**: Tasks as vertices and dependencies as directed edges
    - **State machines**: States as vertices and transitions as directed edges
    - **Data flow diagrams**: Processes as vertices and data flows as directed edges
    - **Control flow in programs**: Basic blocks as vertices and control transfers as directed edges
    - **Communication networks**: Devices as vertices and one-way communications as directed edges

## Graph Representations

=== "Adjacency Matrix"

    In an adjacency matrix representation for a directed graph with n vertices:
    
    - A is an n × n matrix
    - A[i][j] = 1 if there is a directed edge from vertex i to vertex j
    - A[i][j] = 0 otherwise
    
    For a weighted directed graph, A[i][j] = weight of the edge from i to j.
    
    **Advantages:**
    - Checking if there's an edge from i to j is O(1)
    - Simple implementation
    
    **Disadvantages:**
    - Uses O(V²) space, inefficient for sparse graphs
    - Iterating over all outgoing edges from a vertex takes O(V) time

=== "Adjacency List"

    In an adjacency list representation:
    
    - Each vertex maintains a list of its outgoing neighbors
    - For a vertex u, the list contains all vertices v such that there's a directed edge (u, v)
    
    For a weighted directed graph, each entry in the list also stores the weight.
    
    **Advantages:**
    - Uses O(V + E) space, efficient for sparse graphs
    - Iterating over all outgoing edges from a vertex is efficient
    
    **Disadvantages:**
    - Checking if there's an edge from i to j may take O(E) time in the worst case

=== "Edge List"

    In an edge list representation:
    
    - The graph is represented as a list of all directed edges
    - Each edge is stored as a pair (u, v) indicating a directed edge from u to v
    
    For a weighted directed graph, each edge also stores the weight.
    
    **Advantages:**
    - Simple to implement
    - Efficient for algorithms that need to process all edges
    
    **Disadvantages:**
    - Inefficient for checking if an edge exists or finding all edges from a specific vertex

## Traversal Algorithms

=== "Depth-First Search (DFS)"

    DFS for directed graphs is similar to undirected graphs, but with some important applications:
    
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
    
    **Applications specific to directed graphs:**
    
    - **Cycle detection**: Use DFS with a recursion stack to detect cycles
    - **Topological sorting**: Use DFS to compute a topological order
    - **Strongly connected components**: Use Kosaraju's or Tarjan's algorithm with DFS
    - **Path finding**: Find if a path exists from one vertex to another

=== "Breadth-First Search (BFS)"

    BFS for directed graphs:
    
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
    
    **Applications specific to directed graphs:**
    
    - **Shortest path (unweighted)**: Find shortest path in terms of number of edges
    - **Level-by-level traversal**: Process vertices level by level
    - **Finding all vertices reachable from a source**: Useful in many graph algorithms

## Connectivity in Directed Graphs

=== "Types of Connectivity"

    Directed graphs have different types of connectivity compared to undirected graphs:
    
    1. **Weakly connected**: The graph would be connected if we ignored edge directions
    2. **Strongly connected**: There is a directed path from any vertex to any other vertex
    3. **Unilaterally connected**: For any two vertices, there is a directed path from at least one to the other
    
    Understanding connectivity is crucial for analyzing the structure and properties of directed graphs.

=== "Strongly Connected Components (SCCs)"

    A strongly connected component is a maximal subgraph where there's a directed path from any vertex to any other vertex.
    
    Two main algorithms for finding SCCs:
    
    1. **Kosaraju's Algorithm**:
       - Do a DFS and record the finish times
       - Transpose the graph (reverse all edges)
       - Do a DFS on the transposed graph in order of decreasing finish time
    
    2. **Tarjan's Algorithm**:
       - Do a single DFS using additional data structures (low-link values)
       - More efficient than Kosaraju's algorithm
    
    ```python
    def kosaraju(graph):
        """
        Find strongly connected components using Kosaraju's algorithm.
        
        Args:
            graph: Dictionary representing adjacency list of directed graph
            
        Returns:
            list: List of strongly connected components (each component is a list of vertices)
        """
        # Step 1: Do DFS and store vertices in order of finish time
        visited = set()
        finish_order = []
        
        def dfs1(vertex):
            visited.add(vertex)
            for neighbor in graph.get(vertex, []):
                if neighbor not in visited:
                    dfs1(neighbor)
            finish_order.append(vertex)
        
        for vertex in graph:
            if vertex not in visited:
                dfs1(vertex)
        
        # Step 2: Transpose the graph
        transposed = {}
        for u in graph:
            for v in graph.get(u, []):
                transposed.setdefault(v, []).append(u)
        
        # Step 3: DFS on transposed graph in order of decreasing finish time
        visited.clear()
        components = []
        
        def dfs2(vertex, component):
            visited.add(vertex)
            component.append(vertex)
            for neighbor in transposed.get(vertex, []):
                if neighbor not in visited:
                    dfs2(neighbor, component)
        
        # Process vertices in reverse order of finish time
        for vertex in reversed(finish_order):
            if vertex not in visited:
                component = []
                dfs2(vertex, component)
                components.append(component)
        
        return components
    ```

=== "Applications of SCCs"

    Strongly connected components have various applications:
    
    - **Component graph**: Condensing each SCC into a single vertex creates a DAG
    - **Web page analysis**: Identifying communities of related pages
    - **Social network analysis**: Finding groups where everyone is connected
    - **Circuit analysis**: Identifying feedback loops in electronic circuits
    - **Compiler optimization**: Identifying variables with interdependencies

## Cycles in Directed Graphs

=== "Cycle Detection"

    Detecting cycles in directed graphs is an important problem:
    
    ```python
    def has_cycle(graph):
        """
        Check if a directed graph contains a cycle.
        
        Args:
            graph: Dictionary representing adjacency list of directed graph
            
        Returns:
            bool: True if cycle exists, False otherwise
        """
        visited = set()
        recursion_stack = set()
        
        def dfs_cycle(vertex):
            visited.add(vertex)
            recursion_stack.add(vertex)
            
            for neighbor in graph.get(vertex, []):
                if neighbor not in visited:
                    if dfs_cycle(neighbor):
                        return True
                elif neighbor in recursion_stack:
                    return True
            
            recursion_stack.remove(vertex)
            return False
        
        for vertex in graph:
            if vertex not in visited:
                if dfs_cycle(vertex):
                    return True
        
        return False
    ```

=== "Finding All Cycles"

    Finding all cycles in a directed graph is more complex:
    
    - Johnson's algorithm efficiently finds all elementary cycles
    - For simple cases, a modified DFS can enumerate all cycles
    - The number of cycles can be exponential in the worst case

=== "Applications of Cycle Detection"

    Cycle detection has many applications:
    
    - **Deadlock detection** in operating systems
    - **Circular dependencies** in software modules
    - **Infinite loops** in program analysis
    - **Feedback loops** in control systems
    - **Mutual recursion** in function calls

## Topological Sorting

=== "Definition and Properties"

    A topological sort of a directed acyclic graph (DAG) is a linear ordering of vertices such that for every directed edge (u, v), vertex u comes before v in the ordering.
    
    **Key properties:**
    
    - Only possible for DAGs (graphs without cycles)
    - Not unique (there can be multiple valid topological orderings)
    - Useful for scheduling tasks with dependencies

=== "Kahn's Algorithm"

    Kahn's algorithm is an efficient approach for topological sorting:
    
    1. Compute in-degree for each vertex
    2. Enqueue vertices with in-degree 0
    3. Repeatedly dequeue a vertex, add it to the result, and reduce in-degrees of its neighbors
    
    ```python
    from collections import deque
    
    def topological_sort_kahn(graph):
        """
        Perform topological sort using Kahn's algorithm.
        
        Args:
            graph: Dictionary representing adjacency list of directed graph
            
        Returns:
            list: Vertices in topological order, or empty list if a cycle exists
        """
        # Compute in-degree for each vertex
        in_degree = {vertex: 0 for vertex in graph}
        for vertex in graph:
            for neighbor in graph[vertex]:
                in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
        
        # Enqueue vertices with in-degree 0
        queue = deque([vertex for vertex, degree in in_degree.items() if degree == 0])
        result = []
        
        # Process vertices
        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            
            for neighbor in graph.get(vertex, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check if all vertices are included (if not, there's a cycle)
        if len(result) != len(graph):
            return []  # Cycle detected
        
        return result
    ```

=== "DFS-based Approach"

    A DFS-based approach for topological sorting:
    
    ```python
    def topological_sort_dfs(graph):
        """
        Perform topological sort using DFS.
        
        Args:
            graph: Dictionary representing adjacency list of directed graph
            
        Returns:
            list: Vertices in topological order, or empty list if a cycle exists
        """
        visited = set()
        temp_mark = set()  # For cycle detection
        result = []
        
        def dfs(vertex):
            if vertex in temp_mark:
                return False  # Cycle detected
            if vertex in visited:
                return True
                
            temp_mark.add(vertex)
            
            for neighbor in graph.get(vertex, []):
                if not dfs(neighbor):
                    return False
            
            temp_mark.remove(vertex)
            visited.add(vertex)
            result.append(vertex)
            return True
        
        for vertex in graph:
            if vertex not in visited:
                if not dfs(vertex):
                    return []  # Cycle detected
        
        return result[::-1]  # Reverse to get correct topological order
    ```

=== "Applications"

    Topological sorting has numerous applications:
    
    - **Scheduling tasks** with dependencies (e.g., course prerequisites)
    - **Build systems** for determining compilation order
    - **Data serialization** for objects with references
    - **Resolving symbol dependencies** in linkers
    - **Evaluating formulas** in spreadsheets
    - **Job scheduling** in distributed systems

## Shortest Paths in Directed Graphs

=== "Single-Source Shortest Paths"

    For directed graphs, the main algorithms are:
    
    1. **Dijkstra's Algorithm**: For graphs with non-negative edge weights
    2. **Bellman-Ford Algorithm**: When negative edge weights are present
    
    Both algorithms have specific considerations for directed graphs, particularly regarding negative cycles.

=== "All-Pairs Shortest Paths"

    For finding shortest paths between all pairs of vertices:
    
    1. **Floyd-Warshall Algorithm**: Efficient for dense graphs
    2. **Johnson's Algorithm**: Better for sparse graphs
    
    These algorithms work directly on the directed graph structure.

## Transitive Closure

=== "Definition"

    The transitive closure of a directed graph G is a new graph G* where there is an edge from vertex u to vertex v if there is a directed path from u to v in G.
    
    This is useful for determining reachability between vertices.

=== "Floyd-Warshall for Transitive Closure"

    The Floyd-Warshall algorithm can be modified to compute transitive closure:
    
    ```python
    def transitive_closure(graph):
        """
        Compute the transitive closure of a directed graph.
        
        Args:
            graph: Dictionary representing adjacency list of directed graph
            
        Returns:
            dict: Transitive closure as an adjacency list
        """
        vertices = list(graph.keys())
        n = len(vertices)
        
        # Initialize closure with direct edges
        closure = {u: set(graph.get(u, [])) for u in vertices}
        
        # Add self-loops
        for u in vertices:
            closure[u].add(u)
        
        # Apply Floyd-Warshall
        for k in vertices:
            for i in vertices:
                for j in vertices:
                    if k in closure[i] and j in closure[k]:
                        closure[i].add(j)
        
        return closure
    ```

## Comparison with Undirected Graphs

=== "Key Differences"

    | Aspect | Directed Graphs | Undirected Graphs |
    |--------|----------------|-------------------|
    | Edge representation | Ordered pair (u, v) | Unordered pair {u, v} |
    | Connectivity | Strong vs. weak | Connected vs. disconnected |
    | Cycles | Directed cycles | Simple cycles |
    | Paths | Directed paths | Simple paths |
    | Degree | In-degree and out-degree | Degree |
    | Traversal | Can only follow edge directions | Can go in either direction |
    | Eulerian conditions | In-degree = out-degree for all vertices | Even degree for all vertices |
    | Hamiltonian conditions | More complex | More complex |
    | Applications | Workflows, dependencies | Networks, connections |

=== "Algorithm Adaptations"

    Many graph algorithms need adaptations for directed graphs:
    
    - **DFS/BFS**: Follow only outgoing edges
    - **Minimum Spanning Tree**: Not applicable (use Minimum Arborescence instead)
    - **Connectivity**: Check for strong connectivity
    - **Shortest Paths**: Directed edges may create one-way streets
    - **Bipartiteness**: Definition changes for directed graphs

## Advanced Topics

=== "Minimum Spanning Arborescence"

    The directed equivalent of a minimum spanning tree is a minimum spanning arborescence:
    
    - A directed tree where all edges point away from the root
    - Edmonds' algorithm (also known as Chu-Liu/Edmonds algorithm) can find it

=== "Eulerian Path and Circuit"

    In directed graphs:
    
    - For an Eulerian circuit: Every vertex must have equal in-degree and out-degree
    - For an Eulerian path: Either all vertices have equal in-degree and out-degree, or exactly one vertex has out-degree = in-degree + 1 and exactly one has in-degree = out-degree + 1

=== "Network Flow"

    Directed graphs are essential for network flow problems:
    
    - Maximum flow from a source to a sink
    - Minimum cost flow
    - Bipartite matching using flows
    - Ford-Fulkerson, Edmonds-Karp, and Dinic's algorithms

## References

- [Directed Graph on Wikipedia](https://en.wikipedia.org/wiki/Directed_graph)
- [Strongly Connected Components](https://en.wikipedia.org/wiki/Strongly_connected_component)
- [Topological Sorting](https://en.wikipedia.org/wiki/Topological_sorting)
- [Introduction to Algorithms (CLRS)](https://en.wikipedia.org/wiki/Introduction_to_Algorithms) - Chapters 22-26
- [Graph Theory by Reinhard Diestel](https://diestel-graph-theory.com/)
- [Algorithms by Robert Sedgewick and Kevin Wayne](https://algs4.cs.princeton.edu/)
