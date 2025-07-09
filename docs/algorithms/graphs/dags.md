# Directed Acyclic Graphs (DAGs)

A Directed Acyclic Graph (DAG) is a directed graph with no directed cycles. This special structure makes DAGs fundamental in computer science and many other fields, enabling efficient algorithms and elegant solutions to a wide range of problems.

## Overview

=== "Definition"

    A Directed Acyclic Graph (DAG) is a directed graph that does not contain any directed cycles. In other words, it is impossible to start at a vertex, follow a sequence of directed edges, and return to the starting vertex.
    
    ![Directed Acyclic Graph](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Tred-G.svg/440px-Tred-G.svg.png)
    
    **Key properties:**
    
    - Has directed edges
    - Contains no directed cycles
    - Always has at least one source (vertex with no incoming edges) and at least one sink (vertex with no outgoing edges)
    - Always admits a topological ordering of vertices
    - Can represent partial orders

=== "Applications"

    DAGs are widely used to model various systems and solve numerous problems:
    
    - **Dependency resolution**: Software dependencies, build systems
    - **Task scheduling**: Project management, job scheduling
    - **Data processing pipelines**: ETL workflows, computational workflows
    - **Bayesian networks**: Probabilistic graphical models
    - **Compilation**: Instruction scheduling, expression evaluation
    - **Version control systems**: Git commit history
    - **Citation networks**: Academic paper citations
    - **Spreadsheet calculations**: Cell dependencies
    - **Static analysis of programs**: Control flow and data flow
    - **Dynamic programming**: Subproblem dependencies

=== "Examples"

    Common examples of DAGs in practice:
    
    - **Make build files**: Dependencies between compilation units
    - **Course prerequisites**: Dependencies between academic courses
    - **Task schedulers**: Dependencies between tasks in a workflow
    - **Package managers**: Dependencies between software packages
    - **Git commit graph**: History of commits in Git (when considering only parent-child relationships)
    - **Call graphs**: Function calls in programs (without recursion)
    - **Ancestry trees**: Family relationships (without loops due to intermarriage)

## Topological Sorting

=== "Definition"

    A topological sort or topological ordering of a DAG is a linear ordering of its vertices such that for every directed edge (u, v), vertex u comes before vertex v in the ordering.
    
    Every DAG has at least one topological sort, and it may have many. For example, the DAG {1→2, 1→3, 3→4, 2→4} has two valid topological sorts: [1,2,3,4] and [1,3,2,4].
    
    Topological sorting is one of the most fundamental algorithms for DAGs with numerous applications.

=== "Kahn's Algorithm"

    Kahn's algorithm is a simple method for topological sorting:
    
    1. Identify all vertices with no incoming edges (in-degree = 0) and add them to a queue
    2. While the queue is not empty:
       a. Remove a vertex from the queue and add it to the result
       b. For each of its neighbors, reduce their in-degree by 1
       c. If a neighbor's in-degree becomes 0, add it to the queue
    
    ```python
    from collections import deque
    
    def topological_sort_kahn(graph):
        """
        Perform topological sort using Kahn's algorithm.
        
        Args:
            graph: Dictionary representing adjacency list of DAG
            
        Returns:
            list: Vertices in topological order
        """
        # Calculate in-degree for each vertex
        in_degree = {vertex: 0 for vertex in graph}
        for vertex in graph:
            for neighbor in graph[vertex]:
                in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
        
        # Initialize queue with vertices that have no incoming edges
        queue = deque([vertex for vertex, degree in in_degree.items() if degree == 0])
        result = []
        
        # Process vertices
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # Reduce in-degree of neighbors
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If result doesn't include all vertices, there must be a cycle
        if len(result) != len(graph):
            raise ValueError("Graph contains a cycle, not a DAG")
        
        return result
    ```

=== "DFS-Based Algorithm"

    The DFS-based algorithm for topological sorting:
    
    1. Run DFS
    2. As each vertex finishes (all its descendants have been processed), prepend it to the result list
    3. After all vertices are processed, the result is a valid topological sort
    
    ```python
    def topological_sort_dfs(graph):
        """
        Perform topological sort using DFS.
        
        Args:
            graph: Dictionary representing adjacency list of DAG
            
        Returns:
            list: Vertices in topological order
        """
        visited = set()
        temp_mark = set()  # For cycle detection
        result = []
        
        def dfs(vertex):
            if vertex in temp_mark:
                raise ValueError("Graph contains a cycle, not a DAG")
            if vertex in visited:
                return
                
            temp_mark.add(vertex)
            
            for neighbor in graph.get(vertex, []):
                dfs(neighbor)
            
            temp_mark.remove(vertex)
            visited.add(vertex)
            result.append(vertex)
        
        # Process all vertices
        for vertex in graph:
            if vertex not in visited:
                dfs(vertex)
        
        return result[::-1]  # Reverse to get correct topological order
    ```

## Critical Path Analysis

=== "Definition"

    In a DAG where vertices represent tasks and edges represent dependencies, the critical path is the longest path through the graph. It determines the minimum time required to complete all tasks, assuming tasks can be executed in parallel when dependencies allow.
    
    Finding the critical path is important in project management to identify which tasks directly affect the project duration.

=== "Algorithm"

    The algorithm to find the critical path:
    
    1. Perform a topological sort of the DAG
    2. For each vertex in topological order, compute its earliest completion time:
       - earliest[v] = max(earliest[u] + duration[v]) for all edges (u,v)
    3. The critical path length is the maximum earliest completion time
    4. Backtrack to find the critical path
    
    ```python
    def critical_path(graph, durations):
        """
        Find the critical path in a DAG.
        
        Args:
            graph: Dictionary representing adjacency list of DAG
            durations: Dictionary of task durations
            
        Returns:
            list: Critical path (list of vertices)
            int: Length of the critical path
        """
        # Perform topological sort
        topo_order = topological_sort_dfs(graph)
        
        # Compute earliest completion time for each vertex
        earliest = {v: 0 for v in graph}
        parent = {v: None for v in graph}
        
        for v in topo_order:
            for neighbor in graph.get(v, []):
                if earliest[v] + durations[v] > earliest[neighbor]:
                    earliest[neighbor] = earliest[v] + durations[v]
                    parent[neighbor] = v
        
        # Find vertex with maximum completion time
        end_time = max(earliest[v] + durations[v] for v in graph)
        end_vertex = max(graph.keys(), key=lambda v: earliest[v] + durations[v])
        
        # Backtrack to find the critical path
        path = []
        current = end_vertex
        while current is not None:
            path.append(current)
            current = parent[current]
        
        return path[::-1], end_time
    ```

## Shortest and Longest Paths in DAGs

=== "Single-Source Shortest Path"

    Finding single-source shortest paths in a DAG is simpler than in general graphs:
    
    1. Perform a topological sort of the DAG
    2. Process vertices in topological order:
       - For each vertex, update the distances of its adjacent vertices
    
    This algorithm runs in O(V + E) time, more efficient than Dijkstra's algorithm, and works even with negative edge weights.
    
    ```python
    def shortest_paths_dag(graph, weights, start):
        """
        Find shortest paths from start vertex in a DAG.
        
        Args:
            graph: Dictionary representing adjacency list of DAG
            weights: Dictionary of edge weights {(u,v): weight, ...}
            start: Starting vertex
            
        Returns:
            dict: Dictionary of shortest distances
            dict: Dictionary of predecessors for path reconstruction
        """
        # Perform topological sort
        topo_order = topological_sort_dfs(graph)
        
        # Initialize distances
        dist = {vertex: float('inf') for vertex in graph}
        dist[start] = 0
        pred = {vertex: None for vertex in graph}
        
        # Process vertices in topological order
        for u in topo_order:
            if dist[u] != float('inf'):
                for v in graph.get(u, []):
                    if dist[u] + weights.get((u, v), 0) < dist[v]:
                        dist[v] = dist[u] + weights.get((u, v), 0)
                        pred[v] = u
        
        return dist, pred
    ```

=== "Single-Source Longest Path"

    Finding longest paths in general graphs is NP-hard, but in a DAG, it can be solved efficiently:
    
    1. Negate all edge weights
    2. Run the shortest path algorithm
    3. Negate the resulting distances
    
    This approach works because a DAG has no cycles, so there are no negative cycles after negating the weights.
    
    ```python
    def longest_paths_dag(graph, weights, start):
        """
        Find longest paths from start vertex in a DAG.
        
        Args:
            graph: Dictionary representing adjacency list of DAG
            weights: Dictionary of edge weights {(u,v): weight, ...}
            start: Starting vertex
            
        Returns:
            dict: Dictionary of longest distances
            dict: Dictionary of predecessors for path reconstruction
        """
        # Negate weights
        neg_weights = {edge: -weight for edge, weight in weights.items()}
        
        # Find shortest paths with negated weights
        dist, pred = shortest_paths_dag(graph, neg_weights, start)
        
        # Negate distances back
        longest_dist = {vertex: -distance for vertex, distance in dist.items()}
        
        return longest_dist, pred
    ```

## Dynamic Programming on DAGs

=== "Subproblem Dependency"

    Many dynamic programming problems can be represented as finding a path or value in a DAG:
    
    - Vertices represent subproblems
    - Edges represent dependencies between subproblems
    - Edge weights or vertex values represent costs or values
    - The solution involves finding a path or value optimization in this DAG
    
    Examples include sequence alignment, optimal binary search trees, and matrix chain multiplication.

=== "Matrix Chain Multiplication"

    The matrix chain multiplication problem can be solved using a DAG:
    
    - Vertices represent partial products A_i × A_{i+1} × ... × A_j
    - Edges represent ways to split the multiplication
    - The goal is to find the minimum cost multiplication order
    
    ```python
    def matrix_chain_order(dimensions):
        """
        Find optimal order for matrix chain multiplication using dynamic programming.
        
        Args:
            dimensions: List of matrix dimensions [d0, d1, d2, ..., dn] where
                        matrix i has dimensions dimensions[i] × dimensions[i+1]
            
        Returns:
            int: Minimum number of scalar multiplications
            list: Optimal parenthesization
        """
        n = len(dimensions) - 1  # Number of matrices
        
        # dp[i][j] = minimum cost of multiplying matrices i through j
        dp = [[0] * n for _ in range(n)]
        
        # split[i][j] = optimal split point k for matrices i through j
        split = [[0] * n for _ in range(n)]
        
        # Build up the dp table
        for length in range(2, n + 1):  # Length of subchain
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = float('inf')
                for k in range(i, j):
                    cost = dp[i][k] + dp[k + 1][j] + dimensions[i] * dimensions[k + 1] * dimensions[j + 1]
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        split[i][j] = k
        
        # Reconstruct the optimal parenthesization
        def print_optimal_parens(i, j):
            if i == j:
                return f"A{i}"
            else:
                k = split[i][j]
                left = print_optimal_parens(i, k)
                right = print_optimal_parens(k + 1, j)
                return f"({left} × {right})"
        
        return dp[0][n - 1], print_optimal_parens(0, n - 1)
    ```

## Checking if a Graph is a DAG

=== "Cycle Detection"

    To check if a directed graph is a DAG, we need to check if it contains any cycles:
    
    ```python
    def is_dag(graph):
        """
        Check if a directed graph is a DAG (has no cycles).
        
        Args:
            graph: Dictionary representing adjacency list of directed graph
            
        Returns:
            bool: True if the graph is a DAG, False otherwise
        """
        visited = set()
        rec_stack = set()
        
        def dfs_cycle_check(vertex):
            visited.add(vertex)
            rec_stack.add(vertex)
            
            for neighbor in graph.get(vertex, []):
                if neighbor not in visited:
                    if dfs_cycle_check(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(vertex)
            return False
        
        for vertex in graph:
            if vertex not in visited:
                if dfs_cycle_check(vertex):
                    return False
        
        return True
    ```

=== "Using Topological Sort"

    Another approach is to attempt a topological sort. If the topological sort includes all vertices, the graph is a DAG; otherwise, it contains a cycle.
    
    ```python
    def is_dag_using_topo_sort(graph):
        """
        Check if a directed graph is a DAG using topological sort.
        
        Args:
            graph: Dictionary representing adjacency list of directed graph
            
        Returns:
            bool: True if the graph is a DAG, False otherwise
        """
        # Calculate in-degree for each vertex
        in_degree = {vertex: 0 for vertex in graph}
        for vertex in graph:
            for neighbor in graph[vertex]:
                in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
        
        # Initialize queue with vertices that have no incoming edges
        queue = deque([vertex for vertex, degree in in_degree.items() if degree == 0])
        count = 0
        
        # Process vertices
        while queue:
            current = queue.popleft()
            count += 1
            
            # Reduce in-degree of neighbors
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If count equals the number of vertices, it's a DAG
        return count == len(graph)
    ```

## Transitive Reduction and Closure

=== "Transitive Closure"

    The transitive closure of a DAG G is a graph G+ that has an edge (u, v) whenever there is a path from u to v in G.
    
    ```python
    def transitive_closure(graph):
        """
        Compute the transitive closure of a DAG.
        
        Args:
            graph: Dictionary representing adjacency list of DAG
            
        Returns:
            dict: Transitive closure as an adjacency list
        """
        vertices = list(graph.keys())
        n = len(vertices)
        
        # Initialize closure with direct edges
        closure = {u: set(graph.get(u, [])) for u in vertices}
        
        # Apply Floyd-Warshall algorithm
        for k in vertices:
            for i in vertices:
                for j in vertices:
                    if j in closure.get(k, set()) and k in closure.get(i, set()):
                        closure.setdefault(i, set()).add(j)
        
        return closure
    ```

=== "Transitive Reduction"

    The transitive reduction of a DAG G is the smallest graph G- that has the same reachability relation as G.
    
    ```python
    def transitive_reduction(graph):
        """
        Compute the transitive reduction of a DAG.
        
        Args:
            graph: Dictionary representing adjacency list of DAG
            
        Returns:
            dict: Transitive reduction as an adjacency list
        """
        # Compute transitive closure
        closure = transitive_closure(graph)
        
        # Initialize reduction with direct edges
        reduction = {u: set(graph.get(u, [])) for u in graph}
        
        # Remove redundant edges
        for u in graph:
            for v in list(reduction.get(u, [])):
                for w in list(reduction.get(u, [])):
                    if v != w and v in closure.get(w, set()):
                        reduction[u].remove(v)
                        break
        
        return reduction
    ```

## Special Types of DAGs

=== "Trees and Forests"

    A directed tree is a DAG where:
    
    - There is a single root (a vertex with no incoming edges)
    - Every other vertex has exactly one incoming edge
    
    A directed forest is a collection of directed trees.
    
    Trees and forests are special cases of DAGs with additional structural constraints.

=== "Series-Parallel Graphs"

    Series-parallel graphs are DAGs that can be constructed recursively by:
    
    - A single edge is a series-parallel graph
    - The series composition of two series-parallel graphs is a series-parallel graph
    - The parallel composition of two series-parallel graphs is a series-parallel graph
    
    These graphs have efficient algorithms for many problems and appear in circuit design.

=== "Lattices"

    A lattice is a DAG where:
    
    - Every pair of elements has a unique supremum (join) and infimum (meet)
    - The graph represents a partial order
    
    Lattices are important in order theory and have applications in programming language semantics and optimization.

## Applications of DAGs

=== "Build Systems"

    Build systems like Make, Gradle, and Bazel use DAGs to represent dependencies between build targets:
    
    - Vertices are build targets (files, libraries, executables)
    - Edges represent dependencies
    - Topological sort determines the build order
    - Parallel builds can process independent targets simultaneously

=== "Task Scheduling"

    Task scheduling algorithms use DAGs to represent task dependencies:
    
    - Vertices are tasks
    - Edges represent dependencies (task A must complete before task B)
    - Critical path analysis finds the minimum completion time
    - Schedulers can optimize resource allocation

=== "Data Processing Pipelines"

    Data processing frameworks like Apache Airflow and Luigi model workflows as DAGs:
    
    - Vertices are processing steps
    - Edges represent data flow or dependencies
    - Topological sort determines execution order
    - Fault tolerance mechanisms can recover from failures

=== "Program Analysis"

    Compilers and static analysis tools use various DAGs:
    
    - Control flow graphs (without loops): Represent possible execution paths
    - Data flow graphs: Represent data dependencies
    - Call graphs (without recursion): Represent function call relationships
    - These DAGs enable optimization, parallelization, and error detection

## Algorithms on DAGs

=== "Transitive Closure"

    The transitive closure of a DAG represents all reachability relationships:
    
    - Can be computed using Floyd-Warshall algorithm in O(V³)
    - DFS-based approaches run in O(V(V+E))
    - Used for reachability queries and dependency analysis

=== "Minimum Path Cover"

    A path cover is a set of paths such that every vertex belongs to at least one path. The minimum path cover is the path cover with the minimum number of paths.
    
    This problem can be solved by:
    
    1. Constructing a bipartite graph
    2. Finding the maximum bipartite matching
    3. The minimum number of paths equals V minus the size of the maximum matching

=== "Maximum Antichain"

    An antichain is a set of vertices such that no two are comparable (no path between them). The maximum antichain is the largest such set.
    
    By Dilworth's theorem, the size of the maximum antichain equals the minimum number of chains needed to cover the DAG (dual of the minimum path cover).

## References

- [Directed Acyclic Graph on Wikipedia](https://en.wikipedia.org/wiki/Directed_acyclic_graph)
- [Topological Sorting](https://en.wikipedia.org/wiki/Topological_sorting)
- [Critical Path Method](https://en.wikipedia.org/wiki/Critical_path_method)
- [Introduction to Algorithms (CLRS)](https://en.wikipedia.org/wiki/Introduction_to_Algorithms) - Chapter 22
- [Dilworth's Theorem](https://en.wikipedia.org/wiki/Dilworth%27s_theorem)
- [Algorithms by Robert Sedgewick and Kevin Wayne](https://algs4.cs.princeton.edu/)
