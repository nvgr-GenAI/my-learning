# Hamilton Path & Circuit

A Hamilton path is a path in a graph that visits each vertex exactly once. A Hamilton circuit (or Hamilton cycle) is a Hamilton path that returns to the starting vertex.

## Overview

=== "Definition"

    - **Hamilton Path**: A path that visits every vertex in a graph exactly once
    - **Hamilton Circuit**: A Hamilton path that is a cycle (starts and ends at the same vertex)
    
    Unlike Euler paths which visit every edge exactly once, Hamilton paths focus on visiting every vertex exactly once.
    
    ![Hamilton Path Example](https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Hamiltonian_path.svg/440px-Hamiltonian_path.svg.png)
    
    Unlike Euler paths, there is no simple necessary and sufficient condition to determine if a graph contains a Hamilton path or circuit. The problem of finding a Hamilton path or circuit is NP-complete, making it significantly more difficult than finding Euler paths.

=== "History"

    The concept of Hamilton paths and circuits is named after Irish mathematician Sir William Rowan Hamilton, who invented a mathematical game called the "Icosian game" in 1857. The goal of the game was to find a Hamilton circuit along the edges of a dodecahedron.
    
    Hamilton marketed this game as a puzzle, challenging players to "travel around the world" by visiting a set of cities exactly once and returning to the starting point.
    
    This problem later became one of the foundational problems in graph theory and computational complexity theory.

=== "Applications"

    - **Traveling Salesman Problem**: Finding the shortest Hamilton circuit (visits every city once and returns to start)
    - **Circuit design**: Placing components to minimize connection lengths
    - **Genome sequencing**: Reconstructing DNA sequences
    - **Operations research**: Scheduling and planning problems
    - **Network design**: Designing resilient networks with backup paths
    - **Game theory**: Analyzing certain types of games and puzzles
    - **Computer graphics**: Drawing paths without lifting the pen or visiting a vertex twice

## Detecting Hamilton Paths and Circuits

=== "Necessary Conditions"

    While there is no simple characterization for Hamiltonian graphs, there are several necessary conditions:
    
    1. **Connectivity**: The graph must be connected
    2. **Dirac's Theorem**: If every vertex has degree ≥ n/2 (where n is the number of vertices), the graph has a Hamilton circuit
    3. **Ore's Theorem**: If for every pair of non-adjacent vertices, the sum of their degrees is ≥ n, the graph has a Hamilton circuit
    4. **Bondy-Chvátal Theorem**: A generalization of Dirac's and Ore's theorems using the concept of graph closure
    
    However, these conditions are sufficient but not necessary—a graph might have a Hamilton circuit even if it doesn't satisfy these conditions.

=== "Complete Analysis"

    To determine with certainty whether a graph has a Hamilton path or circuit, we generally need to search for one explicitly. This is an NP-complete problem, meaning there is no known polynomial-time algorithm to solve it for all graphs.
    
    The most common approaches are:
    
    1. **Backtracking**: Systematically explore all possible paths
    2. **Dynamic programming**: Useful for smaller graphs
    3. **Heuristic algorithms**: For large graphs where exact solutions are infeasible
    4. **Special case algorithms**: For specific graph types (like complete graphs, grid graphs, etc.)

## Finding Hamilton Paths and Circuits

=== "Backtracking Algorithm"

    The backtracking approach for finding a Hamilton path:
    
    1. Start at any vertex
    2. Recursively explore adjacent unvisited vertices
    3. Backtrack if no solution is found
    4. For a Hamilton circuit, check if the last vertex is adjacent to the first
    
    This has exponential time complexity in the worst case, but works well for small graphs and can be improved with pruning techniques.

=== "Dynamic Programming Approach"

    For small to medium graphs (up to about 20-25 vertices), dynamic programming can efficiently find Hamilton paths or circuits:
    
    1. Define subproblems as: can we visit a subset of vertices S ending at vertex v?
    2. Build solutions bottom-up, starting with small subsets and expanding
    3. Use bit manipulation to efficiently represent vertex subsets
    
    This approach has a time complexity of O(n²·2ⁿ) and space complexity of O(n·2ⁿ), where n is the number of vertices.

=== "Implementation"

    ```python
    def find_hamiltonian_path(graph):
        """
        Find a Hamiltonian path using backtracking.
        
        Args:
            graph: Dictionary representing adjacency list of graph
            
        Returns:
            list: Hamiltonian path if one exists, otherwise empty list
        """
        vertices = list(graph.keys())
        n = len(vertices)
        
        # Initialize path with the first vertex
        path = [vertices[0]]
        visited = {vertices[0]: True}
        for v in vertices[1:]:
            visited[v] = False
            
        def backtrack():
            # If all vertices are visited, we've found a Hamiltonian path
            if len(path) == n:
                return True
                
            # Try each unvisited neighbor of the last vertex in the path
            current = path[-1]
            for neighbor in graph[current]:
                if not visited.get(neighbor, False):
                    path.append(neighbor)
                    visited[neighbor] = True
                    
                    if backtrack():
                        return True
                        
                    # Backtrack if no solution found
                    path.pop()
                    visited[neighbor] = False
                    
            return False
            
        # If backtracking succeeds, return the path; otherwise, return empty list
        if backtrack():
            return path
        return []
        
    def find_hamiltonian_circuit(graph):
        """
        Find a Hamiltonian circuit using backtracking.
        
        Args:
            graph: Dictionary representing adjacency list of graph
            
        Returns:
            list: Hamiltonian circuit if one exists, otherwise empty list
        """
        path = find_hamiltonian_path(graph)
        
        # Check if the last vertex is connected to the first to form a circuit
        if path and path[-1] in graph[path[0]]:
            return path + [path[0]]  # Add first vertex again to complete the circuit
        return []
    
    # Example usage
    graph = {
        'A': ['B', 'C', 'D'],
        'B': ['A', 'C', 'E'],
        'C': ['A', 'B', 'D', 'E'],
        'D': ['A', 'C', 'E'],
        'E': ['B', 'C', 'D']
    }
    
    print("Hamiltonian Path:", find_hamiltonian_path(graph))
    # Possible output: ['A', 'B', 'E', 'D', 'C']
    
    print("Hamiltonian Circuit:", find_hamiltonian_circuit(graph))
    # Possible output: ['A', 'B', 'E', 'D', 'C', 'A']
    ```

=== "Dynamic Programming Implementation"

    ```python
    def hamiltonian_path_dp(graph):
        """
        Find a Hamiltonian path using dynamic programming.
        
        Args:
            graph: Dictionary representing adjacency list of graph
            
        Returns:
            list: Hamiltonian path if one exists, otherwise empty list
        """
        vertices = list(graph.keys())
        n = len(vertices)
        
        # Map vertices to indices
        vertex_to_index = {vertices[i]: i for i in range(n)}
        
        # Initialize DP table
        # dp[mask][i] = True if there's a path visiting vertices in mask and ending at vertex i
        dp = {}
        
        # Initialize base cases: paths with a single vertex
        for i in range(n):
            mask = 1 << i
            dp[(mask, i)] = [i]
        
        # Fill DP table
        for subset_size in range(2, n+1):
            for subset in get_subsets(n, subset_size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                for end in subset:
                    # Skip if this is the only vertex in the subset
                    if mask == (1 << end):
                        continue
                    
                    # Try all possible previous vertices
                    prev_mask = mask & ~(1 << end)
                    for prev in subset:
                        if prev == end or not (1 << prev) & mask:
                            continue
                            
                        # Check if there's an edge from prev to end
                        if vertices[end] not in graph[vertices[prev]]:
                            continue
                            
                        # Check if we can create a path ending at 'prev'
                        if (prev_mask, prev) in dp:
                            dp[(mask, end)] = dp[(prev_mask, prev)] + [end]
                            break
        
        # Check if we found a Hamiltonian path
        full_mask = (1 << n) - 1
        for end in range(n):
            if (full_mask, end) in dp:
                # Convert indices back to vertices
                return [vertices[i] for i in dp[(full_mask, end)]]
                
        return []  # No Hamiltonian path found
    
    def get_subsets(n, k):
        """Generate all subsets of {0, 1, ..., n-1} with size k"""
        result = []
        
        def backtrack(start, subset):
            if len(subset) == k:
                result.append(subset[:])
                return
                
            for i in range(start, n):
                subset.append(i)
                backtrack(i + 1, subset)
                subset.pop()
                
        backtrack(0, [])
        return result
    ```

## The Traveling Salesman Problem

=== "Problem Definition"

    The Traveling Salesman Problem (TSP) is a famous extension of the Hamilton circuit problem where we want to find the shortest Hamilton circuit in a weighted graph.
    
    Given a list of cities and the distances between each pair of cities, the TSP asks for the shortest possible route that visits each city exactly once and returns to the origin city.
    
    This is one of the most studied NP-hard problems in combinatorial optimization.

=== "Exact Algorithms"

    For small instances (typically up to 20-30 cities), several exact algorithms can solve the TSP:
    
    1. **Dynamic Programming**: Using the Held-Karp algorithm, with time complexity O(n²·2ⁿ)
    2. **Branch and Bound**: Systematically explores the solution space, pruning suboptimal branches
    3. **Integer Linear Programming**: Formulates the problem as an integer program and uses solvers
    
    These methods guarantee the optimal solution but become impractical for large instances.

=== "Approximation Algorithms"

    For larger instances, approximation algorithms can find near-optimal solutions:
    
    1. **Christofides Algorithm**: Guarantees a solution within 3/2 of the optimal
    2. **2-Approximation Algorithm**: Using minimum spanning tree and shortcutting
    3. **Nearest Neighbor**: Simple greedy approach that works well in practice
    4. **Lin-Kernighan Heuristic**: Local search method that iteratively improves the tour

## Special Cases and Properties

=== "Complete Graphs"

    In a complete graph (where every vertex is connected to every other vertex), a Hamilton path always exists, and finding one is trivial. A Hamilton circuit also always exists for complete graphs with 3 or more vertices.

=== "Bipartite Graphs"

    A bipartite graph can have a Hamilton path or circuit only if the two parts are of similar size:
    
    - For a Hamilton path: The sizes of the two parts can differ by at most 1
    - For a Hamilton circuit: The two parts must be of equal size

=== "Planar Graphs"

    Not all planar graphs contain Hamilton paths or circuits. However, Tutte's theorem states that every 4-connected planar graph contains a Hamilton circuit.

=== "Dirac's and Ore's Theorems"

    **Dirac's Theorem**: If G is a graph with n ≥ 3 vertices and every vertex has degree ≥ n/2, then G has a Hamilton circuit.
    
    **Ore's Theorem**: If G is a graph with n ≥ 3 vertices and for every pair of non-adjacent vertices u and v, deg(u) + deg(v) ≥ n, then G has a Hamilton circuit.

## Comparison with Euler Paths

=== "Key Differences"

    | Aspect | Euler Path/Circuit | Hamilton Path/Circuit |
    |--------|-------------------|----------------------|
    | Focus | Visits every edge exactly once | Visits every vertex exactly once |
    | Detection | Polynomial time (easy) | NP-complete (hard) |
    | Condition | Simple necessary and sufficient conditions exist | No simple characterization |
    | Applications | Network traversal, Chinese postman problem | TSP, circuit design |
    | Named after | Leonhard Euler | William Rowan Hamilton |

## References

- [Hamilton Path on Wikipedia](https://en.wikipedia.org/wiki/Hamiltonian_path)
- [Traveling Salesman Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem)
- [Dirac's Theorem](https://en.wikipedia.org/wiki/Dirac%27s_theorem)
- [Ore's Theorem](https://en.wikipedia.org/wiki/Ore%27s_theorem)
- [Held-Karp Algorithm](https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm)
- [Christofides Algorithm](https://en.wikipedia.org/wiki/Christofides_algorithm)
