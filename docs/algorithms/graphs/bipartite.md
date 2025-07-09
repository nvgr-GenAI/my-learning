# Bipartite Graphs

A bipartite graph is a graph whose vertices can be divided into two disjoint sets such that every edge connects vertices from different sets. These sets are often called "parts," hence the name bipartite.

## Overview

=== "Definition"

    A bipartite graph (or bigraph) is a graph whose vertices can be divided into two disjoint and independent sets $U$ and $V$ such that every edge connects a vertex in $U$ to one in $V$. Equivalently, a bipartite graph is a graph that does not contain any odd-length cycles.

    ![Bipartite Graph](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Simple-bipartite-graph.svg/440px-Simple-bipartite-graph.svg.png)

    **Key properties:**
    
    - No edge connects vertices within the same set
    - The graph can be colored using only two colors (vertices in the same set have the same color)
    - Every cycle in a bipartite graph has even length

=== "Applications"

    Bipartite graphs naturally model many real-world scenarios where relationships exist between two distinct types of entities:
    
    1. **Matching problems**: Job assignments, college admissions, organ donation matching
    2. **Network flow problems**: Determining maximum throughput in distribution networks
    3. **Task scheduling**: Assigning resources to tasks
    4. **Recommendation systems**: Modeling user-item interactions
    5. **Social networks**: Modeling relationships between different groups
    6. **Chemistry**: Representing molecular structures

=== "Real-world Examples"

    - **Employment networks**: Workers (set U) and jobs (set V)
    - **Dating applications**: Men (set U) and women (set V) in a heterosexual dating app
    - **Recommender systems**: Users (set U) and products (set V)
    - **Document classification**: Documents (set U) and keywords (set V)
    - **Chemical compounds**: Carbon atoms (set U) and hydrogen atoms (set V) in certain hydrocarbons

## Detection Algorithm

=== "Algorithm"

    **Checking if a graph is bipartite:**
    
    The most common approach is to use a breadth-first search (BFS) or depth-first search (DFS) to color the vertices with two colors. If we can successfully color the entire graph with two colors such that no adjacent vertices have the same color, the graph is bipartite.
    
    **Steps:**
    
    1. Start at any vertex and assign it color 1
    2. For each unvisited adjacent vertex, assign the opposite color
    3. If we ever find adjacent vertices with the same color, the graph is not bipartite
    4. Repeat for all connected components

=== "Complexity"

    - **Time Complexity**: $O(V + E)$ where $V$ is the number of vertices and $E$ is the number of edges
    - **Space Complexity**: $O(V)$ for storing the color of each vertex and the queue/stack for BFS/DFS

=== "Edge Cases"

    - Empty graph: Trivially bipartite
    - Single vertex: Trivially bipartite
    - Disconnected graph: Each connected component must be bipartite
    - Trees: Always bipartite (no cycles)

## Implementation

=== "Python"

    ```python
    from collections import deque

    def is_bipartite(graph):
        """
        Check if an undirected graph is bipartite using BFS.
        
        Args:
            graph: Dictionary representing adjacency list of graph
            
        Returns:
            bool: True if graph is bipartite, False otherwise
        """
        if not graph:
            return True
            
        # Initialize colors: -1 means uncolored, 0 and 1 represent two different colors
        colors = {node: -1 for node in graph}
        
        # Process each connected component
        for start in graph:
            if colors[start] == -1:  # If uncolored
                # Start BFS from this node
                queue = deque([start])
                colors[start] = 0
                
                while queue:
                    node = queue.popleft()
                    
                    # Check all neighbors
                    for neighbor in graph[node]:
                        # If uncolored, assign opposite color
                        if colors[neighbor] == -1:
                            colors[neighbor] = 1 - colors[node]
                            queue.append(neighbor)
                        # If already colored with same color, not bipartite
                        elif colors[neighbor] == colors[node]:
                            return False
        
        return True
    
    # Example usage
    graph = {
        1: [2, 3],
        2: [1, 4],
        3: [1, 4],
        4: [2, 3]
    }
    
    print(is_bipartite(graph))  # Should print: False (contains a 3-cycle)
    
    bipartite_graph = {
        1: [2, 4],
        2: [1, 3, 5],
        3: [2, 6],
        4: [1, 5],
        5: [2, 4, 6],
        6: [3, 5]
    }
    
    print(is_bipartite(bipartite_graph))  # Should print: True
    ```

=== "Java"

    ```java
    import java.util.*;

    public class BipartiteGraph {
        
        public static boolean isBipartite(Map<Integer, List<Integer>> graph) {
            if (graph.isEmpty()) {
                return true;
            }
            
            // Colors map: -1 = uncolored, 0 and 1 represent two different colors
            Map<Integer, Integer> colors = new HashMap<>();
            for (int node : graph.keySet()) {
                colors.put(node, -1);
            }
            
            // Process each connected component
            for (int start : graph.keySet()) {
                if (colors.get(start) == -1) {
                    // Start BFS from this node
                    Queue<Integer> queue = new LinkedList<>();
                    queue.add(start);
                    colors.put(start, 0);
                    
                    while (!queue.isEmpty()) {
                        int node = queue.poll();
                        
                        // Check all neighbors
                        for (int neighbor : graph.get(node)) {
                            // If uncolored, assign opposite color
                            if (colors.get(neighbor) == -1) {
                                colors.put(neighbor, 1 - colors.get(node));
                                queue.add(neighbor);
                            }
                            // If already colored with same color, not bipartite
                            else if (colors.get(neighbor) == colors.get(node)) {
                                return false;
                            }
                        }
                    }
                }
            }
            
            return true;
        }
        
        public static void main(String[] args) {
            // Example: Graph that is not bipartite (contains a 3-cycle)
            Map<Integer, List<Integer>> graph = new HashMap<>();
            graph.put(1, Arrays.asList(2, 3));
            graph.put(2, Arrays.asList(1, 4));
            graph.put(3, Arrays.asList(1, 4));
            graph.put(4, Arrays.asList(2, 3));
            
            System.out.println("Is bipartite? " + isBipartite(graph));  // Should print: false
            
            // Example: Bipartite graph
            Map<Integer, List<Integer>> bipartiteGraph = new HashMap<>();
            bipartiteGraph.put(1, Arrays.asList(2, 4));
            bipartiteGraph.put(2, Arrays.asList(1, 3, 5));
            bipartiteGraph.put(3, Arrays.asList(2, 6));
            bipartiteGraph.put(4, Arrays.asList(1, 5));
            bipartiteGraph.put(5, Arrays.asList(2, 4, 6));
            bipartiteGraph.put(6, Arrays.asList(3, 5));
            
            System.out.println("Is bipartite? " + isBipartite(bipartiteGraph));  // Should print: true
        }
    }
    ```

=== "JavaScript"

    ```javascript
    function isBipartite(graph) {
        if (Object.keys(graph).length === 0) {
            return true;
        }
        
        // Colors map: -1 = uncolored, 0 and 1 represent two different colors
        const colors = {};
        for (const node in graph) {
            colors[node] = -1;
        }
        
        // Process each connected component
        for (const start in graph) {
            if (colors[start] === -1) {
                // Start BFS from this node
                const queue = [start];
                colors[start] = 0;
                
                while (queue.length > 0) {
                    const node = queue.shift();
                    
                    // Check all neighbors
                    for (const neighbor of graph[node]) {
                        // If uncolored, assign opposite color
                        if (colors[neighbor] === -1) {
                            colors[neighbor] = 1 - colors[node];
                            queue.push(neighbor);
                        }
                        // If already colored with same color, not bipartite
                        else if (colors[neighbor] === colors[node]) {
                            return false;
                        }
                    }
                }
            }
        }
        
        return true;
    }
    
    // Example usage
    const graph = {
        '1': ['2', '3'],
        '2': ['1', '4'],
        '3': ['1', '4'],
        '4': ['2', '3']
    };
    
    console.log(isBipartite(graph));  // Should print: false (contains a 3-cycle)
    
    const bipartiteGraph = {
        '1': ['2', '4'],
        '2': ['1', '3', '5'],
        '3': ['2', '6'],
        '4': ['1', '5'],
        '5': ['2', '4', '6'],
        '6': ['3', '5']
    };
    
    console.log(isBipartite(bipartiteGraph));  // Should print: true
    ```

## Maximum Bipartite Matching

=== "Concept"

    Maximum Bipartite Matching is the problem of finding the maximum number of edges in a bipartite graph such that no two edges share an endpoint.
    
    This is equivalent to finding the maximum number of people who can be assigned to jobs in the employment network example, where each person can do at most one job, and each job can be done by at most one person.
    
    **Applications:**
    
    - Job assignments
    - College admissions
    - Organ donor matching
    - Resource allocation

=== "Algorithm: Ford-Fulkerson"

    The standard algorithm for Maximum Bipartite Matching is the Ford-Fulkerson algorithm for maximum flow:
    
    1. Create a source node $s$ and add directed edges from $s$ to all nodes in set $U$
    2. Create a sink node $t$ and add directed edges from all nodes in set $V$ to $t$
    3. Make all original edges in the bipartite graph directed from $U$ to $V$ with capacity 1
    4. Run the Ford-Fulkerson algorithm to find the maximum flow from $s$ to $t$
    5. The maximum flow value equals the maximum matching size

=== "Implementation"

    ```python
    def max_bipartite_matching(graph, u_set, v_set):
        """
        Find maximum bipartite matching using Ford-Fulkerson algorithm.
        
        Args:
            graph: Dictionary representing adjacency list of graph
            u_set: Set of vertices in first partition
            v_set: Set of vertices in second partition
            
        Returns:
            int: Size of maximum matching
            dict: Matched pairs (u -> v)
        """
        # Initialize matching
        matching = {}  # Maps vertices in u_set to their matched vertex in v_set
        
        def dfs(u, visited):
            for v in graph.get(u, []):
                if v not in visited:
                    visited.add(v)
                    
                    # If v is not matched or its match can be reassigned
                    if v not in reverse_matching or dfs(reverse_matching[v], visited):
                        matching[u] = v
                        reverse_matching[v] = u
                        return True
            return False
        
        # Run matching algorithm
        match_count = 0
        reverse_matching = {}  # Maps vertices in v_set to their matched vertex in u_set
        
        for u in u_set:
            if u not in matching:
                if dfs(u, set()):
                    match_count += 1
        
        return match_count, matching
    
    # Example usage
    graph = {
        'A': ['1', '2'],
        'B': ['1', '3'],
        'C': ['2'],
        'D': ['2', '3', '4'],
        'E': ['4']
    }
    
    u_set = {'A', 'B', 'C', 'D', 'E'}  # Workers
    v_set = {'1', '2', '3', '4'}       # Jobs
    
    max_count, assignments = max_bipartite_matching(graph, u_set, v_set)
    print(f"Maximum matching: {max_count}")
    print("Assignments:")
    for worker, job in assignments.items():
        print(f"Worker {worker} assigned to job {job}")
    ```

## Related Problems

1. **Maximum Bipartite Matching**: Find the maximum number of edges with no shared endpoints
2. **Minimum Vertex Cover in Bipartite Graphs**: Find the smallest set of vertices such that each edge has at least one endpoint in the set
3. **Maximum Independent Set in Bipartite Graphs**: Find the largest set of vertices with no edges between them
4. **Bipartite Coloring**: Color the vertices of a bipartite graph with exactly two colors
5. **König's Theorem**: In bipartite graphs, the size of the minimum vertex cover equals the size of the maximum matching

## References

- [Bipartite Graph on Wikipedia](https://en.wikipedia.org/wiki/Bipartite_graph)
- [Maximum Bipartite Matching on GeeksforGeeks](https://www.geeksforgeeks.org/maximum-bipartite-matching/)
- [Hopcroft-Karp Algorithm for Maximum Matching](https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm)
- [König's Theorem](https://en.wikipedia.org/wiki/K%C5%91nig%27s_theorem_(graph_theory))
