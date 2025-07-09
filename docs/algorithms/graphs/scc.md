# Strongly Connected Components

## Overview

In a directed graph, a strongly connected component (SCC) is a subgraph where every vertex is reachable from every other vertex within that subgraph. Identifying SCCs is fundamental to understanding the structure of directed graphs.

## Definition

A strongly connected component of a directed graph G is a maximal subgraph where for every pair of vertices u and v in the subgraph, there exists a path from u to v and a path from v to u.

## Algorithms for Finding SCCs

### Kosaraju's Algorithm

Kosaraju's algorithm uses two passes of Depth-First Search (DFS) to find all SCCs in linear time O(V+E).

#### Algorithm Steps:
1. Perform a DFS on the original graph to compute finishing times for each vertex
2. Create a transposed graph (reverse all edges)
3. Perform another DFS on the transposed graph, starting with vertices in decreasing order of finishing time
4. Each tree in the second DFS forest corresponds to a strongly connected component

#### Implementation:

```python
def kosaraju_scc(graph):
    """
    Find strongly connected components using Kosaraju's algorithm.
    
    Parameters:
    - graph: A dictionary where keys are vertices and values are lists of adjacent vertices
    
    Returns:
    - A list of lists, each inner list representing vertices in one SCC
    """
    def dfs_first_pass(node):
        """DFS to compute finishing times."""
        visited[node] = True
        for neighbor in graph.get(node, []):
            if not visited[neighbor]:
                dfs_first_pass(neighbor)
        finish_time.append(node)
    
    def dfs_second_pass(node):
        """DFS to identify components."""
        visited[node] = True
        current_scc.append(node)
        for neighbor in reversed_graph.get(node, []):
            if not visited[neighbor]:
                dfs_second_pass(neighbor)
    
    # Step 1: First DFS to compute finishing times
    visited = {node: False for node in graph}
    finish_time = []
    
    for node in graph:
        if not visited[node]:
            dfs_first_pass(node)
    
    # Step 2: Create reversed graph
    reversed_graph = {}
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if neighbor not in reversed_graph:
                reversed_graph[neighbor] = []
            reversed_graph[neighbor].append(node)
    
    # Step 3: Second DFS in order of decreasing finish time
    visited = {node: False for node in graph}
    scc_list = []
    
    while finish_time:
        node = finish_time.pop()  # Get node with highest finish time
        if not visited[node]:
            current_scc = []
            dfs_second_pass(node)
            scc_list.append(current_scc)
    
    return scc_list

# Example usage
if __name__ == "__main__":
    # Example directed graph
    graph = {
        0: [1],
        1: [2],
        2: [0, 3],
        3: [4],
        4: [5],
        5: [3]
    }
    
    sccs = kosaraju_scc(graph)
    print("Strongly Connected Components:")
    for i, scc in enumerate(sccs):
        print(f"SCC {i+1}: {scc}")
```

### Tarjan's Algorithm

Tarjan's algorithm finds SCCs in a single pass of DFS, making it often more efficient in practice.

#### Algorithm Steps:
1. Perform a DFS traversal
2. Keep track of discovery time and lowest reachable vertex for each node
3. Use a stack to track the vertices in the current potential SCC
4. When a node's discovery time equals its lowest reachable vertex, it's the root of an SCC

#### Implementation:

```python
def tarjan_scc(graph):
    """
    Find strongly connected components using Tarjan's algorithm.
    
    Parameters:
    - graph: A dictionary where keys are vertices and values are lists of adjacent vertices
    
    Returns:
    - A list of lists, each inner list representing vertices in one SCC
    """
    disc = {}  # Discovery times
    low = {}   # Lowest reachable vertex
    stack = []  # Stack for vertices
    on_stack = {}  # Track if vertex is on stack
    scc_list = []  # List of SCCs
    
    time = [0]  # Use list for pass-by-reference in Python
    
    def dfs(node):
        # Initialize discovery time and low value
        disc[node] = low[node] = time[0]
        time[0] += 1
        stack.append(node)
        on_stack[node] = True
        
        # Visit all neighbors
        for neighbor in graph.get(node, []):
            # If neighbor not visited yet
            if neighbor not in disc:
                dfs(neighbor)
                low[node] = min(low[node], low[neighbor])
            # If neighbor is on stack (back edge)
            elif on_stack[neighbor]:
                low[node] = min(low[node], disc[neighbor])
        
        # If node is root of SCC
        if disc[node] == low[node]:
            current_scc = []
            
            # Pop vertices from stack until we find the root
            while True:
                vertex = stack.pop()
                on_stack[vertex] = False
                current_scc.append(vertex)
                if vertex == node:
                    break
            
            scc_list.append(current_scc)
    
    # Run DFS from each unvisited vertex
    for node in graph:
        if node not in disc:
            dfs(node)
    
    return scc_list

# Example usage (same example as above)
if __name__ == "__main__":
    # Example directed graph
    graph = {
        0: [1],
        1: [2],
        2: [0, 3],
        3: [4],
        4: [5],
        5: [3]
    }
    
    sccs = tarjan_scc(graph)
    print("Strongly Connected Components:")
    for i, scc in enumerate(sccs):
        print(f"SCC {i+1}: {scc}")
```

## Applications of SCCs

1. **Graph Condensation**: Converting a graph into a DAG of strongly connected components
2. **Social Network Analysis**: Finding communities in social networks
3. **Web Page Ranking**: Used in algorithms like PageRank to analyze web link structure
4. **Circuit Analysis**: Identifying components in electronic circuits
5. **Compiler Optimization**: Determining data dependencies for optimization

## Condensation Graph

The condensation graph G' of a directed graph G is formed by:
1. Contracting each SCC into a single vertex
2. Creating an edge between two SCCs if there was at least one edge between vertices in the original graph

The condensation graph is always a directed acyclic graph (DAG).

## Time and Space Complexity

For both Kosaraju's and Tarjan's algorithms:

- **Time Complexity**: O(V + E) where V is the number of vertices and E is the number of edges
- **Space Complexity**: O(V) for storing the discovery times, low values, and stack

## Comparison of Algorithms

| Algorithm | Passes | Time Complexity | Space Complexity | Notes |
|-----------|--------|----------------|------------------|-------|
| Kosaraju's | 2 DFS | O(V + E) | O(V) | Simpler to implement but requires two passes |
| Tarjan's | 1 DFS | O(V + E) | O(V) | More efficient in practice with a single pass |
| Gabow's | 1 DFS | O(V + E) | O(V) | Similar to Tarjan's but uses two stacks |

## Important Properties of SCCs

1. **Component Graph is a DAG**: The graph of SCCs always forms a directed acyclic graph
2. **Mutual Reachability**: Within an SCC, every vertex can reach every other vertex
3. **Maximality**: SCCs are maximal - they cannot be expanded further
4. **Unique Partition**: The partition of vertices into SCCs is unique

## Related Concepts

- **Weakly Connected Components**: Connected components in the undirected version of the graph
- **Bi-connected Components**: Subgraphs that remain connected even after removing any single vertex
- **Articulation Points**: Vertices whose removal increases the number of connected components
