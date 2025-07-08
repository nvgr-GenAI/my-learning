# Floyd-Warshall Algorithm

## Overview

The Floyd-Warshall algorithm is used to find the shortest paths between all pairs of vertices in a weighted graph. It can handle negative edge weights but cannot handle negative cycles.

## Algorithm Steps

1. Create a distance matrix where the entry at [i][j] is the weight of the edge from vertex i to vertex j
2. For each vertex k as an intermediate point, update the distance matrix if going through k results in a shorter path
3. After processing all vertices, the matrix contains the shortest distances between all pairs of vertices

## Implementation

```python
def floyd_warshall(graph):
    """
    Implementation of Floyd-Warshall algorithm
    
    Args:
        graph: 2D adjacency matrix where graph[i][j] is the weight from i to j
               or float('infinity') if no direct edge exists
               
    Returns:
        2D matrix of shortest distances between all pairs of vertices
    """
    n = len(graph)
    dist = [row[:] for row in graph]  # Create a copy of the graph
    
    # Initialize the distance matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
    
    # Consider each vertex as an intermediate
    for k in range(n):
        # Consider all pairs (i,j) of vertices
        for i in range(n):
            for j in range(n):
                # If vertex k is on the shortest path from i to j, update dist[i][j]
                if dist[i][k] != float('infinity') and dist[k][j] != float('infinity'):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    # Check for negative cycles
    for i in range(n):
        if dist[i][i] < 0:
            print("Graph contains negative cycle")
            return None
    
    return dist
```

## Time Complexity

- **Time Complexity**: O(|V|³), where |V| is the number of vertices
- **Space Complexity**: O(|V|²)

## Applications

- Finding all-pairs shortest paths in a graph
- Transitive closure of a graph
- Detecting negative cycles in a graph
- Solving graph problems with constraints on intermediate vertices
- Network routing tables for all destinations

## Comparison with Other Algorithms

| Algorithm | Floyd-Warshall | Dijkstra (for all pairs) | Bellman-Ford (for all pairs) |
|-----------|---------------|-------------------------|----------------------------|
| Time Complexity | O(V³) | O(V² log V + VE) | O(V²E) |
| Handles negative edges | Yes | No | Yes |
| All-pairs calculation | Direct | Requires V executions | Requires V executions |
| Space efficiency | O(V²) | O(V²) | O(V²) |

## Example Problems

1. **Shortest Path Between All Cities**: Finding the shortest distance between every pair of cities in a road network
2. **Network Latency Analysis**: Calculating the minimum delay between all pairs of nodes in a network
3. **Transitive Closure**: Determining if there exists a path between every pair of vertices in a graph

## Common Pitfalls

- Inefficient for sparse graphs compared to running Dijkstra's algorithm for each vertex
- Cannot correctly handle graphs with negative cycles
- Memory intensive for large graphs due to O(V²) space requirement

## Optimizations

- For sparse graphs, using Johnson's algorithm may be more efficient
- Parallel processing of iterations for large graphs
- Using bit manipulation for problems involving transitive closure
