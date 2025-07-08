# Bellman-Ford Algorithm

## Overview

The Bellman-Ford algorithm finds the shortest path from a source vertex to all other vertices in a weighted graph. Unlike Dijkstra's algorithm, it can handle graphs with negative weight edges and detect negative cycles.

## Algorithm Steps

1. Initialize distances to all vertices as infinite, except source (distance = 0)
2. Relax all edges |V|-1 times, where |V| is the number of vertices
3. Check for negative-weight cycles
4. If no negative cycles exist, return the shortest path distances

## Implementation

```python
def bellman_ford(graph, source):
    """
    Implementation of the Bellman-Ford algorithm
    
    Args:
        graph: Dictionary containing edges and weights {(u,v): weight}
        source: Starting vertex
        
    Returns:
        Dictionary of shortest distances, or None if negative cycle exists
    """
    # Get list of vertices
    vertices = set()
    for u, v in graph:
        vertices.add(u)
        vertices.add(v)
    
    # Initialize distances
    distances = {vertex: float('infinity') for vertex in vertices}
    distances[source] = 0
    
    # Relax edges |V|-1 times
    for _ in range(len(vertices) - 1):
        for (u, v), weight in graph.items():
            if distances[u] != float('infinity') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
    
    # Check for negative cycles
    for (u, v), weight in graph.items():
        if distances[u] != float('infinity') and distances[u] + weight < distances[v]:
            # Negative cycle detected
            return None
    
    return distances
```

## Time Complexity

- **Time Complexity**: O(|V| × |E|), where |V| is the number of vertices and |E| is the number of edges
- **Space Complexity**: O(|V|)

## Applications

- Finding shortest paths in graphs with negative edge weights
- Detecting negative cycles in a graph
- Network routing protocols (like RIP - Routing Information Protocol)
- Arbitrage detection in currency exchange

## Comparison with Dijkstra's Algorithm

| Feature | Bellman-Ford | Dijkstra |
|---------|-------------|----------|
| Handles negative edges | Yes | No |
| Detects negative cycles | Yes | No |
| Time Complexity | O(VE) | O(V log V + E) with priority queue |
| Use case | When negative edges exist | When all edges are positive |

## Example Problems

1. **Currency Arbitrage Detection**: Find if there's a way to exchange currencies to make a profit due to inconsistent exchange rates
2. **Network Routing**: Determine the shortest path for data packets to travel
3. **Traffic Flow Optimization**: Find the quickest path considering road closures (negative edges)

## Common Pitfalls

- Inefficient for large graphs compared to Dijkstra's algorithm when all edges are positive
- May not terminate if applied to a graph with a negative cycle reachable from the source
- Results may be misleading if negative cycles exist but are not properly detected

## Optimizations

- Early termination if no relaxation occurs in an iteration
- SPFA (Shortest Path Faster Algorithm) - an optimization that uses a queue to process only vertices that might lead to a relaxation
