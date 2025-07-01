# Dijkstra's Algorithm

## Overview

Dijkstra's algorithm finds the shortest path from a source vertex to all other vertices in a weighted graph with non-negative edge weights.

## Algorithm Steps

1. Initialize distances to all vertices as infinite, except source (distance = 0)
2. Create a priority queue with all vertices
3. While priority queue is not empty:
   - Extract vertex with minimum distance
   - For each adjacent vertex, calculate tentative distance
   - If tentative distance is less than current distance, update it

## Implementation

### Using Priority Queue

```python
import heapq
from collections import defaultdict

def dijkstra(graph, start):
    # Initialize distances
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    
    # Priority queue: (distance, node)
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        # Check neighbors
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances
```

### With Path Reconstruction

```python
def dijkstra_with_path(graph, start, end):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}
    
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_node == end:
            break
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))
    
    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    return distances[end], path if path[0] == start else []
```

## Example Usage

```python
# Graph representation: {node: [(neighbor, weight), ...]}
graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('C', 1), ('D', 5)],
    'C': [('D', 8), ('E', 10)],
    'D': [('E', 2)],
    'E': []
}

distances = dijkstra(graph, 'A')
print(distances)  # {'A': 0, 'B': 4, 'C': 2, 'D': 9, 'E': 11}

distance, path = dijkstra_with_path(graph, 'A', 'E')
print(f"Distance: {distance}, Path: {path}")  # Distance: 11, Path: ['A', 'C', 'D', 'E']
```

## Time and Space Complexity

- **Time Complexity**: O((V + E) log V) with binary heap
- **Space Complexity**: O(V) for distances and priority queue

## Applications

1. **GPS Navigation Systems**
2. **Network Routing Protocols**
3. **Social Network Analysis**
4. **Game AI Pathfinding**
5. **Flight Connection Systems**

## Variations

### A* Algorithm

Extension of Dijkstra's with heuristic function for faster pathfinding.

### Floyd-Warshall

Finds shortest paths between all pairs of vertices.

## Common Problems

1. **Network Delay Time**
2. **Cheapest Flights Within K Stops**
3. **Path with Maximum Probability**
4. **Minimum Cost to Make Array Equal**

## Practice Problems

- [ ] Network Delay Time
- [ ] Path With Minimum Effort
- [ ] Swim in Rising Water
- [ ] Cheapest Flights Within K Stops
