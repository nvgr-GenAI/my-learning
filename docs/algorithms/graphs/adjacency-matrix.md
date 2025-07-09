# Adjacency Matrix

## Overview

An adjacency matrix is a square matrix used to represent a finite graph. The elements of the matrix indicate whether pairs of vertices are adjacent or not in the graph.

## Definition

For a graph with $n$ vertices, an adjacency matrix is an $n \times n$ matrix where:

- $A[i][j] = 1$ if there is an edge from vertex $i$ to vertex $j$
- $A[i][j] = 0$ otherwise

For weighted graphs, the value can represent the weight or cost of the edge:

- $A[i][j] = w$ if there is an edge from vertex $i$ to vertex $j$ with weight $w$
- $A[i][j] = \infty$ (or a sentinel value) if there is no edge

## Implementation

```python
# Adjacency Matrix representation of a graph
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        # Initialize matrix with zeros
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]
    
    def add_edge(self, u, v, weight=1):
        # Add edge from u to v
        self.graph[u][v] = weight
        
        # For undirected graph, add edge from v to u as well
        # self.graph[v][u] = weight
    
    def remove_edge(self, u, v):
        self.graph[u][v] = 0
        # For undirected graph
        # self.graph[v][u] = 0
    
    def has_edge(self, u, v):
        return self.graph[u][v] != 0
    
    def get_edge_weight(self, u, v):
        return self.graph[u][v]
    
    def get_neighbors(self, u):
        neighbors = []
        for v in range(self.V):
            if self.graph[u][v] != 0:
                neighbors.append(v)
        return neighbors
    
    def print_graph(self):
        for i in range(self.V):
            for j in range(self.V):
                print(f"{self.graph[i][j]}", end=" ")
            print()

# Example usage
if __name__ == "__main__":
    g = Graph(5)
    g.add_edge(0, 1)
    g.add_edge(0, 4)
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(1, 4)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    
    print("Adjacency Matrix:")
    g.print_graph()
    
    print("Neighbors of vertex 1:", g.get_neighbors(1))
```

## Time and Space Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Add Edge  | O(1)           | -               |
| Remove Edge | O(1)         | -               |
| Check if edge exists | O(1) | -              |
| Get all edges | O(V²)      | -               |
| Get neighbors of a vertex | O(V) | -          |
| Storage   | -              | O(V²)           |

## Advantages

1. **Simple Implementation**: Easy to represent and understand
2. **Edge Weight Lookup**: O(1) time to check if an edge exists or get its weight
3. **Dense Graphs**: Efficient for dense graphs where |E| approaches |V|²
4. **Matrix Operations**: Can use linear algebra operations for graph algorithms

## Disadvantages

1. **Space Inefficiency**: Requires O(V²) space even for sparse graphs
2. **Vertex Addition**: Adding new vertices requires recreating the matrix
3. **Iteration Overhead**: Takes O(V) time to find all neighbors of a vertex

## When to Use

Use adjacency matrix when:

- The graph is dense (many edges)
- You need fast edge weight lookups
- The graph is small (fewer vertices)
- You need to perform matrix operations on the graph

## Common Applications

1. **Floyd-Warshall Algorithm**: For finding shortest paths between all pairs of vertices
2. **Graph Coloring Problems**: Where checking edge existence is frequent
3. **Network Flow Problems**: When matrix operations are beneficial
4. **Dynamic Programming on Graphs**: When the state depends on existence of specific edges

## Comparison with Other Representations

| Aspect | Adjacency Matrix | Adjacency List | Edge List |
|--------|-----------------|---------------|-----------|
| Space  | O(V²)           | O(V + E)      | O(E)      |
| Check Edge | O(1)        | O(degree(v))  | O(E)      |
| Find Neighbors | O(V)    | O(degree(v))  | O(E)      |
| Add Edge | O(1)          | O(1)          | O(1)      |
| Remove Edge | O(1)       | O(degree(v))  | O(E)      |
| Best For | Dense graphs  | Sparse graphs | Very sparse graphs |
