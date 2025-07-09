# Adjacency List

## Overview

An adjacency list is a collection of unordered lists used to represent a finite graph. Each list describes the neighbors of a particular vertex in the graph.

## Definition

For a graph with vertices V, an adjacency list consists of |V| lists, one for each vertex. For each vertex v, the adjacency list contains all the vertices that are adjacent to v (i.e., there's an edge from v to those vertices).

## Implementation

```python
# Adjacency List representation of a graph
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        # Initialize empty adjacency list for each vertex
        self.graph = [[] for _ in range(vertices)]
    
    def add_edge(self, u, v, weight=None):
        # For unweighted graph
        if weight is None:
            self.graph[u].append(v)
            # For undirected graph, uncomment below
            # self.graph[v].append(u)
        else:
            # For weighted graph
            self.graph[u].append((v, weight))
            # For undirected graph, uncomment below
            # self.graph[v].append((u, weight))
    
    def remove_edge(self, u, v):
        # For unweighted graph
        if v in self.graph[u]:
            self.graph[u].remove(v)
            # For undirected graph, uncomment below
            # self.graph[v].remove(u)
        
        # For weighted graph
        # for i, (vertex, weight) in enumerate(self.graph[u]):
        #     if vertex == v:
        #         self.graph[u].pop(i)
        #         break
        # For undirected graph, do the same for v
    
    def has_edge(self, u, v):
        # For unweighted graph
        return v in self.graph[u]
        
        # For weighted graph
        # for vertex, weight in self.graph[u]:
        #     if vertex == v:
        #         return True
        # return False
    
    def get_edge_weight(self, u, v):
        # For weighted graph only
        for vertex, weight in self.graph[u]:
            if vertex == v:
                return weight
        return None
    
    def get_neighbors(self, u):
        # For unweighted graph
        return self.graph[u]
        
        # For weighted graph
        # return [v for v, w in self.graph[u]]
    
    def print_graph(self):
        for i in range(self.V):
            print(f"Adjacency list of vertex {i}:")
            print(f"head", end="")
            for j in self.graph[i]:
                print(f" -> {j}", end="")
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
    
    print("Adjacency List:")
    g.print_graph()
    
    print("Neighbors of vertex 1:", g.get_neighbors(1))
```

## Weighted Graph Implementation

For weighted graphs, we can store pairs of (vertex, weight) in the lists:

```python
# For weighted graph
class WeightedGraph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[] for _ in range(vertices)]
    
    def add_edge(self, u, v, weight):
        self.graph[u].append((v, weight))
        # For undirected graph
        # self.graph[v].append((u, weight))
    
    # Other methods would need to be adapted for the (vertex, weight) tuple format
```

## Time and Space Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Add Edge  | O(1)           | -               |
| Remove Edge | O(degree(v)) | -               |
| Check if edge exists | O(degree(v)) | -      |
| Get all edges | O(V + E)   | -               |
| Get neighbors of a vertex | O(1) to access list, O(degree(v)) to copy | - |
| Storage   | -              | O(V + E)        |

## Advantages

1. **Space Efficiency**: Uses O(V + E) space, which is efficient for sparse graphs
2. **Vertex Neighbors**: Fast to iterate through all neighbors of a vertex
3. **Vertex Addition**: Easy to add new vertices
4. **Memory Usage**: Memory-friendly for large, sparse graphs

## Disadvantages

1. **Edge Lookup**: O(degree(v)) time to check if an edge exists or get its weight
2. **Asymmetric Space Usage**: Memory usage varies by vertex degrees

## When to Use

Use adjacency list when:

- The graph is sparse (|E| << |V|²)
- Memory efficiency is important
- You frequently need to iterate through neighbors of vertices
- The graph is large with many vertices

## Common Applications

1. **Breadth-First Search (BFS)**: Efficiently explore neighbors
2. **Depth-First Search (DFS)**: Traverse graph with minimal memory overhead
3. **Dijkstra's Algorithm**: Find shortest paths from a source vertex
4. **Prim's Algorithm**: Find minimum spanning tree

## Comparison with Other Representations

| Aspect | Adjacency Matrix | Adjacency List | Edge List |
|--------|-----------------|---------------|-----------|
| Space  | O(V²)           | O(V + E)      | O(E)      |
| Check Edge | O(1)        | O(degree(v))  | O(E)      |
| Find Neighbors | O(V)    | O(1) to access, O(degree(v)) to iterate | O(E) |
| Add Edge | O(1)          | O(1)          | O(1)      |
| Remove Edge | O(1)       | O(degree(v))  | O(E)      |
| Best For | Dense graphs  | Most general purpose uses | Very sparse graphs |
