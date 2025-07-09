# Edge List

## Overview

An edge list is a simple way to represent a graph as a list of its edges. It's one of the most compact representations for sparse graphs.

## Definition

An edge list is a collection of all edges in the graph. For an unweighted graph, each edge is represented as a pair of vertices (u, v). For a weighted graph, each edge is represented as a triplet (u, v, weight).

## Implementation

```python
# Edge List representation of a graph
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.edges = []  # List to store edges
    
    def add_edge(self, u, v, weight=None):
        # For unweighted graph
        if weight is None:
            self.edges.append((u, v))
        else:
            # For weighted graph
            self.edges.append((u, v, weight))
    
    def remove_edge(self, u, v):
        # For unweighted graph
        if (u, v) in self.edges:
            self.edges.remove((u, v))
        
        # For weighted graph
        # for i, (src, dst, w) in enumerate(self.edges):
        #     if src == u and dst == v:
        #         self.edges.pop(i)
        #         break
    
    def has_edge(self, u, v):
        # For unweighted graph
        return (u, v) in self.edges
        
        # For weighted graph
        # for src, dst, w in self.edges:
        #     if src == u and dst == v:
        #         return True
        # return False
    
    def get_edge_weight(self, u, v):
        # For weighted graph only
        for src, dst, weight in self.edges:
            if src == u and dst == v:
                return weight
        return None
    
    def get_neighbors(self, u):
        # Get all vertices adjacent to u
        neighbors = []
        for edge in self.edges:
            if edge[0] == u:
                neighbors.append(edge[1])
        return neighbors
    
    def print_graph(self):
        print("Edge List:")
        for edge in self.edges:
            print(edge)

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
    
    g.print_graph()
    
    print("Neighbors of vertex 1:", g.get_neighbors(1))
    print("Edge (1,2) exists:", g.has_edge(1, 2))
```

## Weighted Graph Implementation

For weighted graphs, we include the weight as the third element in each edge:

```python
# For weighted graph
class WeightedGraph:
    def __init__(self, vertices):
        self.V = vertices
        self.edges = []  # List to store weighted edges
    
    def add_edge(self, u, v, weight):
        self.edges.append((u, v, weight))
    
    # Other methods would need to be adapted for the (u, v, weight) format
```

## Time and Space Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Add Edge  | O(1)           | -               |
| Remove Edge | O(E)         | -               |
| Check if edge exists | O(E) | -              |
| Get all edges | O(1)       | -               |
| Get neighbors of a vertex | O(E) | -         |
| Storage   | -              | O(E)            |

## Advantages

1. **Space Efficiency**: Uses O(E) space, minimal for very sparse graphs
2. **Simplicity**: Simple to implement and understand
3. **Edge Operations**: Natural for algorithms that process edges (e.g., Kruskal's algorithm)
4. **Edge Addition**: Very easy to add new edges

## Disadvantages

1. **Edge Lookup**: O(E) time to check if an edge exists
2. **Finding Neighbors**: O(E) time to find all neighbors of a vertex
3. **Not Vertex-Centric**: Not optimized for operations centered on vertices

## When to Use

Use edge list when:

- The graph is extremely sparse
- Edge-based operations are dominant (e.g., sorting edges by weight)
- Simplicity is more important than query performance
- Memory is very constrained

## Common Applications

1. **Kruskal's Algorithm**: For finding minimum spanning tree (requires sorting edges)
2. **Edge-Based Graph Algorithms**: Where iterating over all edges is the primary operation
3. **Simple Graph Representations**: Where query performance is less important
4. **External Graph Processing**: When the graph doesn't fit in memory and is processed edge by edge

## Comparison with Other Representations

| Aspect | Adjacency Matrix | Adjacency List | Edge List |
|--------|-----------------|---------------|-----------|
| Space  | O(V²)           | O(V + E)      | O(E)      |
| Check Edge | O(1)        | O(degree(v))  | O(E)      |
| Find Neighbors | O(V)    | O(degree(v))  | O(E)      |
| Add Edge | O(1)          | O(1)          | O(1)      |
| Remove Edge | O(1)       | O(degree(v))  | O(E)      |
| Best For | Dense graphs  | Sparse graphs | Algorithms that process edges |
