# Minimum Spanning Tree (MST) Algorithms

## Overview

A Minimum Spanning Tree (MST) is a subset of edges of a connected, edge-weighted undirected graph that connects all vertices together without cycles while minimizing the total edge weight.

## Key MST Algorithms

### Kruskal's Algorithm

Kruskal's algorithm builds the MST by adding edges in order of increasing weight, skipping edges that would create cycles.

#### Algorithm Steps

1. Sort all edges in non-decreasing order of weight
2. Initialize MST as empty
3. For each edge (u,v) in sorted order:
   - If including edge (u,v) doesn't create a cycle, add it to the MST
   - Otherwise, discard it
4. Continue until MST has (V-1) edges

#### Implementation

```python
def kruskal_mst(graph, vertices):
    """
    Kruskal's algorithm for finding the MST
    
    Args:
        graph: List of edges in the format [(weight, u, v), ...]
        vertices: Number of vertices in the graph
        
    Returns:
        List of edges in the MST
    """
    # Sort edges by weight
    graph.sort()
    
    # Initialize Union-Find data structure
    uf = UnionFind(vertices)
    
    mst = []
    
    # Process edges in sorted order
    for weight, u, v in graph:
        # If including this edge doesn't create a cycle
        if not uf.connected(u, v):
            uf.union(u, v)
            mst.append((u, v, weight))
            
            # Stop when we have V-1 edges
            if len(mst) == vertices - 1:
                break
                
    return mst
```

### Prim's Algorithm

Prim's algorithm builds the MST by starting with a single vertex and repeatedly adding the lowest-weight edge that connects a vertex in the tree to a vertex outside the tree.

#### Algorithm Steps

1. Start with any vertex
2. Repeatedly add the minimum-weight edge that connects a vertex in the tree to a vertex outside the tree
3. Continue until all vertices are included

#### Implementation

```python
import heapq

def prim_mst(graph, start):
    """
    Prim's algorithm for finding the MST
    
    Args:
        graph: Adjacency list of the form {node: [(neighbor, weight), ...]}
        start: Starting vertex
        
    Returns:
        List of edges in the MST
    """
    # Priority queue: (weight, vertex, parent)
    pq = [(0, start, None)]
    
    visited = set()
    mst = []
    
    while pq and len(visited) < len(graph):
        weight, vertex, parent = heapq.heappop(pq)
        
        if vertex in visited:
            continue
            
        visited.add(vertex)
        
        if parent is not None:
            mst.append((parent, vertex, weight))
        
        # Add all edges from this vertex to the priority queue
        for neighbor, edge_weight in graph[vertex]:
            if neighbor not in visited:
                heapq.heappush(pq, (edge_weight, neighbor, vertex))
                
    return mst
```

## Time Complexity

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Kruskal's | O(E log E) or O(E log V) | O(V + E) |
| Prim's    | O(E log V) with binary heap | O(V + E) |

## Comparison

| Feature | Kruskal's | Prim's |
|---------|-----------|--------|
| Approach | Edge-based | Vertex-based |
| Best for | Sparse graphs | Dense graphs |
| Data Structure | Union-Find | Priority Queue |
| Implementation | Simpler | More complex |

## Applications

1. **Network Design**: Designing the minimum cost network infrastructure
2. **Cluster Analysis**: Hierarchical clustering algorithms
3. **Circuit Design**: Minimizing total wire length in circuit design
4. **Approximation Algorithms**: For NP-hard problems like Traveling Salesman Problem
5. **Image Segmentation**: Region-based segmentation in computer vision

## Variations and Special Cases

- **Maximum Spanning Tree**: Reverse the edge weights to find the maximum spanning tree
- **Bottleneck Spanning Tree**: Minimize the maximum edge weight in the tree
- **Directed Minimum Spanning Tree**: Finding a minimum spanning arborescence in directed graphs
- **Minimum Spanning Forest**: For disconnected graphs

## Common Interview Problems

1. **Network Repair**: Given a damaged network, find the minimum cost to repair it
2. **Water Supply Network**: Design an efficient water pipe system to minimize pipe costs
3. **Island Connection**: Connect a group of islands with bridges while minimizing total bridge length
