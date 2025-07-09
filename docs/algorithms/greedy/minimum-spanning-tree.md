# Minimum Spanning Tree Algorithms 🌳

Find the minimum spanning tree of a weighted graph using greedy algorithms.

## 🎯 Problem Statement

Given a connected, weighted graph, find a spanning tree with minimum total edge weight.

**Input**: Weighted graph G = (V, E)
**Output**: Minimum spanning tree T with minimum total weight

## 🧠 Algorithm Approaches

### 1. Kruskal's Algorithm
- **Strategy**: Sort edges by weight, add edges that don't create cycles
- **Data Structure**: Union-Find (Disjoint Set Union)

### 2. Prim's Algorithm
- **Strategy**: Start from vertex, grow tree by adding minimum weight edges
- **Data Structure**: Priority Queue (Min-Heap)

## 📝 Implementation

### Kruskal's Algorithm

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        self.components -= 1
        return True
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)

def kruskal_mst(vertices, edges):
    """
    Kruskal's algorithm for Minimum Spanning Tree
    
    Args:
        vertices: Number of vertices
        edges: List of (weight, u, v) tuples
        
    Returns:
        List of edges in MST and total weight
    """
    # Sort edges by weight
    edges.sort(key=lambda x: x[0])
    
    uf = UnionFind(vertices)
    mst_edges = []
    total_weight = 0
    
    for weight, u, v in edges:
        if uf.union(u, v):
            mst_edges.append((weight, u, v))
            total_weight += weight
            
            # MST has exactly n-1 edges
            if len(mst_edges) == vertices - 1:
                break
    
    return mst_edges, total_weight

# Example usage
if __name__ == "__main__":
    # Graph representation: (weight, vertex1, vertex2)
    edges = [
        (10, 0, 1), (6, 0, 2), (5, 0, 3),
        (15, 1, 3), (4, 2, 3)
    ]
    
    mst_edges, total_weight = kruskal_mst(4, edges)
    print("Kruskal's MST:")
    for weight, u, v in mst_edges:
        print(f"Edge ({u}, {v}): {weight}")
    print(f"Total weight: {total_weight}")
```

### Prim's Algorithm

```python
import heapq
from collections import defaultdict

def prim_mst(graph, start=0):
    """
    Prim's algorithm for Minimum Spanning Tree
    
    Args:
        graph: Adjacency list representation {vertex: [(weight, neighbor), ...]}
        start: Starting vertex
        
    Returns:
        List of edges in MST and total weight
    """
    mst_edges = []
    total_weight = 0
    visited = set()
    
    # Priority queue: (weight, vertex, parent)
    pq = [(0, start, -1)]
    
    while pq:
        weight, vertex, parent = heapq.heappop(pq)
        
        if vertex in visited:
            continue
        
        visited.add(vertex)
        
        if parent != -1:
            mst_edges.append((weight, parent, vertex))
            total_weight += weight
        
        # Add all neighbors to priority queue
        for edge_weight, neighbor in graph[vertex]:
            if neighbor not in visited:
                heapq.heappush(pq, (edge_weight, neighbor, vertex))
    
    return mst_edges, total_weight

# Graph representation for Prim's
def create_graph_from_edges(edges):
    """Convert edge list to adjacency list"""
    graph = defaultdict(list)
    
    for weight, u, v in edges:
        graph[u].append((weight, v))
        graph[v].append((weight, u))
    
    return graph

# Example usage
if __name__ == "__main__":
    edges = [
        (10, 0, 1), (6, 0, 2), (5, 0, 3),
        (15, 1, 3), (4, 2, 3)
    ]
    
    graph = create_graph_from_edges(edges)
    mst_edges, total_weight = prim_mst(graph)
    
    print("Prim's MST:")
    for weight, u, v in mst_edges:
        print(f"Edge ({u}, {v}): {weight}")
    print(f"Total weight: {total_weight}")
```

### Advanced Implementation with Visualization

```python
import matplotlib.pyplot as plt
import networkx as nx

class MST:
    def __init__(self, vertices):
        self.vertices = vertices
        self.edges = []
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v, weight):
        self.edges.append((weight, u, v))
        self.graph[u].append((weight, v))
        self.graph[v].append((weight, u))
    
    def kruskal(self):
        """Kruskal's algorithm implementation"""
        self.edges.sort()
        uf = UnionFind(self.vertices)
        mst = []
        total_weight = 0
        
        for weight, u, v in self.edges:
            if uf.union(u, v):
                mst.append((u, v, weight))
                total_weight += weight
                
                if len(mst) == self.vertices - 1:
                    break
        
        return mst, total_weight
    
    def prim(self, start=0):
        """Prim's algorithm implementation"""
        mst = []
        total_weight = 0
        visited = set([start])
        
        # Initialize priority queue with edges from start vertex
        pq = [(weight, start, neighbor) for weight, neighbor in self.graph[start]]
        heapq.heapify(pq)
        
        while pq and len(mst) < self.vertices - 1:
            weight, u, v = heapq.heappop(pq)
            
            if v in visited:
                continue
            
            visited.add(v)
            mst.append((u, v, weight))
            total_weight += weight
            
            # Add new edges from v
            for edge_weight, neighbor in self.graph[v]:
                if neighbor not in visited:
                    heapq.heappush(pq, (edge_weight, v, neighbor))
        
        return mst, total_weight
    
    def visualize_mst(self, mst_edges, title="Minimum Spanning Tree"):
        """Visualize the MST using NetworkX and Matplotlib"""
        G = nx.Graph()
        
        # Add all edges
        for weight, u, v in self.edges:
            G.add_edge(u, v, weight=weight)
        
        # Create MST graph
        mst_graph = nx.Graph()
        for u, v, weight in mst_edges:
            mst_graph.add_edge(u, v, weight=weight)
        
        plt.figure(figsize=(12, 5))
        
        # Plot original graph
        plt.subplot(1, 2, 1)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=10, font_weight='bold')
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
        plt.title("Original Graph")
        
        # Plot MST
        plt.subplot(1, 2, 2)
        nx.draw(mst_graph, pos, with_labels=True, node_color='lightgreen', 
                node_size=500, font_size=10, font_weight='bold', 
                edge_color='red', width=2)
        
        # Draw MST edge labels
        mst_edge_labels = nx.get_edge_attributes(mst_graph, 'weight')
        nx.draw_networkx_edge_labels(mst_graph, pos, mst_edge_labels)
        plt.title(title)
        
        plt.tight_layout()
        plt.show()

# Example usage with visualization
if __name__ == "__main__":
    # Create example graph
    mst = MST(6)
    mst.add_edge(0, 1, 4)
    mst.add_edge(0, 2, 3)
    mst.add_edge(1, 2, 1)
    mst.add_edge(1, 3, 2)
    mst.add_edge(2, 3, 4)
    mst.add_edge(3, 4, 2)
    mst.add_edge(4, 5, 6)
    
    # Run Kruskal's algorithm
    kruskal_mst, kruskal_weight = mst.kruskal()
    print("Kruskal's MST:")
    for u, v, weight in kruskal_mst:
        print(f"Edge ({u}, {v}): {weight}")
    print(f"Total weight: {kruskal_weight}")
    
    # Run Prim's algorithm
    prim_mst, prim_weight = mst.prim()
    print("\\nPrim's MST:")
    for u, v, weight in prim_mst:
        print(f"Edge ({u}, {v}): {weight}")
    print(f"Total weight: {prim_weight}")
    
    # Visualize results
    mst.visualize_mst(kruskal_mst, "Kruskal's MST")
    mst.visualize_mst(prim_mst, "Prim's MST")
```

## ⚡ Time Complexity Analysis

### Kruskal's Algorithm
- **Time Complexity**: O(E log E) or O(E log V)
  - Sorting edges: O(E log E)
  - Union-Find operations: O(E α(V)) ≈ O(E)
- **Space Complexity**: O(V) for Union-Find structure

### Prim's Algorithm
- **Time Complexity**: O(E log V) with binary heap
  - Each vertex: O(log V) to extract min
  - Each edge: O(log V) to update priority
- **Space Complexity**: O(V) for priority queue

### Comparison
| Algorithm | Time Complexity | Space Complexity | Best For |
|-----------|----------------|------------------|----------|
| Kruskal's | O(E log E) | O(V) | Sparse graphs |
| Prim's | O(E log V) | O(V) | Dense graphs |

## 🔄 Step-by-Step Example

```text
Graph: 6 vertices, edges with weights
(0,1,4), (0,2,3), (1,2,1), (1,3,2), (2,3,4), (3,4,2), (4,5,6)

Kruskal's Algorithm:
1. Sort edges: (1,2,1), (1,3,2), (3,4,2), (0,2,3), (0,1,4), (2,3,4), (4,5,6)
2. Add (1,2,1) - components: {0}, {1,2}, {3}, {4}, {5}
3. Add (1,3,2) - components: {0}, {1,2,3}, {4}, {5}
4. Add (3,4,2) - components: {0}, {1,2,3,4}, {5}
5. Add (0,2,3) - components: {0,1,2,3,4}, {5}
6. Add (4,5,6) - components: {0,1,2,3,4,5}
7. MST: {(1,2,1), (1,3,2), (3,4,2), (0,2,3), (4,5,6)} - Weight: 14

Prim's Algorithm (starting from vertex 0):
1. Start with vertex 0, visited = {0}
2. Add edge (0,2,3) - visited = {0,2}
3. Add edge (1,2,1) - visited = {0,1,2}
4. Add edge (1,3,2) - visited = {0,1,2,3}
5. Add edge (3,4,2) - visited = {0,1,2,3,4}
6. Add edge (4,5,6) - visited = {0,1,2,3,4,5}
7. MST: {(0,2,3), (1,2,1), (1,3,2), (3,4,2), (4,5,6)} - Weight: 14
```

## 🎯 Key Insights

1. **Greedy Property**: Both algorithms make locally optimal choices
2. **Cut Property**: MST algorithms respect the cut property of graphs
3. **Unique MST**: MST is unique if all edge weights are distinct
4. **Cycle Property**: Adding any non-MST edge creates exactly one cycle

## 💡 Applications

- **Network Design**: Minimum cost to connect all locations
- **Clustering**: Hierarchical clustering algorithms
- **Approximation Algorithms**: TSP, Steiner tree approximations
- **Circuit Design**: Minimizing wire length in VLSI
- **Social Networks**: Finding most influential connections

## 🚀 Advanced Variants

1. **Borůvka's Algorithm**: Parallel-friendly MST algorithm
2. **Reverse-Delete Algorithm**: Start with all edges, remove maximum weight edges
3. **Degree-Constrained MST**: MST with maximum degree constraints
4. **Dynamic MST**: Maintain MST with edge insertions/deletions

---

*Minimum spanning tree algorithms demonstrate the power of greedy approaches in graph optimization problems.*
