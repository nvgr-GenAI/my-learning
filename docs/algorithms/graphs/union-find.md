# Union-Find (Disjoint Set) Algorithm

## Overview

Union-Find, also known as Disjoint Set Union (DSU), is a data structure that keeps track of elements partitioned into a number of disjoint (non-overlapping) sets. It provides near-constant time operations for:

> **Note:** Union-Find can be viewed both as a graph algorithm and a tree-based data structure. For a more detailed explanation of DSU as a tree structure, see [Disjoint Set Union in Trees](../trees/dsu.md).

1. Finding which set an element belongs to
2. Merging two sets together

## Algorithm Components

### Key Operations

1. **Make-Set(x)**: Create a new set with single element x
2. **Find(x)**: Return the representative (or root) of the set containing element x
3. **Union(x, y)**: Merge the sets containing elements x and y

### Optimizations

- **Path Compression**: Flattens the structure of the tree during Find operations
- **Union by Rank/Size**: Always attach the smaller tree to the root of the larger tree

## Implementation

```python
class UnionFind:
    def __init__(self, n):
        """
        Initialize Union-Find data structure with n elements
        """
        self.parent = list(range(n))  # Each element is its own parent initially
        self.rank = [0] * n  # Rank (approximate height) of each subtree
        self.count = n  # Number of disjoint sets
        
    def find(self, x):
        """
        Find the representative (root) of the set containing element x
        with path compression
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
        
    def union(self, x, y):
        """
        Union the sets containing elements x and y
        using union by rank
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in the same set
            
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
            
        self.count -= 1  # Decrease count of disjoint sets
        return True
        
    def connected(self, x, y):
        """
        Check if elements x and y are in the same set
        """
        return self.find(x) == self.find(y)
```

## Time Complexity

With both path compression and union by rank:

- **Make-Set**: O(1)
- **Find**: O(α(n)) amortized, where α is the inverse Ackermann function (practically constant)
- **Union**: O(α(n)) amortized
- **Space Complexity**: O(n)

## Applications

1. **Kruskal's Minimum Spanning Tree Algorithm**: To detect cycles in undirected graphs
2. **Network Connectivity**: Determining if two nodes in a network are connected
3. **Image Processing**: Connected component labeling
4. **Dynamic Graph Connectivity**: Tracking connected components in a dynamic graph
5. **Percolation Analysis**: Studying connectivity in grid-based systems

## Example Problems

### Detecting Cycles in a Graph

```python
def has_cycle(edges, n):
    """
    Detect if an undirected graph has a cycle using Union-Find
    
    Args:
        edges: List of edges [(u, v), ...]
        n: Number of vertices
        
    Returns:
        True if the graph contains a cycle, False otherwise
    """
    uf = UnionFind(n)
    
    for u, v in edges:
        # If u and v are already connected, adding edge (u,v) creates a cycle
        if uf.connected(u, v):
            return True
        uf.union(u, v)
    
    return False
```

### Finding Number of Connected Components

```python
def count_components(edges, n):
    """
    Count the number of connected components in an undirected graph
    
    Args:
        edges: List of edges [(u, v), ...]
        n: Number of vertices
        
    Returns:
        Number of connected components
    """
    uf = UnionFind(n)
    
    for u, v in edges:
        uf.union(u, v)
    
    return uf.count
```

## Common Variations

1. **Union-Find with Path Halving**: An alternative to path compression
2. **Weighted Union-Find**: Uses size instead of rank for merging decision
3. **Union-Find with Parent Pointers**: Simpler implementation without ranks
4. **Persistent Union-Find**: Supports undo operations

## Advanced Applications

- **Dynamic Graph Algorithms**: Supporting edge insertions and connectivity queries
- **Lowest Common Ancestor**: Finding LCA in trees efficiently
- **Image Segmentation**: Region growing algorithms in computer vision
