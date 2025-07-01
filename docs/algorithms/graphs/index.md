# Graph Algorithms

## 📋 Overview

Graph algorithms are essential for solving problems involving relationships, networks, and connections. Graphs consist of vertices (nodes) connected by edges, making them perfect for modeling everything from social networks to road systems.

## 🔍 What You'll Learn

- **Graph Representations**: Adjacency list, adjacency matrix, edge list
- **Traversal Algorithms**: DFS, BFS and their applications
- **Shortest Path**: Dijkstra, Bellman-Ford, Floyd-Warshall
- **Advanced Topics**: Topological sort, minimum spanning trees, strongly connected components

## 📚 Section Contents

### 🎯 Fundamentals

- **[Graph Fundamentals](fundamentals.md)** - Representations, terminology, basic operations
- **[Graph Traversal Patterns](patterns.md)** - Common traversal techniques and when to use them

### 🌊 Traversal Algorithms

- **[Depth-First Search (DFS)](dfs.md)** - Explore deep before wide, find paths and cycles
- **[Breadth-First Search (BFS)](bfs.md)** - Explore level by level, shortest unweighted paths

### 🛣️ Shortest Path Algorithms

- **[Dijkstra's Algorithm](dijkstra.md)** - Single-source shortest path for weighted graphs
- **[Bellman-Ford Algorithm](bellman-ford.md)** - Handle negative weights, detect negative cycles
- **[Floyd-Warshall Algorithm](floyd-warshall.md)** - All-pairs shortest path

### 🔗 Graph Properties & Algorithms

- **[Topological Sort](topological-sort.md)** - Linear ordering for directed acyclic graphs
- **[Union-Find (Disjoint Set)](union-find.md)** - Detect cycles, connected components
- **[Minimum Spanning Tree](mst.md)** - Kruskal's and Prim's algorithms

### 💪 Practice by Difficulty

#### 🟢 Easy Problems
- **[Easy Graph Problems](easy-problems.md)**
  - Graph validation, simple traversals, basic connectivity
  - Island counting, path existence

#### 🟡 Medium Problems
- **[Medium Graph Problems](medium-problems.md)**
  - Shortest path variants, cycle detection
  - Graph coloring, course scheduling

#### 🔴 Hard Problems
- **[Hard Graph Problems](hard-problems.md)**
  - Advanced shortest path, complex graph problems
  - Network flow, strongly connected components

### 🎨 Advanced Topics

- **[Strongly Connected Components](scc.md)** - Tarjan's and Kosaraju's algorithms
- **[Network Flow](network-flow.md)** - Max flow, min cut problems
- **[Graph Matching](matching.md)** - Bipartite matching, maximum matching

## 🧠 Core Concepts

### Graph Representations

```python
# Adjacency List (most common)
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

# Adjacency Matrix
graph_matrix = [
    [0, 1, 1, 0, 0, 0],  # A -> B, C
    [0, 0, 0, 1, 1, 0],  # B -> D, E
    [0, 0, 0, 0, 0, 1],  # C -> F
    [0, 0, 0, 0, 0, 0],  # D -> none
    [0, 0, 0, 0, 0, 1],  # E -> F
    [0, 0, 0, 0, 0, 0]   # F -> none
]

# Edge List with weights
edges = [
    ('A', 'B', 4),
    ('A', 'C', 2),
    ('B', 'D', 5),
    ('C', 'F', 1)
]
```

### Basic Traversal Templates

```python
def dfs_recursive(graph, node, visited):
    """DFS using recursion"""
    visited.add(node)
    print(node)  # Process node
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)

def bfs_iterative(graph, start):
    """BFS using queue"""
    from collections import deque
    queue = deque([start])
    visited = {start}
    
    while queue:
        node = queue.popleft()
        print(node)  # Process node
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

## 📊 Algorithm Comparison

| Algorithm | Time Complexity | Space | Use Case |
|-----------|----------------|-------|----------|
| **DFS** | O(V + E) | O(V) | Paths, cycles, components |
| **BFS** | O(V + E) | O(V) | Shortest unweighted paths |
| **Dijkstra** | O((V + E) log V) | O(V) | Shortest weighted paths |
| **Bellman-Ford** | O(VE) | O(V) | Negative weights allowed |
| **Floyd-Warshall** | O(V³) | O(V²) | All pairs shortest path |
| **Topological Sort** | O(V + E) | O(V) | DAG ordering |

## 🔧 Problem-Solving Framework

### Step 1: Identify Graph Problem
- Nodes represent entities (cities, people, states)
- Edges represent relationships (connections, transitions)
- Need to find paths, connectivity, or optimization

### Step 2: Choose Representation
```python
# Dense graphs: Adjacency matrix O(V²) space
# Sparse graphs: Adjacency list O(V + E) space
# Weighted graphs: Include weights in representation
```

### Step 3: Select Algorithm
- **Unweighted shortest path**: BFS
- **Weighted shortest path**: Dijkstra (no negative weights)
- **Detect cycles**: DFS with recursion stack
- **Connected components**: DFS or Union-Find

### Step 4: Handle Edge Cases
- Empty graphs, single nodes
- Disconnected components
- Self-loops, multiple edges

## 🎨 Common Graph Patterns

### 1. **Island/Component Counting**
```python
def count_islands(grid):
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                dfs_mark_island(grid, i, j)
                count += 1
    return count
```

### 2. **Path Finding**
```python
def has_path(graph, start, end):
    visited = set()
    return dfs_path_exists(graph, start, end, visited)
```

### 3. **Cycle Detection**
```python
def has_cycle(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}
    
    def dfs(node):
        color[node] = GRAY
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:  # Back edge found
                return True
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False
    
    for node in graph:
        if color[node] == WHITE and dfs(node):
            return True
    return False
```

## 🚀 Getting Started

**New to graphs?** Start with:
1. **[Graph Fundamentals](fundamentals.md)** - Learn representations and terminology
2. **[DFS](dfs.md)** - Master depth-first traversal
3. **[Easy Problems](easy-problems.md)** - Practice basic patterns

**Have graph experience?** Jump to:
- **[Shortest Path Algorithms](dijkstra.md)** for weighted graphs
- **[Medium Problems](medium-problems.md)** for optimization challenges

**Advanced practitioner?** Challenge yourself with:
- **[Network Flow](network-flow.md)** for complex optimization
- **[Hard Problems](hard-problems.md)** for competitive programming

## 💡 Pro Tips

1. **Visualization**: Draw small examples to understand the problem
2. **State Tracking**: Use colors (white/gray/black) for complex traversals
3. **Edge Cases**: Always consider disconnected graphs and edge cases
4. **Space-Time Tradeoffs**: Adjacency list vs matrix based on graph density
5. **Problem Patterns**: Many problems reduce to standard graph algorithms

---

*Graphs are everywhere in computer science and real-world applications. Master these algorithms to solve complex connectivity and optimization problems!*
