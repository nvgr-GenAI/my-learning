# Graph Algorithms - Easy Problems

## 🎯 Learning Objectives

Master basic graph operations and simple traversal patterns:

- Graph representation and construction
- Basic DFS and BFS implementations
- Connectivity and path existence
- Simple graph properties

---

## Problem 1: Number of Islands

**Difficulty**: 🟢 Easy  
**Pattern**: DFS/BFS on Grid  
**Time**: O(m×n), **Space**: O(m×n)

### Problem Statement

Given a 2D binary grid representing a map of '1's (land) and '0's (water), count the number of islands.

```
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"], 
  ["0","0","0","0","0"]
]
Output: 1
```

### Solution - DFS Approach

```python
def numIslands(grid):
    """
    Use DFS to explore each connected component (island)
    Mark visited cells to avoid recounting
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    islands = 0
    
    def dfs(r, c):
        # Base cases: out of bounds or water
        if (r < 0 or r >= rows or 
            c < 0 or c >= cols or 
            grid[r][c] == '0'):
            return
        
        # Mark as visited (sink the island)
        grid[r][c] = '0'
        
        # Explore all 4 directions
        dfs(r + 1, c)  # down
        dfs(r - 1, c)  # up
        dfs(r, c + 1)  # right
        dfs(r, c - 1)  # left
    
    # Check each cell
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                islands += 1
                dfs(r, c)  # Sink the entire island
    
    return islands

# Test
grid = [
    ["1","1","1","1","0"],
    ["1","1","0","1","0"],
    ["1","1","0","0","0"],
    ["0","0","0","0","0"]
]
print(numIslands(grid))  # Output: 1
```

### Solution - BFS Approach

```python
from collections import deque

def numIslandsBFS(grid):
    """
    Use BFS to explore each connected component
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    islands = 0
    
    def bfs(start_r, start_c):
        queue = deque([(start_r, start_c)])
        grid[start_r][start_c] = '0'  # Mark as visited
        
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        
        while queue:
            r, c = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and 
                    grid[nr][nc] == '1'):
                    grid[nr][nc] = '0'  # Mark as visited
                    queue.append((nr, nc))
    
    # Check each cell
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                islands += 1
                bfs(r, c)
    
    return islands
```

### 🔍 Key Insights
- **Grid as Graph**: Each cell is a node, adjacent cells are connected
- **Connected Components**: Each island is a connected component
- **Marking Visited**: Prevent infinite loops and double counting
- **DFS vs BFS**: Both work, DFS is more memory efficient for this problem

---

## Problem 2: Valid Graph Tree

**Difficulty**: 🟢 Easy  
**Pattern**: Graph Validation  
**Time**: O(V+E), **Space**: O(V+E)

### Problem Statement

Given `n` nodes labeled from `0` to `n-1` and a list of undirected edges, check if these edges form a valid tree.

```
Input: n = 5, edges = [[0,1],[0,2],[0,3],[1,4]]
Output: true

Input: n = 5, edges = [[0,1],[1,2],[2,3],[1,3],[1,4]]
Output: false (has cycle)
```

### Solution

```python
def validTree(n, edges):
    """
    A valid tree must have:
    1. Exactly n-1 edges
    2. All nodes connected (no isolated components)
    3. No cycles
    """
    # Tree with n nodes must have exactly n-1 edges
    if len(edges) != n - 1:
        return False
    
    # Build adjacency list
    graph = {i: [] for i in range(n)}
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    
    # DFS to check connectivity and cycles
    visited = set()
    
    def dfs(node, parent):
        if node in visited:
            return False  # Cycle detected
        
        visited.add(node)
        
        for neighbor in graph[node]:
            if neighbor != parent:  # Don't go back to parent
                if not dfs(neighbor, node):
                    return False
        
        return True
    
    # Start DFS from node 0
    if not dfs(0, -1):
        return False
    
    # Check if all nodes are visited (connected)
    return len(visited) == n

# Test
print(validTree(5, [[0,1],[0,2],[0,3],[1,4]]))  # True
print(validTree(5, [[0,1],[1,2],[2,3],[1,3],[1,4]]))  # False
```

### 🔍 Key Insights
- **Tree Properties**: n nodes, n-1 edges, connected, acyclic
- **Cycle Detection**: If we visit a node we've seen before (except parent), there's a cycle
- **Connectivity**: All nodes must be reachable from any starting node

---

## Problem 3: Find if Path Exists

**Difficulty**: 🟢 Easy  
**Pattern**: Path Finding  
**Time**: O(V+E), **Space**: O(V+E)

### Problem Statement

Given edges of a graph and two nodes `source` and `destination`, determine if there's a valid path from source to destination.

```
Input: n = 3, edges = [[0,1],[1,2],[2,0]], source = 0, destination = 2
Output: true
```

### Solution - DFS

```python
def validPath(n, edges, source, destination):
    """
    Use DFS to find if path exists from source to destination
    """
    if source == destination:
        return True
    
    # Build adjacency list
    graph = {i: [] for i in range(n)}
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    
    visited = set()
    
    def dfs(node):
        if node == destination:
            return True
        
        visited.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
        
        return False
    
    return dfs(source)

# Test
edges = [[0,1],[1,2],[2,0]]
print(validPath(3, edges, 0, 2))  # True
```

### Solution - BFS

```python
from collections import deque

def validPathBFS(n, edges, source, destination):
    """
    Use BFS to find if path exists
    """
    if source == destination:
        return True
    
    # Build adjacency list
    graph = {i: [] for i in range(n)}
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    
    visited = set()
    queue = deque([source])
    visited.add(source)
    
    while queue:
        node = queue.popleft()
        
        if node == destination:
            return True
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return False
```

### Solution - Union Find

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)

def validPathUnionFind(n, edges, source, destination):
    """
    Use Union-Find to check connectivity
    """
    uf = UnionFind(n)
    
    # Union all connected components
    for a, b in edges:
        uf.union(a, b)
    
    return uf.connected(source, destination)
```

---

## Problem 4: Clone Graph (Simple)

**Difficulty**: 🟢 Easy  
**Pattern**: Graph Traversal + Construction  
**Time**: O(V+E), **Space**: O(V)

### Problem Statement

Clone an undirected connected graph. Each node contains a value and a list of neighbors.

### Solution

```python
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def cloneGraph(node):
    """
    Use DFS with hash map to clone graph
    """
    if not node:
        return None
    
    # Map original node to cloned node
    cloned = {}
    
    def dfs(original):
        if original in cloned:
            return cloned[original]
        
        # Create clone of current node
        clone = Node(original.val)
        cloned[original] = clone
        
        # Clone all neighbors
        for neighbor in original.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)
```

---

## Problem 5: All Paths From Source to Target

**Difficulty**: 🟢 Easy  
**Pattern**: Path Enumeration  
**Time**: O(2^V × V), **Space**: O(2^V × V)

### Problem Statement

Find all possible paths from node 0 to node n-1 in a directed acyclic graph.

```
Input: graph = [[1,2],[3],[3],[]]
Output: [[0,1,3],[0,2,3]]
```

### Solution

```python
def allPathsSourceTarget(graph):
    """
    Use DFS to find all paths from 0 to n-1
    """
    target = len(graph) - 1
    paths = []
    
    def dfs(node, path):
        if node == target:
            paths.append(path[:])  # Add copy of current path
            return
        
        for neighbor in graph[node]:
            path.append(neighbor)
            dfs(neighbor, path)
            path.pop()  # Backtrack
    
    dfs(0, [0])
    return paths

# Test
graph = [[1,2],[3],[3],[]]
print(allPathsSourceTarget(graph))  # [[0,1,3],[0,2,3]]
```

---

## 📝 Summary

### Key Patterns Learned

1. **Grid as Graph** - Treat 2D grids as graphs with implicit edges
2. **Graph Validation** - Check tree properties and connectivity
3. **Path Finding** - Use DFS/BFS to find if paths exist
4. **Graph Cloning** - Traverse and construct simultaneously
5. **Path Enumeration** - Find all possible paths with backtracking

### Essential Techniques

- **DFS**: Good for path finding, connected components
- **BFS**: Good for shortest unweighted paths, level exploration
- **Union-Find**: Efficient for connectivity queries
- **Backtracking**: Required for enumerating all solutions

### Time Complexities

- **Graph Traversal**: O(V + E) where V = vertices, E = edges
- **Grid Problems**: O(m × n) where m, n are grid dimensions
- **Path Enumeration**: Exponential in worst case

---

Ready for more challenging problems? Move on to **[Medium Graph Problems](medium-problems.md)** to tackle shortest paths, cycle detection, and graph coloring!

### 📚 What's Next

- **[Graph Fundamentals](fundamentals.md)** - Review core concepts
- **[DFS Deep Dive](dfs.md)** - Master depth-first search
- **[BFS Deep Dive](bfs.md)** - Master breadth-first search
- **[Medium Problems](medium-problems.md)** - Take on harder challenges
