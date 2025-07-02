# Graph Algorithms - Medium Problems

## 🎯 Learning Objectives

Master intermediate graph algorithms and patterns:

- Shortest path algorithms (Dijkstra, BFS)
- Cycle detection in directed/undirected graphs
- Topological sorting
- Graph coloring and bipartite checking
- Advanced DFS/BFS applications

---

## Problem 1: Course Schedule

**Difficulty**: 🟡 Medium  
**Pattern**: Topological Sort / Cycle Detection  
**Time**: O(V+E), **Space**: O(V+E)

### Problem Description

There are `numCourses` courses labeled from `0` to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [ai, bi]` indicates that you must take course `bi` first if you want to take course `ai`.

Return `true` if you can finish all courses, otherwise return `false`.

**Example:**
```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: Take course 0, then course 1.

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: Circular dependency.
```

### Solution - DFS Cycle Detection

```python
def canFinish(numCourses, prerequisites):
    """
    Detect cycle in directed graph using DFS
    - WHITE: not visited
    - GRAY: visiting (in current path)
    - BLACK: visited (processed)
    """
    # Build adjacency list
    graph = {i: [] for i in range(numCourses)}
    for course, prereq in prerequisites:
        graph[course].append(prereq)
    
    # 0: WHITE, 1: GRAY, 2: BLACK
    colors = [0] * numCourses
    
    def dfs(node):
        if colors[node] == 1:  # GRAY - cycle detected
            return False
        if colors[node] == 2:  # BLACK - already processed
            return True
        
        colors[node] = 1  # Mark as GRAY
        
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        
        colors[node] = 2  # Mark as BLACK
        return True
    
    for i in range(numCourses):
        if colors[i] == 0:  # WHITE
            if not dfs(i):
                return False
    
    return True
```

### Solution - Kahn's Algorithm (BFS)

```python
from collections import deque, defaultdict

def canFinishBFS(numCourses, prerequisites):
    """
    Topological sort using Kahn's algorithm
    """
    # Build graph and in-degree count
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # Queue for nodes with no incoming edges
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    processed = 0
    
    while queue:
        node = queue.popleft()
        processed += 1
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return processed == numCourses
```

---

## Problem 2: Shortest Path in Binary Matrix

**Difficulty**: 🟡 Medium  
**Pattern**: BFS Shortest Path  
**Time**: O(n²), **Space**: O(n²)

### Problem Description

Given an `n x n` binary matrix `grid`, return the length of the shortest clear path from top-left to bottom-right. If no such path exists, return `-1`.

A clear path is a path from top-left to bottom-right such that all visited cells are `0`.

### Solution

```python
from collections import deque

def shortestPathBinaryMatrix(grid):
    """
    Use BFS to find shortest path in unweighted grid
    """
    n = len(grid)
    
    # Check if start or end is blocked
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1
    
    # Special case: single cell
    if n == 1:
        return 1
    
    # 8 directions (including diagonals)
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    queue = deque([(0, 0, 1)])  # (row, col, distance)
    visited = set([(0, 0)])
    
    while queue:
        row, col, dist = queue.popleft()
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds and if cell is clear
            if (0 <= new_row < n and 0 <= new_col < n and 
                grid[new_row][new_col] == 0 and 
                (new_row, new_col) not in visited):
                
                # Check if we reached the destination
                if new_row == n-1 and new_col == n-1:
                    return dist + 1
                
                visited.add((new_row, new_col))
                queue.append((new_row, new_col, dist + 1))
    
    return -1

# Test
grid = [[0,0,0],[1,1,0],[1,1,0]]
print(shortestPathBinaryMatrix(grid))  # 4
```

---

## Problem 3: Network Delay Time

**Difficulty**: 🟡 Medium  
**Pattern**: Dijkstra's Algorithm  
**Time**: O(E log V), **Space**: O(V+E)

### Problem Description

You are given a network of `n` nodes, labeled from `1` to `n`. You are also given `times`, a list of travel times as directed edges `times[i] = (ui, vi, wi)`, where `ui` is the source node, `vi` is the target node, and `wi` is the time it takes for a signal to travel from source to target.

Return the minimum time it takes for all `n` nodes to receive the signal. If it is impossible for all nodes to receive the signal, return `-1`.

### Solution

```python
import heapq
from collections import defaultdict

def networkDelayTime(times, n, k):
    """
    Use Dijkstra's algorithm to find shortest paths from source k
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    
    # Dijkstra's algorithm
    distances = {}
    heap = [(0, k)]  # (distance, node)
    
    while heap:
        dist, node = heapq.heappop(heap)
        
        if node in distances:
            continue
        
        distances[node] = dist
        
        for neighbor, weight in graph[node]:
            if neighbor not in distances:
                heapq.heappush(heap, (dist + weight, neighbor))
    
    # Check if all nodes are reachable
    if len(distances) != n:
        return -1
    
    return max(distances.values())

# Test
times = [[2,1,1],[2,3,1],[3,4,1]]
print(networkDelayTime(times, 4, 2))  # 2
```

---

## Problem 4: Is Graph Bipartite?

**Difficulty**: 🟡 Medium  
**Pattern**: Graph Coloring / BFS  
**Time**: O(V+E), **Space**: O(V)

### Problem Description

Given an undirected `graph`, return `true` if and only if it is bipartite.

A graph is bipartite if we can color its nodes using two colors such that no two adjacent nodes have the same color.

### Solution - BFS Coloring

```python
from collections import deque

def isBipartite(graph):
    """
    Use BFS to color graph with two colors
    If we can color without conflicts, it's bipartite
    """
    n = len(graph)
    colors = [-1] * n  # -1: uncolored, 0: red, 1: blue
    
    for start in range(n):
        if colors[start] == -1:  # Uncolored component
            queue = deque([start])
            colors[start] = 0  # Start with color 0
            
            while queue:
                node = queue.popleft()
                
                for neighbor in graph[node]:
                    if colors[neighbor] == -1:
                        # Color with opposite color
                        colors[neighbor] = 1 - colors[node]
                        queue.append(neighbor)
                    elif colors[neighbor] == colors[node]:
                        # Same color - not bipartite
                        return False
    
    return True
```

### Solution - DFS Coloring

```python
def isBipartiteDFS(graph):
    """
    Use DFS to color graph with two colors
    """
    n = len(graph)
    colors = [-1] * n
    
    def dfs(node, color):
        colors[node] = color
        
        for neighbor in graph[node]:
            if colors[neighbor] == -1:
                # Color with opposite color
                if not dfs(neighbor, 1 - color):
                    return False
            elif colors[neighbor] == color:
                # Same color - not bipartite
                return False
        
        return True
    
    for i in range(n):
        if colors[i] == -1:
            if not dfs(i, 0):
                return False
    
    return True
```

---

## Problem 5: Surrounded Regions

**Difficulty**: 🟡 Medium  
**Pattern**: DFS/BFS from Boundary  
**Time**: O(m×n), **Space**: O(m×n)

### Problem Description

Given an `m x n` matrix `board` containing `'X'` and `'O'`, capture all regions that are 4-directionally surrounded by `'X'`.

A region is captured by flipping all `'O'`s into `'X'`s in that surrounded region.

### Solution

```python
def solve(board):
    """
    Start DFS from boundary 'O's to mark safe regions
    Then flip all remaining 'O's to 'X's
    """
    if not board or not board[0]:
        return
    
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            board[r][c] != 'O'):
            return
        
        board[r][c] = 'S'  # Mark as safe
        
        # Check all 4 directions
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    # Mark all boundary-connected 'O's as safe
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows-1 or c == 0 or c == cols-1) and board[r][c] == 'O':
                dfs(r, c)
    
    # Convert remaining 'O's to 'X's and restore safe 'O's
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 'O':
                board[r][c] = 'X'
            elif board[r][c] == 'S':
                board[r][c] = 'O'
```

---

## Problem 6: Find the Town Judge

**Difficulty**: 🟡 Medium  
**Pattern**: Graph In-degree/Out-degree  
**Time**: O(E), **Space**: O(V)

### Problem Description

In a town, there are `n` people labeled from `1` to `n`. There is a rumor that one of these people is secretly the town judge.

If the town judge exists, then:
1. The town judge trusts nobody.
2. Everybody (except for the town judge) trusts the town judge.
3. There is exactly one person that satisfies properties 1 and 2.

### Solution

```python
def findJudge(n, trust):
    """
    Count in-degree and out-degree for each person
    Judge has in-degree = n-1 and out-degree = 0
    """
    if n == 1:
        return 1
    
    # trust_count[i] = in_degree[i] - out_degree[i]
    trust_count = [0] * (n + 1)
    
    for a, b in trust:
        trust_count[a] -= 1  # a trusts someone (out-degree)
        trust_count[b] += 1  # b is trusted by someone (in-degree)
    
    for i in range(1, n + 1):
        if trust_count[i] == n - 1:
            return i
    
    return -1

# Test
trust = [[1,3],[2,3]]
print(findJudge(3, trust))  # 3
```

---

## Problem 7: Rotting Oranges

**Difficulty**: 🟡 Medium  
**Pattern**: Multi-source BFS  
**Time**: O(m×n), **Space**: O(m×n)

### Problem Description

You are given an `m x n` grid where each cell can have one of three values:
- `0` representing an empty cell,
- `1` representing a fresh orange, or
- `2` representing a rotten orange.

Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

Return the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return `-1`.

### Solution

```python
from collections import deque

def orangesRotting(grid):
    """
    Use multi-source BFS starting from all rotten oranges
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh_count = 0
    
    # Find all rotten oranges and count fresh ones
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c))
            elif grid[r][c] == 1:
                fresh_count += 1
    
    # If no fresh oranges, return 0
    if fresh_count == 0:
        return 0
    
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    minutes = 0
    
    while queue and fresh_count > 0:
        minutes += 1
        
        # Process all oranges that rot in this minute
        for _ in range(len(queue)):
            r, c = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and 
                    grid[nr][nc] == 1):
                    grid[nr][nc] = 2  # Rot the orange
                    fresh_count -= 1
                    queue.append((nr, nc))
    
    return minutes if fresh_count == 0 else -1

# Test
grid = [[2,1,1],[1,1,0],[0,1,1]]
print(orangesRotting(grid))  # 4
```

---

## 📝 Summary

### Key Patterns Mastered

1. **Topological Sort** - Detect cycles in directed graphs, course scheduling
2. **Shortest Path** - BFS for unweighted, Dijkstra for weighted graphs
3. **Graph Coloring** - Bipartite checking, conflict resolution
4. **Boundary Processing** - Start from edges to identify safe regions
5. **Multi-source BFS** - Simultaneous spreading from multiple starting points
6. **Degree Counting** - In-degree/out-degree for graph properties

### Algorithm Comparison

| **Algorithm** | **Use Case** | **Time** | **Space** |
|---------------|--------------|----------|-----------|
| **DFS** | Cycle detection, path finding | O(V+E) | O(V) |
| **BFS** | Shortest unweighted path, level traversal | O(V+E) | O(V) |
| **Dijkstra** | Shortest weighted path (non-negative) | O(E log V) | O(V) |
| **Topological Sort** | Dependency resolution, cycle detection | O(V+E) | O(V) |
| **Union-Find** | Connectivity, cycle detection | O(α(V)) | O(V) |

### When to Use Each Pattern

- **DFS**: Deep exploration, backtracking, cycle detection
- **BFS**: Level-by-level, shortest path, spreading patterns
- **Dijkstra**: Weighted shortest path, network routing
- **Topological Sort**: Task scheduling, dependency resolution
- **Graph Coloring**: Conflict resolution, resource allocation

---

Ready for the ultimate challenge? Move on to **[Hard Graph Problems](hard-problems.md)** to master advanced algorithms like strongly connected components, maximum flow, and complex optimization problems!

### 📚 What's Next

- **[Hard Problems](hard-problems.md)** - Master advanced graph algorithms
- **[Dijkstra's Algorithm](dijkstra.md)** - Deep dive into shortest paths
- **[Topological Sort](topological-sort.md)** - Dependency resolution
- **[Union-Find](union-find.md)** - Efficient connectivity queries
