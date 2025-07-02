# Graph Algorithms - Hard Problems

## 🎯 Learning Objectives

Master advanced graph algorithms and complex optimization techniques:

- Strongly connected components (Tarjan's, Kosaraju's)
- Maximum flow algorithms (Ford-Fulkerson, Edmonds-Karp)
- Advanced shortest path algorithms
- Complex graph construction and manipulation
- Optimization problems with multiple constraints

---

## Problem 1: Alien Dictionary

**Difficulty**: 🔴 Hard  
**Pattern**: Topological Sort + Graph Construction  
**Time**: O(C), **Space**: O(1) where C = total characters

### Problem Overview

There is a new alien language that uses the English alphabet. However, the order among letters is unknown to you.

You are given a list of strings `words` from the dictionary, where words are sorted lexicographically by the rules of this new language.

Derive the order of letters in this alien language.

**Example:**
```
Input: words = ["wrt","wrf","er","ett","rftt"]
Output: "wertf"
```

### Solution

```python
from collections import defaultdict, deque

def alienOrder(words):
    """
    1. Build graph from character ordering constraints
    2. Use topological sort to find valid ordering
    3. Handle edge cases (cycles, impossible orderings)
    """
    # Step 1: Initialize in-degree for all characters
    in_degree = {c: 0 for word in words for c in word}
    graph = defaultdict(list)
    
    # Step 2: Build graph from adjacent word pairs
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        min_len = min(len(word1), len(word2))
        
        # Check for invalid case: longer word is prefix of shorter word
        if len(word1) > len(word2) and word1[:min_len] == word2[:min_len]:
            return ""
        
        # Find first differing character
        for j in range(min_len):
            if word1[j] != word2[j]:
                if word2[j] not in graph[word1[j]]:
                    graph[word1[j]].append(word2[j])
                    in_degree[word2[j]] += 1
                break
    
    # Step 3: Topological sort using Kahn's algorithm
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []
    
    while queue:
        char = queue.popleft()
        result.append(char)
        
        for neighbor in graph[char]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check if all characters are processed (no cycle)
    if len(result) != len(in_degree):
        return ""
    
    return "".join(result)

# Test
words = ["wrt","wrf","er","ett","rftt"]
print(alienOrder(words))  # "wertf"
```

---

## Problem 2: Critical Connections in a Network

**Difficulty**: 🔴 Hard  
**Pattern**: Tarjan's Bridge-Finding Algorithm  
**Time**: O(V+E), **Space**: O(V+E)

### Problem Overview

There are `n` servers numbered from `0` to `n-1` connected by undirected server-to-server connections forming a network where `connections[i] = [a, b]` represents a connection between servers `a` and `b`.

A critical connection is a connection that, if removed, will make some server unable to reach some other server.

Return all critical connections in the network in any order.

### Solution

```python
def criticalConnections(n, connections):
    """
    Use Tarjan's algorithm to find bridges in undirected graph
    """
    # Build adjacency list
    graph = [[] for _ in range(n)]
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)
    
    # Tarjan's algorithm variables
    discovery = [-1] * n  # Discovery time
    low = [-1] * n        # Low-link value
    parent = [-1] * n     # Parent in DFS tree
    bridges = []
    time = [0]           # Use list to make it mutable in nested function
    
    def tarjan_dfs(u):
        # Mark current node as visited
        discovery[u] = low[u] = time[0]
        time[0] += 1
        
        for v in graph[u]:
            if discovery[v] == -1:  # Tree edge
                parent[v] = u
                tarjan_dfs(v)
                
                # Update low value
                low[u] = min(low[u], low[v])
                
                # Check if edge u-v is a bridge
                if low[v] > discovery[u]:
                    bridges.append([u, v])
            
            elif v != parent[u]:  # Back edge (not to parent)
                low[u] = min(low[u], discovery[v])
    
    # Run DFS from all unvisited nodes
    for i in range(n):
        if discovery[i] == -1:
            tarjan_dfs(i)
    
    return bridges

# Test
connections = [[0,1],[1,2],[2,0],[1,3]]
print(criticalConnections(4, connections))  # [[1,3]]
```

---

## Problem 3: Reconstruct Itinerary

**Difficulty**: 🔴 Hard  
**Pattern**: Eulerian Path (Hierholzer's Algorithm)  
**Time**: O(E log E), **Space**: O(E)

### Problem Overview

You are given a list of airline `tickets` where `tickets[i] = [fromi, toi]` represent the departure and arrival airports of one flight.

Reconstruct the itinerary in order and return it. All tickets must be used exactly once.

If there are multiple valid itineraries, return the lexically smallest one.

### Solution

```python
from collections import defaultdict
import heapq

def findItinerary(tickets):
    """
    Find Eulerian path using Hierholzer's algorithm
    Use min-heap to ensure lexicographical order
    """
    # Build graph with min-heap for each departure airport
    graph = defaultdict(list)
    for src, dst in tickets:
        heapq.heappush(graph[src], dst)
    
    # Hierholzer's algorithm for Eulerian path
    def dfs(airport):
        while graph[airport]:
            next_airport = heapq.heappop(graph[airport])
            dfs(next_airport)
        path.append(airport)
    
    path = []
    dfs("JFK")  # Start from JFK
    
    return path[::-1]  # Return reversed path

# Alternative implementation with explicit stack
def findItineraryStack(tickets):
    """
    Iterative version using explicit stack
    """
    graph = defaultdict(list)
    for src, dst in tickets:
        graph[src].append(dst)
    
    # Sort destinations in reverse order for correct popping
    for src in graph:
        graph[src].sort(reverse=True)
    
    stack = ["JFK"]
    path = []
    
    while stack:
        while graph[stack[-1]]:
            stack.append(graph[stack[-1]].pop())
        path.append(stack.pop())
    
    return path[::-1]

# Test
tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
print(findItinerary(tickets))  # ["JFK","MUC","LHR","SFO","SJC"]
```

---

## Problem 4: Minimum Cost to Make at Least One Valid Path

**Difficulty**: 🔴 Hard  
**Pattern**: Modified Dijkstra / 0-1 BFS  
**Time**: O(mn), **Space**: O(mn)

### Problem Overview

Given a `m x n` grid where each cell has a direction (1=right, 2=left, 3=down, 4=up), find the minimum cost to make at least one path from top-left to bottom-right valid.

You can change the direction of a cell at cost 1.

### Solution

```python
from collections import deque

def minCost(grid):
    """
    Use 0-1 BFS (deque) for shortest path with 0/1 edge weights
    - Cost 0: follow existing direction
    - Cost 1: change direction
    """
    m, n = len(grid), len(grid[0])
    
    # Direction mappings: 1=right, 2=left, 3=down, 4=up
    directions = {1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)}
    all_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # 0-1 BFS using deque
    dq = deque([(0, 0, 0)])  # (row, col, cost)
    visited = set()
    
    while dq:
        r, c, cost = dq.popleft()
        
        if (r, c) in visited:
            continue
        
        visited.add((r, c))
        
        if r == m - 1 and c == n - 1:
            return cost
        
        # Try all 4 directions
        for i, (dr, dc) in enumerate(all_dirs):
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in visited:
                # Check if this direction matches current cell's direction
                if directions[grid[r][c]] == (dr, dc):
                    # Cost 0: following existing direction
                    dq.appendleft((nr, nc, cost))
                else:
                    # Cost 1: changing direction
                    dq.append((nr, nc, cost + 1))
    
    return -1  # Should not reach here given problem constraints

# Test
grid = [[1,1,1,1],[2,2,2,2],[1,1,1,1],[2,2,2,2]]
print(minCost(grid))  # 3
```

---

## Problem 5: Swim in Rising Water

**Difficulty**: 🔴 Hard  
**Pattern**: Binary Search + BFS/DFS  
**Time**: O(n² log n), **Space**: O(n²)

### Problem Overview

On an `n x n` grid, each square has an elevation given by `grid[i][j]`.

You start at the top left corner and want to reach the bottom right corner. In one step, you can move to one of the four adjacent squares.

You can only move to a square if its elevation is at most `t` (water level). Find the minimum `t` such that you can reach the destination.

### Solution

```python
def swimInWater(grid):
    """
    Binary search on water level + BFS/DFS to check reachability
    """
    n = len(grid)
    
    def canReach(water_level):
        """Check if we can reach destination with given water level"""
        if grid[0][0] > water_level:
            return False
        
        visited = set()
        stack = [(0, 0)]
        
        while stack:
            r, c = stack.pop()
            
            if (r, c) in visited:
                continue
            
            visited.add((r, c))
            
            if r == n - 1 and c == n - 1:
                return True
            
            # Try all 4 directions
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < n and 0 <= nc < n and 
                    (nr, nc) not in visited and 
                    grid[nr][nc] <= water_level):
                    stack.append((nr, nc))
        
        return False
    
    # Binary search on water level
    left, right = 0, n * n - 1
    
    while left < right:
        mid = (left + right) // 2
        
        if canReach(mid):
            right = mid
        else:
            left = mid + 1
    
    return left

# Alternative: Dijkstra-based solution
import heapq

def swimInWaterDijkstra(grid):
    """
    Use modified Dijkstra to find path with minimum maximum elevation
    """
    n = len(grid)
    
    # Priority queue: (max_elevation_so_far, row, col)
    heap = [(grid[0][0], 0, 0)]
    visited = set()
    
    while heap:
        max_elev, r, c = heapq.heappop(heap)
        
        if (r, c) in visited:
            continue
        
        visited.add((r, c))
        
        if r == n - 1 and c == n - 1:
            return max_elev
        
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            nr, nc = r + dr, c + dc
            
            if (0 <= nr < n and 0 <= nc < n and (nr, nc) not in visited):
                new_max_elev = max(max_elev, grid[nr][nc])
                heapq.heappush(heap, (new_max_elev, nr, nc))
    
    return -1

# Test
grid = [[0,2],[1,3]]
print(swimInWater(grid))  # 3
```

---

## Problem 6: Bus Routes

**Difficulty**: 🔴 Hard  
**Pattern**: Graph Modeling + BFS  
**Time**: O(N²M), **Space**: O(NM) where N=routes, M=max bus stops

### Problem Overview

You have an array of `routes` where `routes[i]` is a bus route that the `i`th bus repeats. You will start at the bus stop `source` and want to go to the bus stop `target`.

Return the least number of buses you must take to reach your destination. Return `-1` if it is not possible.

### Solution

```python
from collections import defaultdict, deque

def numBusesToDestination(routes, source, target):
    """
    Model as graph where nodes are bus routes, not bus stops
    Use BFS to find minimum number of route changes
    """
    if source == target:
        return 0
    
    # Build mappings
    stop_to_routes = defaultdict(list)  # stop -> list of routes
    for i, route in enumerate(routes):
        for stop in route:
            stop_to_routes[stop].append(i)
    
    # BFS on routes (not stops)
    queue = deque()
    visited_routes = set()
    
    # Start from all routes that contain source
    for route_id in stop_to_routes[source]:
        queue.append((route_id, 1))  # (route_id, bus_count)
        visited_routes.add(route_id)
    
    while queue:
        current_route, bus_count = queue.popleft()
        
        # Check if current route contains target
        if target in routes[current_route]:
            return bus_count
        
        # Explore neighboring routes
        for stop in routes[current_route]:
            for next_route in stop_to_routes[stop]:
                if next_route not in visited_routes:
                    visited_routes.add(next_route)
                    queue.append((next_route, bus_count + 1))
    
    return -1

# Test
routes = [[1,2,7],[3,6,7]]
source, target = 1, 6
print(numBusesToDestination(routes, source, target))  # 2
```

---

## Problem 7: Word Ladder II

**Difficulty**: 🔴 Hard  
**Pattern**: BFS + Backtracking  
**Time**: O(M²×N), **Space**: O(M²×N)

### Problem Overview

Given two words, `beginWord` and `endWord`, and a dictionary `wordList`, return all the shortest transformation sequences from `beginWord` to `endWord`.

### Solution

```python
from collections import defaultdict, deque

def findLadders(beginWord, endWord, wordList):
    """
    1. BFS to find shortest distance to each word
    2. DFS/Backtracking to construct all shortest paths
    """
    if endWord not in wordList:
        return []
    
    wordList = set(wordList)
    
    # BFS to find distances
    queue = deque([beginWord])
    distances = {beginWord: 0}
    
    # Build adjacency graph during BFS
    graph = defaultdict(list)
    found = False
    
    while queue and not found:
        level_words = set()
        
        # Process all words at current level
        for _ in range(len(queue)):
            word = queue.popleft()
            
            # Try all possible one-character changes
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == word[i]:
                        continue
                    
                    next_word = word[:i] + c + word[i+1:]
                    
                    if next_word in wordList:
                        if next_word == endWord:
                            found = True
                        
                        if next_word not in distances:
                            if next_word not in level_words:
                                level_words.add(next_word)
                                queue.append(next_word)
                                distances[next_word] = distances[word] + 1
                        
                        # Build graph for path reconstruction
                        if distances.get(next_word, float('inf')) == distances[word] + 1:
                            graph[word].append(next_word)
    
    # DFS to find all shortest paths
    def dfs(word, path, target):
        if word == target:
            result.append(path)
            return
        
        for next_word in graph[word]:
            dfs(next_word, path + [next_word], target)
    
    result = []
    if endWord in distances:
        dfs(beginWord, [beginWord], endWord)
    
    return result

# Test
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log","cog"]
print(findLadders(beginWord, endWord, wordList))
# [["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]
```

---

## Problem 8: Cheapest Flights Within K Stops

**Difficulty**: 🔴 Hard  
**Pattern**: Modified Dijkstra with Constraints  
**Time**: O(E + V × K), **Space**: O(V × K)

### Problem Overview

There are `n` cities connected by some number of flights. Find the cheapest price from `src` to `dst` with at most `k` stops.

### Solution

```python
import heapq
from collections import defaultdict

def findCheapestPrice(n, flights, src, dst, k):
    """
    Modified Dijkstra considering stops constraint
    Use heap to track (cost, city, stops_remaining)
    """
    # Build graph
    graph = defaultdict(list)
    for u, v, w in flights:
        graph[u].append((v, w))
    
    # Heap: (cost, city, stops_remaining)
    heap = [(0, src, k + 1)]
    
    # Track best cost to reach each (city, stops) state
    best = {}
    
    while heap:
        cost, city, stops = heapq.heappop(heap)
        
        if city == dst:
            return cost
        
        if stops == 0:
            continue
        
        # Skip if we've seen this state with better cost
        if (city, stops) in best and best[(city, stops)] <= cost:
            continue
        
        best[(city, stops)] = cost
        
        # Explore neighbors
        for next_city, price in graph[city]:
            new_cost = cost + price
            heapq.heappush(heap, (new_cost, next_city, stops - 1))
    
    return -1

# Alternative: Bellman-Ford approach
def findCheapestPriceBF(n, flights, src, dst, k):
    """
    Bellman-Ford with k+1 iterations
    Time: O(k × E), Space: O(V)
    """
    # Initialize distances
    dist = [float('inf')] * n
    dist[src] = 0
    
    # Relax edges k+1 times
    for _ in range(k + 1):
        temp_dist = dist[:]
        
        for u, v, w in flights:
            if dist[u] != float('inf'):
                temp_dist[v] = min(temp_dist[v], dist[u] + w)
        
        dist = temp_dist
    
    return dist[dst] if dist[dst] != float('inf') else -1

# Test
n = 3
flights = [[0,1,100],[1,2,100],[0,2,500]]
src, dst, k = 0, 2, 1
print(findCheapestPrice(n, flights, src, dst, k))  # 200
```

---

## Problem 9: Network Delay Time

**Difficulty**: 🔴 Hard  
**Pattern**: Dijkstra's Shortest Path  
**Time**: O(E log V), **Space**: O(V + E)

### Problem Overview

You have a network of `n` nodes. Find the minimum time for signal to reach all nodes from node `k`.

### Solution

```python
import heapq
from collections import defaultdict

def networkDelayTime(times, n, k):
    """
    Use Dijkstra to find shortest paths from source k
    Return max distance (time for signal to reach all nodes)
    """
    # Build graph
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    
    # Dijkstra's algorithm
    heap = [(0, k)]  # (time, node)
    distances = {}
    
    while heap:
        time, node = heapq.heappop(heap)
        
        if node in distances:
            continue
        
        distances[node] = time
        
        # Explore neighbors
        for neighbor, weight in graph[node]:
            if neighbor not in distances:
                heapq.heappush(heap, (time + weight, neighbor))
    
    # Check if all nodes are reachable
    if len(distances) != n:
        return -1
    
    return max(distances.values())

# Test
times = [[2,1,1],[2,3,1],[3,4,1]]
n, k = 4, 2
print(networkDelayTime(times, n, k))  # 2
```

---

## Problem 10: Path with Maximum Probability

**Difficulty**: 🔴 Hard  
**Pattern**: Modified Dijkstra (Max Probability)  
**Time**: O(E log V), **Space**: O(V + E)

### Problem Overview

Find the path with maximum probability from `start` to `end` in an undirected weighted graph.

### Solution

```python
import heapq
from collections import defaultdict

def maxProbability(n, edges, succProb, start, end):
    """
    Modified Dijkstra to maximize probability instead of minimize distance
    Use negative probabilities with max heap simulation
    """
    # Build graph with probabilities
    graph = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        prob = succProb[i]
        graph[u].append((v, prob))
        graph[v].append((u, prob))
    
    # Max heap simulation using negative values
    heap = [(-1.0, start)]  # (-probability, node)
    visited = set()
    
    while heap:
        neg_prob, node = heapq.heappop(heap)
        prob = -neg_prob
        
        if node in visited:
            continue
        
        visited.add(node)
        
        if node == end:
            return prob
        
        # Explore neighbors
        for neighbor, edge_prob in graph[node]:
            if neighbor not in visited:
                new_prob = prob * edge_prob
                heapq.heappush(heap, (-new_prob, neighbor))
    
    return 0.0

# Test
n = 3
edges = [[0,1],[1,2],[0,2]]
succProb = [0.5,0.5,0.2]
start, end = 0, 2
print(maxProbability(n, edges, succProb, start, end))  # 0.25
```

---

## Problem 11: Minimum Spanning Tree (Prim's Algorithm)

**Difficulty**: 🔴 Hard  
**Pattern**: Greedy + Min Heap  
**Time**: O(E log V), **Space**: O(V + E)

### Problem Overview

Find minimum spanning tree using Prim's algorithm with priority queue optimization.

### Solution

```python
import heapq
from collections import defaultdict

def minimumSpanningTreePrim(n, edges):
    """
    Prim's algorithm using min heap
    Build MST by always adding minimum weight edge to new vertex
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, weight in edges:
        graph[u].append((v, weight))
        graph[v].append((u, weight))
    
    mst = []
    mst_weight = 0
    visited = set()
    
    # Start from node 0
    visited.add(0)
    heap = []
    
    # Add all edges from node 0
    for neighbor, weight in graph[0]:
        heapq.heappush(heap, (weight, 0, neighbor))
    
    while heap and len(mst) < n - 1:
        weight, u, v = heapq.heappop(heap)
        
        if v in visited:
            continue
        
        # Add edge to MST
        mst.append((u, v, weight))
        mst_weight += weight
        visited.add(v)
        
        # Add all edges from new vertex v
        for neighbor, edge_weight in graph[v]:
            if neighbor not in visited:
                heapq.heappush(heap, (edge_weight, v, neighbor))
    
    return mst, mst_weight

def connectCitiesMinCost(connections):
    """
    LeetCode: Connecting Cities with Minimum Cost
    """
    # Extract unique cities and build edges
    cities = set()
    for city1, city2, cost in connections:
        cities.add(city1)
        cities.add(city2)
    
    n = len(cities)
    city_to_idx = {city: i for i, city in enumerate(sorted(cities))}
    
    # Convert to indexed edges
    edges = []
    for city1, city2, cost in connections:
        u = city_to_idx[city1]
        v = city_to_idx[city2]
        edges.append((u, v, cost))
    
    # Build graph
    graph = defaultdict(list)
    for u, v, cost in edges:
        graph[u].append((v, cost))
        graph[v].append((u, cost))
    
    # Check if all cities are connected
    if len(graph) != n:
        return -1
    
    visited = set()
    heap = [(0, 0)]  # (cost, city)
    total_cost = 0
    
    while heap:
        cost, city = heapq.heappop(heap)
        
        if city in visited:
            continue
        
        visited.add(city)
        total_cost += cost
        
        for neighbor, edge_cost in graph[city]:
            if neighbor not in visited:
                heapq.heappush(heap, (edge_cost, neighbor))
    
    return total_cost if len(visited) == n else -1

# Test
edges = [(0, 1, 10), (0, 2, 6), (0, 3, 5), (1, 3, 15), (2, 3, 4)]
mst, weight = minimumSpanningTreePrim(4, edges)
print(f"MST weight: {weight}")  # MST weight: 19
```

---

## Problem 12: Dijkstra with Path Reconstruction

**Difficulty**: 🔴 Hard  
**Pattern**: Dijkstra + Path Tracking  
**Time**: O(E log V), **Space**: O(V + E)

### Problem Overview

Find shortest path and reconstruct the actual path, not just the distance.

### Solution

```python
import heapq
from collections import defaultdict

def dijkstraWithPath(graph, start, end):
    """
    Dijkstra with path reconstruction
    Returns both distance and actual path
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}
    
    heap = [(0, start)]
    visited = set()
    
    while heap:
        current_dist, current = heapq.heappop(heap)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        if current == end:
            break
        
        for neighbor, weight in graph[current]:
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current
                heapq.heappush(heap, (distance, neighbor))
    
    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    
    path.reverse()
    
    if path[0] != start:
        return float('inf'), []  # No path exists
    
    return distances[end], path

def shortestPathAllNodes(graph, start):
    """
    Find shortest paths from start to all other nodes
    Returns distances and paths
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}
    
    heap = [(0, start)]
    visited = set()
    
    while heap:
        current_dist, current = heapq.heappop(heap)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        for neighbor, weight in graph[current]:
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current
                heapq.heappush(heap, (distance, neighbor))
    
    # Reconstruct all paths
    paths = {}
    for node in graph:
        if distances[node] != float('inf'):
            path = []
            current = node
            while current is not None:
                path.append(current)
                current = previous[current]
            path.reverse()
            paths[node] = path
    
    return distances, paths

# Test
graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('C', 1), ('D', 5)],
    'C': [('D', 8), ('E', 10)],
    'D': [('E', 2)],
    'E': []
}

distance, path = dijkstraWithPath(graph, 'A', 'E')
print(f"Distance: {distance}, Path: {path}")  # Distance: 11, Path: ['A', 'B', 'C', 'D', 'E']
```

---

## 📝 Summary

### Advanced Algorithms Mastered

1. **Tarjan's Algorithm** - Finding bridges and articulation points
2. **Hierholzer's Algorithm** - Eulerian path/circuit construction
3. **0-1 BFS** - Shortest path with 0/1 edge weights
4. **Binary Search + Graph** - Optimization problems on graphs
5. **Graph Modeling** - Transform complex problems into graph problems
6. **Multi-level BFS** - Layer-by-layer exploration with backtracking

### Key Insights

| **Problem Type** | **Algorithm Choice** | **Key Insight** |
|------------------|---------------------|-----------------|
| **Bridge Finding** | Tarjan's Algorithm | Track discovery time and low-link values |
| **Eulerian Path** | Hierholzer's Algorithm | All vertices have even degree (except start/end) |
| **0-1 Weighted Paths** | 0-1 BFS with Deque | Use deque: append for weight 1, appendleft for weight 0 |
| **Min-Max Optimization** | Binary Search + BFS | Binary search on answer, verify with graph traversal |
| **Complex Modeling** | Abstract to Graph | Identify what represents nodes and edges |
| **Shortest Path Variants** | BFS + Backtracking | Find distances first, then reconstruct paths |

### Complexity Analysis

- **Tarjan's Algorithm**: O(V + E) - Linear time bridge finding
- **Hierholzer's Algorithm**: O(E) - Linear time Eulerian path
- **0-1 BFS**: O(V + E) - Each edge processed once
- **Binary Search + BFS**: O(log(max_value) × (V + E))
- **Multi-level BFS**: O(V + E) per level, may have exponential paths

### Problem-Solving Strategies

1. **Identify Graph Properties** - Is it directed? Weighted? Dense?
2. **Choose Right Algorithm** - Match algorithm to problem constraints
3. **Handle Edge Cases** - Empty graphs, single nodes, disconnected components
4. **Optimize Space/Time** - Use appropriate data structures
5. **Verify Correctness** - Test with various inputs and edge cases

---

## 🏆 Congratulations!

You've mastered the most challenging graph algorithms! These advanced techniques are used in:

- **Compiler Design** - Dependency analysis, optimization
- **Network Analysis** - Finding critical connections, routing
- **Game Development** - Path finding, AI decision making  
- **Distributed Systems** - Load balancing, fault tolerance
- **Operations Research** - Supply chain, resource allocation

### 📚 What's Next

- **[System Design](../../../system-design/index.md)** - Apply graph algorithms at scale
- **[Advanced Topics](../advanced/index.md)** - Network flows, matching algorithms
- **Practice More** - LeetCode Hard, competitive programming
- **Real Projects** - Build graph-based applications

---

*You're now equipped to tackle any graph problem in interviews and real-world applications!*
