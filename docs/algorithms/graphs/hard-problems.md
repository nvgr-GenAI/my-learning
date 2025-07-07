# Graph Algorithms - Hard Problems

## 🎯 Learning Objectives

Master advanced graph algorithms and complex optimization techniques:

- Strongly connected components (Tarjan's, Kosaraju's)
- Maximum flow algorithms (Ford-Fulkerson, Edmonds-Karp)
- Advanced shortest path algorithms
- Complex graph construction and manipulation
- Optimization problems with multiple constraints

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Alien Dictionary | Topological Sort + Graph | Hard | O(C) | O(1) |
    | 2 | Critical Connections | Tarjan's Bridge Finding | Hard | O(V+E) | O(V+E) |
    | 3 | Reconstruct Itinerary | Eulerian Path (Hierholzer) | Hard | O(E log E) | O(E) |
    | 4 | Min Cost Valid Path | Modified Dijkstra / 0-1 BFS | Hard | O(mn) | O(mn) |
    | 5 | Swim in Rising Water | Binary Search + BFS/DFS | Hard | O(n² log n) | O(n²) |
    | 6 | Bus Routes | Graph Modeling + BFS | Hard | O(N²M) | O(NM) |
    | 7 | Word Ladder II | BFS + Backtracking | Hard | O(M²×N) | O(M²×N) |
    | 8 | Cheapest Flights K Stops | Modified Dijkstra | Hard | O(E + V×K) | O(V×K) |
    | 9 | Network Delay Time | Dijkstra's Shortest Path | Hard | O(E log V) | O(V+E) |
    | 10 | Max Probability Path | Modified Dijkstra (Max) | Hard | O(E log V) | O(V+E) |
    | 11 | Minimum Spanning Tree | Prim's Algorithm | Hard | O(E log V) | O(V+E) |
    | 12 | Dijkstra Path Reconstruction | Dijkstra + Path Tracking | Hard | O(E log V) | O(V+E) |
    | 13 | Maximum Flow | Ford-Fulkerson Algorithm | Hard | O(E × max_flow) | O(V²) |
    | 14 | Strongly Connected Components | Tarjan's Algorithm | Hard | O(V+E) | O(V) |
    | 15 | Minimum Cut | Stoer-Wagner Algorithm | Hard | O(V³) | O(V²) |

=== "🎯 Advanced Patterns"

    **🔗 Complex Graph Properties:**
    - Strongly connected components analysis
    - Bridge and articulation point detection
    - Eulerian paths and circuits
    
    **🌊 Network Flow:**
    - Maximum flow algorithms
    - Minimum cut problems
    - Bipartite matching applications
    
    **🎯 Advanced Search:**
    - Modified Dijkstra for constraints
    - 0-1 BFS for binary weights
    - Binary search on graph properties
    
    **🧩 Graph Modeling:**
    - Transform complex problems to graphs
    - Multi-level graph representations
    - Abstract graph constructions

=== "💡 Solutions"

    === "Alien Dictionary"
        ```python
        from collections import defaultdict, deque

        def alienOrder(words):
            """
            Derive alien language character order from dictionary
            Use topological sort on character dependency graph
            """
            # Initialize in-degree for all characters
            in_degree = {c: 0 for word in words for c in word}
            graph = defaultdict(list)
            
            # Build graph from adjacent word pairs
            for i in range(len(words) - 1):
                word1, word2 = words[i], words[i + 1]
                min_len = min(len(word1), len(word2))
                
                # Invalid case: longer word is prefix of shorter
                if len(word1) > len(word2) and word1[:min_len] == word2[:min_len]:
                    return ""
                
                # Find first differing character
                for j in range(min_len):
                    if word1[j] != word2[j]:
                        if word2[j] not in graph[word1[j]]:
                            graph[word1[j]].append(word2[j])
                            in_degree[word2[j]] += 1
                        break
            
            # Topological sort using Kahn's algorithm
            queue = deque([c for c in in_degree if in_degree[c] == 0])
            result = []
            
            while queue:
                char = queue.popleft()
                result.append(char)
                
                for neighbor in graph[char]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            
            # Check for cycle
            return "".join(result) if len(result) == len(in_degree) else ""
        ```
    
    === "Critical Connections"
        ```python
        def criticalConnections(n, connections):
            """
            Find bridges using Tarjan's algorithm
            Bridge: edge whose removal increases connected components
            """
            # Build adjacency list
            graph = [[] for _ in range(n)]
            for u, v in connections:
                graph[u].append(v)
                graph[v].append(u)
            
            discovery = [-1] * n  # Discovery time
            low = [-1] * n        # Low-link value
            parent = [-1] * n     # Parent in DFS tree
            bridges = []
            time = [0]
            
            def tarjan_dfs(u):
                discovery[u] = low[u] = time[0]
                time[0] += 1
                
                for v in graph[u]:
                    if discovery[v] == -1:  # Tree edge
                        parent[v] = u
                        tarjan_dfs(v)
                        low[u] = min(low[u], low[v])
                        
                        # Bridge condition
                        if low[v] > discovery[u]:
                            bridges.append([u, v])
                    elif v != parent[u]:  # Back edge
                        low[u] = min(low[u], discovery[v])
            
            for i in range(n):
                if discovery[i] == -1:
                    tarjan_dfs(i)
            
            return bridges
        ```
    
    === "Reconstruct Itinerary"
        ```python
        from collections import defaultdict
        import heapq

        def findItinerary(tickets):
            """
            Find Eulerian path using Hierholzer's algorithm
            Use heap for lexicographical ordering
            """
            # Build graph with min-heap for each departure
            graph = defaultdict(list)
            for src, dst in tickets:
                heapq.heappush(graph[src], dst)
            
            def dfs(airport):
                while graph[airport]:
                    next_airport = heapq.heappop(graph[airport])
                    dfs(next_airport)
                path.append(airport)
            
            path = []
            dfs("JFK")
            return path[::-1]
        ```
    
    === "Min Cost Valid Path"
        ```python
        from collections import deque

        def minCost(grid):
            """
            0-1 BFS for minimum cost path
            Cost 0: follow existing direction, Cost 1: change direction
            """
            m, n = len(grid), len(grid[0])
            directions = {1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)}
            all_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            
            dq = deque([(0, 0, 0)])  # (row, col, cost)
            visited = set()
            
            while dq:
                r, c, cost = dq.popleft()
                
                if (r, c) in visited:
                    continue
                visited.add((r, c))
                
                if r == m - 1 and c == n - 1:
                    return cost
                
                for dr, dc in all_dirs:
                    nr, nc = r + dr, c + dc
                    
                    if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in visited:
                        if directions[grid[r][c]] == (dr, dc):
                            dq.appendleft((nr, nc, cost))  # Cost 0
                        else:
                            dq.append((nr, nc, cost + 1))  # Cost 1
            
            return -1
        ```
    
    === "Swim in Rising Water"
        ```python
        import heapq

        def swimInWater(grid):
            """
            Binary search on water level + BFS reachability check
            Alternative: Dijkstra for min-max path
            """
            n = len(grid)
            
            # Dijkstra approach: minimize maximum elevation
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
                    
                    if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in visited:
                        new_max = max(max_elev, grid[nr][nc])
                        heapq.heappush(heap, (new_max, nr, nc))
            
            return -1
        ```
    
    === "Bus Routes"
        ```python
        from collections import defaultdict, deque

        def numBusesToDestination(routes, source, target):
            """
            Model as graph of routes (not stops)
            BFS to find minimum route changes
            """
            if source == target:
                return 0
            
            # Map stops to routes containing them
            stop_to_routes = defaultdict(list)
            for i, route in enumerate(routes):
                for stop in route:
                    stop_to_routes[stop].append(i)
            
            # BFS on routes
            queue = deque()
            visited_routes = set()
            
            for route_id in stop_to_routes[source]:
                queue.append((route_id, 1))
                visited_routes.add(route_id)
            
            while queue:
                route_id, buses = queue.popleft()
                
                if target in routes[route_id]:
                    return buses
                
                # Explore connected routes
                for stop in routes[route_id]:
                    for next_route in stop_to_routes[stop]:
                        if next_route not in visited_routes:
                            visited_routes.add(next_route)
                            queue.append((next_route, buses + 1))
            
            return -1
        ```
    
    === "Word Ladder II"
        ```python
        from collections import defaultdict, deque

        def findLadders(beginWord, endWord, wordList):
            """
            BFS to find shortest distances + DFS to construct paths
            """
            if endWord not in wordList:
                return []
            
            wordList = set(wordList)
            queue = deque([beginWord])
            distances = {beginWord: 0}
            graph = defaultdict(list)
            found = False
            
            while queue and not found:
                level_words = set()
                
                for _ in range(len(queue)):
                    word = queue.popleft()
                    
                    # Try all single-character changes
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
                                
                                # Build transformation graph
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
        ```
    
    === "Maximum Flow"
        ```python
        from collections import defaultdict, deque

        class MaxFlow:
            def __init__(self, edges):
                self.graph = defaultdict(dict)
                for u, v, capacity in edges:
                    self.graph[u][v] = capacity
                    if v not in self.graph or u not in self.graph[v]:
                        self.graph[v][u] = 0  # Reverse edge
            
            def bfs_find_path(self, source, sink, parent):
                """Find augmenting path using BFS (Edmonds-Karp)"""
                visited = {source}
                queue = deque([source])
                
                while queue:
                    u = queue.popleft()
                    
                    for v in self.graph[u]:
                        if v not in visited and self.graph[u][v] > 0:
                            visited.add(v)
                            parent[v] = u
                            if v == sink:
                                return True
                            queue.append(v)
                return False
            
            def max_flow(self, source, sink):
                """Ford-Fulkerson with BFS (Edmonds-Karp)"""
                parent = {}
                max_flow_value = 0
                
                while self.bfs_find_path(source, sink, parent):
                    # Find bottleneck capacity
                    path_flow = float('inf')
                    s = sink
                    while s != source:
                        path_flow = min(path_flow, self.graph[parent[s]][s])
                        s = parent[s]
                    
                    # Update residual capacities
                    max_flow_value += path_flow
                    v = sink
                    while v != source:
                        u = parent[v]
                        self.graph[u][v] -= path_flow
                        self.graph[v][u] += path_flow
                        v = parent[v]
                    
                    parent.clear()
                
                return max_flow_value
        ```

=== "📊 Advanced Techniques"

    **🔧 Algorithm Design Principles:**
    - **Tarjan's Algorithm**: Use DFS with discovery times and low-link values
    - **Hierholzer's Algorithm**: Find Eulerian paths through edge removal
    - **0-1 BFS**: Use deque for graphs with binary edge weights
    - **Network Flow**: Model capacity constraints and flow conservation
    - **Graph Modeling**: Abstract complex problems into graph representations
    
    **⚡ Optimization Strategies:**
    - **Binary Search on Graphs**: Search answer space with graph validation
    - **Modified Dijkstra**: Adapt for different optimization criteria
    - **Multi-level BFS**: Process layers with different objectives
    - **Path Reconstruction**: Track parent pointers during search
    - **State Space Reduction**: Use appropriate data structures and pruning
    
    **🎯 Pattern Recognition:**
    - **Bridge/Articulation**: Look for critical graph components
    - **Eulerian Paths**: Check degree conditions and connectivity
    - **Flow Networks**: Identify source, sink, and capacity constraints
    - **Graph Construction**: Build graphs from problem constraints
    - **Multi-dimensional**: Use additional state dimensions for complex problems

=== "🚀 Expert Tips"

    **💡 Problem-Solving Strategy:**
    1. **Identify Graph Structure**: What are nodes and edges?
    2. **Choose Algorithm**: Match problem type to known algorithms
    3. **Handle Edge Cases**: Empty graphs, disconnected components
    4. **Optimize Implementation**: Use appropriate data structures
    5. **Verify Correctness**: Test with various graph configurations
    
    **🔍 Common Challenges:**
    - **Complexity Management**: Keep track of multiple algorithm components
    - **State Representation**: Design efficient state encoding
    - **Memory Optimization**: Balance time vs space trade-offs
    - **Numerical Stability**: Handle large numbers in flow algorithms
    
    **🏆 Advanced Applications:**
    - **Compiler Design**: Dependency analysis and optimization
    - **Network Analysis**: Critical infrastructure identification
    - **Game Development**: Pathfinding and AI decision trees
    - **Distributed Systems**: Load balancing and fault tolerance
    - **Operations Research**: Supply chain and logistics optimization

## 📝 Summary

These hard graph problems demonstrate:

- **Advanced Graph Theory** with sophisticated algorithms
- **Network Flow** for optimization and matching problems
- **Complex Modeling** to transform real-world problems
- **Algorithmic Engineering** for performance optimization
- **Theoretical Foundations** for problem-solving approaches

These techniques are essential for:

- **System Architecture** requiring graph-based solutions
- **Research and Development** in algorithms and optimization
- **Competitive Programming** at the highest levels
- **Industry Leadership** in technical problem solving
- **Academic Research** in graph theory and algorithms

Master these patterns to tackle the most challenging graph problems in computer science and become an expert in graph algorithms!
