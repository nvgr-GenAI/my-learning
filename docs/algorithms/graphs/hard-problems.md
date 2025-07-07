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

=== "� Study Plan"

    **Week 1: Advanced Graph Theory (Problems 1-5)**
    - Master topological sort and Tarjan's algorithms
    - Practice complex graph construction
    
    **Week 2: Network Flow & Optimization (Problems 6-10)**
    - Learn flow algorithms and path optimization
    - Focus on multi-constraint problems
    
    **Week 3: Expert Level (Problems 11-15)**
    - Advanced algorithms and complex modeling
    - Competitive programming techniques

=== "Alien Dictionary"

    **Problem Statement:**
    There is a new alien language that uses the English alphabet. However, the order among the letters is unknown to you. You are given a list of strings words from the alien language's dictionary, where the strings in words are sorted lexicographically by the rules of this new language.

    **Example:**
    ```text
    Input: words = ["wrt","wrf","er","ett","rftt"]
    Output: "wertf"
    ```

    **Solution:**
    ```python
    from collections import defaultdict, deque

    def alienOrder(words):
        """
        Derive alien language character order from dictionary
        Use topological sort on character dependency graph
        
        Time: O(C) where C is total content of words
        Space: O(1) as max 26 characters
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

    **Key Insights:**
    - Build graph from character ordering constraints
    - Use topological sort to find valid ordering
    - Handle invalid cases like cycles or impossible orderings
    - Kahn's algorithm efficiently detects cycles

=== "Critical Connections"

    **Problem Statement:**
    There are n servers numbered from 0 to n - 1 connected by undirected server-to-server connections forming a network where connections[i] = [ai, bi] represents a connection between servers ai and bi. Any server can reach other servers directly or indirectly through the network. A critical connection is a connection that, if removed, will make some servers unable to reach some other servers.

    **Example:**
    ```text
    Input: n = 4, connections = [[0,1],[1,2],[2,0],[1,3]]
    Output: [[1,3]]
    ```

    **Solution:**
    ```python
    def criticalConnections(n, connections):
        """
        Find bridges using Tarjan's algorithm
        Bridge: edge whose removal increases connected components
        
        Time: O(V+E) - single DFS traversal
        Space: O(V+E) - graph and arrays
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

    **Key Insights:**
    - Tarjan's algorithm finds bridges in single DFS pass
    - Low-link values track earliest reachable vertex
    - Bridge exists when low[v] > discovery[u] for tree edge
    - Essential for network reliability analysis

=== "Reconstruct Itinerary"

    **Problem Statement:**
    You are given a list of airline tickets where tickets[i] = [fromi, toi] represent the departure and the arrival airports of one ticket. Reconstruct the itinerary in order and return it. All of the tickets belong to a man who departs from "JFK", thus, the itinerary must begin with "JFK".

    **Example:**
    ```text
    Input: tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
    Output: ["JFK","MUC","LHR","SFO","SJC"]
    ```

    **Solution:**
    ```python
    from collections import defaultdict
    import heapq

    def findItinerary(tickets):
        """
        Find Eulerian path using Hierholzer's algorithm
        Use heap for lexicographical ordering
        
        Time: O(E log E) - sorting edges
        Space: O(E) - graph storage
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

    **Key Insights:**
    - Problem is finding Eulerian path in directed graph
    - Use heap to ensure lexicographical order
    - Hierholzer's algorithm builds path in reverse
    - DFS with edge removal handles the traversal

=== "Min Cost Valid Path"

    **Problem Statement:**
    Given a m x n grid where each cell has a sign pointing to the next cell you should visit if you are currently in this cell. The sign of grid[i][j] can be: 1 which means go to the cell to the right, 2 which means go to the cell to the left, 3 which means go to the cell below, 4 which means go to the cell above. Return the minimum cost to make the path from the top-left cell to the bottom-right cell valid.

    **Example:**
    ```text
    Input: grid = [[1,1,1,1],[2,2,2,2],[1,1,1,1],[2,2,2,2]]
    Output: 3
    ```

    **Solution:**
    ```python
    from collections import deque

    def minCost(grid):
        """
        0-1 BFS for minimum cost path
        Cost 0: follow existing direction, Cost 1: change direction
        
        Time: O(mn) - each cell processed once
        Space: O(mn) - deque storage
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

    **Key Insights:**
    - 0-1 BFS optimal for binary edge weights
    - Following sign has cost 0, changing direction costs 1
    - Use deque: appendleft for cost 0, append for cost 1
    - Guarantees shortest path in linear time

=== "Swim in Rising Water"

    **Problem Statement:**
    On an N x N grid, each square grid[i][j] represents the elevation at that point (i,j). Now rain starts to fall. At time t, the depth of the water everywhere is t. You can swim from a square to another 4-directionally adjacent square if and only if the elevation of both squares individually are at most t. You can swim infinite distance in zero time. Of course, you must stay within the boundaries of the grid during your swim.

    **Example:**
    ```text
    Input: grid = [[0,2],[1,3]]
    Output: 3
    ```

    **Solution:**
    ```python
    import heapq

    def swimInWater(grid):
        """
        Dijkstra approach: minimize maximum elevation
        
        Time: O(n² log n) - priority queue operations
        Space: O(n²) - heap and visited set
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

    **Alternative (Binary Search):**
    ```python
    def swimInWater(grid):
        """Binary search on answer + BFS validation"""
        n = len(grid)
        
        def canSwim(time):
            if grid[0][0] > time:
                return False
            
            visited = [[False] * n for _ in range(n)]
            queue = [(0, 0)]
            visited[0][0] = True
            
            while queue:
                new_queue = []
                for r, c in queue:
                    if r == n-1 and c == n-1:
                        return True
                    
                    for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < n and 0 <= nc < n and 
                            not visited[nr][nc] and grid[nr][nc] <= time):
                            visited[nr][nc] = True
                            new_queue.append((nr, nc))
                queue = new_queue
            
            return False
        
        left, right = 0, n * n - 1
        while left < right:
            mid = (left + right) // 2
            if canSwim(mid):
                right = mid
            else:
                left = mid + 1
        
        return left
    ```

    **Key Insights:**
    - Problem asks for minimum time to reach destination
    - Dijkstra minimizes maximum elevation in path
    - Binary search approach validates reachability at each time
    - Both approaches achieve optimal time complexity

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
