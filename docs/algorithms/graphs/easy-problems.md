# Graph Algorithms - Easy Problems

## 🎯 Learning Objectives

Master basic graph operations and traversal patterns. These 15 problems cover essential graph algorithms frequently asked in technical interviews.

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Number of Islands | DFS/BFS on Grid | Easy | O(m×n) | O(m×n) |
    | 2 | Find if Path Exists | Graph Traversal | Easy | O(V+E) | O(V+E) |
    | 3 | Clone Graph | DFS/BFS with Cloning | Easy | O(V+E) | O(V) |
    | 4 | All Paths Source to Target | DFS Backtracking | Easy | O(2^V × V) | O(V) |
    | 5 | Find Center of Star Graph | Graph Properties | Easy | O(1) | O(1) |
    | 6 | Find Town Judge | In/Out Degree | Easy | O(V+E) | O(V) |
    | 7 | Redundant Connection | Union-Find | Easy | O(V×α(V)) | O(V) |
    | 8 | Valid Tree (Connected + No Cycle) | DFS/Union-Find | Easy | O(V+E) | O(V) |
    | 9 | Connected Components Count | DFS/BFS | Easy | O(V+E) | O(V) |
    | 10 | Is Graph Bipartite | BFS/DFS Coloring | Easy | O(V+E) | O(V) |
    | 11 | Course Schedule I | Cycle Detection | Easy | O(V+E) | O(V) |
    | 12 | Minimum Height Trees | Tree Properties | Easy | O(V) | O(V) |
    | 13 | Employee Importance | DFS/BFS | Easy | O(V+E) | O(V) |
    | 14 | Keys and Rooms | DFS/BFS | Easy | O(V+E) | O(V) |
    | 15 | Flood Fill | DFS/BFS on Matrix | Easy | O(m×n) | O(m×n) |

=== "🎯 Core Graph Patterns"

    **🌊 Graph Traversal:**
    - DFS (Depth-First Search): Recursive exploration
    - BFS (Breadth-First Search): Level-by-level exploration
    
    **🔗 Connectivity:**
    - Connected components counting
    - Path existence between nodes
    - Union-Find for dynamic connectivity
    
    **🎨 Graph Coloring:**
    - Bipartite graph detection
    - Conflict resolution problems
    
    **🌳 Tree Properties:**
    - Cycle detection in undirected graphs
    - Tree validation and construction

=== "⚡ Interview Strategy"

    **💡 Problem Recognition:**
    
    - **Grid problems**: Often graphs in disguise (4/8-directional movement)
    - **Dependency problems**: Usually require topological sorting
    - **Connectivity questions**: DFS/BFS or Union-Find
    - **Shortest path**: BFS for unweighted, Dijkstra for weighted
    
    **🎪 Common Patterns:**
    
    - **Visited array**: Track explored nodes to avoid cycles
    - **Adjacency representation**: List vs Matrix tradeoffs
    - **DFS for connectivity**: Recursive exploration of components
    - **BFS for levels**: When distance/steps matter

---

## Problem 1: Number of Islands

**Difficulty**: 🟢 Easy  
**Pattern**: DFS/BFS on Grid  
**Time**: O(m×n), **Space**: O(m×n)

=== "Problem Statement"

    Given a 2D binary grid representing a map of '1's (land) and '0's (water), count the number of islands. An island is formed by connecting adjacent lands horizontally or vertically.

    **Example:**
    ```text
    Input: grid = [
      ["1","1","1","1","0"],
      ["1","1","0","1","0"],
      ["1","1","0","0","0"], 
      ["0","0","0","0","0"]
    ]
    Output: 1
    ```

=== "DFS Solution"

    ```python
    def numIslands(grid):
        """
        DFS approach - explore each island completely when found.
        Time: O(m×n), Space: O(m×n) for recursion stack
        """
        if not grid or not grid[0]:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        islands = 0
        
        def dfs(r, c):
            # Base cases: out of bounds or water/visited
            if (r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1'):
                return
            
            # Mark as visited by setting to '0'
            grid[r][c] = '0'
            
            # Explore all 4 directions
            dfs(r + 1, c)  # Down
            dfs(r - 1, c)  # Up
            dfs(r, c + 1)  # Right
            dfs(r, c - 1)  # Left
        
        # Check each cell
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':
                    islands += 1
                    dfs(r, c)  # Sink the entire island
        
        return islands
    ```

=== "BFS Solution"

    ```python
    def numIslands(grid):
        """
        BFS approach - explore islands level by level.
        Time: O(m×n), Space: O(min(m,n)) for queue
        """
        if not grid or not grid[0]:
            return 0
        
        from collections import deque
        
        rows, cols = len(grid), len(grid[0])
        islands = 0
        
        def bfs(start_r, start_c):
            queue = deque([(start_r, start_c)])
            grid[start_r][start_c] = '0'  # Mark as visited
            
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            
            while queue:
                r, c = queue.popleft()
                
                # Check all 4 directions
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    
                    if (0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1'):
                        grid[nr][nc] = '0'  # Mark as visited
                        queue.append((nr, nc))
        
        # Check each cell
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':
                    islands += 1
                    bfs(r, c)  # Explore the entire island
        
        return islands
    ```

=== "💡 Tips & Insights"

    **Key Insights:**
    
    - Grid problems are graphs in disguise - each cell is a node
    - Mark visited cells to avoid infinite loops and recounting
    - DFS uses recursion stack, BFS uses explicit queue
    - Both approaches modify the grid - make a copy if needed
    
    **Interview Notes:**
    
    - Ask about modifying input vs using visited set
    - Clarify connectivity (4-directional vs 8-directional)
    - Consider edge cases: empty grid, single cell
    - Time complexity is always O(m×n) - visit each cell once

---

## Problem 2: Find if Path Exists

**Difficulty**: 🟢 Easy  
**Pattern**: Graph Traversal  
**Time**: O(V+E), **Space**: O(V+E)

=== "Problem Statement"

    Given a graph represented by edges and two nodes `source` and `destination`, determine if there exists a valid path from `source` to `destination`.

    **Example:**
    ```text
    Input: n = 3, edges = [[0,1],[1,2],[2,0]], source = 0, destination = 2
    Output: true
    Explanation: There are two paths from 0 to 2: 0 → 1 → 2 and 0 → 2
    ```

=== "DFS Solution"

    ```python
    def validPath(n, edges, source, destination):
        """
        DFS approach to find path between nodes.
        Time: O(V+E), Space: O(V+E)
        """
        if source == destination:
            return True
        
        # Build adjacency list
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
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
    ```

=== "BFS Solution"

    ```python
    def validPath(n, edges, source, destination):
        """
        BFS approach to find path between nodes.
        Time: O(V+E), Space: O(V+E)
        """
        if source == destination:
            return True
        
        from collections import deque
        
        # Build adjacency list
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        visited = set([source])
        queue = deque([source])
        
        while queue:
            node = queue.popleft()
            
            for neighbor in graph[node]:
                if neighbor == destination:
                    return True
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False
    ```

=== "Union-Find Solution"

    ```python
    def validPath(n, edges, source, destination):
        """
        Union-Find approach for connectivity queries.
        Time: O(E×α(n)), Space: O(n)
        """
        if source == destination:
            return True
        
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            
            # Union by rank
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
        
        # Union all edges
        for u, v in edges:
            union(u, v)
        
        return find(source) == find(destination)
    ```

=== "💡 Tips & Insights"

    **Key Insights:**
    
    - Three main approaches: DFS, BFS, Union-Find
    - DFS/BFS: Good for single query, O(V+E) time
    - Union-Find: Better for multiple connectivity queries
    - Early termination when destination found
    
    **Interview Notes:**
    
    - Ask about graph representation (adjacency list vs matrix)
    - Consider number of queries - Union-Find for multiple
    - Edge case: source equals destination
    - Space-time tradeoffs between approaches

---

## Problem 3: Clone Graph

**Difficulty**: 🟢 Easy  
**Pattern**: DFS/BFS with Cloning  
**Time**: O(V+E), **Space**: O(V)

=== "Problem Statement"

    Given a reference to a node in a connected undirected graph, return a deep copy (clone) of the graph. Each node contains a value and a list of neighbors.

    **Example:**
    ```text
    Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
    Output: [[2,4],[1,3],[2,4],[1,3]]
    ```

=== "DFS Solution"

    ```python
    class Node:
        def __init__(self, val=0, neighbors=None):
            self.val = val
            self.neighbors = neighbors if neighbors is not None else []
    
    def cloneGraph(node):
        """
        DFS approach to clone graph.
        Time: O(V+E), Space: O(V)
        """
        if not node:
            return None
        
        clones = {}  # Original node -> Cloned node mapping
        
        def dfs(original):
            if original in clones:
                return clones[original]
            
            # Create clone for current node
            clone = Node(original.val)
            clones[original] = clone
            
            # Clone all neighbors
            for neighbor in original.neighbors:
                clone.neighbors.append(dfs(neighbor))
            
            return clone
        
        return dfs(node)
    ```

=== "BFS Solution"

    ```python
    def cloneGraph(node):
        """
        BFS approach to clone graph.
        Time: O(V+E), Space: O(V)
        """
        if not node:
            return None
        
        from collections import deque
        
        clones = {node: Node(node.val)}
        queue = deque([node])
        
        while queue:
            original = queue.popleft()
            
            for neighbor in original.neighbors:
                if neighbor not in clones:
                    # Create clone for neighbor
                    clones[neighbor] = Node(neighbor.val)
                    queue.append(neighbor)
                
                # Connect clone to neighbor clone
                clones[original].neighbors.append(clones[neighbor])
        
        return clones[node]
    ```

=== "💡 Tips & Insights"

    **Key Insights:**
    
    - Use hashmap to track original -> clone mapping
    - Create clone nodes on first visit
    - Connect neighbors after all nodes are created
    - Handle cycles naturally with visited tracking
    
    **Interview Notes:**
    
    - Ask about node structure and constraints
    - Clarify if values are unique (affects cloning strategy)
    - Consider memory usage for large graphs
    - Both DFS and BFS work - choose based on stack/heap preferences

---

## Problem 4: All Paths Source to Target

**Difficulty**: 🟢 Easy  
**Pattern**: DFS Backtracking  
**Time**: O(2^V × V), **Space**: O(V)

=== "Problem Statement"

    Given a directed acyclic graph (DAG) with n nodes labeled from 0 to n-1, find all possible paths from node 0 to node n-1.

    **Example:**
    ```text
    Input: graph = [[1,2],[3],[3],[]]
    Output: [[0,1,3],[0,2,3]]
    ```

=== "DFS Backtracking"

    ```python
    def allPathsSourceTarget(graph):
        """
        DFS backtracking to find all paths.
        Time: O(2^V × V), Space: O(V) for recursion
        """
        target = len(graph) - 1
        result = []
        
        def dfs(node, path):
            if node == target:
                result.append(path[:])  # Add copy of current path
                return
            
            for neighbor in graph[node]:
                path.append(neighbor)
                dfs(neighbor, path)
                path.pop()  # Backtrack
        
        dfs(0, [0])
        return result
    ```

=== "Iterative DFS"

    ```python
    def allPathsSourceTarget(graph):
        """
        Iterative DFS using stack.
        Time: O(2^V × V), Space: O(2^V × V)
        """
        target = len(graph) - 1
        stack = [(0, [0])]  # (current_node, current_path)
        result = []
        
        while stack:
            node, path = stack.pop()
            
            if node == target:
                result.append(path)
                continue
            
            for neighbor in graph[node]:
                stack.append((neighbor, path + [neighbor]))
        
        return result
    ```

=== "💡 Tips & Insights"

    **Key Insights:**
    
    - DAG property ensures no cycles - no visited set needed
    - Backtracking naturally explores all possible paths
    - Path copying is crucial for correct results
    - Exponential time complexity due to potentially many paths
    
    **Interview Notes:**
    
    - Confirm graph is DAG (no cycle handling needed)
    - Ask about memory constraints for large path counts
    - Consider iterative vs recursive based on stack depth
    - Time complexity depends on number of possible paths

---

## Problem 5: Find Center of Star Graph

**Difficulty**: 🟢 Easy  
**Pattern**: Graph Properties  
**Time**: O(1), **Space**: O(1)

=== "Problem Statement"

    There is an undirected star graph consisting of n nodes labeled from 1 to n. A star graph is a graph where there is one center node and exactly n-1 edges that connect the center node with every other node.

    **Example:**
    ```text
    Input: edges = [[1,2],[2,3],[4,2]]
    Output: 2
    Explanation: Node 2 is connected to every other node.
    ```

=== "Optimal Solution"

    ```python
    def findCenter(edges):
        """
        In a star graph, center appears in every edge.
        Check first two edges to find common node.
        Time: O(1), Space: O(1)
        """
        # The center must be in both first and second edge
        first_edge = edges[0]
        second_edge = edges[1]
        
        # Find common node between first two edges
        if first_edge[0] in second_edge:
            return first_edge[0]
        else:
            return first_edge[1]
    ```

=== "Degree Counting"

    ```python
    def findCenter(edges):
        """
        Alternative: Count degrees (less efficient but shows concept).
        Time: O(E), Space: O(V)
        """
        from collections import defaultdict
        
        degree = defaultdict(int)
        
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        
        # Find node with maximum degree
        return max(degree, key=degree.get)
    ```

=== "💡 Tips & Insights"

    **Key Insights:**
    
    - Star graph property: center connected to all other nodes
    - Center appears in every edge of the graph
    - Only need to check first two edges for common node
    - Degree of center = n-1, others = 1
    
    **Interview Notes:**
    
    - Leverage star graph properties for O(1) solution
    - Ask if graph is guaranteed to be a star
    - Consider edge cases: minimum star has 2 nodes
    - Optimal solution beats brute force by orders of magnitude

---

## Problem 6: Find Town Judge

**Difficulty**: 🟢 Easy  
**Pattern**: In/Out Degree Analysis  
**Time**: O(V+E), **Space**: O(V)

=== "Problem Statement"

    In a town, there are n people labeled from 1 to n. There is a rumor that one of these people is secretly the town judge. The town judge:
    1. Trusts nobody
    2. Is trusted by everyone else

    **Example:**
    ```text
    Input: n = 3, trust = [[1,3],[2,3]]
    Output: 3
    Explanation: Person 3 is trusted by 1 and 2, and trusts nobody.
    ```

=== "Degree Counting"

    ```python
    def findJudge(n, trust):
        """
        Count trust relationships as in/out degrees.
        Judge: in-degree = n-1, out-degree = 0
        Time: O(V+E), Space: O(V)
        """
        if n == 1:
            return 1  # Single person is the judge
        
        trust_count = [0] * (n + 1)  # Index 0 unused
        
        for a, b in trust:
            trust_count[a] -= 1  # a trusts someone (out-degree)
            trust_count[b] += 1  # b is trusted (in-degree)
        
        # Find person with trust_count = n-1
        for i in range(1, n + 1):
            if trust_count[i] == n - 1:
                return i
        
        return -1
    ```

=== "Two Array Approach"

    ```python
    def findJudge(n, trust):
        """
        Separate arrays for in-degree and out-degree.
        Time: O(V+E), Space: O(V)
        """
        if n == 1:
            return 1
        
        trusted_by = [0] * (n + 1)  # How many trust this person
        trusts = [0] * (n + 1)      # How many this person trusts
        
        for a, b in trust:
            trusts[a] += 1
            trusted_by[b] += 1
        
        # Judge: trusted by n-1 people, trusts 0 people
        for i in range(1, n + 1):
            if trusted_by[i] == n - 1 and trusts[i] == 0:
                return i
        
        return -1
    ```

=== "💡 Tips & Insights"

    **Key Insights:**
    
    - Model as directed graph with trust relationships
    - Judge has unique degree pattern: in-degree n-1, out-degree 0
    - Single array optimization: combine in/out degrees
    - Edge case: single person town
    
    **Interview Notes:**
    
    - Ask about 1-indexed vs 0-indexed people
    - Clarify constraints: exactly one judge or none
    - Consider memory optimization with single array
    - Time complexity linear in people + trust relationships

---

## Problem 7: Redundant Connection

**Difficulty**: 🟢 Easy  
**Pattern**: Union-Find (Cycle Detection)  
**Time**: O(V×α(V)), **Space**: O(V)

=== "Problem Statement"

    Given a graph that started as a tree but has one additional edge added, find the edge that can be removed to restore the tree property.

    **Example:**
    ```text
    Input: edges = [[1,2],[1,3],[2,3]]
    Output: [2,3]
    Explanation: Removing [2,3] leaves a valid tree.
    ```

=== "Union-Find Solution"

    ```python
    def findRedundantConnection(edges):
        """
        Use Union-Find to detect the first edge that creates a cycle.
        Time: O(V×α(V)), Space: O(V)
        """
        parent = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False  # Already connected - would create cycle
            parent[px] = py
            return True
        
        for u, v in edges:
            if not union(u, v):
                return [u, v]  # This edge creates a cycle
        
        return []
    ```

=== "DFS Cycle Detection"

    ```python
    def findRedundantConnection(edges):
        """
        Alternative: DFS-based cycle detection.
        Time: O(V×E), Space: O(V+E)
        """
        graph = {}
        
        def has_path(start, end, visited):
            if start == end:
                return True
            
            visited.add(start)
            
            for neighbor in graph.get(start, []):
                if neighbor not in visited:
                    if has_path(neighbor, end, visited):
                        return True
            
            return False
        
        for u, v in edges:
            # Check if path already exists
            if u in graph and v in graph:
                if has_path(u, v, set()):
                    return [u, v]
            
            # Add edge to graph
            if u not in graph:
                graph[u] = []
            if v not in graph:
                graph[v] = []
            graph[u].append(v)
            graph[v].append(u)
        
        return []
    ```

=== "💡 Tips & Insights"

    **Key Insights:**
    
    - Tree has V-1 edges; extra edge creates exactly one cycle
    - Union-Find detects cycle when connecting already-connected nodes
    - Process edges in order - first cycle-creating edge is answer
    - Path compression optimizes Union-Find performance
    
    **Interview Notes:**
    
    - Ask if edges are processed in specific order
    - Consider Union-Find vs DFS tradeoffs
    - Clarify if solution should be lexicographically smallest
    - Union-Find preferred for this specific problem pattern

---

## Problem 8: Valid Tree (Connected + No Cycle)

**Difficulty**: 🟢 Easy  
**Pattern**: DFS/Union-Find  
**Time**: O(V+E), **Space**: O(V)

=== "Problem Statement"

    Given n nodes and a list of undirected edges, determine if these edges make up a valid tree. A valid tree is connected and has no cycles.

    **Example:**
    ```text
    Input: n = 5, edges = [[0,1],[0,2],[0,3],[1,4]]
    Output: true
    ```

=== "DFS Solution"

    ```python
    def validTree(n, edges):
        """
        DFS to check connectivity and cycle detection.
        Tree: exactly n-1 edges, connected, no cycles.
        Time: O(V+E), Space: O(V+E)
        """
        # Tree must have exactly n-1 edges
        if len(edges) != n - 1:
            return False
        
        # Build adjacency list
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        visited = set()
        
        def dfs(node, parent):
            visited.add(node)
            
            for neighbor in graph[node]:
                if neighbor == parent:
                    continue  # Skip edge back to parent
                
                if neighbor in visited:
                    return False  # Cycle detected
                
                if not dfs(neighbor, node):
                    return False
            
            return True
        
        # Check for cycles and ensure connectivity
        return dfs(0, -1) and len(visited) == n
    ```

=== "Union-Find Solution"

    ```python
    def validTree(n, edges):
        """
        Union-Find approach for cycle detection and connectivity.
        Time: O(E×α(n)), Space: O(n)
        """
        # Tree must have exactly n-1 edges
        if len(edges) != n - 1:
            return False
        
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False  # Cycle detected
            parent[px] = py
            return True
        
        # Process all edges
        for u, v in edges:
            if not union(u, v):
                return False
        
        return True  # No cycles and correct edge count = connected tree
    ```

=== "💡 Tips & Insights"

    **Key Insights:**
    
    - Valid tree: exactly n-1 edges, connected, no cycles
    - Edge count check eliminates many invalid cases quickly
    - DFS with parent tracking avoids false cycle detection
    - Union-Find naturally detects cycles during edge processing
    
    **Interview Notes:**
    
    - Tree properties: n nodes, n-1 edges, connected, acyclic
    - Ask about empty graph case (n=0 or n=1)
    - Consider efficiency: Union-Find vs DFS
    - Edge count optimization eliminates many test cases early

---

## Problem 9: Connected Components Count

**Difficulty**: 🟢 Easy  
**Pattern**: DFS/BFS  
**Time**: O(V+E), **Space**: O(V)

=== "Problem Statement"

    Given an undirected graph represented by an adjacency list, count the number of connected components.

    **Example:**
    ```text
    Input: n = 5, edges = [[0,1],[1,2],[3,4]]
    Output: 2
    Explanation: Components: {0,1,2} and {3,4}
    ```

=== "DFS Solution"

    ```python
    def countComponents(n, edges):
        """
        DFS to explore each component completely.
        Time: O(V+E), Space: O(V+E)
        """
        # Build adjacency list
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        visited = set()
        components = 0
        
        def dfs(node):
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor)
        
        # Visit each unvisited node
        for i in range(n):
            if i not in visited:
                dfs(i)
                components += 1
        
        return components
    ```

=== "Union-Find Solution"

    ```python
    def countComponents(n, edges):
        """
        Union-Find to merge components and count roots.
        Time: O(E×α(n)), Space: O(n)
        """
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union all edges
        for u, v in edges:
            union(u, v)
        
        # Count unique roots
        return len(set(find(i) for i in range(n)))
    ```

=== "BFS Solution"

    ```python
    def countComponents(n, edges):
        """
        BFS to explore each component level by level.
        Time: O(V+E), Space: O(V+E)
        """
        from collections import deque
        
        # Build adjacency list
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        visited = set()
        components = 0
        
        def bfs(start):
            queue = deque([start])
            visited.add(start)
            
            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        # Visit each unvisited node
        for i in range(n):
            if i not in visited:
                bfs(i)
                components += 1
        
        return components
    ```

=== "💡 Tips & Insights"

    **Key Insights:**
    
    - Each DFS/BFS traversal discovers one complete component
    - Union-Find naturally groups nodes into components
    - Count number of traversals needed to visit all nodes
    - Isolated nodes form single-node components
    
    **Interview Notes:**
    
    - All three approaches (DFS, BFS, Union-Find) work well
    - Choose based on problem context and follow-up queries
    - Consider space-time tradeoffs for large graphs
    - Union-Find better for dynamic connectivity queries

---

## Problem 10: Is Graph Bipartite

**Difficulty**: 🟢 Easy  
**Pattern**: BFS/DFS Coloring  
**Time**: O(V+E), **Space**: O(V)

=== "Problem Statement"

    Given an undirected graph, determine if it can be colored with two colors such that no two adjacent nodes have the same color.

    **Example:**
    ```text
    Input: graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
    Output: false
    Explanation: Cannot color with two colors.
    ```

=== "BFS Coloring"

    ```python
    def isBipartite(graph):
        """
        BFS with 2-coloring to check bipartiteness.
        Time: O(V+E), Space: O(V)
        """
        from collections import deque
        
        n = len(graph)
        color = [-1] * n  # -1: uncolored, 0/1: colors
        
        def bfs(start):
            queue = deque([start])
            color[start] = 0
            
            while queue:
                node = queue.popleft()
                
                for neighbor in graph[node]:
                    if color[neighbor] == -1:
                        # Color with opposite color
                        color[neighbor] = 1 - color[node]
                        queue.append(neighbor)
                    elif color[neighbor] == color[node]:
                        return False  # Same color conflict
            
            return True
        
        # Check all connected components
        for i in range(n):
            if color[i] == -1:
                if not bfs(i):
                    return False
        
        return True
    ```

=== "DFS Coloring"

    ```python
    def isBipartite(graph):
        """
        DFS with 2-coloring to check bipartiteness.
        Time: O(V+E), Space: O(V)
        """
        n = len(graph)
        color = [-1] * n
        
        def dfs(node, c):
            color[node] = c
            
            for neighbor in graph[node]:
                if color[neighbor] == -1:
                    # Color with opposite color
                    if not dfs(neighbor, 1 - c):
                        return False
                elif color[neighbor] == c:
                    return False  # Same color conflict
            
            return True
        
        # Check all connected components
        for i in range(n):
            if color[i] == -1:
                if not dfs(i, 0):
                    return False
        
        return True
    ```

=== "💡 Tips & Insights"

    **Key Insights:**
    
    - Bipartite ⟺ can be 2-colored ⟺ no odd-length cycles
    - Use BFS/DFS to assign alternating colors
    - Conflict detection: adjacent nodes with same color
    - Handle multiple connected components separately
    
    **Interview Notes:**
    
    - Ask about graph representation (adjacency list vs matrix)
    - Consider disconnected graphs - check all components
    - Both DFS and BFS work - choose based on preference
    - Time complexity always O(V+E) for complete traversal

---

## Problem 11: Course Schedule I

**Difficulty**: 🟢 Easy  
**Pattern**: Cycle Detection (Topological Sort)  
**Time**: O(V+E), **Space**: O(V)

=== "Problem Statement"

    There are numCourses courses labeled from 0 to numCourses-1. Given prerequisites array where prerequisites[i] = [ai, bi] indicates you must take course bi before ai, determine if you can finish all courses.

    **Example:**
    ```text
    Input: numCourses = 2, prerequisites = [[1,0]]
    Output: true
    Explanation: Take course 0, then course 1.
    ```

=== "DFS Cycle Detection"

    ```python
    def canFinish(numCourses, prerequisites):
        """
        DFS-based cycle detection in directed graph.
        Time: O(V+E), Space: O(V+E)
        """
        # Build adjacency list
        graph = [[] for _ in range(numCourses)]
        for course, prereq in prerequisites:
            graph[prereq].append(course)
        
        # 0: unvisited, 1: visiting (in current path), 2: visited
        state = [0] * numCourses
        
        def dfs(node):
            if state[node] == 1:
                return False  # Back edge - cycle detected
            if state[node] == 2:
                return True   # Already processed
            
            state[node] = 1  # Mark as visiting
            
            for neighbor in graph[node]:
                if not dfs(neighbor):
                    return False
            
            state[node] = 2  # Mark as visited
            return True
        
        # Check all courses for cycles
        for i in range(numCourses):
            if state[i] == 0:
                if not dfs(i):
                    return False
        
        return True
    ```

=== "Kahn's Algorithm (BFS)"

    ```python
    def canFinish(numCourses, prerequisites):
        """
        Kahn's algorithm using in-degree and BFS.
        Time: O(V+E), Space: O(V+E)
        """
        from collections import deque
        
        # Build graph and calculate in-degrees
        graph = [[] for _ in range(numCourses)]
        in_degree = [0] * numCourses
        
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        # Start with courses having no prerequisites
        queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
        processed = 0
        
        while queue:
            course = queue.popleft()
            processed += 1
            
            # Process all dependent courses
            for dependent in graph[course]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return processed == numCourses
    ```

=== "💡 Tips & Insights"

    **Key Insights:**
    
    - Course scheduling ⟺ cycle detection in directed graph
    - DFS: Use 3-state coloring to detect back edges
    - BFS (Kahn's): Process nodes with zero in-degree iteratively
    - If cycle exists, impossible to complete all courses
    
    **Interview Notes:**
    
    - Ask about self-loops and duplicate prerequisites
    - Two main approaches: DFS cycle detection vs topological sort
    - Kahn's algorithm also produces valid course order
    - Consider follow-up: return actual course schedule

---

## Problem 12: Minimum Height Trees

**Difficulty**: 🟢 Easy  
**Pattern**: Tree Properties (Leaf Removal)  
**Time**: O(V), **Space**: O(V)

=== "Problem Statement"

    For an undirected graph that forms a tree, find all nodes that can serve as root to create minimum height trees. At most 2 such nodes exist.

    **Example:**
    ```text
    Input: n = 4, edges = [[1,0],[1,2],[1,3]]
    Output: [1]
    Explanation: Root at node 1 gives height 1.
    ```

=== "Leaf Removal Algorithm"

    ```python
    def findMinHeightTrees(n, edges):
        """
        Remove leaves iteratively until 1-2 nodes remain.
        These are the centroids with minimum height.
        Time: O(V), Space: O(V)
        """
        if n <= 2:
            return list(range(n))
        
        from collections import deque
        
        # Build adjacency list
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        # Find initial leaves (degree 1)
        leaves = deque([i for i in range(n) if len(graph[i]) == 1])
        remaining = n
        
        # Remove leaves level by level
        while remaining > 2:
            leaf_count = len(leaves)
            remaining -= leaf_count
            
            # Process current level of leaves
            for _ in range(leaf_count):
                leaf = leaves.popleft()
                
                # Remove leaf from its neighbor
                neighbor = graph[leaf][0]
                graph[neighbor].remove(leaf)
                
                # If neighbor becomes leaf, add to queue
                if len(graph[neighbor]) == 1:
                    leaves.append(neighbor)
        
        return list(range(n)) if remaining == n else [leaves[i] for i in range(len(leaves))]
    ```

=== "Centroid Property"

    ```python
    def findMinHeightTrees(n, edges):
        """
        Alternative implementation with cleaner leaf tracking.
        Time: O(V), Space: O(V)
        """
        if n <= 2:
            return list(range(n))
        
        from collections import defaultdict
        
        # Build adjacency list with degree tracking
        graph = defaultdict(set)
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
        
        # Start with leaves
        leaves = [i for i in range(n) if len(graph[i]) == 1]
        
        while n > 2:
            n -= len(leaves)
            new_leaves = []
            
            # Remove current leaves
            for leaf in leaves:
                neighbor = graph[leaf].pop()
                graph[neighbor].remove(leaf)
                
                if len(graph[neighbor]) == 1:
                    new_leaves.append(neighbor)
            
            leaves = new_leaves
        
        return leaves
    ```

=== "💡 Tips & Insights"

    **Key Insights:**
    
    - Tree centroid property: at most 2 nodes minimize max distance
    - Remove leaves iteratively - they can't be optimal roots
    - Last 1-2 remaining nodes are the centroids
    - Linear time algorithm using BFS-like level processing
    
    **Interview Notes:**
    
    - Ask about tree guarantee (connected, n-1 edges)
    - Consider edge cases: n=1, n=2 (special handling)
    - Explain centroid property and why it works
    - Alternative: brute force has O(V²) complexity

---

## Problem 13: Employee Importance

**Difficulty**: 🟢 Easy  
**Pattern**: DFS/BFS  
**Time**: O(V+E), **Space**: O(V)

=== "Problem Statement"

    You have employees with unique IDs and importance values. Each employee has subordinates. Given an employee ID, calculate total importance of the employee and all subordinates.

    **Example:**
    ```text
    Input: employees = [[1,5,[2,3]],[2,3,[]],[3,3,[]]], id = 1
    Output: 11
    Explanation: Employee 1 has importance 5, and subordinates 2,3 with importance 3 each.
    ```

=== "DFS Solution"

    ```python
    class Employee:
        def __init__(self, id, importance, subordinates):
            self.id = id
            self.importance = importance
            self.subordinates = subordinates
    
    def getImportance(employees, id):
        """
        DFS to traverse employee hierarchy.
        Time: O(V+E), Space: O(V)
        """
        # Build employee map for O(1) lookup
        emp_map = {emp.id: emp for emp in employees}
        
        def dfs(emp_id):
            emp = emp_map[emp_id]
            total = emp.importance
            
            # Add importance of all subordinates
            for sub_id in emp.subordinates:
                total += dfs(sub_id)
            
            return total
        
        return dfs(id)
    ```

=== "BFS Solution"

    ```python
    def getImportance(employees, id):
        """
        BFS to traverse employee hierarchy iteratively.
        Time: O(V+E), Space: O(V)
        """
        from collections import deque
        
        # Build employee map
        emp_map = {emp.id: emp for emp in employees}
        
        queue = deque([id])
        total = 0
        
        while queue:
            emp_id = queue.popleft()
            emp = emp_map[emp_id]
            
            total += emp.importance
            
            # Add all subordinates to queue
            for sub_id in emp.subordinates:
                queue.append(sub_id)
        
        return total
    ```

=== "💡 Tips & Insights"

    **Key Insights:**
    
    - Tree/DAG traversal problem in disguise
    - Build lookup map for O(1) employee access
    - Both DFS and BFS work - DFS more natural for hierarchy
    - No cycle detection needed (organizational hierarchy)
    
    **Interview Notes:**
    
    - Ask about data structure format and constraints
    - Consider memory usage for large organizations
    - Clarify if cycles possible (unusual but worth asking)
    - Preprocessing with map improves lookup efficiency

---

## Problem 14: Keys and Rooms

**Difficulty**: 🟢 Easy  
**Pattern**: DFS/BFS  
**Time**: O(V+E), **Space**: O(V)

=== "Problem Statement"

    There are n rooms numbered 0 to n-1. Room 0 is unlocked initially. Each room contains keys to other rooms. Determine if you can visit all rooms.

    **Example:**
    ```text
    Input: rooms = [[1],[2],[3],[]]
    Output: true
    Explanation: Start in room 0, get key to room 1, then 2, then 3.
    ```

=== "DFS Solution"

    ```python
    def canVisitAllRooms(rooms):
        """
        DFS to explore all reachable rooms.
        Time: O(V+E), Space: O(V)
        """
        visited = set()
        
        def dfs(room):
            visited.add(room)
            
            # Use keys to visit other rooms
            for key in rooms[room]:
                if key not in visited:
                    dfs(key)
        
        dfs(0)  # Start from room 0
        return len(visited) == len(rooms)
    ```

=== "BFS Solution"

    ```python
    def canVisitAllRooms(rooms):
        """
        BFS to explore rooms level by level.
        Time: O(V+E), Space: O(V)
        """
        from collections import deque
        
        visited = set([0])
        queue = deque([0])
        
        while queue:
            room = queue.popleft()
            
            # Use keys to visit other rooms
            for key in rooms[room]:
                if key not in visited:
                    visited.add(key)
                    queue.append(key)
        
        return len(visited) == len(rooms)
    ```

=== "Iterative DFS with Stack"

    ```python
    def canVisitAllRooms(rooms):
        """
        Iterative DFS using explicit stack.
        Time: O(V+E), Space: O(V)
        """
        visited = set()
        stack = [0]
        
        while stack:
            room = stack.pop()
            
            if room in visited:
                continue
            
            visited.add(room)
            
            # Add all keys to stack
            for key in rooms[room]:
                if key not in visited:
                    stack.append(key)
        
        return len(visited) == len(rooms)
    ```

=== "💡 Tips & Insights"

    **Key Insights:**
    
    - Graph reachability problem - can reach all nodes from node 0
    - Rooms are nodes, keys are directed edges
    - Standard DFS/BFS traversal with visited tracking
    - Count visited rooms and compare with total
    
    **Interview Notes:**
    
    - Ask about room numbering (0-indexed, consecutive)
    - Consider duplicate keys (doesn't affect algorithm)
    - All three approaches (DFS, BFS, iterative) work equally well
    - Edge case: single room (always true)

---

## Problem 15: Flood Fill

**Difficulty**: 🟢 Easy  
**Pattern**: DFS/BFS on Matrix  
**Time**: O(m×n), **Space**: O(m×n)

=== "Problem Statement"

    Given a 2D image, a starting pixel, and a new color, perform flood fill. Change the color of the starting pixel and all connected pixels of the same original color.

    **Example:**
    ```text
    Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, color = 2
    Output: [[2,2,2],[2,2,0],[2,0,1]]
    ```

=== "DFS Solution"

    ```python
    def floodFill(image, sr, sc, color):
        """
        DFS flood fill starting from given pixel.
        Time: O(m×n), Space: O(m×n)
        """
        if not image or not image[0]:
            return image
        
        rows, cols = len(image), len(image[0])
        original_color = image[sr][sc]
        
        # If new color same as original, no change needed
        if original_color == color:
            return image
        
        def dfs(r, c):
            # Check bounds and color match
            if (r < 0 or r >= rows or c < 0 or c >= cols or 
                image[r][c] != original_color):
                return
            
            # Fill with new color
            image[r][c] = color
            
            # Fill all 4 directions
            dfs(r + 1, c)  # Down
            dfs(r - 1, c)  # Up
            dfs(r, c + 1)  # Right
            dfs(r, c - 1)  # Left
        
        dfs(sr, sc)
        return image
    ```

=== "BFS Solution"

    ```python
    def floodFill(image, sr, sc, color):
        """
        BFS flood fill using queue.
        Time: O(m×n), Space: O(m×n)
        """
        if not image or not image[0]:
            return image
        
        from collections import deque
        
        rows, cols = len(image), len(image[0])
        original_color = image[sr][sc]
        
        if original_color == color:
            return image
        
        queue = deque([(sr, sc)])
        image[sr][sc] = color
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            r, c = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and 
                    image[nr][nc] == original_color):
                    image[nr][nc] = color
                    queue.append((nr, nc))
        
        return image
    ```

=== "💡 Tips & Insights"

    **Key Insights:**
    
    - Matrix traversal problem similar to island counting
    - Check if new color equals original to avoid infinite loops
    - Both DFS and BFS work - DFS more memory efficient
    - 4-directional connectivity (not 8-directional)
    
    **Interview Notes:**
    
    - Ask about in-place modification vs creating new image
    - Clarify connectivity type (4-way vs 8-way)
    - Consider edge case: new color equals original color
    - Time complexity always O(m×n) in worst case

---

## 🎯 Practice Recommendations

### **Easy Level Mastery Path:**

1. **Start with grid problems** (1, 15) - foundational DFS/BFS
2. **Basic connectivity** (2, 8, 9) - core graph traversal
3. **Property-based** (5, 6, 12) - leverage graph characteristics
4. **Advanced patterns** (7, 10, 11) - union-find and coloring

### **Key Patterns to Master:**

- **DFS/BFS**: Universal graph traversal tools
- **Union-Find**: Dynamic connectivity queries
- **Visited tracking**: Avoid cycles and infinite loops
- **State machines**: 3-state coloring for cycle detection

### **Interview Strategy:**

- Always ask about graph representation
- Consider edge cases (empty, single node, disconnected)
- Choose algorithm based on problem constraints
- Practice both recursive and iterative approaches
