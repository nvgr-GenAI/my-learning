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

=== "Problem Statement"
    **Difficulty**: 🟡 Medium  
    **Pattern**: Topological Sort / Cycle Detection  
    **Time**: O(V+E), **Space**: O(V+E)

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

=== "Solution"
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
            graph[prereq].append(course)
        
        # Color states: 0=white, 1=gray, 2=black
        color = [0] * numCourses
        
        def has_cycle(node):
            if color[node] == 1:  # Gray - cycle detected
                return True
            if color[node] == 2:  # Black - already processed
                return False
            
            # Mark as gray (visiting)
            color[node] = 1
            
            # Visit all neighbors
            for neighbor in graph[node]:
                if has_cycle(neighbor):
                    return True
            
            # Mark as black (processed)
            color[node] = 2
            return False
        
        # Check each component
        for i in range(numCourses):
            if color[i] == 0 and has_cycle(i):
                return False
        
        return True
    ```

=== "Alternative Solutions"
    **Kahn's Algorithm (BFS Topological Sort):**
    ```python
    def canFinish(numCourses, prerequisites):
        from collections import deque, defaultdict
        
        # Build graph and indegree
        graph = defaultdict(list)
        indegree = [0] * numCourses
        
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            indegree[course] += 1
        
        # BFS with nodes having indegree 0
        queue = deque([i for i in range(numCourses) if indegree[i] == 0])
        processed = 0
        
        while queue:
            node = queue.popleft()
            processed += 1
            
            for neighbor in graph[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
        
        return processed == numCourses
    ```

=== "Tips & Insights"
    **Pattern Recognition:**
    - Prerequisite problems → Topological sort
    - Cycle detection needed → Use DFS with coloring or BFS with indegree
    - Directed graph problems often need state tracking

    **Common Mistakes:**
    - Forgetting to handle disconnected components
    - Mixing up course and prerequisite order
    - Not properly resetting visited states

    **Interview Tips:**
    - Explain both DFS and BFS approaches
    - Discuss time/space complexity trade-offs
    - Mention real-world applications (build systems, academic planning)

---

## Problem 2: Shortest Path in Binary Matrix

=== "Problem Statement"
    **Difficulty**: 🟡 Medium  
    **Pattern**: BFS / Shortest Path  
    **Time**: O(N²), **Space**: O(N²)

    Given an `n x n` binary matrix `grid`, return the length of the shortest clear path from top-left to bottom-right. If no such path exists, return `-1`.

    A clear path is a path from `(0, 0)` to `(n-1, n-1)` where all visited cells are `0`. You can move 8-directionally.

    **Example:**
    ```
    Input: grid = [[0,0,0],[1,1,0],[1,1,0]]
    Output: 4
    Explanation: Path is (0,0)→(0,1)→(0,2)→(1,2)→(2,2)
    ```

=== "Solution"
    ```python
    def shortestPathBinaryMatrix(grid):
        from collections import deque
        
        n = len(grid)
        if grid[0][0] == 1 or grid[n-1][n-1] == 1:
            return -1
        
        # 8 directions
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        
        queue = deque([(0, 0, 1)])  # (row, col, path_length)
        visited = {(0, 0)}
        
        while queue:
            row, col, path_len = queue.popleft()
            
            if row == n-1 and col == n-1:
                return path_len
            
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                
                if (0 <= nr < n and 0 <= nc < n and 
                    grid[nr][nc] == 0 and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc, path_len + 1))
        
        return -1
    ```

=== "Alternative Solutions"
    **A* Algorithm (More efficient for large grids):**
    ```python
    def shortestPathBinaryMatrix(grid):
        import heapq
        
        n = len(grid)
        if grid[0][0] == 1 or grid[n-1][n-1] == 1:
            return -1
        
        def heuristic(row, col):
            return max(abs(row - (n-1)), abs(col - (n-1)))
        
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        heap = [(1 + heuristic(0, 0), 1, 0, 0)]  # (f_score, path_len, row, col)
        visited = {(0, 0)}
        
        while heap:
            _, path_len, row, col = heapq.heappop(heap)
            
            if row == n-1 and col == n-1:
                return path_len
            
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                
                if (0 <= nr < n and 0 <= nc < n and 
                    grid[nr][nc] == 0 and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    f_score = path_len + 1 + heuristic(nr, nc)
                    heapq.heappush(heap, (f_score, path_len + 1, nr, nc))
        
        return -1
    ```

=== "Tips & Insights"
    **Pattern Recognition:**
    - Shortest path in unweighted graph → BFS
    - 8-directional movement → Update direction vectors
    - Early termination possible → Check goal on dequeue

    **Common Mistakes:**
    - Forgetting to check start/end cells for obstacles
    - Using wrong direction vectors
    - Not handling edge cases (single cell, blocked path)

    **Interview Tips:**
    - Explain why BFS gives shortest path in unweighted graphs
    - Discuss A* optimization for larger grids
    - Mention space optimization with bidirectional BFS

---

## Problem 3: Network Delay Time

=== "Problem Statement"
    **Difficulty**: 🟡 Medium  
    **Pattern**: Dijkstra's Algorithm / Shortest Path  
    **Time**: O(E log V), **Space**: O(V + E)

    You are given a network of `n` nodes, labeled from `1` to `n`. You are also given `times`, a list of travel times as directed edges `times[i] = (ui, vi, wi)`, where `ui` is the source node, `vi` is the target node, and `wi` is the time it takes for a signal to travel from source to target.

    We will send a signal from a given node `k`. Return the time it takes for all the `n` nodes to receive the signal. If it is impossible for all the nodes to receive the signal, return `-1`.

    **Example:**
    ```
    Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
    Output: 2
    Explanation: Signal path: 2→1 (time 1), 2→3→4 (time 2)
    ```

=== "Solution"
    ```python
    def networkDelayTime(times, n, k):
        import heapq
        from collections import defaultdict
        
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
    ```

=== "Alternative Solutions"
    **Bellman-Ford Algorithm:**
    ```python
    def networkDelayTime(times, n, k):
        distances = [float('inf')] * (n + 1)
        distances[k] = 0
        
        # Relax edges n-1 times
        for _ in range(n - 1):
            for u, v, w in times:
                if distances[u] + w < distances[v]:
                    distances[v] = distances[u] + w
        
        # Find maximum distance
        max_dist = 0
        for i in range(1, n + 1):
            if distances[i] == float('inf'):
                return -1
            max_dist = max(max_dist, distances[i])
        
        return max_dist
    ```

=== "Tips & Insights"
    **Pattern Recognition:**
    - Shortest path with positive weights → Dijkstra's
    - Single source to all nodes → Classic shortest path problem
    - Find maximum of all shortest paths → Broadcasting scenario

    **Common Mistakes:**
    - Not handling unreachable nodes properly
    - Forgetting that nodes are 1-indexed
    - Using wrong data structures for priority queue

    **Interview Tips:**
    - Compare Dijkstra vs Bellman-Ford trade-offs
    - Explain why we need maximum of shortest distances
    - Discuss applications in network protocols

---

## Problem 4: Is Graph Bipartite?

=== "Problem Statement"
    **Difficulty**: 🟡 Medium  
    **Pattern**: Graph Coloring / BFS/DFS  
    **Time**: O(V + E), **Space**: O(V)

    There is an undirected graph with `n` nodes, where each node is numbered between `0` and `n - 1`. You are given a 2D array `graph`, where `graph[i]` is an array of the nodes that are adjacent to node `i`.

    Return `true` if and only if it is bipartite.

    A graph is bipartite if we can split its set of nodes into two independent subsets A and B such that every edge in the graph connects a node in set A and a node in set B.

    **Example:**
    ```
    Input: graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
    Output: false
    Explanation: Cannot partition into two sets without adjacent nodes in same set.
    ```

=== "Solution"
    ```python
    def isBipartite(graph):
        """
        Color nodes with alternating colors.
        If we can color without conflicts, graph is bipartite.
        """
        n = len(graph)
        colors = {}
        
        def dfs(node, color):
            if node in colors:
                return colors[node] == color
            
            colors[node] = color
            
            # Color all neighbors with opposite color
            for neighbor in graph[node]:
                if not dfs(neighbor, 1 - color):
                    return False
            
            return True
        
        # Check each component
        for i in range(n):
            if i not in colors:
                if not dfs(i, 0):
                    return False
        
        return True
    ```

=== "Alternative Solutions"
    **BFS Approach:**
    ```python
    def isBipartite(graph):
        from collections import deque
        
        n = len(graph)
        colors = {}
        
        for start in range(n):
            if start in colors:
                continue
            
            queue = deque([start])
            colors[start] = 0
            
            while queue:
                node = queue.popleft()
                
                for neighbor in graph[node]:
                    if neighbor in colors:
                        if colors[neighbor] == colors[node]:
                            return False
                    else:
                        colors[neighbor] = 1 - colors[node]
                        queue.append(neighbor)
        
        return True
    ```

=== "Tips & Insights"
    **Pattern Recognition:**
    - Two-coloring problem → Bipartite checking
    - Alternating properties → Graph coloring
    - Conflict detection → Use DFS/BFS with state

    **Common Mistakes:**
    - Not handling disconnected components
    - Forgetting to check color conflicts properly
    - Mixing up color assignment logic

    **Interview Tips:**
    - Explain bipartite graph definition clearly
    - Discuss applications (matching problems, scheduling)
    - Compare DFS vs BFS approaches

---

## Problem 5: Surrounded Regions

=== "Problem Statement"
    **Difficulty**: 🟡 Medium  
    **Pattern**: DFS/BFS / Border Traversal  
    **Time**: O(M×N), **Space**: O(M×N)

    Given an `m x n` matrix `board` containing `'X'` and `'O'`, capture all regions that are 4-directionally surrounded by `'X'`.

    A region is captured by flipping all `'O'`s into `'X'`s in that surrounded region.

    **Example:**
    ```
    Input: board = [["X","X","X","X"],
                   ["X","O","O","X"],
                   ["X","X","O","X"],
                   ["X","O","X","X"]]
    Output: [["X","X","X","X"],
            ["X","X","X","X"],
            ["X","X","X","X"],
            ["X","O","X","X"]]
    ```

=== "Solution"
    ```python
    def solve(board):
        """
        1. Mark all 'O's connected to border as safe
        2. Convert remaining 'O's to 'X's
        3. Restore safe 'O's
        """
        if not board or not board[0]:
            return
        
        m, n = len(board), len(board[0])
        
        def dfs(row, col):
            if (row < 0 or row >= m or col < 0 or col >= n or 
                board[row][col] != 'O'):
                return
            
            board[row][col] = 'S'  # Mark as safe
            
            # Check 4 directions
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                dfs(row + dr, col + dc)
        
        # Mark border-connected 'O's as safe
        for i in range(m):
            if board[i][0] == 'O':
                dfs(i, 0)
            if board[i][n-1] == 'O':
                dfs(i, n-1)
        
        for j in range(n):
            if board[0][j] == 'O':
                dfs(0, j)
            if board[m-1][j] == 'O':
                dfs(m-1, j)
        
        # Convert remaining 'O's to 'X's and restore safe 'O's
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == 'S':
                    board[i][j] = 'O'
    ```

=== "Alternative Solutions"
    **BFS Approach:**
    ```python
    def solve(board):
        from collections import deque
        
        if not board or not board[0]:
            return
        
        m, n = len(board), len(board[0])
        queue = deque()
        
        # Find all border 'O's
        for i in range(m):
            for j in range(n):
                if ((i == 0 or i == m-1 or j == 0 or j == n-1) and 
                    board[i][j] == 'O'):
                    queue.append((i, j))
                    board[i][j] = 'S'
        
        # BFS to mark all connected 'O's as safe
        directions = [(0,1), (0,-1), (1,0), (-1,0)]
        while queue:
            row, col = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if (0 <= nr < m and 0 <= nc < n and board[nr][nc] == 'O'):
                    board[nr][nc] = 'S'
                    queue.append((nr, nc))
        
        # Final conversion
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == 'S':
                    board[i][j] = 'O'
    ```

=== "Tips & Insights"
    **Pattern Recognition:**
    - Border-connected elements are special → Start from borders
    - Need to preserve some elements → Use temporary marking
    - Region-based problems → DFS/BFS traversal

    **Common Mistakes:**
    - Not starting from border cells
    - Modifying board while traversing
    - Forgetting to restore safe elements

    **Interview Tips:**
    - Explain the "reverse thinking" approach
    - Discuss in-place vs extra space solutions
    - Mention applications in image processing

---

## Problem 6: Find the Town Judge

=== "Problem Statement"
    **Difficulty**: 🟡 Medium  
    **Pattern**: Graph Properties / In-degree/Out-degree  
    **Time**: O(E), **Space**: O(V)

    In a town, there are `n` people labeled from `1` to `n`. There is a rumor that one of these people is secretly the town judge.

    If the town judge exists, then:
    1. The town judge trusts nobody.
    2. Everybody (except for the town judge) trusts the town judge.
    3. There is exactly one person that satisfies properties 1 and 2.

    You are given an array `trust` where `trust[i] = [ai, bi]` representing that person `ai` trusts person `bi`.

    Return the label of the town judge if the town judge exists and can be identified, or return `-1` otherwise.

    **Example:**
    ```
    Input: n = 3, trust = [[1,3],[2,3]]
    Output: 3
    Explanation: Person 3 is trusted by 1 and 2, and trusts nobody.
    ```

=== "Solution"
    ```python
    def findJudge(n, trust):
        """
        Judge has in-degree n-1 and out-degree 0
        """
        if n == 1:
            return 1 if not trust else -1
        
        in_degree = [0] * (n + 1)
        out_degree = [0] * (n + 1)
        
        for a, b in trust:
            out_degree[a] += 1
            in_degree[b] += 1
        
        for i in range(1, n + 1):
            if in_degree[i] == n - 1 and out_degree[i] == 0:
                return i
        
        return -1
    ```

=== "Alternative Solutions"
    **Space-optimized approach:**
    ```python
    def findJudge(n, trust):
        """
        Use net trust score: in_degree - out_degree
        Judge has score n-1
        """
        if n == 1:
            return 1 if not trust else -1
        
        trust_score = [0] * (n + 1)
        
        for a, b in trust:
            trust_score[a] -= 1  # Trusts someone (out-degree)
            trust_score[b] += 1  # Is trusted (in-degree)
        
        for i in range(1, n + 1):
            if trust_score[i] == n - 1:
                return i
        
        return -1
    ```

=== "Tips & Insights"
    **Pattern Recognition:**
    - Authority/centrality problems → Check in-degree/out-degree
    - Unique node with special properties → Graph vertex analysis
    - Trust relationships → Directed graph modeling

    **Common Mistakes:**
    - Not handling n=1 edge case
    - Counting degrees incorrectly
    - Not checking both conditions (in-degree AND out-degree)

    **Interview Tips:**
    - Explain graph representation of trust relationships
    - Discuss what in-degree and out-degree represent
    - Mention applications in social networks, authority ranking

---

## Problem 7: Rotting Oranges

=== "Problem Statement"
    **Difficulty**: 🟡 Medium  
    **Pattern**: Multi-source BFS  
    **Time**: O(M×N), **Space**: O(M×N)

    You are given an `m x n` grid where each cell can have one of three values:
    - `0` representing an empty cell,
    - `1` representing a fresh orange, or
    - `2` representing a rotten orange.

    Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

    Return the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return `-1`.

    **Example:**
    ```
    Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
    Output: 4
    Explanation: Oranges rot in this order: minute 1, 2, 3, 4
    ```

=== "Solution"
    ```python
    def orangesRotting(grid):
        from collections import deque
        
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        queue = deque()
        fresh_count = 0
        
        # Find all initially rotten oranges and count fresh ones
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    queue.append((i, j, 0))  # (row, col, time)
                elif grid[i][j] == 1:
                    fresh_count += 1
        
        if fresh_count == 0:
            return 0
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        max_time = 0
        
        while queue:
            row, col, time = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                
                if (0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1):
                    grid[nr][nc] = 2  # Make it rotten
                    fresh_count -= 1
                    max_time = time + 1
                    queue.append((nr, nc, time + 1))
        
        return max_time if fresh_count == 0 else -1
    ```

=== "Alternative Solutions"
    **Without modifying input:**
    ```python
    def orangesRotting(grid):
        from collections import deque
        
        m, n = len(grid), len(grid[0])
        queue = deque()
        fresh_oranges = set()
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    queue.append((i, j, 0))
                elif grid[i][j] == 1:
                    fresh_oranges.add((i, j))
        
        if not fresh_oranges:
            return 0
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        max_time = 0
        
        while queue:
            row, col, time = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                
                if (nr, nc) in fresh_oranges:
                    fresh_oranges.remove((nr, nc))
                    max_time = time + 1
                    queue.append((nr, nc, time + 1))
        
        return max_time if not fresh_oranges else -1
    ```

=== "Tips & Insights"
    **Pattern Recognition:**
    - Multiple starting points → Multi-source BFS
    - Time-based propagation → BFS with time tracking
    - Check if all nodes reachable → Count remaining targets

    **Common Mistakes:**
    - Forgetting to handle case where no fresh oranges exist
    - Not tracking time correctly in BFS
    - Not checking if all fresh oranges were reached

    **Interview Tips:**
    - Explain multi-source BFS concept
    - Discuss why BFS gives minimum time
    - Mention applications in disease spread, fire propagation

---

## Problem 8: Clone Graph

=== "Problem Statement"
    **Difficulty**: 🟡 Medium  
    **Pattern**: Graph Traversal / Deep Copy  
    **Time**: O(V + E), **Space**: O(V)

    Given a reference of a node in a connected undirected graph, return a deep copy (clone) of the graph.

    Each node in the graph contains a value (`int`) and a list (`List[Node]`) of its neighbors.

    **Example:**
    ```
    Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
    Output: [[2,4],[1,3],[2,4],[1,3]]
    Explanation: Graph with 4 nodes cloned successfully.
    ```

=== "Solution"
    ```python
    def cloneGraph(node):
        """
        DFS approach with memoization
        """
        if not node:
            return None
        
        cloned = {}
        
        def dfs(original):
            if original in cloned:
                return cloned[original]
            
            # Create clone for current node
            clone = Node(original.val)
            cloned[original] = clone
            
            # Clone all neighbors
            for neighbor in original.neighbors:
                clone.neighbors.append(dfs(neighbor))
            
            return clone
        
        return dfs(node)
    ```

=== "Alternative Solutions"
    **BFS Approach:**
    ```python
    def cloneGraph(node):
        from collections import deque
        
        if not node:
            return None
        
        cloned = {node: Node(node.val)}
        queue = deque([node])
        
        while queue:
            original = queue.popleft()
            
            for neighbor in original.neighbors:
                if neighbor not in cloned:
                    cloned[neighbor] = Node(neighbor.val)
                    queue.append(neighbor)
                
                cloned[original].neighbors.append(cloned[neighbor])
        
        return cloned[node]
    ```

=== "Tips & Insights"
    **Pattern Recognition:**
    - Deep copy with references → Use hash map for memoization
    - Graph traversal → DFS or BFS
    - Avoid infinite loops → Track visited nodes

    **Common Mistakes:**
    - Creating multiple clones for same node
    - Not handling None input
    - Forgetting to clone neighbor relationships

    **Interview Tips:**
    - Explain why we need memoization
    - Compare DFS vs BFS approaches
    - Discuss space complexity considerations

---

## Problem 9: Word Ladder

=== "Problem Statement"
    **Difficulty**: 🟡 Medium  
    **Pattern**: BFS / Shortest Path  
    **Time**: O(M²×N), **Space**: O(M²×N)

    A transformation sequence from word `beginWord` to word `endWord` using a dictionary `wordList` is a sequence of words where:
    1. The first word is `beginWord`
    2. The last word is `endWord`
    3. Only one letter is different between consecutive words
    4. Every intermediate word exists in `wordList`

    Return the length of the shortest transformation sequence. If no such sequence exists, return `0`.

    **Example:**
    ```
    Input: beginWord = "hit", endWord = "cog", 
           wordList = ["hot","dot","dog","lot","log","cog"]
    Output: 5
    Explanation: "hit" → "hot" → "dot" → "dog" → "cog"
    ```

=== "Solution"
    ```python
    def ladderLength(beginWord, endWord, wordList):
        from collections import deque
        
        if endWord not in wordList:
            return 0
        
        wordSet = set(wordList)
        queue = deque([(beginWord, 1)])
        visited = {beginWord}
        
        while queue:
            word, length = queue.popleft()
            
            if word == endWord:
                return length
            
            # Try all possible single character changes
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c != word[i]:
                        new_word = word[:i] + c + word[i+1:]
                        
                        if new_word in wordSet and new_word not in visited:
                            visited.add(new_word)
                            queue.append((new_word, length + 1))
        
        return 0
    ```

=== "Alternative Solutions"
    **Bidirectional BFS:**
    ```python
    def ladderLength(beginWord, endWord, wordList):
        if endWord not in wordList:
            return 0
        
        wordSet = set(wordList)
        if beginWord in wordSet:
            wordSet.remove(beginWord)
        
        front = {beginWord}
        back = {endWord}
        length = 1
        
        while front:
            length += 1
            next_front = set()
            
            for word in front:
                for i in range(len(word)):
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        if c != word[i]:
                            new_word = word[:i] + c + word[i+1:]
                            
                            if new_word in back:
                                return length
                            
                            if new_word in wordSet:
                                next_front.add(new_word)
                                wordSet.remove(new_word)
            
            front = next_front
            if len(front) > len(back):
                front, back = back, front
        
        return 0
    ```

=== "Tips & Insights"
    **Pattern Recognition:**
    - Shortest path in unweighted graph → BFS
    - Word transformation → Generate neighbors by changing characters
    - Optimization possible → Bidirectional BFS

    **Common Mistakes:**
    - Not checking if endWord exists in wordList
    - Generating invalid neighbors
    - Not using visited set (infinite loops)

    **Interview Tips:**
    - Explain why BFS gives shortest path
    - Discuss bidirectional BFS optimization
    - Mention time complexity depends on word length and alphabet size

---

## Problem 10: Minimum Height Trees

=== "Problem Statement"
    **Difficulty**: 🟡 Medium  
    **Pattern**: Topological Sort / Tree Centers  
    **Time**: O(V), **Space**: O(V)

    A tree is an undirected graph in which any two vertices are connected by exactly one path. Given such a tree with `n` nodes labeled from `0` to `n - 1` and an array of `n - 1` edges, find all possible roots such that the resulting rooted tree has minimum height.

    **Example:**
    ```
    Input: n = 4, edges = [[1,0],[1,2],[1,3]]
    Output: [1]
    Explanation: Root at node 1 gives tree height 1, others give height 2.
    ```

=== "Solution"
    ```python
    def findMinHeightTrees(n, edges):
        """
        Tree centers are the roots that minimize height.
        Remove leaf nodes iteratively until 1 or 2 nodes remain.
        """
        if n <= 2:
            return list(range(n))
        
        from collections import defaultdict, deque
        
        # Build adjacency list
        graph = defaultdict(set)
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
        
        # Initialize leaves (nodes with degree 1)
        leaves = deque([i for i in range(n) if len(graph[i]) == 1])
        remaining = n
        
        # Remove leaves layer by layer
        while remaining > 2:
            leaf_count = len(leaves)
            remaining -= leaf_count
            
            for _ in range(leaf_count):
                leaf = leaves.popleft()
                
                # Remove leaf from its neighbor
                for neighbor in graph[leaf]:
                    graph[neighbor].remove(leaf)
                    if len(graph[neighbor]) == 1:
                        leaves.append(neighbor)
        
        return list(leaves)
    ```

=== "Alternative Solutions"
    **DFS from each node (Brute Force):**
    ```python
    def findMinHeightTrees(n, edges):
        from collections import defaultdict
        
        if n <= 2:
            return list(range(n))
        
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def get_height(root):
            visited = {root}
            
            def dfs(node):
                max_depth = 0
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        max_depth = max(max_depth, 1 + dfs(neighbor))
                        visited.remove(neighbor)
                return max_depth
            
            return dfs(root)
        
        min_height = float('inf')
        result = []
        
        for i in range(n):
            height = get_height(i)
            if height < min_height:
                min_height = height
                result = [i]
            elif height == min_height:
                result.append(i)
        
        return result
    ```

=== "Tips & Insights"
    **Pattern Recognition:**
    - Tree center problem → Remove leaves iteratively
    - Minimize tree height → Find centroid(s)
    - At most 2 centers in any tree → Mathematical property

    **Common Mistakes:**
    - Not handling edge cases (n ≤ 2)
    - Incorrect leaf removal logic
    - Not understanding why there are at most 2 centers

    **Interview Tips:**
    - Explain tree center concept
    - Discuss why iterative leaf removal works
    - Mention applications in network design, hierarchical structures

---

## Problem 11: Course Schedule II

=== "Problem Statement"
    **Difficulty**: 🟡 Medium  
    **Pattern**: Topological Sort  
    **Time**: O(V + E), **Space**: O(V + E)

    There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [ai, bi]` indicates that you must take course `bi` first if you want to take course `ai`.

    Return the ordering of courses you should take to finish all courses. If there are many valid answers, return any of them. If it is impossible to finish all courses, return an empty array.

    **Example:**
    ```
    Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
    Output: [0,2,1,3] or [0,1,2,3]
    Explanation: Course 0 has no prerequisites. Courses 1,2 depend on 0. Course 3 depends on 1,2.
    ```

=== "Solution"
    ```python
    def findOrder(numCourses, prerequisites):
        """
        Kahn's Algorithm for topological sorting
        """
        from collections import defaultdict, deque
        
        # Build graph and indegree array
        graph = defaultdict(list)
        indegree = [0] * numCourses
        
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            indegree[course] += 1
        
        # Start with courses having no prerequisites
        queue = deque([i for i in range(numCourses) if indegree[i] == 0])
        result = []
        
        while queue:
            course = queue.popleft()
            result.append(course)
            
            # Remove this course and update indegrees
            for next_course in graph[course]:
                indegree[next_course] -= 1
                if indegree[next_course] == 0:
                    queue.append(next_course)
        
        return result if len(result) == numCourses else []
    ```

=== "Alternative Solutions"
    **DFS-based Topological Sort:**
    ```python
    def findOrder(numCourses, prerequisites):
        graph = {i: [] for i in range(numCourses)}
        for course, prereq in prerequisites:
            graph[prereq].append(course)
        
        # 0: white, 1: gray, 2: black
        color = [0] * numCourses
        result = []
        
        def dfs(node):
            if color[node] == 1:  # Cycle detected
                return False
            if color[node] == 2:  # Already processed
                return True
            
            color[node] = 1  # Mark as visiting
            
            for neighbor in graph[node]:
                if not dfs(neighbor):
                    return False
            
            color[node] = 2  # Mark as processed
            result.append(node)
            return True
        
        for i in range(numCourses):
            if color[i] == 0 and not dfs(i):
                return []
        
        return result[::-1]  # Reverse for correct order
    ```

=== "Tips & Insights"
    **Pattern Recognition:**
    - Need actual ordering → Topological sort (not just cycle detection)
    - Dependency resolution → Build and process graph
    - Multiple valid answers → Any valid topological order works

    **Common Mistakes:**
    - Forgetting to reverse DFS result
    - Not handling cycles properly
    - Mixing up course and prerequisite order

    **Interview Tips:**
    - Compare Kahn's vs DFS approaches
    - Explain why we need topological sorting
    - Discuss applications in build systems, task scheduling

---

## Problem 12: Pacific Atlantic Water Flow

=== "Problem Statement"
    **Difficulty**: 🟡 Medium  
    **Pattern**: Multi-source DFS/BFS  
    **Time**: O(M×N), **Space**: O(M×N)

    There is an `m x n` rectangular island that borders both the Pacific Ocean and Atlantic Ocean. The Pacific Ocean touches the island's left and top edges, and the Atlantic Ocean touches the island's right and bottom edges.

    The island is partitioned into a grid of square cells. You are given an `m x n` integer matrix `heights` where `heights[r][c]` represents the height above sea level of the cell at coordinate `(r, c)`.

    Return a list of grid coordinates where water can flow to both the Pacific and Atlantic oceans.

    **Example:**
    ```
    Input: heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
    Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
    ```

=== "Solution"
    ```python
    def pacificAtlantic(heights):
        """
        Work backwards: start from oceans and find reachable cells
        """
        if not heights or not heights[0]:
            return []
        
        m, n = len(heights), len(heights[0])
        
        def dfs(row, col, reachable, prev_height):
            if (row < 0 or row >= m or col < 0 or col >= n or
                (row, col) in reachable or heights[row][col] < prev_height):
                return
            
            reachable.add((row, col))
            
            # Explore 4 directions
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                dfs(row + dr, col + dc, reachable, heights[row][col])
        
        pacific = set()
        atlantic = set()
        
        # Start DFS from Pacific borders (top and left)
        for i in range(m):
            dfs(i, 0, pacific, heights[i][0])
        for j in range(n):
            dfs(0, j, pacific, heights[0][j])
        
        # Start DFS from Atlantic borders (bottom and right)
        for i in range(m):
            dfs(i, n-1, atlantic, heights[i][n-1])
        for j in range(n):
            dfs(m-1, j, atlantic, heights[m-1][j])
        
        # Find intersection
        return list(pacific & atlantic)
    ```

=== "Alternative Solutions"
    **BFS Approach:**
    ```python
    def pacificAtlantic(heights):
        from collections import deque
        
        if not heights or not heights[0]:
            return []
        
        m, n = len(heights), len(heights[0])
        
        def bfs(queue, reachable):
            while queue:
                row, col = queue.popleft()
                
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = row + dr, col + dc
                    
                    if (0 <= nr < m and 0 <= nc < n and 
                        (nr, nc) not in reachable and
                        heights[nr][nc] >= heights[row][col]):
                        reachable.add((nr, nc))
                        queue.append((nr, nc))
        
        pacific = set()
        atlantic = set()
        pacific_queue = deque()
        atlantic_queue = deque()
        
        # Initialize border cells
        for i in range(m):
            pacific.add((i, 0))
            pacific_queue.append((i, 0))
            atlantic.add((i, n-1))
            atlantic_queue.append((i, n-1))
        
        for j in range(n):
            pacific.add((0, j))
            pacific_queue.append((0, j))
            atlantic.add((m-1, j))
            atlantic_queue.append((m-1, j))
        
        bfs(pacific_queue, pacific)
        bfs(atlantic_queue, atlantic)
        
        return list(pacific & atlantic)
    ```

=== "Tips & Insights"
    **Pattern Recognition:**
    - Multiple targets → Work backwards from targets
    - Water flow problems → Think about reverse flow
    - Intersection of reachable sets → Two separate traversals

    **Common Mistakes:**
    - Working forward instead of backward
    - Incorrect boundary conditions
    - Not handling height comparison properly

    **Interview Tips:**
    - Explain the "reverse thinking" approach
    - Discuss why we start from oceans
    - Compare DFS vs BFS trade-offs

---

## Problem 13: Accounts Merge

=== "Problem Statement"
    **Difficulty**: 🟡 Medium  
    **Pattern**: Union-Find / DFS  
    **Time**: O(N log N), **Space**: O(N)

    Given a list of `accounts` where each element `accounts[i]` is a list of strings, where the first element `accounts[i][0]` is a name, and the rest of the elements are emails representing emails of the account.

    Merge accounts that belong to the same person. Two accounts belong to the same person if there is some common email to both accounts.

    **Example:**
    ```
    Input: accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],
                      ["John","johnsmith@mail.com","john00@mail.com"],
                      ["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
    Output: [["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],
            ["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
    ```

=== "Solution"
    ```python
    def accountsMerge(accounts):
        """
        Union-Find approach
        """
        class UnionFind:
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0] * n
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return
                if self.rank[px] < self.rank[py]:
                    px, py = py, px
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
        
        uf = UnionFind(len(accounts))
        email_to_account = {}
        
        # Map emails to account indices
        for i, account in enumerate(accounts):
            for email in account[1:]:
                if email in email_to_account:
                    uf.union(i, email_to_account[email])
                else:
                    email_to_account[email] = i
        
        # Group emails by root account
        from collections import defaultdict
        merged = defaultdict(set)
        for email, account_idx in email_to_account.items():
            root = uf.find(account_idx)
            merged[root].add(email)
        
        # Build result
        result = []
        for account_idx, emails in merged.items():
            name = accounts[account_idx][0]
            result.append([name] + sorted(emails))
        
        return result
    ```

=== "Alternative Solutions"
    **DFS Approach:**
    ```python
    def accountsMerge(accounts):
        from collections import defaultdict
        
        # Build email to accounts mapping
        email_to_accounts = defaultdict(list)
        for i, account in enumerate(accounts):
            for email in account[1:]:
                email_to_accounts[email].append(i)
        
        visited = set()
        result = []
        
        def dfs(account_idx, emails):
            if account_idx in visited:
                return
            
            visited.add(account_idx)
            
            for email in accounts[account_idx][1:]:
                emails.add(email)
                for next_account in email_to_accounts[email]:
                    dfs(next_account, emails)
        
        for i in range(len(accounts)):
            if i not in visited:
                emails = set()
                dfs(i, emails)
                name = accounts[i][0]
                result.append([name] + sorted(emails))
        
        return result
    ```

=== "Tips & Insights"
    **Pattern Recognition:**
    - Grouping connected components → Union-Find or DFS
    - Common elements create connections → Build graph from shared items
    - Merge requirement → Connected components problem

    **Common Mistakes:**
    - Not handling multiple accounts with same name properly
    - Forgetting to sort emails in result
    - Incorrect graph construction from shared emails

    **Interview Tips:**
    - Compare Union-Find vs DFS approaches
    - Explain how shared emails create connections
    - Discuss applications in social networks, data deduplication

---

## Problem 14: Cheapest Flights Within K Stops

=== "Problem Statement"
    **Difficulty**: 🟡 Medium  
    **Pattern**: Modified Dijkstra / Bellman-Ford  
    **Time**: O(E × K), **Space**: O(V)

    There are `n` cities connected by some number of flights. You are given an array `flights` where `flights[i] = [fromi, toi, pricei]` indicates that there is a flight from city `fromi` to city `toi` with cost `pricei`.

    You are also given three integers `src`, `dst`, and `k`, return the cheapest price from `src` to `dst` with at most `k` stops. If there is no such route, return `-1`.

    **Example:**
    ```
    Input: n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], 
           src = 0, dst = 3, k = 1
    Output: 700
    Explanation: Optimal path: 0→1→3 with cost 100+600=700 (1 stop)
    ```

=== "Solution"
    ```python
    def findCheapestPrice(n, flights, src, dst, k):
        """
        Bellman-Ford with at most k+1 edges
        """
        # Initialize distances
        distances = [float('inf')] * n
        distances[src] = 0
        
        # Relax edges at most k+1 times
        for _ in range(k + 1):
            temp_distances = distances[:]
            
            for u, v, price in flights:
                if distances[u] + price < temp_distances[v]:
                    temp_distances[v] = distances[u] + price
            
            distances = temp_distances
        
        return distances[dst] if distances[dst] != float('inf') else -1
    ```

=== "Alternative Solutions"
    **Modified Dijkstra with stops tracking:**
    ```python
    def findCheapestPrice(n, flights, src, dst, k):
        import heapq
        from collections import defaultdict
        
        graph = defaultdict(list)
        for u, v, price in flights:
            graph[u].append((v, price))
        
        # (cost, node, stops_used)
        heap = [(0, src, 0)]
        visited = {}
        
        while heap:
            cost, node, stops = heapq.heappop(heap)
            
            if node == dst:
                return cost
            
            if stops > k:
                continue
            
            # Skip if we've seen this (node, stops) with better cost
            if (node, stops) in visited and visited[(node, stops)] <= cost:
                continue
            
            visited[(node, stops)] = cost
            
            for neighbor, price in graph[node]:
                if stops + 1 <= k + 1:  # Can still make stops
                    heapq.heappush(heap, (cost + price, neighbor, stops + 1))
        
        return -1
    ```

=== "Tips & Insights"
    **Pattern Recognition:**
    - Shortest path with constraints → Modified shortest path algorithms
    - Limited steps/stops → Bellman-Ford or constrained Dijkstra
    - State includes both position and constraint → Multi-dimensional DP

    **Common Mistakes:**
    - Not handling the "at most k stops" constraint properly
    - Incorrect termination conditions
    - Not considering that k stops means k+1 edges

    **Interview Tips:**
    - Explain why standard Dijkstra doesn't work
    - Discuss Bellman-Ford vs Modified Dijkstra trade-offs
    - Mention applications in travel planning, network routing

---

## Problem 15: Evaluate Division

=== "Problem Statement"
    **Difficulty**: 🟡 Medium  
    **Pattern**: Weighted Graph / Union-Find with weights  
    **Time**: O(E + Q × V), **Space**: O(V)

    You are given an array of variable pairs `equations` and an array of real numbers `values`, where `equations[i] = [Ai, Bi]` and `values[i]` represent the equation `Ai / Bi = values[i]`. Each `Ai` or `Bi` is a string that represents a single variable.

    You are also given some `queries`, where `queries[j] = [Cj, Dj]` represents the `j`th query where you must find the answer for `Cj / Dj = ?`.

    Return the answers to all queries. If a single answer cannot be determined, return `-1.0`.

    **Example:**
    ```
    Input: equations = [["a","b"],["b","c"]], values = [2.0,3.0], 
           queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
    Output: [6.00000,0.50000,-1.00000,1.00000,-1.00000]
    Explanation: a/b=2.0, b/c=3.0 → a/c=6.0, b/a=0.5, etc.
    ```

=== "Solution"
    ```python
    def calcEquation(equations, values, queries):
        """
        Build weighted directed graph and use DFS to find paths
        """
        from collections import defaultdict
        
        graph = defaultdict(dict)
        
        # Build graph: if a/b = value, then a→b with weight value, b→a with weight 1/value
        for (a, b), value in zip(equations, values):
            graph[a][b] = value
            graph[b][a] = 1.0 / value
        
        def dfs(start, end, visited):
            if start not in graph or end not in graph:
                return -1.0
            
            if start == end:
                return 1.0
            
            visited.add(start)
            
            for neighbor, weight in graph[start].items():
                if neighbor not in visited:
                    result = dfs(neighbor, end, visited)
                    if result != -1.0:
                        return weight * result
            
            return -1.0
        
        results = []
        for c, d in queries:
            visited = set()
            results.append(dfs(c, d, visited))
        
        return results
    ```

=== "Alternative Solutions"
    **Union-Find with weights:**
    ```python
    def calcEquation(equations, values, queries):
        class UnionFind:
            def __init__(self):
                self.parent = {}
                self.weight = {}
            
            def find(self, x):
                if x not in self.parent:
                    self.parent[x] = x
                    self.weight[x] = 1.0
                    return x
                
                if self.parent[x] != x:
                    original_parent = self.parent[x]
                    self.parent[x] = self.find(self.parent[x])
                    self.weight[x] *= self.weight[original_parent]
                
                return self.parent[x]
            
            def union(self, x, y, value):
                root_x, root_y = self.find(x), self.find(y)
                
                if root_x != root_y:
                    self.parent[root_x] = root_y
                    self.weight[root_x] = self.weight[y] * value / self.weight[x]
            
            def query(self, x, y):
                if x not in self.parent or y not in self.parent:
                    return -1.0
                
                root_x, root_y = self.find(x), self.find(y)
                
                if root_x != root_y:
                    return -1.0
                
                return self.weight[x] / self.weight[y]
        
        uf = UnionFind()
        
        for (a, b), value in zip(equations, values):
            uf.union(a, b, value)
        
        return [uf.query(c, d) for c, d in queries]
    ```

=== "Tips & Insights"
    **Pattern Recognition:**
    - Division relationships → Weighted directed graph
    - Transitive relationships → Graph traversal or Union-Find
    - Ratio calculations → Multiply edge weights along path

    **Common Mistakes:**
    - Not handling bidirectional relationships properly
    - Forgetting edge cases (same variable, unknown variables)
    - Incorrect weight calculations in Union-Find

    **Interview Tips:**
    - Explain how division creates weighted edges
    - Compare DFS vs Union-Find approaches
    - Discuss handling of floating point precision

---

## Summary & Key Patterns

### 🎯 Core Patterns Mastered

| **Pattern** | **Key Problems** | **When to Use** |
|-------------|------------------|-----------------|
| **BFS Shortest Path** | Shortest Path in Matrix, Word Ladder | Unweighted shortest path, level-by-level exploration |
| **Topological Sort** | Course Schedule I/II, Minimum Height Trees | Dependency resolution, cycle detection in DAG |
| **Graph Coloring** | Is Graph Bipartite | Two-coloring problems, conflict detection |
| **Multi-source BFS** | Rotting Oranges, Pacific Atlantic | Multiple starting points, simultaneous propagation |
| **Union-Find** | Accounts Merge, Evaluate Division | Connectivity, grouping, dynamic equivalence |
| **Dijkstra/Modified** | Network Delay, Cheapest Flights | Weighted shortest path, constrained optimization |

### 📈 Time Complexity Patterns

- **BFS/DFS**: O(V + E) for basic traversal
- **Dijkstra**: O(E log V) with priority queue
- **Topological Sort**: O(V + E) using Kahn's or DFS
- **Union-Find**: O(α(V)) amortized per operation

### 🔧 Implementation Tips

1. **Graph Representation**: Use adjacency list for sparse graphs
2. **Visited Tracking**: Set for fast lookup, list for ordered processing
3. **Direction Vectors**: Store common patterns (4-dir, 8-dir, etc.)
4. **State Management**: Include necessary constraints in state tuple

---

Ready for the ultimate challenge? Move on to **[Hard Graph Problems](hard-problems.md)** to master advanced algorithms like strongly connected components, maximum flow, and complex optimization problems!

### 📚 What's Next

- **[Hard Problems](hard-problems.md)** - Master advanced graph algorithms
- **[Graph Theory Fundamentals](../fundamentals/graph-theory.md)** - Deep theoretical understanding
- **[Advanced Algorithms](../advanced/network-flow.md)** - Specialized graph algorithms
