# Graph Algorithms - Fundamentals

## 🎯 Overview

Graph algorithms are fundamental to computer science and have widespread applications in networking, social media, transportation, and many other domains. This section covers essential graph concepts, representations, and core algorithms.

=== "📋 Core Graph Concepts"

    ## **Graph Terminology**
    
    | Term | Definition | Example |
    |------|------------|---------|
    | **Vertex (Node)** | Basic unit of a graph | Cities in a map |
    | **Edge** | Connection between vertices | Roads between cities |
    | **Degree** | Number of edges connected to a vertex | Number of roads from a city |
    | **Path** | Sequence of vertices connected by edges | Route from A to B |
    | **Cycle** | Path that starts and ends at same vertex | Circular route |
    | **Connected** | Path exists between any two vertices | All cities are reachable |
    | **Component** | Maximal set of connected vertices | Isolated city clusters |
    | **Weight** | Value associated with an edge | Distance, cost, time |

    ## **Graph Types**
    
    | Type | Properties | Use Cases |
    |------|------------|-----------|
    | **Undirected** | Edges have no direction | Social networks, physical networks |
    | **Directed** | Edges have direction | Web pages, dependency graphs |
    | **Weighted** | Edges have weights | GPS navigation, network routing |
    | **Unweighted** | All edges equal | Friendship networks |
    | **Cyclic** | Contains cycles | General graphs |
    | **Acyclic** | No cycles | Trees, DAGs |
    | **Complete** | Every pair connected | Fully connected networks |
    | **Bipartite** | Two sets, edges only between sets | Matching problems |

=== "🔗 Graph Representations"

    ## **Adjacency Matrix**
    
    ```python
    class GraphMatrix:
        def __init__(self, vertices):
            self.V = vertices
            self.graph = [[0 for _ in range(vertices)] 
                         for _ in range(vertices)]
        
        def add_edge(self, u, v, weight=1):
            self.graph[u][v] = weight
            # For undirected graph:
            # self.graph[v][u] = weight
        
        def print_graph(self):
            for row in self.graph:
                print(row)
    ```
    
    **Advantages:**
    - O(1) edge lookup
    - Simple to implement
    - Works well for dense graphs
    
    **Disadvantages:**
    - O(V²) space complexity
    - Inefficient for sparse graphs
    
    ## **Adjacency List**
    
    ```python
    from collections import defaultdict
    
    class GraphList:
        def __init__(self):
            self.graph = defaultdict(list)
        
        def add_edge(self, u, v, weight=1):
            self.graph[u].append((v, weight))
            # For undirected graph:
            # self.graph[v].append((u, weight))
        
        def print_graph(self):
            for vertex in self.graph:
                print(f"{vertex}: {self.graph[vertex]}")
    ```
    
    **Advantages:**
    - O(V + E) space complexity
    - Efficient for sparse graphs
    - Easy to iterate over neighbors
    
    **Disadvantages:**
    - O(V) edge lookup in worst case
    - Slightly more complex implementation
    
    ## **Edge List**
    
    ```python
    class GraphEdgeList:
        def __init__(self):
            self.edges = []
        
        def add_edge(self, u, v, weight=1):
            self.edges.append((u, v, weight))
        
        def print_graph(self):
            for edge in self.edges:
                print(f"{edge[0]} -> {edge[1]} (weight: {edge[2]})")
    ```
    
    **Use Cases:**
    - Algorithms that process all edges
    - Kruskal's MST algorithm
    - Simple representation

=== "🔍 Graph Traversal Algorithms"

    ## **Depth-First Search (DFS)**
    
    ```python
    def dfs_recursive(graph, start, visited=None):
        if visited is None:
            visited = set()
        
        visited.add(start)
        print(start, end=' ')
        
        for neighbor in graph[start]:
            if neighbor not in visited:
                dfs_recursive(graph, neighbor, visited)
    
    def dfs_iterative(graph, start):
        visited = set()
        stack = [start]
        
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                print(vertex, end=' ')
                
                # Add neighbors to stack
                for neighbor in graph[vertex]:
                    if neighbor not in visited:
                        stack.append(neighbor)
    ```
    
    **Applications:**
    - Topological sorting
    - Finding connected components
    - Cycle detection
    - Path finding
    
    **Time Complexity:** O(V + E)
    **Space Complexity:** O(V)
    
    ## **Breadth-First Search (BFS)**
    
    ```python
    from collections import deque
    
    def bfs(graph, start):
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            vertex = queue.popleft()
            print(vertex, end=' ')
            
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    def bfs_shortest_path(graph, start, end):
        visited = set()
        queue = deque([(start, [start])])
        visited.add(start)
        
        while queue:
            vertex, path = queue.popleft()
            
            if vertex == end:
                return path
            
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    ```
    
    **Applications:**
    - Shortest path in unweighted graphs
    - Level-order traversal
    - Finding all nodes at given distance
    - Web crawling
    
    **Time Complexity:** O(V + E)
    **Space Complexity:** O(V)

=== "🛣️ Shortest Path Algorithms"

    ## **Dijkstra's Algorithm**
    
    ```python
    import heapq
    
    def dijkstra(graph, start):
        distances = {vertex: float('infinity') for vertex in graph}
        distances[start] = 0
        pq = [(0, start)]
        
        while pq:
            current_distance, current_vertex = heapq.heappop(pq)
            
            if current_distance > distances[current_vertex]:
                continue
            
            for neighbor, weight in graph[current_vertex]:
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
        
        return distances
    ```
    
    **Use Cases:**
    - GPS navigation
    - Network routing
    - Flight connections
    
    **Time Complexity:** O((V + E) log V)
    **Space Complexity:** O(V)
    
    ## **Bellman-Ford Algorithm**
    
    ```python
    def bellman_ford(graph, start):
        distances = {vertex: float('infinity') for vertex in graph}
        distances[start] = 0
        
        # Relax edges V-1 times
        for _ in range(len(graph) - 1):
            for vertex in graph:
                for neighbor, weight in graph[vertex]:
                    if distances[vertex] + weight < distances[neighbor]:
                        distances[neighbor] = distances[vertex] + weight
        
        # Check for negative cycles
        for vertex in graph:
            for neighbor, weight in graph[vertex]:
                if distances[vertex] + weight < distances[neighbor]:
                    return None  # Negative cycle detected
        
        return distances
    ```
    
    **Advantages:**
    - Works with negative weights
    - Detects negative cycles
    
    **Time Complexity:** O(VE)
    **Space Complexity:** O(V)

=== "🌳 Tree Algorithms"

    ## **Minimum Spanning Tree (MST)**
    
    ```python
    # Kruskal's Algorithm
    def find_parent(parent, i):
        if parent[i] == i:
            return i
        return find_parent(parent, parent[i])
    
    def union(parent, rank, x, y):
        xroot = find_parent(parent, x)
        yroot = find_parent(parent, y)
        
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1
    
    def kruskal_mst(graph):
        result = []
        edges = []
        
        # Create edge list
        for u in graph:
            for v, weight in graph[u]:
                edges.append((weight, u, v))
        
        edges.sort()
        
        parent = {}
        rank = {}
        for vertex in graph:
            parent[vertex] = vertex
            rank[vertex] = 0
        
        for weight, u, v in edges:
            if find_parent(parent, u) != find_parent(parent, v):
                result.append((u, v, weight))
                union(parent, rank, u, v)
        
        return result
    ```
    
    **Applications:**
    - Network design
    - Clustering
    - Approximation algorithms
    
    **Time Complexity:** O(E log E)
    **Space Complexity:** O(V)

=== "🎯 Topological Sorting"

    ## **Kahn's Algorithm (BFS-based)**
    
    ```python
    from collections import deque, defaultdict
    
    def topological_sort_bfs(graph):
        in_degree = defaultdict(int)
        
        # Calculate in-degrees
        for vertex in graph:
            for neighbor in graph[vertex]:
                in_degree[neighbor] += 1
        
        # Start with vertices having no incoming edges
        queue = deque([vertex for vertex in graph if in_degree[vertex] == 0])
        result = []
        
        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            
            for neighbor in graph[vertex]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result if len(result) == len(graph) else None
    ```
    
    ## **DFS-based Topological Sort**
    
    ```python
    def topological_sort_dfs(graph):
        visited = set()
        stack = []
        
        def dfs(vertex):
            visited.add(vertex)
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    dfs(neighbor)
            stack.append(vertex)
        
        for vertex in graph:
            if vertex not in visited:
                dfs(vertex)
        
        return stack[::-1]
    ```
    
    **Applications:**
    - Task scheduling
    - Dependency resolution
    - Course prerequisites
    
    **Time Complexity:** O(V + E)
    **Space Complexity:** O(V)

=== "🔄 Cycle Detection"

    ## **Cycle Detection in Undirected Graphs**
    
    ```python
    def has_cycle_undirected(graph):
        visited = set()
        
        def dfs(vertex, parent):
            visited.add(vertex)
            
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    if dfs(neighbor, vertex):
                        return True
                elif neighbor != parent:
                    return True
            
            return False
        
        for vertex in graph:
            if vertex not in visited:
                if dfs(vertex, -1):
                    return True
        
        return False
    ```
    
    ## **Cycle Detection in Directed Graphs**
    
    ```python
    def has_cycle_directed(graph):
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {vertex: WHITE for vertex in graph}
        
        def dfs(vertex):
            color[vertex] = GRAY
            
            for neighbor in graph[vertex]:
                if color[neighbor] == GRAY:
                    return True
                if color[neighbor] == WHITE and dfs(neighbor):
                    return True
            
            color[vertex] = BLACK
            return False
        
        for vertex in graph:
            if color[vertex] == WHITE:
                if dfs(vertex):
                    return True
        
        return False
    ```

=== "📊 Complexity Analysis"

    ## **Time Complexities**
    
    | Algorithm | Time Complexity | Space Complexity |
    |-----------|-----------------|------------------|
    | **DFS/BFS** | O(V + E) | O(V) |
    | **Dijkstra's** | O((V + E) log V) | O(V) |
    | **Bellman-Ford** | O(VE) | O(V) |
    | **Floyd-Warshall** | O(V³) | O(V²) |
    | **Kruskal's MST** | O(E log E) | O(V) |
    | **Prim's MST** | O((V + E) log V) | O(V) |
    | **Topological Sort** | O(V + E) | O(V) |
    
    ## **Space Complexities by Representation**
    
    | Representation | Space Complexity | Best For |
    |----------------|------------------|----------|
    | **Adjacency Matrix** | O(V²) | Dense graphs |
    | **Adjacency List** | O(V + E) | Sparse graphs |
    | **Edge List** | O(E) | Simple operations |

=== "🎯 Problem-Solving Strategies"

    ## **Common Graph Problem Patterns**
    
    | Pattern | When to Use | Example Problems |
    |---------|-------------|------------------|
    | **BFS** | Shortest path, level-order | Word ladder, shortest path |
    | **DFS** | Exploration, backtracking | Connected components, cycle detection |
    | **Dijkstra's** | Weighted shortest path | GPS navigation, network routing |
    | **Topological Sort** | Dependency resolution | Course scheduling, build systems |
    | **Union-Find** | Connectivity, MST | Network connectivity, Kruskal's |
    | **Two-Coloring** | Bipartite checking | Conflict resolution, matching |
    
    ## **Algorithm Selection Guide**
    
    ```python
    def choose_algorithm(graph_type, problem_type):
        """
        Guide for choosing the right graph algorithm
        """
        if problem_type == "shortest_path":
            if graph_type == "unweighted":
                return "BFS"
            elif graph_type == "weighted_no_negative":
                return "Dijkstra"
            elif graph_type == "weighted_with_negative":
                return "Bellman-Ford"
            elif graph_type == "all_pairs":
                return "Floyd-Warshall"
        
        elif problem_type == "connectivity":
            return "DFS or Union-Find"
        
        elif problem_type == "mst":
            return "Kruskal or Prim"
        
        elif problem_type == "topological_ordering":
            return "DFS or Kahn's Algorithm"
        
        elif problem_type == "cycle_detection":
            return "DFS with coloring"
        
        return "Analyze problem requirements"
    ```

---

*Master these graph fundamentals to solve complex network problems and build scalable graph-based applications!*
