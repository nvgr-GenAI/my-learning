# Weighted Graphs

Weighted graphs are graphs where each edge has an associated weight or cost. These weights represent quantities like distance, time, capacity, or any other metric relevant to the problem domain. This section explores algorithms, applications, and concepts specific to weighted graphs.

## Overview

=== "Definition"

    A weighted graph G = (V, E, w) consists of:
    
    - A set V of vertices (or nodes)
    - A set E of edges connecting pairs of vertices
    - A weight function w: E → R that assigns a real value (weight) to each edge
    
    Weighted graphs can be either directed or undirected. In a weighted directed graph, an edge (u, v) has a weight that may be different from the edge (v, u). In a weighted undirected graph, the edge {u, v} has a single weight.
    
    ![Weighted Graph](https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Weighted_graph.svg/440px-Weighted_graph.svg.png)
    
    In the above example, numbers on the edges indicate their weights.

=== "Applications"

    Weighted graphs model many real-world systems:
    
    - **Transportation networks**: Vertices are locations, edges are roads, weights are distances or travel times
    - **Communication networks**: Vertices are devices, weights are bandwidth or latency
    - **Social networks**: Vertices are people, weights are strength of relationships
    - **Electrical networks**: Vertices are components, weights are resistance or capacity
    - **Flow networks**: Vertices are locations, weights are flow capacities
    - **Financial systems**: Vertices are entities, weights are transaction costs
    - **Project scheduling**: Vertices are tasks, weights are completion times
    - **Resource allocation**: Vertices are resources and consumers, weights are costs or values

=== "Types of Weights"

    Weights can represent various quantities:
    
    - **Distance/Length**: Physical distance between locations
    - **Cost**: Monetary cost of using a connection
    - **Time**: Duration required to traverse a connection
    - **Capacity**: Maximum flow possible through a connection
    - **Reliability**: Probability that a connection works
    - **Bandwidth**: Data transmission capacity
    - **Similarity**: Degree of similarity between entities
    - **Strength**: Strength of a relationship or connection

## Representation of Weighted Graphs

=== "Adjacency Matrix"

    In an adjacency matrix representation for a weighted graph with n vertices:
    
    - A is an n × n matrix
    - A[i][j] = weight of edge from vertex i to vertex j
    - A[i][j] = ∞ (or a special value) if no edge exists
    - For undirected graphs, A[i][j] = A[j][i]
    
    ```python
    def create_adjacency_matrix(vertices, edges, directed=False):
        """
        Create an adjacency matrix for a weighted graph.
        
        Args:
            vertices: List of vertices
            edges: List of edges as tuples (u, v, weight)
            directed: Boolean indicating if the graph is directed
            
        Returns:
            list: Adjacency matrix
        """
        n = len(vertices)
        # Create a mapping from vertex labels to indices
        vertex_to_idx = {vertices[i]: i for i in range(n)}
        
        # Initialize with infinity (or a large value)
        inf = float('inf')
        matrix = [[inf] * n for _ in range(n)]
        
        # Set diagonal elements to 0
        for i in range(n):
            matrix[i][i] = 0
        
        # Add edges
        for u, v, weight in edges:
            matrix[vertex_to_idx[u]][vertex_to_idx[v]] = weight
            if not directed:
                matrix[vertex_to_idx[v]][vertex_to_idx[u]] = weight
        
        return matrix
    ```

=== "Adjacency List"

    In an adjacency list representation for a weighted graph:
    
    - Each vertex maintains a list of (neighbor, weight) pairs
    - For a vertex u, the list contains all pairs (v, w) such that there's an edge from u to v with weight w
    
    ```python
    def create_adjacency_list(vertices, edges, directed=False):
        """
        Create an adjacency list for a weighted graph.
        
        Args:
            vertices: List of vertices
            edges: List of edges as tuples (u, v, weight)
            directed: Boolean indicating if the graph is directed
            
        Returns:
            dict: Adjacency list
        """
        graph = {v: [] for v in vertices}
        
        for u, v, weight in edges:
            graph[u].append((v, weight))
            if not directed:
                graph[v].append((u, weight))
        
        return graph
    ```

=== "Edge List"

    In an edge list representation for a weighted graph:
    
    - The graph is represented as a list of all edges with their weights
    - Each edge is stored as a tuple (u, v, weight)
    
    ```python
    def create_edge_list(edges):
        """
        Create an edge list for a weighted graph.
        
        Args:
            edges: List of edges as tuples (u, v, weight)
            
        Returns:
            list: Edge list
        """
        return edges
    ```

## Shortest Path Algorithms

=== "Dijkstra's Algorithm"

    Dijkstra's algorithm finds the shortest path from a source vertex to all other vertices in a graph with non-negative edge weights:
    
    ```python
    import heapq
    
    def dijkstra(graph, start):
        """
        Find shortest paths from start vertex to all vertices using Dijkstra's algorithm.
        
        Args:
            graph: Dictionary representing adjacency list of weighted graph as {u: [(v, weight), ...], ...}
            start: Starting vertex
            
        Returns:
            dict: Dictionary of shortest distances from start to each vertex
            dict: Dictionary of parent pointers for reconstructing paths
        """
        # Initialize distances with infinity
        distances = {vertex: float('inf') for vertex in graph}
        distances[start] = 0
        
        # Parent pointers for path reconstruction
        parents = {vertex: None for vertex in graph}
        
        # Priority queue for selecting next vertex to process
        priority_queue = [(0, start)]
        
        # Set to track processed vertices
        processed = set()
        
        while priority_queue:
            # Get vertex with minimum distance
            current_distance, current_vertex = heapq.heappop(priority_queue)
            
            # Skip if already processed
            if current_vertex in processed:
                continue
                
            processed.add(current_vertex)
            
            # Check all neighbors
            for neighbor, weight in graph[current_vertex]:
                if neighbor in processed:
                    continue
                    
                # Calculate potential new distance
                distance = current_distance + weight
                
                # If found a shorter path, update
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    parents[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (distance, neighbor))
        
        return distances, parents
    
    def reconstruct_path(parents, target):
        """
        Reconstruct path from start to target using parent pointers.
        
        Args:
            parents: Dictionary of parent pointers
            target: Target vertex
            
        Returns:
            list: Path from start to target
        """
        path = []
        while target is not None:
            path.append(target)
            target = parents[target]
        return path[::-1]  # Reverse to get path from start to target
    ```

=== "Bellman-Ford Algorithm"

    The Bellman-Ford algorithm finds the shortest paths from a source vertex to all other vertices, and works with negative edge weights (as long as there are no negative cycles):
    
    ```python
    def bellman_ford(graph, vertices, edges, start):
        """
        Find shortest paths from start vertex to all vertices using Bellman-Ford algorithm.
        
        Args:
            graph: Dictionary representing adjacency list of weighted graph
            vertices: List of all vertices
            edges: List of all edges as tuples (u, v, weight)
            start: Starting vertex
            
        Returns:
            dict: Dictionary of shortest distances from start to each vertex
            dict: Dictionary of parent pointers for reconstructing paths
            bool: True if no negative cycle reachable from start, False otherwise
        """
        # Initialize distances with infinity
        distances = {vertex: float('inf') for vertex in vertices}
        distances[start] = 0
        
        # Parent pointers for path reconstruction
        parents = {vertex: None for vertex in vertices}
        
        # Relax edges |V|-1 times
        for _ in range(len(vertices) - 1):
            for u, v, weight in edges:
                if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    parents[v] = u
        
        # Check for negative cycles
        for u, v, weight in edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                return None, None, False  # Negative cycle detected
        
        return distances, parents, True
    ```

=== "Floyd-Warshall Algorithm"

    The Floyd-Warshall algorithm finds shortest paths between all pairs of vertices:
    
    ```python
    def floyd_warshall(graph, vertices):
        """
        Find shortest paths between all pairs of vertices using Floyd-Warshall algorithm.
        
        Args:
            graph: Adjacency matrix representation of weighted graph
            vertices: List of all vertices
            
        Returns:
            list: Matrix of shortest distances between all pairs of vertices
            list: Matrix of next vertices for path reconstruction
        """
        n = len(vertices)
        
        # Initialize distance matrix and next vertex matrix
        dist = [row[:] for row in graph]  # Create a copy
        next_vertex = [[None] * n for _ in range(n)]
        
        # Initialize next vertex matrix
        for i in range(n):
            for j in range(n):
                if i != j and dist[i][j] < float('inf'):
                    next_vertex[i][j] = j
        
        # Update distances considering each vertex as intermediate
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] < float('inf') and dist[k][j] < float('inf'):
                        if dist[i][k] + dist[k][j] < dist[i][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]
                            next_vertex[i][j] = next_vertex[i][k]
        
        # Check for negative cycles
        for i in range(n):
            if dist[i][i] < 0:
                raise ValueError("Graph contains a negative cycle")
        
        return dist, next_vertex
    
    def reconstruct_all_paths(next_vertex, vertices):
        """
        Reconstruct all shortest paths using next vertex matrix.
        
        Args:
            next_vertex: Matrix of next vertices for path reconstruction
            vertices: List of all vertices
            
        Returns:
            dict: Dictionary of all shortest paths between all pairs of vertices
        """
        n = len(vertices)
        paths = {}
        
        for i in range(n):
            for j in range(n):
                if i != j and next_vertex[i][j] is not None:
                    path = [vertices[i]]
                    current = i
                    while current != j:
                        current = next_vertex[current][j]
                        path.append(vertices[current])
                    paths[(vertices[i], vertices[j])] = path
        
        return paths
    ```

## Minimum Spanning Tree Algorithms

=== "Kruskal's Algorithm"

    Kruskal's algorithm finds a minimum spanning tree for a connected, undirected weighted graph:
    
    ```python
    def kruskal(vertices, edges):
        """
        Find minimum spanning tree using Kruskal's algorithm.
        
        Args:
            vertices: List of all vertices
            edges: List of all edges as tuples (u, v, weight)
            
        Returns:
            list: Edges in the minimum spanning tree
        """
        # Sort edges by weight
        edges = sorted(edges, key=lambda x: x[2])
        
        # Initialize disjoint set for each vertex
        parent = {v: v for v in vertices}
        rank = {v: 0 for v in vertices}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            
            if root_x == root_y:
                return
            
            # Union by rank
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
        
        mst = []
        for u, v, weight in edges:
            if find(u) != find(v):
                union(u, v)
                mst.append((u, v, weight))
                
                # Early termination when MST is complete
                if len(mst) == len(vertices) - 1:
                    break
        
        return mst
    ```

=== "Prim's Algorithm"

    Prim's algorithm also finds a minimum spanning tree, but starts from a vertex and grows the tree:
    
    ```python
    def prim(graph, vertices):
        """
        Find minimum spanning tree using Prim's algorithm.
        
        Args:
            graph: Dictionary representing adjacency list of weighted graph as {u: [(v, weight), ...], ...}
            vertices: List of all vertices
            
        Returns:
            list: Edges in the minimum spanning tree
        """
        if not vertices:
            return []
        
        # Start with the first vertex
        start = vertices[0]
        
        # Initialize priority queue with edges from start
        priority_queue = [(weight, start, neighbor) for neighbor, weight in graph[start]]
        heapq.heapify(priority_queue)
        
        visited = {start}
        mst = []
        
        while priority_queue and len(visited) < len(vertices):
            weight, u, v = heapq.heappop(priority_queue)
            
            if v in visited:
                continue
                
            visited.add(v)
            mst.append((u, v, weight))
            
            # Add edges from the newly added vertex
            for neighbor, edge_weight in graph[v]:
                if neighbor not in visited:
                    heapq.heappush(priority_queue, (edge_weight, v, neighbor))
        
        return mst
    ```

## Network Flow Algorithms

=== "Ford-Fulkerson Algorithm"

    The Ford-Fulkerson algorithm finds the maximum flow in a weighted directed graph (flow network):
    
    ```python
    def ford_fulkerson(graph, source, sink):
        """
        Find maximum flow in a network using Ford-Fulkerson algorithm.
        
        Args:
            graph: Dictionary representing adjacency list with capacities
            source: Source vertex
            sink: Sink vertex
            
        Returns:
            int: Maximum flow value
            dict: Flow on each edge
        """
        # Initialize flow network
        flow = {u: {v: 0 for v in graph[u]} for u in graph}
        
        def bfs():
            """Find an augmenting path using BFS."""
            visited = {source: True}
            queue = [source]
            parent = {}
            
            while queue:
                u = queue.pop(0)
                
                for v in graph[u]:
                    # If there's residual capacity and v is not visited
                    residual_capacity = graph[u][v] - flow[u].get(v, 0)
                    if residual_capacity > 0 and v not in visited:
                        visited[v] = True
                        parent[v] = u
                        queue.append(v)
                        
                        # If we reached the sink, return the path
                        if v == sink:
                            path = []
                            current = sink
                            while current != source:
                                path.append((parent[current], current))
                                current = parent[current]
                            return path[::-1]  # Reverse to get path from source to sink
            
            return None  # No augmenting path found
        
        # Main algorithm
        max_flow = 0
        
        while True:
            path = bfs()
            if not path:
                break
                
            # Find bottleneck capacity
            bottleneck = float('inf')
            for u, v in path:
                residual_capacity = graph[u][v] - flow[u].get(v, 0)
                bottleneck = min(bottleneck, residual_capacity)
            
            # Augment flow along the path
            for u, v in path:
                flow[u][v] = flow[u].get(v, 0) + bottleneck
                flow.setdefault(v, {})[u] = flow.get(v, {}).get(u, 0) - bottleneck
            
            max_flow += bottleneck
        
        return max_flow, flow
    ```

=== "Edmonds-Karp Algorithm"

    The Edmonds-Karp algorithm is an implementation of Ford-Fulkerson that specifically uses BFS to find augmenting paths, guaranteeing polynomial time complexity.

=== "Dinic's Algorithm"

    Dinic's algorithm is an improved version of Edmonds-Karp that uses level graphs and blocking flows, providing better theoretical time complexity.

## Weighted Graph Problems

=== "Traveling Salesman Problem (TSP)"

    The TSP is a classic NP-hard problem: find the shortest possible route that visits each city exactly once and returns to the origin city.
    
    For small instances, dynamic programming can be used:
    
    ```python
    def tsp_dp(graph, start):
        """
        Solve Traveling Salesman Problem using dynamic programming.
        
        Args:
            graph: Adjacency matrix of distances
            start: Starting vertex index
            
        Returns:
            int: Length of shortest tour
            list: Order of vertices in the tour
        """
        n = len(graph)
        
        # dp[mask][i] = shortest path visiting all vertices in mask and ending at vertex i
        dp = {}
        
        # Base case: Start at the starting vertex
        dp[(1 << start, start)] = 0
        
        def solve_tsp(mask, pos):
            if (mask, pos) in dp:
                return dp[(mask, pos)]
            
            # If all vertices are visited
            if mask == (1 << n) - 1:
                return graph[pos][start] if graph[pos][start] != float('inf') else float('inf')
            
            ans = float('inf')
            
            # Try to go to an unvisited vertex
            for next_city in range(n):
                if mask & (1 << next_city) == 0 and graph[pos][next_city] != float('inf'):
                    new_ans = graph[pos][next_city] + solve_tsp(mask | (1 << next_city), next_city)
                    ans = min(ans, new_ans)
            
            dp[(mask, pos)] = ans
            return ans
        
        # Calculate shortest tour length
        shortest_length = solve_tsp(1 << start, start)
        
        # Reconstruct tour
        mask = 1 << start
        pos = start
        tour = [start]
        
        for _ in range(n - 1):
            next_city = -1
            best_cost = float('inf')
            
            for candidate in range(n):
                if mask & (1 << candidate) == 0 and graph[pos][candidate] != float('inf'):
                    cost = graph[pos][candidate] + dp.get((mask | (1 << candidate), candidate), float('inf'))
                    if cost < best_cost:
                        best_cost = cost
                        next_city = candidate
            
            tour.append(next_city)
            mask |= (1 << next_city)
            pos = next_city
        
        tour.append(start)  # Return to start
        return shortest_length, tour
    ```

=== "Shortest Path with Constraints"

    Many variants of shortest path problems have additional constraints:
    
    - **Resource-constrained shortest path**: Find the shortest path subject to resource constraints
    - **Time-dependent shortest path**: Edge weights vary with time
    - **k-shortest paths**: Find k different shortest paths
    - **Constrained shortest path**: Additional constraints on the path properties

=== "Maximum Flow with Multiple Sources/Sinks"

    To handle multiple sources and sinks in flow networks, create super-source and super-sink vertices:
    
    ```python
    def max_flow_multi_source_sink(graph, sources, sinks):
        """
        Find maximum flow with multiple sources and sinks.
        
        Args:
            graph: Dictionary representing adjacency list with capacities
            sources: List of source vertices
            sinks: List of sink vertices
            
        Returns:
            int: Maximum flow value
        """
        # Add super source and super sink
        graph['super_source'] = {src: float('inf') for src in sources}
        for sink in sinks:
            if sink not in graph:
                graph[sink] = {}
            graph[sink]['super_sink'] = float('inf')
        
        # Find maximum flow from super source to super sink
        max_flow, _ = ford_fulkerson(graph, 'super_source', 'super_sink')
        
        return max_flow
    ```

## Special Weighted Graphs

=== "Complete Weighted Graphs"

    A complete weighted graph has an edge between every pair of vertices:
    
    - Often used in optimization problems like TSP
    - Facilitates greedy algorithms like nearest neighbor
    - Enables triangle inequality-based approximations

=== "Euclidean Graphs"

    In Euclidean graphs, vertices are points in Euclidean space and edge weights are distances between points:
    
    - Satisfy the triangle inequality
    - Enable geometric algorithms
    - Important in computational geometry and spatial planning

=== "Flow Networks"

    Flow networks are directed weighted graphs where:
    
    - Each edge has a capacity constraint
    - There is a source vertex (flow producer)
    - There is a sink vertex (flow consumer)
    - Flow conservation applies at intermediate vertices

## Graph Theory Concepts for Weighted Graphs

=== "Weight Functions and Properties"

    Properties of weight functions in different applications:
    
    - **Metric weights**: Satisfy triangle inequality (w(u,v) ≤ w(u,x) + w(x,v))
    - **Euclidean weights**: Derived from Euclidean distance between points
    - **Non-negative weights**: All weights are non-negative
    - **Integer weights**: All weights are integers
    - **Uniform weights**: All weights are the same (reduces to unweighted graph)

=== "Path Properties"

    Important concepts related to paths in weighted graphs:
    
    - **Path weight**: Sum of weights of edges in the path
    - **Shortest path**: Path with minimum total weight
    - **Bottleneck path**: Path where the maximum edge weight is minimized
    - **Critical path**: Longest path in a directed acyclic graph (important in project scheduling)
    - **Negative cycles**: Cycles with negative total weight (problematic for some algorithms)

=== "Cut Properties"

    A cut in a graph divides vertices into two disjoint sets:
    
    - **Cut weight**: Sum of weights of edges crossing the cut
    - **Minimum cut**: Cut with minimum total weight
    - **s-t cut**: Divides graph such that source s and sink t are in different sets
    - **Max-flow min-cut theorem**: Maximum flow equals minimum s-t cut capacity

## Applications of Weighted Graphs

=== "Transportation Networks"

    - **Route planning**: Finding shortest or fastest routes
    - **Traffic flow optimization**: Managing congestion
    - **Public transit scheduling**: Planning bus/train schedules
    - **Logistics optimization**: Planning deliveries

=== "Telecommunication Networks"

    - **Network design**: Minimizing costs while meeting requirements
    - **Routing protocols**: Finding optimal paths for data
    - **Bandwidth allocation**: Managing limited bandwidth
    - **Network reliability**: Ensuring robust connections

=== "Facility Location"

    - **Warehouse placement**: Minimizing transportation costs
    - **Emergency service location**: Maximizing coverage
    - **Cell tower placement**: Optimizing signal coverage
    - **Hub location**: Designing efficient transportation hubs

=== "Image Processing"

    - **Image segmentation**: Separating image regions
    - **Feature extraction**: Identifying image features
    - **Object recognition**: Matching image features
    - **Image restoration**: Removing noise and artifacts

## References

- [Weighted Graph on Wikipedia](https://en.wikipedia.org/wiki/Weighted_graph)
- [Dijkstra's Algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
- [Bellman-Ford Algorithm](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm)
- [Floyd-Warshall Algorithm](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm)
- [Kruskal's Algorithm](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm)
- [Prim's Algorithm](https://en.wikipedia.org/wiki/Prim%27s_algorithm)
- [Maximum Flow Problem](https://en.wikipedia.org/wiki/Maximum_flow_problem)
- [Traveling Salesman Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem)
- [Introduction to Algorithms (CLRS)](https://en.wikipedia.org/wiki/Introduction_to_Algorithms) - Chapters 23-26
