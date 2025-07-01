# Graph Algorithms 🕸️

Master the algorithms that power social networks, navigation systems, and the internet itself.

## Overview

Graph algorithms solve problems involving networks of connected entities. From finding shortest paths to detecting communities, these algorithms are fundamental to many real-world applications.

## 📊 Algorithm Complexity

| Algorithm | Time Complexity | Space Complexity | Use Case |
|-----------|----------------|------------------|----------|
| **BFS** | O(V + E) | O(V) | Shortest path (unweighted), level-order traversal |
| **DFS** | O(V + E) | O(V) | Topological sort, cycle detection, connectivity |
| **Dijkstra** | O((V + E) log V) | O(V) | Shortest path (weighted, non-negative) |
| **Bellman-Ford** | O(VE) | O(V) | Shortest path (negative weights) |
| **Floyd-Warshall** | O(V³) | O(V²) | All-pairs shortest paths |
| **Kruskal's MST** | O(E log E) | O(V) | Minimum spanning tree |
| **Prim's MST** | O((V + E) log V) | O(V) | Minimum spanning tree |
| **Tarjan's SCC** | O(V + E) | O(V) | Strongly connected components |

## 🏗️ Graph Representation

### Adjacency List vs Adjacency Matrix

```python
class Graph:
    """Comprehensive graph implementation supporting both representations."""
    
    def __init__(self, num_vertices=0, directed=False, use_matrix=False):
        self.num_vertices = num_vertices
        self.directed = directed
        self.use_matrix = use_matrix
        
        if use_matrix:
            # Adjacency Matrix representation
            self.matrix = [[0] * num_vertices for _ in range(num_vertices)]
        else:
            # Adjacency List representation
            self.adj_list = [[] for _ in range(num_vertices)]
        
        self.weights = {}  # For weighted graphs
    
    def add_edge(self, u, v, weight=1):
        """Add edge between vertices u and v."""
        if self.use_matrix:
            self.matrix[u][v] = weight
            if not self.directed:
                self.matrix[v][u] = weight
        else:
            self.adj_list[u].append(v)
            if not self.directed:
                self.adj_list[v].append(u)
        
        # Store weights for algorithms that need them
        self.weights[(u, v)] = weight
        if not self.directed:
            self.weights[(v, u)] = weight
    
    def get_neighbors(self, vertex):
        """Get neighbors of a vertex."""
        if self.use_matrix:
            neighbors = []
            for i in range(self.num_vertices):
                if self.matrix[vertex][i] != 0:
                    neighbors.append(i)
            return neighbors
        else:
            return self.adj_list[vertex]
    
    def has_edge(self, u, v):
        """Check if edge exists between u and v."""
        if self.use_matrix:
            return self.matrix[u][v] != 0
        else:
            return v in self.adj_list[u]
    
    def get_edge_weight(self, u, v):
        """Get weight of edge between u and v."""
        return self.weights.get((u, v), 0)
    
    def display(self):
        """Display graph representation."""
        if self.use_matrix:
            print("Adjacency Matrix:")
            for row in self.matrix:
                print(row)
        else:
            print("Adjacency List:")
            for i, neighbors in enumerate(self.adj_list):
                print(f"{i}: {neighbors}")

# Example usage
graph = Graph(5, directed=False, use_matrix=False)
edges = [(0, 1), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (3, 4)]

for u, v in edges:
    graph.add_edge(u, v)

graph.display()
```

## 🔍 Graph Traversal Algorithms

### Breadth-First Search (BFS)

Explores the graph level by level, visiting all neighbors before moving to the next level.

```python
from collections import deque

def bfs(graph, start_vertex):
    """
    Breadth-First Search traversal.
    Time: O(V + E), Space: O(V)
    """
    visited = set()
    queue = deque([start_vertex])
    result = []
    
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            
            # Add all unvisited neighbors to queue
            for neighbor in graph.get_neighbors(vertex):
                if neighbor not in visited:
                    queue.append(neighbor)
    
    return result

def bfs_shortest_path(graph, start, end):
    """Find shortest path using BFS (unweighted graph)."""
    if start == end:
        return [start]
    
    visited = set()
    queue = deque([(start, [start])])
    
    while queue:
        vertex, path = queue.popleft()
        
        if vertex not in visited:
            visited.add(vertex)
            
            for neighbor in graph.get_neighbors(vertex):
                new_path = path + [neighbor]
                
                if neighbor == end:
                    return new_path
                
                if neighbor not in visited:
                    queue.append((neighbor, new_path))
    
    return None  # No path found

def bfs_levels(graph, start_vertex):
    """BFS that returns vertices grouped by level."""
    visited = set()
    current_level = [start_vertex]
    levels = []
    
    while current_level:
        levels.append(current_level[:])
        next_level = []
        
        for vertex in current_level:
            if vertex not in visited:
                visited.add(vertex)
                
                for neighbor in graph.get_neighbors(vertex):
                    if neighbor not in visited:
                        next_level.append(neighbor)
        
        current_level = next_level
    
    return levels

def bfs_connected_components(graph):
    """Find all connected components using BFS."""
    visited = set()
    components = []
    
    for vertex in range(graph.num_vertices):
        if vertex not in visited:
            component = []
            queue = deque([vertex])
            
            while queue:
                v = queue.popleft()
                if v not in visited:
                    visited.add(v)
                    component.append(v)
                    
                    for neighbor in graph.get_neighbors(v):
                        if neighbor not in visited:
                            queue.append(neighbor)
            
            components.append(component)
    
    return components

# Example usage
graph = Graph(6, directed=False)
edges = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5)]
for u, v in edges:
    graph.add_edge(u, v)

print("BFS traversal:", bfs(graph, 0))
print("Shortest path 0->3:", bfs_shortest_path(graph, 0, 3))
print("Connected components:", bfs_connected_components(graph))
```

### Depth-First Search (DFS)

Explores as far as possible along each branch before backtracking.

```python
def dfs_recursive(graph, start_vertex, visited=None):
    """
    Recursive DFS traversal.
    Time: O(V + E), Space: O(V)
    """
    if visited is None:
        visited = set()
    
    visited.add(start_vertex)
    result = [start_vertex]
    
    for neighbor in graph.get_neighbors(start_vertex):
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))
    
    return result

def dfs_iterative(graph, start_vertex):
    """Iterative DFS using explicit stack."""
    visited = set()
    stack = [start_vertex]
    result = []
    
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            
            # Add neighbors in reverse order for consistent traversal
            neighbors = graph.get_neighbors(vertex)
            for neighbor in reversed(neighbors):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return result

def dfs_with_timestamps(graph, start_vertex):
    """DFS with discovery and finish timestamps."""
    visited = set()
    discovery_time = {}
    finish_time = {}
    time = [0]  # Use list to make it mutable
    
    def dfs_visit(vertex):
        time[0] += 1
        discovery_time[vertex] = time[0]
        visited.add(vertex)
        
        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in visited:
                dfs_visit(neighbor)
        
        time[0] += 1
        finish_time[vertex] = time[0]
    
    dfs_visit(start_vertex)
    
    return discovery_time, finish_time

def has_cycle_undirected(graph):
    """Detect cycle in undirected graph using DFS."""
    visited = set()
    
    def dfs_cycle_check(vertex, parent):
        visited.add(vertex)
        
        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in visited:
                if dfs_cycle_check(neighbor, vertex):
                    return True
            elif neighbor != parent:
                return True  # Back edge found
        
        return False
    
    # Check all components
    for vertex in range(graph.num_vertices):
        if vertex not in visited:
            if dfs_cycle_check(vertex, -1):
                return True
    
    return False

def has_cycle_directed(graph):
    """Detect cycle in directed graph using DFS."""
    WHITE, GRAY, BLACK = 0, 1, 2
    colors = [WHITE] * graph.num_vertices
    
    def dfs_cycle_check(vertex):
        if colors[vertex] == GRAY:
            return True  # Back edge found
        if colors[vertex] == BLACK:
            return False
        
        colors[vertex] = GRAY
        
        for neighbor in graph.get_neighbors(vertex):
            if dfs_cycle_check(neighbor):
                return True
        
        colors[vertex] = BLACK
        return False
    
    for vertex in range(graph.num_vertices):
        if colors[vertex] == WHITE:
            if dfs_cycle_check(vertex):
                return True
    
    return False

def topological_sort(graph):
    """Topological sort using DFS."""
    if not graph.directed:
        raise ValueError("Topological sort only applies to directed graphs")
    
    visited = set()
    stack = []
    
    def dfs_topological(vertex):
        visited.add(vertex)
        
        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in visited:
                dfs_topological(neighbor)
        
        stack.append(vertex)
    
    # Visit all vertices
    for vertex in range(graph.num_vertices):
        if vertex not in visited:
            dfs_topological(vertex)
    
    return stack[::-1]  # Reverse to get topological order

# Example usage
directed_graph = Graph(6, directed=True)
directed_edges = [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)]
for u, v in directed_edges:
    directed_graph.add_edge(u, v)

print("DFS traversal:", dfs_recursive(directed_graph, 5))
print("Topological sort:", topological_sort(directed_graph))
print("Has cycle:", has_cycle_directed(directed_graph))
```

## 🛣️ Shortest Path Algorithms

### Dijkstra's Algorithm

Finds shortest paths from a source vertex to all other vertices in a weighted graph with non-negative edge weights.

```python
import heapq
from collections import defaultdict

def dijkstra(graph, start_vertex):
    """
    Dijkstra's shortest path algorithm.
    Time: O((V + E) log V), Space: O(V)
    """
    # Initialize distances and previous vertices
    distances = defaultdict(lambda: float('inf'))
    distances[start_vertex] = 0
    previous = {}
    
    # Priority queue: (distance, vertex)
    pq = [(0, start_vertex)]
    visited = set()
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        
        if current_vertex in visited:
            continue
        
        visited.add(current_vertex)
        
        # Check all neighbors
        for neighbor in graph.get_neighbors(current_vertex):
            weight = graph.get_edge_weight(current_vertex, neighbor)
            distance = current_distance + weight
            
            # If we found a shorter path
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))
    
    return dict(distances), previous

def dijkstra_shortest_path(graph, start, end):
    """Get shortest path between two vertices."""
    distances, previous = dijkstra(graph, start)
    
    if end not in previous and end != start:
        return None, float('inf')  # No path exists
    
    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous.get(current)
    
    path.reverse()
    return path, distances[end]

def dijkstra_all_paths(graph, start_vertex):
    """Get shortest paths to all vertices."""
    distances, previous = dijkstra(graph, start_vertex)
    paths = {}
    
    for vertex in distances:
        if vertex == start_vertex:
            paths[vertex] = [start_vertex]
        elif vertex in previous:
            path = []
            current = vertex
            while current is not None:
                path.append(current)
                current = previous.get(current)
            paths[vertex] = path[::-1]
    
    return paths, distances

# Modified Dijkstra for finding k shortest paths
def k_shortest_paths(graph, start, end, k):
    """Find k shortest paths using modified Dijkstra."""
    paths = []
    
    # Priority queue: (distance, path)
    pq = [(0, [start])]
    
    while pq and len(paths) < k:
        distance, path = heapq.heappop(pq)
        current_vertex = path[-1]
        
        if current_vertex == end:
            paths.append((path, distance))
            continue
        
        for neighbor in graph.get_neighbors(current_vertex):
            if neighbor not in path:  # Avoid cycles
                weight = graph.get_edge_weight(current_vertex, neighbor)
                new_distance = distance + weight
                new_path = path + [neighbor]
                heapq.heappush(pq, (new_distance, new_path))
    
    return paths

# Example usage
weighted_graph = Graph(5, directed=False)
weighted_edges = [(0, 1, 4), (0, 2, 1), (1, 3, 1), (2, 1, 2), (2, 3, 5), (3, 4, 3)]
for u, v, w in weighted_edges:
    weighted_graph.add_edge(u, v, w)

distances, previous = dijkstra(weighted_graph, 0)
print("Distances from vertex 0:", dict(distances))

path, distance = dijkstra_shortest_path(weighted_graph, 0, 4)
print(f"Shortest path from 0 to 4: {path}, distance: {distance}")
```

### Bellman-Ford Algorithm

Handles negative edge weights and detects negative cycles.

```python
def bellman_ford(graph, start_vertex):
    """
    Bellman-Ford algorithm for shortest paths with negative weights.
    Time: O(VE), Space: O(V)
    """
    # Initialize distances
    distances = defaultdict(lambda: float('inf'))
    distances[start_vertex] = 0
    previous = {}
    
    # Get all edges
    edges = []
    for u in range(graph.num_vertices):
        for v in graph.get_neighbors(u):
            weight = graph.get_edge_weight(u, v)
            edges.append((u, v, weight))
    
    # Relax edges V-1 times
    for _ in range(graph.num_vertices - 1):
        for u, v, weight in edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                previous[v] = u
    
    # Check for negative cycles
    for u, v, weight in edges:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            raise ValueError("Graph contains negative cycle")
    
    return dict(distances), previous

def detect_negative_cycle(graph):
    """Detect if graph contains negative cycle."""
    try:
        bellman_ford(graph, 0)
        return False
    except ValueError:
        return True

# Example with negative weights
negative_graph = Graph(4, directed=True)
negative_edges = [(0, 1, 1), (1, 2, -3), (2, 3, 2), (3, 1, 1)]
for u, v, w in negative_edges:
    negative_graph.add_edge(u, v, w)

try:
    distances, _ = bellman_ford(negative_graph, 0)
    print("Bellman-Ford distances:", distances)
except ValueError as e:
    print("Error:", e)
```

### Floyd-Warshall Algorithm

Finds shortest paths between all pairs of vertices.

```python
def floyd_warshall(graph):
    """
    Floyd-Warshall algorithm for all-pairs shortest paths.
    Time: O(V³), Space: O(V²)
    """
    n = graph.num_vertices
    
    # Initialize distance matrix
    dist = [[float('inf')] * n for _ in range(n)]
    
    # Distance from vertex to itself is 0
    for i in range(n):
        dist[i][i] = 0
    
    # Fill in direct edge weights
    for u in range(n):
        for v in graph.get_neighbors(u):
            dist[u][v] = graph.get_edge_weight(u, v)
    
    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist

def print_shortest_paths_matrix(dist_matrix):
    """Print the shortest paths matrix."""
    n = len(dist_matrix)
    print("Shortest distances between all pairs:")
    print("    ", end="")
    for j in range(n):
        print(f"{j:4}", end="")
    print()
    
    for i in range(n):
        print(f"{i:2}: ", end="")
        for j in range(n):
            if dist_matrix[i][j] == float('inf'):
                print(" INF", end="")
            else:
                print(f"{dist_matrix[i][j]:4}", end="")
        print()

# Example usage
fw_graph = Graph(4, directed=True)
fw_edges = [(0, 1, 3), (0, 3, 7), (1, 0, 8), (1, 2, 2), (2, 0, 5), (2, 3, 1), (3, 0, 2)]
for u, v, w in fw_edges:
    fw_graph.add_edge(u, v, w)

distances_matrix = floyd_warshall(fw_graph)
print_shortest_paths_matrix(distances_matrix)
```

## 🌳 Minimum Spanning Tree Algorithms

### Kruskal's Algorithm

Finds MST by sorting edges and using Union-Find data structure.

```python
class UnionFind:
    """Union-Find (Disjoint Set) data structure."""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x):
        """Find root of x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union two sets by rank."""
        root_x, root_y = self.find(x), self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.components -= 1
        return True
    
    def connected(self, x, y):
        """Check if x and y are in same component."""
        return self.find(x) == self.find(y)

def kruskals_mst(graph):
    """
    Kruskal's algorithm for Minimum Spanning Tree.
    Time: O(E log E), Space: O(V)
    """
    # Get all edges with weights
    edges = []
    for u in range(graph.num_vertices):
        for v in graph.get_neighbors(u):
            if u < v:  # Avoid duplicate edges in undirected graph
                weight = graph.get_edge_weight(u, v)
                edges.append((weight, u, v))
    
    # Sort edges by weight
    edges.sort()
    
    # Initialize Union-Find
    uf = UnionFind(graph.num_vertices)
    mst_edges = []
    total_weight = 0
    
    # Process edges in order
    for weight, u, v in edges:
        if uf.union(u, v):
            mst_edges.append((u, v, weight))
            total_weight += weight
            
            # Stop when we have V-1 edges
            if len(mst_edges) == graph.num_vertices - 1:
                break
    
    return mst_edges, total_weight

# Example usage
mst_graph = Graph(4, directed=False)
mst_edges = [(0, 1, 10), (0, 2, 6), (0, 3, 5), (1, 3, 15), (2, 3, 4)]
for u, v, w in mst_edges:
    mst_graph.add_edge(u, v, w)

mst, weight = kruskals_mst(mst_graph)
print(f"MST edges: {mst}")
print(f"Total weight: {weight}")
```

### Prim's Algorithm

Grows MST from a starting vertex by adding minimum weight edges.

```python
def prims_mst(graph, start_vertex=0):
    """
    Prim's algorithm for Minimum Spanning Tree.
    Time: O((V + E) log V), Space: O(V)
    """
    mst_edges = []
    total_weight = 0
    visited = set()
    
    # Priority queue: (weight, from_vertex, to_vertex)
    pq = []
    
    # Start with the given vertex
    visited.add(start_vertex)
    
    # Add all edges from start vertex to priority queue
    for neighbor in graph.get_neighbors(start_vertex):
        weight = graph.get_edge_weight(start_vertex, neighbor)
        heapq.heappush(pq, (weight, start_vertex, neighbor))
    
    while pq and len(visited) < graph.num_vertices:
        weight, u, v = heapq.heappop(pq)
        
        # Skip if both vertices are already in MST
        if v in visited:
            continue
        
        # Add edge to MST
        mst_edges.append((u, v, weight))
        total_weight += weight
        visited.add(v)
        
        # Add all edges from new vertex to priority queue
        for neighbor in graph.get_neighbors(v):
            if neighbor not in visited:
                edge_weight = graph.get_edge_weight(v, neighbor)
                heapq.heappush(pq, (edge_weight, v, neighbor))
    
    return mst_edges, total_weight

# Example usage with same graph
mst_prim, weight_prim = prims_mst(mst_graph)
print(f"Prim's MST edges: {mst_prim}")
print(f"Total weight: {weight_prim}")
```

## 🔗 Advanced Graph Algorithms

### Strongly Connected Components (Tarjan's Algorithm)

```python
def tarjans_scc(graph):
    """
    Tarjan's algorithm for strongly connected components.
    Time: O(V + E), Space: O(V)
    """
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    on_stack = {}
    scc_list = []
    
    def strongconnect(v):
        # Set the depth index for v
        index[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        
        # Consider successors of v
        for w in graph.get_neighbors(v):
            if w not in index:
                # Successor w has not been visited; recurse
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif on_stack[w]:
                # Successor w is in stack and hence in current SCC
                lowlinks[v] = min(lowlinks[v], index[w])
        
        # If v is a root node, pop the stack and create SCC
        if lowlinks[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            scc_list.append(component)
    
    # Find SCCs for all vertices
    for v in range(graph.num_vertices):
        if v not in index:
            strongconnect(v)
    
    return scc_list

# Example usage
scc_graph = Graph(5, directed=True)
scc_edges = [(1, 0), (0, 2), (2, 1), (0, 3), (3, 4)]
for u, v in scc_edges:
    scc_graph.add_edge(u, v)

sccs = tarjans_scc(scc_graph)
print(f"Strongly Connected Components: {sccs}")
```

### Network Flow (Ford-Fulkerson with DFS)

```python
def ford_fulkerson_dfs(graph, source, sink):
    """
    Ford-Fulkerson algorithm for maximum flow using DFS.
    Time: O(E * max_flow), Space: O(V)
    """
    def dfs_find_path(graph, source, sink, visited):
        """Find augmenting path using DFS."""
        if source == sink:
            return [sink]
        
        visited.add(source)
        
        for neighbor in graph.get_neighbors(source):
            capacity = graph.get_edge_weight(source, neighbor)
            if neighbor not in visited and capacity > 0:
                path = dfs_find_path(graph, neighbor, sink, visited)
                if path:
                    return [source] + path
        
        return None
    
    max_flow = 0
    
    # Create residual graph (copy of original)
    residual = Graph(graph.num_vertices, directed=True)
    for u in range(graph.num_vertices):
        for v in graph.get_neighbors(u):
            capacity = graph.get_edge_weight(u, v)
            residual.add_edge(u, v, capacity)
    
    while True:
        # Find augmenting path
        visited = set()
        path = dfs_find_path(residual, source, sink, visited)
        
        if not path:
            break  # No more augmenting paths
        
        # Find minimum capacity along the path
        path_flow = float('inf')
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            path_flow = min(path_flow, residual.get_edge_weight(u, v))
        
        # Update residual capacities
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # Decrease forward edge
            current_capacity = residual.get_edge_weight(u, v)
            residual.weights[(u, v)] = current_capacity - path_flow
            
            # Increase backward edge
            backward_capacity = residual.get_edge_weight(v, u)
            residual.weights[(v, u)] = backward_capacity + path_flow
        
        max_flow += path_flow
    
    return max_flow

# Example usage for max flow
flow_graph = Graph(6, directed=True)
flow_edges = [(0, 1, 16), (0, 2, 13), (1, 2, 10), (1, 3, 12), 
              (2, 1, 4), (2, 4, 14), (3, 2, 9), (3, 5, 20), 
              (4, 3, 7), (4, 5, 4)]
for u, v, c in flow_edges:
    flow_graph.add_edge(u, v, c)

max_flow_value = ford_fulkerson_dfs(flow_graph, 0, 5)
print(f"Maximum flow from 0 to 5: {max_flow_value}")
```

## 🎯 Application Examples

### Social Network Analysis

```python
def analyze_social_network(friendships):
    """Analyze a social network represented as friendships."""
    graph = Graph(len(friendships), directed=False)
    
    # Build graph from friendships
    for person, friends in friendships.items():
        for friend in friends:
            graph.add_edge(person, friend)
    
    # Find connected components (friend groups)
    components = bfs_connected_components(graph)
    
    # Calculate centrality measures
    centrality = {}
    for person in range(graph.num_vertices):
        # Degree centrality
        degree = len(graph.get_neighbors(person))
        centrality[person] = degree
    
    return {
        'friend_groups': components,
        'centrality': centrality,
        'most_popular': max(centrality.items(), key=lambda x: x[1])
    }

# Example social network
friendships = {
    0: [1, 2],
    1: [0, 2, 3],
    2: [0, 1, 3],
    3: [1, 2],
    4: [5],
    5: [4]
}

analysis = analyze_social_network(friendships)
print(f"Social network analysis: {analysis}")
```

### Route Planning

```python
def route_planner(cities, roads, start_city, end_city):
    """Plan optimal route between cities."""
    # Create graph
    graph = Graph(len(cities), directed=False)
    city_to_index = {city: i for i, city in enumerate(cities)}
    
    for city1, city2, distance in roads:
        idx1, idx2 = city_to_index[city1], city_to_index[city2]
        graph.add_edge(idx1, idx2, distance)
    
    # Find shortest path
    start_idx = city_to_index[start_city]
    end_idx = city_to_index[end_city]
    
    path_indices, total_distance = dijkstra_shortest_path(graph, start_idx, end_idx)
    
    if path_indices is None:
        return None, float('inf')
    
    # Convert back to city names
    path_cities = [cities[i] for i in path_indices]
    
    return path_cities, total_distance

# Example route planning
cities = ["New York", "Boston", "Philadelphia", "Washington DC"]
roads = [
    ("New York", "Boston", 215),
    ("New York", "Philadelphia", 95),
    ("Philadelphia", "Washington DC", 140),
    ("Boston", "Philadelphia", 300)
]

route, distance = route_planner(cities, roads, "New York", "Washington DC")
print(f"Best route: {' -> '.join(route)}")
print(f"Total distance: {distance} miles")
```

---

**Connect the dots, find the paths! 🗺️**
