# Network Flow

Network flow algorithms model the movement of items through a network with capacity constraints. These algorithms are fundamental in solving many optimization problems involving transportation, resource allocation, and matching.

## Overview

=== "Definition"

    A **flow network** is a directed graph where:
    
    - Each edge has a non-negative capacity (maximum flow that can pass through)
    - There is a source vertex (where flow originates)
    - There is a sink vertex (where flow terminates)
    - Flow conservation: For each vertex (except source and sink), the incoming flow equals the outgoing flow
    
    The **maximum flow problem** asks for the greatest amount of flow that can be sent from source to sink while respecting capacity constraints.
    
    ![Flow Network Example](https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/Max_flow.svg/440px-Max_flow.svg.png)
    
    In the image above, edge labels indicate capacity/flow. The maximum flow value is 5.

=== "Applications"

    Network flow algorithms have diverse applications:
    
    - **Transportation networks**: Modeling traffic flow, logistics, airlines
    - **Resource allocation**: Assigning tasks to workers, machines to jobs
    - **Bipartite matching**: Job assignments, college admissions, organ donation
    - **Image segmentation**: Separating foreground from background
    - **Sports scheduling**: Creating tournament schedules
    - **Network reliability**: Finding the minimum cut in a network
    - **Supply chain optimization**: Managing distribution networks
    - **Communication networks**: Routing packets in computer networks

=== "Key Concepts"

    - **Flow**: The amount of material passing through each edge
    - **Capacity**: The maximum amount that can pass through an edge
    - **Residual capacity**: The additional flow that can be added to an edge
    - **Augmenting path**: A path from source to sink in the residual network
    - **Residual network**: A network showing remaining capacity on each edge
    - **Cut**: A partition of vertices into two sets, one containing the source and one containing the sink
    - **Min-cut**: A cut with minimum capacity (sum of capacities of edges crossing from source side to sink side)
    - **Max-flow min-cut theorem**: The maximum flow equals the minimum cut capacity

## Ford-Fulkerson Algorithm

=== "Algorithm Description"

    The Ford-Fulkerson algorithm finds the maximum flow in a network:
    
    1. Initialize all flows to zero
    2. While there exists an augmenting path from source to sink in the residual network:
       a. Find the bottleneck capacity (minimum residual capacity along the path)
       b. Augment the flow along this path by the bottleneck capacity
       c. Update the residual network
    3. Return the total flow
    
    An augmenting path can be found using BFS (Edmonds-Karp algorithm) or DFS.
    
    The time complexity depends on how augmenting paths are found:
    - Using DFS: O(E·max_flow), where E is the number of edges and max_flow is the maximum flow value
    - Using BFS (Edmonds-Karp): O(V·E²), where V is the number of vertices and E is the number of edges

=== "Implementation"

    ```python
    def ford_fulkerson(graph, source, sink):
        """
        Find maximum flow using Ford-Fulkerson algorithm with BFS (Edmonds-Karp).
        
        Args:
            graph: Dictionary representing adjacency matrix with capacities
            source: Source vertex
            sink: Sink vertex
            
        Returns:
            int: Maximum flow
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
                if v in flow[u]:
                    flow[u][v] += bottleneck
                else:
                    flow[u][v] = bottleneck
                    
                # Add reverse edge for residual network
                if u in flow.get(v, {}):
                    flow[v][u] -= bottleneck
                elif v in graph:  # Ensure v exists in graph
                    if u not in flow.get(v, {}):
                        flow.setdefault(v, {})[u] = 0
                    flow[v][u] -= bottleneck
            
            max_flow += bottleneck
        
        return max_flow, flow
    
    # Example usage
    graph = {
        'S': {'A': 4, 'B': 3},
        'A': {'B': 2, 'T': 2},
        'B': {'T': 3},
        'T': {}
    }
    
    max_flow, flow_dict = ford_fulkerson(graph, 'S', 'T')
    print(f"Maximum flow: {max_flow}")
    print("Flow network:")
    for u in flow_dict:
        for v, f in flow_dict[u].items():
            if f > 0:
                print(f"{u} -> {v}: {f}")
    ```

=== "Visualization"

    Step-by-step visualization of Ford-Fulkerson algorithm on a simple network:
    
    1. **Initial network** (capacities shown on edges)
       ```
       S → A (4)
       S → B (3)
       A → B (2)
       A → T (2)
       B → T (3)
       ```
    
    2. **First augmenting path**: S → A → T with bottleneck 2
       ```
       Updated flows:
       S → A: 2
       A → T: 2
       Current max flow: 2
       ```
    
    3. **Second augmenting path**: S → B → T with bottleneck 3
       ```
       Updated flows:
       S → A: 2
       A → T: 2
       S → B: 3
       B → T: 3
       Current max flow: 5
       ```
    
    4. **Third augmenting path**: S → A → B → T with bottleneck 2
       ```
       Wait, A → B capacity is 2, but we need residual capacity here.
       S → A has 2/4 flow, so 2 more available
       A → B has 0/2 flow, so 2 more available
       B → T has 3/3 flow, so 0 more available
       
       But we can use the reverse flow B → A with capacity equal to current flow A → B, which is 0.
       So this path doesn't work.
       ```
    
    5. **No more augmenting paths**
       ```
       Maximum flow: 5
       ```

## Dinic's Algorithm

=== "Algorithm Description"

    Dinic's algorithm is an improvement over Edmonds-Karp, using the concept of level graphs and blocking flows:
    
    1. Construct a level graph (assign levels to vertices based on their shortest path distance from the source)
    2. Find a blocking flow (a flow where any path from source to sink uses at least one saturated edge)
    3. Update the residual network
    4. Repeat until no more augmenting paths exist
    
    Time Complexity: O(V²·E)

=== "Implementation"

    ```python
    def dinic(graph, source, sink):
        """
        Find maximum flow using Dinic's algorithm.
        
        Args:
            graph: Dictionary representing adjacency matrix with capacities
            source: Source vertex
            sink: Sink vertex
            
        Returns:
            int: Maximum flow
        """
        # Create residual graph
        residual = {u: {v: graph[u][v] for v in graph[u]} for u in graph}
        
        def bfs():
            """Construct level graph and check if more flow is possible."""
            level = {source: 0}
            queue = [source]
            i = 0
            
            while i < len(queue):
                u = queue[i]
                i += 1
                
                for v in residual[u]:
                    if v not in level and residual[u][v] > 0:
                        level[v] = level[u] + 1
                        queue.append(v)
            
            return sink in level, level
        
        def dfs(u, flow, level, next_edge):
            """Find a blocking flow using DFS."""
            if u == sink:
                return flow
            
            while next_edge.setdefault(u, 0) < len(residual[u].keys()):
                v = list(residual[u].keys())[next_edge[u]]
                next_edge[u] += 1
                
                if v in level and level[v] == level[u] + 1 and residual[u][v] > 0:
                    bottleneck = dfs(v, min(flow, residual[u][v]), level, next_edge)
                    
                    if bottleneck > 0:
                        residual[u][v] -= bottleneck
                        residual.setdefault(v, {})[u] = residual.get(v, {}).get(u, 0) + bottleneck
                        return bottleneck
            
            return 0
        
        max_flow = 0
        
        while True:
            has_path, level = bfs()
            if not has_path:
                break
                
            next_edge = {}
            
            while True:
                flow = dfs(source, float('inf'), level, next_edge)
                if flow <= 0:
                    break
                max_flow += flow
        
        return max_flow
    ```

## Push-Relabel Algorithm

=== "Algorithm Description"

    The Push-Relabel algorithm takes a different approach:
    
    1. Maintain a preflow (may violate flow conservation by having excess flow at vertices)
    2. Assign height labels to vertices
    3. Perform two operations:
       - Push: Move excess flow from a vertex to a lower-height neighbor
       - Relabel: Increase the height of a vertex when no more pushes are possible
    4. Repeat until no excess flow remains at any vertex except source and sink
    
    This algorithm has better theoretical time complexity: O(V²·E) or O(V³) depending on implementation.

=== "Implementation"

    ```python
    def push_relabel(graph, source, sink):
        """
        Find maximum flow using Push-Relabel algorithm.
        
        Args:
            graph: Dictionary representing adjacency matrix with capacities
            source: Source vertex
            sink: Sink vertex
            
        Returns:
            int: Maximum flow
        """
        vertices = list(graph.keys())
        n = len(vertices)
        
        # Initialize height and excess flow
        height = {v: 0 for v in vertices}
        excess = {v: 0 for v in vertices}
        flow = {u: {v: 0 for v in graph[u]} for u in graph}
        
        # Initialize source height and push flow from source
        height[source] = n
        
        for v in graph[source]:
            flow[source][v] = graph[source][v]
            flow.setdefault(v, {})[source] = -graph[source][v]  # Reverse edge
            excess[v] = graph[source][v]
            excess[source] -= graph[source][v]
        
        def push(u, v):
            """Push flow from u to v."""
            send = min(excess[u], graph[u].get(v, 0) - flow[u].get(v, 0))
            flow[u][v] += send
            flow.setdefault(v, {})[u] = flow.get(v, {}).get(u, 0) - send
            excess[u] -= send
            excess[v] += send
            return send > 0
            
        def relabel(u):
            """Relabel vertex u."""
            min_height = float('inf')
            for v in graph[u]:
                if graph[u][v] - flow[u].get(v, 0) > 0:
                    min_height = min(min_height, height.get(v, 0))
            
            if min_height < float('inf'):
                height[u] = min_height + 1
                return True
            return False
            
        def discharge(u):
            """Discharge excess flow from vertex u."""
            while excess[u] > 0:
                for v in graph[u]:
                    if graph[u][v] - flow[u].get(v, 0) > 0 and height[u] == height.get(v, 0) + 1:
                        if push(u, v):
                            return True
                
                if not relabel(u):
                    break
            
            return False
        
        # Main algorithm
        active = [v for v in vertices if v != source and v != sink and excess[v] > 0]
        
        while active:
            u = active[0]
            old_excess = excess[u]
            
            discharge(u)
            
            if excess[u] < old_excess:
                # If excess was reduced but not eliminated, move u to the back
                if excess[u] > 0:
                    active.append(active.pop(0))
                else:
                    active.pop(0)
            else:
                # No progress was made, try a different vertex
                active.append(active.pop(0))
                
            # Add new overflowing vertices
            for v in vertices:
                if v != source and v != sink and v not in active and excess[v] > 0:
                    active.append(v)
        
        return sum(flow[source].values())
    ```

## Max-Flow Min-Cut Theorem

=== "Theorem Statement"

    The max-flow min-cut theorem states that in a flow network, the maximum amount of flow is equal to the minimum capacity of a cut.
    
    A **cut** is a partition of the vertices into two disjoint sets S and T such that:
    - The source s is in S
    - The sink t is in T
    
    The **capacity of a cut** is the sum of capacities of the edges from S to T.
    
    This theorem provides a duality between the maximum flow problem and the minimum cut problem: finding one immediately gives the other.

=== "Finding the Minimum Cut"

    After finding the maximum flow with Ford-Fulkerson or any other algorithm, the minimum cut can be determined:
    
    1. Run a depth-first search or breadth-first search from the source in the residual graph
    2. Mark all reachable vertices
    3. The minimum cut consists of all edges from reachable to unreachable vertices in the original graph
    
    ```python
    def find_min_cut(graph, flow, source):
        """
        Find the minimum cut after computing the maximum flow.
        
        Args:
            graph: Original capacity graph
            flow: Flow graph after running max flow algorithm
            source: Source vertex
            
        Returns:
            tuple: Set of vertices reachable from source, set of unreachable vertices
            list: Edges in the min cut
        """
        # Create residual graph
        residual = {u: {} for u in graph}
        for u in graph:
            for v in graph[u]:
                residual[u][v] = graph[u][v] - flow[u].get(v, 0)
        
        # Run DFS to find reachable vertices
        visited = set()
        stack = [source]
        
        while stack:
            u = stack.pop()
            if u not in visited:
                visited.add(u)
                for v in residual[u]:
                    if residual[u][v] > 0:
                        stack.append(v)
        
        # Find unreachable vertices
        unreachable = set(graph.keys()) - visited
        
        # Compute min cut edges
        cut_edges = []
        for u in visited:
            for v in graph[u]:
                if v in unreachable:
                    cut_edges.append((u, v))
        
        return visited, unreachable, cut_edges
    ```

## Applications and Extensions

=== "Bipartite Matching"

    Maximum bipartite matching can be solved using network flow:
    
    1. Create a source node s and add edges from s to all vertices in the first set
    2. Add edges from all vertices in the second set to a sink node t
    3. Set all edge capacities to 1
    4. Find the maximum flow, which equals the size of the maximum matching
    
    ```python
    def max_bipartite_matching(graph, left_set, right_set):
        """
        Solve maximum bipartite matching using network flow.
        
        Args:
            graph: Dictionary representing adjacency list of bipartite graph
            left_set: Vertices in the left partition
            right_set: Vertices in the right partition
            
        Returns:
            int: Size of maximum matching
            dict: Dictionary mapping vertices in left_set to matched vertices in right_set
        """
        # Create flow network
        flow_graph = {
            'source': {u: 1 for u in left_set},
            'sink': {}
        }
        
        for u in left_set:
            flow_graph[u] = {v: 1 for v in graph[u]}
            
        for v in right_set:
            flow_graph[v] = {'sink': 1}
            
        # Find maximum flow
        max_flow, flow_dict = ford_fulkerson(flow_graph, 'source', 'sink')
        
        # Extract matching
        matching = {}
        for u in left_set:
            for v in graph[u]:
                if flow_dict[u].get(v, 0) > 0:
                    matching[u] = v
        
        return max_flow, matching
    ```

=== "Multi-source Multi-sink Flow"

    To solve a problem with multiple sources and multiple sinks:
    
    1. Create a super-source s' and add edges from s' to all sources
    2. Create a super-sink t' and add edges from all sinks to t'
    3. Set the capacities of these new edges to infinity (or a very large value)
    4. Solve the single-source single-sink problem from s' to t'

=== "Minimum Cost Flow"

    In the minimum cost flow problem, each edge has both a capacity and a cost per unit flow. The goal is to find the maximum flow with minimum total cost.
    
    Algorithms like the Successive Shortest Path algorithm or the Network Simplex algorithm can solve this problem.

=== "Circulation Problems"

    In a circulation problem, there is no source or sink, but each vertex may have supply or demand. The goal is to find a flow that satisfies all supplies and demands.
    
    This can be reduced to a standard max flow problem by adding a source and a sink.

## References

- [Maximum Flow Problem on Wikipedia](https://en.wikipedia.org/wiki/Maximum_flow_problem)
- [Ford-Fulkerson Algorithm](https://en.wikipedia.org/wiki/Ford%E2%80%93Fulkerson_algorithm)
- [Edmonds-Karp Algorithm](https://en.wikipedia.org/wiki/Edmonds%E2%80%93Karp_algorithm)
- [Dinic's Algorithm](https://en.wikipedia.org/wiki/Dinic%27s_algorithm)
- [Push-Relabel Algorithm](https://en.wikipedia.org/wiki/Push%E2%80%93relabel_maximum_flow_algorithm)
- [Max-Flow Min-Cut Theorem](https://en.wikipedia.org/wiki/Max-flow_min-cut_theorem)
- [Introduction to Algorithms (CLRS)](https://en.wikipedia.org/wiki/Introduction_to_Algorithms) - Chapter 26: Maximum Flow
