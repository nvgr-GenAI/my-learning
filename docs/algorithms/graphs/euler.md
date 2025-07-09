# Euler Path & Circuit

An Euler path is a path in a graph that visits every edge exactly once. An Euler circuit (or Euler cycle) is an Euler path that starts and ends at the same vertex.

## Overview

=== "Definition"

    - **Euler Path**: A path in a graph that visits every edge exactly once
    - **Euler Circuit**: An Euler path that starts and ends at the same vertex
    
    The existence of Euler paths and circuits depends on the degree of vertices (the number of edges connected to a vertex):
    
    - A connected undirected graph has an Euler circuit if and only if every vertex has an even degree
    - A connected undirected graph has an Euler path if and only if exactly two vertices have odd degree (these vertices must be the start and end points)
    - A connected directed graph has an Euler circuit if and only if every vertex has equal in-degree and out-degree
    - A connected directed graph has an Euler path if and only if at most one vertex has (out-degree) - (in-degree) = 1, at most one vertex has (in-degree) - (out-degree) = 1, and all other vertices have equal in-degree and out-degree

    ![Euler Path Example](https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Labelled_Eulergraph.svg/440px-Labelled_Eulergraph.svg.png)

=== "History"

    The concept of Euler paths and circuits originated from the famous "Seven Bridges of Königsberg" problem, posed and solved by Leonhard Euler in 1736. This problem asked whether it was possible to walk through the city of Königsberg (now Kaliningrad) and cross each of its seven bridges exactly once.
    
    Euler proved that this was impossible and, in doing so, laid the foundations of graph theory. The Königsberg bridge problem can be modeled as a graph, and the question becomes whether this graph contains an Euler path.
    
    ![Königsberg Bridges](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/Konigsberg_bridges.png/440px-Konigsberg_bridges.png)

=== "Applications"

    - **Circuit design**: Designing circuits that traverse each wire exactly once
    - **Network traversal**: Planning routes that use each connection exactly once
    - **DNA fragment assembly**: Reconstructing DNA sequences from fragments
    - **Puzzle solving**: Games like "draw this figure without lifting your pen or retracing lines"
    - **Chinese Postman Problem**: Finding the shortest path that visits every edge at least once (when no Euler path exists)
    - **De Bruijn sequences**: Constructing sequences where every possible subsequence of a given length appears exactly once

## Checking for Euler Paths and Circuits

=== "Undirected Graphs"

    For undirected graphs, the criteria for the existence of Euler paths and circuits are:
    
    **Euler Circuit**:
    1. All vertices with non-zero degree are connected (there's only one connected component with edges)
    2. All vertices have even degree
    
    **Euler Path**:
    1. All vertices with non-zero degree are connected
    2. Either all vertices have even degree (Euler circuit), or exactly two vertices have odd degree (these would be the start and end vertices)
    
    ```python
    def check_euler_undirected(graph):
        """
        Check if an undirected graph has Euler path or circuit.
        
        Args:
            graph: Dictionary representing adjacency list of graph
            
        Returns:
            str: "circuit" if has Euler circuit, "path" if has Euler path, "none" otherwise
        """
        # Count degrees and check connectivity
        odd_count = 0
        for node in graph:
            if len(graph[node]) % 2 == 1:
                odd_count += 1
        
        if odd_count == 0:
            return "circuit"
        elif odd_count == 2:
            return "path"
        else:
            return "none"
    ```

=== "Directed Graphs"

    For directed graphs, the criteria are:
    
    **Euler Circuit**:
    1. All vertices with non-zero degree are in a single strongly connected component
    2. For each vertex, in-degree equals out-degree
    
    **Euler Path**:
    1. All vertices with non-zero degree are connected if the graph is treated as undirected
    2. At most one vertex has out-degree - in-degree = 1 (start vertex)
    3. At most one vertex has in-degree - out-degree = 1 (end vertex)
    4. All other vertices have equal in-degree and out-degree
    
    ```python
    def check_euler_directed(graph):
        """
        Check if a directed graph has Euler path or circuit.
        
        Args:
            graph: Dictionary representing adjacency list of graph
            
        Returns:
            str: "circuit" if has Euler circuit, "path" if has Euler path, "none" otherwise
        """
        # Calculate in-degree and out-degree for each node
        in_degree = {node: 0 for node in graph}
        out_degree = {node: len(neighbors) for node, neighbors in graph.items()}
        
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
        
        # Check degree conditions
        start_node_count = 0
        end_node_count = 0
        for node in graph:
            if out_degree[node] - in_degree[node] == 1:
                start_node_count += 1
            elif in_degree[node] - out_degree[node] == 1:
                end_node_count += 1
            elif in_degree[node] != out_degree[node]:
                return "none"
        
        if start_node_count == 0 and end_node_count == 0:
            return "circuit"
        elif start_node_count == 1 and end_node_count == 1:
            return "path"
        else:
            return "none"
    ```

## Finding Euler Paths and Circuits

=== "Hierholzer's Algorithm"

    Hierholzer's algorithm is the most efficient method for finding Euler paths and circuits:
    
    1. Start from any vertex for an Euler circuit, or from an odd-degree vertex for an Euler path
    2. Follow edges, deleting them as you go, until you get stuck at a vertex
    3. Backtrack to the nearest vertex with unused edges and repeat
    4. The final path is constructed by inserting smaller cycles into the main cycle
    
    **Time Complexity**: O(E) where E is the number of edges
    
    **Space Complexity**: O(E) for storing the path and visited edges

=== "Implementation"

    ```python
    def find_euler_path(graph, directed=False):
        """
        Find an Euler path or circuit in the graph using Hierholzer's algorithm.
        
        Args:
            graph: Dictionary representing adjacency list of graph
            directed: Boolean indicating if the graph is directed
            
        Returns:
            list: The Euler path or circuit as a list of vertices, or empty list if none exists
        """
        # Create a copy of the graph to modify
        graph_copy = {node: list(neighbors) for node, neighbors in graph.items()}
        
        # Find a valid starting vertex
        start_vertex = None
        
        if directed:
            # Calculate in-degree and out-degree for directed graph
            in_degree = {node: 0 for node in graph}
            for node in graph:
                for neighbor in graph[node]:
                    in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
            
            # Find start vertex (prefer vertex with out_degree - in_degree = 1)
            for node in graph:
                out_degree = len(graph[node])
                if out_degree - in_degree.get(node, 0) == 1:
                    start_vertex = node
                    break
            
            # If no suitable starting vertex found, use any vertex with edges
            if start_vertex is None:
                for node in graph:
                    if graph[node]:
                        start_vertex = node
                        break
        else:
            # For undirected graph, prefer vertex with odd degree
            for node in graph:
                if len(graph[node]) % 2 == 1:
                    start_vertex = node
                    break
            
            # If no odd degree vertex, use any vertex with edges
            if start_vertex is None:
                for node in graph:
                    if graph[node]:
                        start_vertex = node
                        break
        
        # If no starting vertex found, graph has no edges
        if start_vertex is None:
            return []
        
        # Apply Hierholzer's algorithm
        path = []
        
        def dfs(node):
            while graph_copy[node]:
                neighbor = graph_copy[node].pop(0)
                
                # For undirected graphs, remove reverse edge too
                if not directed:
                    graph_copy[neighbor].remove(node)
                
                dfs(neighbor)
            
            path.append(node)
        
        dfs(start_vertex)
        
        # Reverse path to get the correct order
        path.reverse()
        
        # Check if all edges are used
        for node in graph_copy:
            if graph_copy[node]:
                return []  # Not all edges could be traversed
        
        return path
    ```

=== "Example"

    ```python
    # Example usage
    # Undirected graph with an Euler circuit
    euler_circuit_graph = {
        'A': ['B', 'C'],
        'B': ['A', 'C', 'D'],
        'C': ['A', 'B', 'D'],
        'D': ['B', 'C']
    }
    
    # Undirected graph with an Euler path but no circuit
    euler_path_graph = {
        'A': ['B'],
        'B': ['A', 'C', 'D'],
        'C': ['B', 'D'],
        'D': ['B', 'C', 'E'],
        'E': ['D']
    }
    
    print("Euler circuit:", find_euler_path(euler_circuit_graph))
    # Possible output: ['A', 'B', 'C', 'D', 'B', 'A', 'C', 'B']
    
    print("Euler path:", find_euler_path(euler_path_graph))
    # Possible output: ['A', 'B', 'C', 'D', 'B', 'D', 'E']
    ```

## Variations and Related Problems

=== "Chinese Postman Problem"

    The Chinese Postman Problem (or Route Inspection Problem) asks for the shortest circuit that visits every edge at least once. When the graph has an Euler circuit, that circuit is the optimal solution. Otherwise, we need to duplicate some edges to create an Euler circuit.
    
    Steps to solve:
    1. Identify all odd-degree vertices
    2. Find the minimum-weight perfect matching between these vertices
    3. Duplicate the edges corresponding to this matching
    4. Find an Euler circuit in the modified graph

=== "De Bruijn Sequences"

    A De Bruijn sequence of order n on a size-k alphabet is a cyclic sequence in which every possible length-n string on the alphabet appears exactly once as a substring. These sequences can be constructed by finding Euler paths in De Bruijn graphs.
    
    For example, the De Bruijn sequence of order 2 on {0,1} is 00110, as it contains all binary strings of length 2: 00, 01, 11, 10 (and cycling back to 00).

=== "DNA Sequencing"

    Euler paths are used in DNA sequencing algorithms. The DNA is broken into overlapping fragments, and these fragments form the vertices of a graph. Edges connect fragments that overlap. Finding an Euler path in this graph helps reconstruct the original DNA sequence.

## Practical Considerations

=== "Implementation Efficiency"

    - Use adjacency lists rather than matrices for sparse graphs
    - For undirected graphs, maintain a separate set of available edges for each vertex to avoid searching for reverse edges
    - Consider using a stack instead of recursion for very large graphs to avoid stack overflow

=== "Edge Cases"

    - Empty graphs: Trivially have an Euler circuit (an empty path)
    - Single vertex graphs: Have an Euler circuit (again, an empty path)
    - Disconnected graphs: Cannot have an Euler path or circuit unless all edges are in a single component
    - Multigraphs (graphs with multiple edges between vertices): The same rules apply, but degrees count the total number of edge ends

## References

- [Euler Path and Circuit on Wikipedia](https://en.wikipedia.org/wiki/Eulerian_path)
- [Seven Bridges of Königsberg](https://en.wikipedia.org/wiki/Seven_Bridges_of_K%C3%B6nigsberg)
- [Hierholzer's Algorithm](https://en.wikipedia.org/wiki/Eulerian_path#Hierholzer's_algorithm)
- [Chinese Postman Problem](https://en.wikipedia.org/wiki/Route_inspection_problem)
- [De Bruijn Sequences](https://en.wikipedia.org/wiki/De_Bruijn_sequence)
