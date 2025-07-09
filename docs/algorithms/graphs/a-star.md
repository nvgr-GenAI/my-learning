# A* Search Algorithm

## Overview

A* (pronounced "A-star") is an informed search algorithm that finds the shortest path between nodes in a graph. It combines the advantages of both Dijkstra's algorithm and Greedy Best-First-Search by using heuristics to guide its search.

## Definition

A* evaluates nodes by combining:
1. **g(n)**: The exact cost to reach the node from the start
2. **h(n)**: A heuristic estimate of the cost to reach the goal from the node

The total evaluation function is **f(n) = g(n) + h(n)**

## Algorithm

1. Initialize an open list (priority queue) with the start node, with f(start) = h(start)
2. Initialize a closed list (set of visited nodes) as empty
3. While the open list is not empty:
   - Get the node with the lowest f(n) value from the open list
   - If this node is the goal, reconstruct and return the path
   - Move this node from the open list to the closed list
   - For each neighbor of the current node:
     - If the neighbor is in the closed list, skip it
     - Calculate tentative g score for the neighbor
     - If the neighbor is not in the open list or the tentative g score is better than the current one:
       - Update the neighbor's g score
       - Update the neighbor's f score: f(n) = g(n) + h(n)
       - If the neighbor is not in the open list, add it
4. If the open list is empty and goal not found, return failure

## Implementation

```python
import heapq

def a_star_search(graph, start, goal, h):
    """
    A* search algorithm to find shortest path from start to goal.
    
    Parameters:
    - graph: A dict of dicts where graph[u][v] gives weight of edge (u,v)
    - start: Starting node
    - goal: Target node
    - h: A heuristic function h(n) that estimates cost from n to goal
    
    Returns:
    - The shortest path from start to goal as a list of nodes
    - The total cost of the path
    """
    # Priority queue for open nodes
    open_set = []
    heapq.heappush(open_set, (h(start), 0, start))  # (f, g, node)
    
    # For path reconstruction
    came_from = {}
    
    # g[n] = exact cost from start to n
    g_score = {node: float('infinity') for node in graph}
    g_score[start] = 0
    
    # f[n] = g[n] + h(n)
    f_score = {node: float('infinity') for node in graph}
    f_score[start] = h(start)
    
    # Set to track visited nodes
    closed_set = set()
    
    while open_set:
        # Get node with lowest f_score
        current_f, current_g, current = heapq.heappop(open_set)
        
        # If we reached the goal, reconstruct and return path
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, current_g
        
        # Mark as visited
        closed_set.add(current)
        
        # Check all neighbors
        for neighbor, weight in graph[current].items():
            # Skip if already evaluated
            if neighbor in closed_set:
                continue
            
            # Calculate tentative g score
            tentative_g = g_score[current] + weight
            
            # If we found a better path to neighbor
            if tentative_g < g_score[neighbor]:
                # Update path
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = g_score[neighbor] + h(neighbor)
                
                # Add to open set if not already there
                if all(neighbor != node for _, _, node in open_set):
                    heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], neighbor))
    
    # If we get here, no path was found
    return None, float('infinity')

# Example usage
if __name__ == "__main__":
    # Example graph as adjacency list with weights
    graph = {
        'A': {'B': 1, 'C': 3},
        'B': {'A': 1, 'D': 5, 'E': 1},
        'C': {'A': 3, 'F': 2},
        'D': {'B': 5, 'G': 2},
        'E': {'B': 1, 'G': 1},
        'F': {'C': 2, 'G': 5},
        'G': {'D': 2, 'E': 1, 'F': 5}
    }
    
    # Manhattan distance heuristic (for grid-like graphs)
    def manhattan_distance(node):
        # In this example, we'll use simple distance to 'G'
        coords = {
            'A': (0, 0), 'B': (1, 0), 'C': (0, 1),
            'D': (2, 0), 'E': (1, 1), 'F': (0, 2),
            'G': (2, 1)
        }
        goal_coords = coords['G']
        node_coords = coords[node]
        return abs(node_coords[0] - goal_coords[0]) + abs(node_coords[1] - goal_coords[1])
    
    path, cost = a_star_search(graph, 'A', 'G', manhattan_distance)
    print(f"Path: {path}")
    print(f"Cost: {cost}")
```

## Heuristic Functions

The efficiency and accuracy of A* heavily depend on the heuristic function. A good heuristic should:

1. **Be admissible**: Never overestimate the actual cost to reach the goal
2. **Be consistent (or monotonic)**: For any nodes n and successor n', h(n) ≤ d(n, n') + h(n')

Common heuristic functions include:

- **Manhattan Distance**: |x1 - x2| + |y1 - y2| (good for grid movements with no diagonals)
- **Euclidean Distance**: sqrt((x1 - x2)² + (y1 - y2)²) (good for unrestricted movement)
- **Diagonal Distance**: max(|x1 - x2|, |y1 - y2|) (good for grid movements with diagonals)
- **Custom Domain-Specific Heuristics**: Based on problem characteristics

## Time and Space Complexity

- **Time Complexity**: O(b^d) in the worst case, where b is the branching factor and d is the depth of the goal
- **Space Complexity**: O(b^d) to store all nodes

With a good heuristic, the performance can be much better in practice, approaching O(d) in the best case.

## Advantages

1. **Optimality**: Guaranteed to find the shortest path if the heuristic is admissible
2. **Efficiency**: More efficient than Dijkstra's algorithm due to heuristic guidance
3. **Completeness**: Will find a path if one exists
4. **Flexibility**: Can be adapted to many different problem domains

## Disadvantages

1. **Heuristic Dependency**: Performance heavily depends on the quality of the heuristic
2. **Memory Usage**: Can use a lot of memory for large graphs
3. **Not Suitable for All Graphs**: Less effective when the heuristic cannot provide good guidance

## Applications

1. **Pathfinding in Games**: Finding routes for characters in grid-based games
2. **Robot Navigation**: Planning paths for robots in known environments
3. **Puzzle Solving**: Solving problems like the 15-puzzle or Rubik's cube
4. **Route Planning**: Finding optimal routes in road networks or transit systems
5. **Network Routing**: Finding efficient paths in computer networks

## Comparison with Other Algorithms

| Algorithm | Description | Optimality | Completeness | Time Complexity | Space Complexity |
|-----------|-------------|------------|--------------|-----------------|------------------|
| A* | Informed search using f(n) = g(n) + h(n) | Yes (with admissible h) | Yes | O(b^d) worst case | O(b^d) |
| Dijkstra | Uniform-cost search (A* with h=0) | Yes | Yes | O(E + V log V) | O(V) |
| Greedy Best-First | Only uses heuristic h(n) | No | No | O(b^d) | O(b^d) |
| BFS | Explores all nodes at each depth | Yes (for unweighted) | Yes | O(b^d) | O(b^d) |
| DFS | Explores as far as possible along each branch | No | No (can get stuck in infinite paths) | O(b^d) | O(d) |
