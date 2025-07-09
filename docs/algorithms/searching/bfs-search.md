# BFS for Searching

## Overview

Breadth-First Search (BFS) is a graph traversal algorithm that explores all vertices at the present depth before moving to vertices at the next depth level. When applied to searching, BFS guarantees that the first occurrence of a target is found using the shortest path from the starting point, making it ideal for shortest path problems and state space exploration.

## Algorithm

1. Initialize a queue with the starting node
2. Mark the starting node as visited
3. While the queue is not empty:
   - Dequeue a node
   - If this node is the target, return it
   - Add all unvisited adjacent nodes to the queue and mark them as visited
4. If the queue becomes empty without finding the target, the target is not reachable

```python
from collections import deque

def bfs_search(graph, start, target):
    """
    Performs a BFS to search for target node from start node in a graph.
    Returns the path to the target if found, otherwise returns None.
    """
    if start == target:
        return [start]
        
    # Queue for BFS
    queue = deque([start])
    
    # To keep track of visited nodes and reconstruct the path
    visited = {start: None}  # Maps each node to its predecessor
    
    while queue:
        current = queue.popleft()
        
        # Check all adjacent vertices
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited[neighbor] = current
                
                if neighbor == target:
                    # Reconstruct the path
                    path = [neighbor]
                    while path[-1] != start:
                        path.append(visited[path[-1]])
                    
                    # Return path in correct order
                    return path[::-1]
                    
                queue.append(neighbor)
    
    # Target not found
    return None
```

## Time and Space Complexity

- **Time Complexity**: O(V + E) where V is the number of vertices and E is the number of edges
- **Space Complexity**: O(V) for the queue and visited set

## Advantages and Disadvantages

### Advantages

- Guarantees the shortest path in unweighted graphs
- Complete algorithm (will find a solution if it exists)
- Well-suited for searching in graphs with large branching factors
- Optimal for searching when the target is expected to be close to the starting point
- Simpler to implement than more complex search algorithms

### Disadvantages

- Can be memory-intensive for large graphs (stores all vertices at a given depth)
- Not suitable for weighted graphs when finding the shortest path
- May explore unnecessary nodes when the target is far from the starting point
- Not efficient when the search space is very large or infinite
- May not be the best choice for deep graphs with limited branching

## Use Cases

- Shortest path problems in unweighted graphs
- Web crawling and network traversal
- Social network connections (e.g., degrees of separation)
- Puzzle solving (e.g., sliding puzzles, maze solving)
- State space exploration in artificial intelligence
- Connected component analysis
- Level-order traversal in trees

## Implementation Details

### Python Implementation

```python
from collections import deque

def bfs_search(graph, start, target):
    """
    Performs a BFS to search for target node from start node in a graph.
    Returns the path to the target if found, otherwise returns None.
    
    Parameters:
    - graph: A dictionary representing the adjacency list of the graph
    - start: The starting node
    - target: The node to search for
    """
    if start == target:
        return [start]
        
    # Queue for BFS
    queue = deque([start])
    
    # To keep track of visited nodes and reconstruct the path
    visited = {start: None}
    
    while queue:
        current = queue.popleft()
        
        # Check all adjacent vertices
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited[neighbor] = current
                
                if neighbor == target:
                    # Reconstruct the path
                    path = [neighbor]
                    while path[-1] != start:
                        path.append(visited[path[-1]])
                    
                    # Return path in correct order
                    return path[::-1]
                    
                queue.append(neighbor)
    
    # Target not found
    return None

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

path = bfs_search(graph, 'A', 'F')
print(f"Path from A to F: {path}")  # Output: Path from A to F: ['A', 'C', 'F']
```

### Java Implementation

```java
import java.util.*;

public class BFSSearch {
    public static List<String> bfsSearch(Map<String, List<String>> graph, String start, String target) {
        if (start.equals(target)) {
            return Collections.singletonList(start);
        }
        
        // Queue for BFS
        Queue<String> queue = new LinkedList<>();
        queue.add(start);
        
        // To keep track of visited nodes and reconstruct the path
        Map<String, String> visited = new HashMap<>();
        visited.put(start, null);
        
        while (!queue.isEmpty()) {
            String current = queue.poll();
            
            // Check all adjacent vertices
            for (String neighbor : graph.getOrDefault(current, new ArrayList<>())) {
                if (!visited.containsKey(neighbor)) {
                    visited.put(neighbor, current);
                    
                    if (neighbor.equals(target)) {
                        // Reconstruct the path
                        List<String> path = new ArrayList<>();
                        String node = neighbor;
                        
                        while (node != null) {
                            path.add(node);
                            node = visited.get(node);
                        }
                        
                        // Return path in correct order
                        Collections.reverse(path);
                        return path;
                    }
                    
                    queue.add(neighbor);
                }
            }
        }
        
        // Target not found
        return null;
    }
    
    public static void main(String[] args) {
        Map<String, List<String>> graph = new HashMap<>();
        graph.put("A", Arrays.asList("B", "C"));
        graph.put("B", Arrays.asList("A", "D", "E"));
        graph.put("C", Arrays.asList("A", "F"));
        graph.put("D", Arrays.asList("B"));
        graph.put("E", Arrays.asList("B", "F"));
        graph.put("F", Arrays.asList("C", "E"));
        
        List<String> path = bfsSearch(graph, "A", "F");
        System.out.println("Path from A to F: " + path);  // Output: Path from A to F: [A, C, F]
    }
}
```

## BFS for 2D Grid Search

BFS is commonly used for searching in 2D grids, such as in maze-solving problems:

```python
from collections import deque

def grid_bfs_search(grid, start, target):
    """
    BFS search in a 2D grid.
    
    Parameters:
    - grid: 2D array where 0 represents free space and 1 represents obstacles
    - start: Tuple (row, col) representing starting position
    - target: Tuple (row, col) representing target position
    
    Returns the shortest path as a list of coordinates.
    """
    if start == target:
        return [start]
    
    rows, cols = len(grid), len(grid[0])
    
    # Possible moves (up, right, down, left)
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    # Queue for BFS
    queue = deque([start])
    
    # To keep track of visited cells and reconstruct the path
    visited = {start: None}
    
    while queue:
        row, col = queue.popleft()
        
        # Try all four directions
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check if the new position is valid
            if (0 <= new_row < rows and 0 <= new_col < cols and 
                grid[new_row][new_col] == 0 and 
                (new_row, new_col) not in visited):
                
                visited[(new_row, new_col)] = (row, col)
                
                if (new_row, new_col) == target:
                    # Reconstruct the path
                    path = [(new_row, new_col)]
                    while path[-1] != start:
                        path.append(visited[path[-1]])
                    
                    return path[::-1]
                
                queue.append((new_row, new_col))
    
    # Target not reachable
    return None
```

## Bidirectional BFS

For faster search in certain cases, we can use bidirectional BFS that searches from both the start and target:

```python
from collections import deque

def bidirectional_bfs(graph, start, target):
    """
    Bidirectional BFS to find the shortest path between start and target.
    """
    if start == target:
        return [start]
    
    # Forward BFS from start
    forward_queue = deque([start])
    forward_visited = {start: None}
    
    # Backward BFS from target
    backward_queue = deque([target])
    backward_visited = {target: None}
    
    # Function to reconstruct path from intersection
    def reconstruct_path(intersection, forward_parents, backward_parents):
        # Reconstruct path from start to intersection
        path_from_start = []
        current = intersection
        while current is not None:
            path_from_start.append(current)
            current = forward_parents[current]
        path_from_start.reverse()
        
        # Reconstruct path from intersection to target
        path_to_target = []
        current = intersection
        while current is not None:
            if current != intersection:  # Avoid duplicating the intersection
                path_to_target.append(current)
            current = backward_parents[current]
        
        # Combine paths
        return path_from_start + path_to_target
    
    while forward_queue and backward_queue:
        # Forward BFS step
        current = forward_queue.popleft()
        for neighbor in graph[current]:
            if neighbor not in forward_visited:
                forward_visited[neighbor] = current
                forward_queue.append(neighbor)
                
                # Check for intersection with backward search
                if neighbor in backward_visited:
                    return reconstruct_path(neighbor, forward_visited, backward_visited)
        
        # Backward BFS step
        current = backward_queue.popleft()
        for neighbor in graph[current]:
            if neighbor not in backward_visited:
                backward_visited[neighbor] = current
                backward_queue.append(neighbor)
                
                # Check for intersection with forward search
                if neighbor in forward_visited:
                    return reconstruct_path(neighbor, forward_visited, backward_visited)
    
    # No path found
    return None
```

## Interview Tips

- Explain the key differences between BFS and DFS for search problems
- Highlight when BFS is the preferred choice (shortest path, state space exploration)
- Discuss how to optimize BFS for memory usage in large graphs
- Explain how to track and reconstruct the path using parent pointers
- Mention bidirectional BFS as an optimization technique for certain problems
- Discuss adaptations of BFS for different problem types (grid search, shortest path, etc.)

## Practice Problems

1. Find the shortest path in a maze represented as a 2D grid
2. Determine the minimum number of moves to solve a sliding puzzle
3. Find the shortest word transformation sequence (word ladder problem)
4. Calculate the minimum number of steps to reach a target in an infinite grid
5. Find the minimum number of knight moves to reach a target position on a chessboard
