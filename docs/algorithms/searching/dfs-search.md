# DFS for Searching

## Overview

Depth-First Search (DFS) is a graph traversal algorithm that explores as far as possible along each branch before backtracking. In the context of search problems, DFS goes deep into one path before trying alternatives. This approach is memory-efficient and well-suited for problems where solutions are likely to be found deep in the search tree.

## Algorithm

1. Start at a given node (root)
2. Explore the first unexplored neighbor completely before moving to the next neighbor
3. Use a stack (or recursion) to keep track of nodes to explore
4. Mark nodes as visited to avoid cycles
5. Continue until the target is found or all reachable nodes are visited

```python
def dfs_search(graph, start, target):
    """
    Performs DFS to find a path from start to target in a graph.
    Returns a path if found, otherwise returns None.
    """
    # Stack for DFS (node, path_so_far)
    stack = [(start, [start])]
    visited = set([start])
    
    while stack:
        current, path = stack.pop()
        
        if current == target:
            return path
            
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))
    
    # Target not found
    return None
```

## Time and Space Complexity

- **Time Complexity**: O(V + E) where V is the number of vertices and E is the number of edges
- **Space Complexity**: O(V) for the stack and visited set (worst case when the graph is a tree)

## Advantages and Disadvantages

### Advantages

- Memory-efficient compared to BFS (only needs to store a single path at any time)
- Well-suited for problems where solutions are deep in the tree
- Good for solving puzzles with many dead ends
- Natural implementation using recursion
- Can be used to detect cycles in a graph
- Easier to implement for backtracking problems

### Disadvantages

- Does not guarantee the shortest path (unlike BFS)
- Can get stuck in deep branches that don't lead to a solution
- Risk of stack overflow for deep graphs when using recursion
- Not optimal for finding all nodes at a given depth
- May not terminate on infinite or very large graphs without proper termination conditions

## Use Cases

- Solving mazes and puzzles
- Topological sorting
- Cycle detection in graphs
- Connected component analysis
- Finding paths in graphs (not necessarily shortest)
- Game trees and decision trees exploration
- Web crawling (depth-first exploration)
- Analyzing code control flow

## Implementation Details

### Recursive DFS Implementation

```python
def dfs_search_recursive(graph, current, target, path=None, visited=None):
    """
    Recursive DFS implementation to find a path from current to target.
    """
    if path is None:
        path = [current]
    if visited is None:
        visited = set([current])
        
    if current == target:
        return path
        
    for neighbor in graph[current]:
        if neighbor not in visited:
            visited.add(neighbor)
            new_path = path + [neighbor]
            result = dfs_search_recursive(graph, neighbor, target, new_path, visited)
            if result:
                return result
    
    return None
```

### Iterative DFS Implementation

```python
def dfs_search_iterative(graph, start, target):
    """
    Iterative DFS implementation using an explicit stack.
    """
    if start == target:
        return [start]
        
    stack = [(start, [start])]
    visited = set([start])
    
    while stack:
        current, path = stack.pop()
        
        for neighbor in graph[current]:
            if neighbor not in visited:
                if neighbor == target:
                    return path + [neighbor]
                    
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))
    
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

path = dfs_search_iterative(graph, 'A', 'F')
print(f"Path from A to F: {path}")  # Possible output: Path from A to F: ['A', 'C', 'F'] or ['A', 'B', 'E', 'F']
```

### Java Implementation

```java
import java.util.*;

public class DFSSearch {
    public static List<String> dfsSearchIterative(Map<String, List<String>> graph, String start, String target) {
        if (start.equals(target)) {
            return Collections.singletonList(start);
        }
        
        // Stack to store node and path pairs
        Stack<Map.Entry<String, List<String>>> stack = new Stack<>();
        List<String> initialPath = new ArrayList<>();
        initialPath.add(start);
        stack.push(new AbstractMap.SimpleEntry<>(start, initialPath));
        
        Set<String> visited = new HashSet<>();
        visited.add(start);
        
        while (!stack.isEmpty()) {
            Map.Entry<String, List<String>> entry = stack.pop();
            String current = entry.getKey();
            List<String> path = entry.getValue();
            
            for (String neighbor : graph.getOrDefault(current, new ArrayList<>())) {
                if (!visited.contains(neighbor)) {
                    List<String> newPath = new ArrayList<>(path);
                    newPath.add(neighbor);
                    
                    if (neighbor.equals(target)) {
                        return newPath;
                    }
                    
                    visited.add(neighbor);
                    stack.push(new AbstractMap.SimpleEntry<>(neighbor, newPath));
                }
            }
        }
        
        return null;  // Target not found
    }
    
    public static void main(String[] args) {
        Map<String, List<String>> graph = new HashMap<>();
        graph.put("A", Arrays.asList("B", "C"));
        graph.put("B", Arrays.asList("A", "D", "E"));
        graph.put("C", Arrays.asList("A", "F"));
        graph.put("D", Arrays.asList("B"));
        graph.put("E", Arrays.asList("B", "F"));
        graph.put("F", Arrays.asList("C", "E"));
        
        List<String> path = dfsSearchIterative(graph, "A", "F");
        System.out.println("Path from A to F: " + path);
    }
}
```

## DFS for Grid Search

DFS is commonly used for searching in mazes and 2D grids:

```python
def grid_dfs_search(grid, start, target):
    """
    DFS search in a 2D grid.
    
    Parameters:
    - grid: 2D array where 0 represents free space and 1 represents obstacles
    - start: Tuple (row, col) representing starting position
    - target: Tuple (row, col) representing target position
    
    Returns a path as a list of coordinates.
    """
    rows, cols = len(grid), len(grid[0])
    visited = set([start])
    
    # Possible moves (up, right, down, left)
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    def dfs(current, path):
        row, col = current
        
        if current == target:
            return path
            
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            new_pos = (new_row, new_col)
            
            # Check if the move is valid
            if (0 <= new_row < rows and 0 <= new_col < cols and
                grid[new_row][new_col] == 0 and new_pos not in visited):
                
                visited.add(new_pos)
                result = dfs(new_pos, path + [new_pos])
                
                if result:
                    return result
                    
                # Backtracking (optional, depending on if we want to find all paths)
                visited.remove(new_pos)
        
        return None
    
    return dfs(start, [start])
```

## Iterative Deepening DFS

For large or infinite search spaces, we can use iterative deepening DFS to get the benefits of both DFS and BFS:

```python
def iterative_deepening_dfs(graph, start, target, max_depth=float('inf')):
    """
    Iterative Deepening DFS that combines the space efficiency of DFS 
    with the completeness of BFS.
    """
    def dfs_with_depth_limit(node, target, depth_limit, path=None, visited=None):
        if path is None:
            path = [node]
        if visited is None:
            visited = set([node])
            
        if node == target:
            return path
            
        if depth_limit <= 0:
            return None
            
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                result = dfs_with_depth_limit(neighbor, target, depth_limit - 1, 
                                            path + [neighbor], visited.copy())
                if result:
                    return result
        
        return None
    
    # Try increasing depths until we find the target or reach max_depth
    for depth in range(1, max_depth + 1):
        result = dfs_with_depth_limit(start, target, depth)
        if result:
            return result
    
    return None
```

## Interview Tips

- Compare DFS with BFS and explain when each is preferable
- Discuss the space efficiency advantage of DFS over BFS
- Explain how to handle cycles in graphs using the visited set
- Demonstrate both recursive and iterative implementations
- Mention iterative deepening DFS as a way to get completeness in large search spaces
- Discuss backtracking as an extension of DFS for constraint satisfaction problems
- Be prepared to apply DFS to common problems like maze solving, topological sorting, or connected components

## Practice Problems

1. Find a path through a maze using DFS
2. Detect cycles in a directed graph
3. Find all possible paths between two vertices in a graph
4. Implement a word search algorithm for a 2D grid of letters
5. Solve the n-Queens problem using DFS and backtracking
6. Find connected components in an undirected graph
7. Implement topological sorting using DFS
