# A* Search Algorithm

## Overview

A* (pronounced "A-star") is an informed search algorithm that combines features of uniform-cost search (Dijkstra's algorithm) and greedy best-first search. It uses a heuristic function to estimate the cost from the current node to the goal, making it more efficient than uninformed search algorithms like BFS or Dijkstra's algorithm. A* guarantees the shortest path if the heuristic is admissible (never overestimates the true cost).

## Algorithm

1. Maintain two lists: an open list (nodes to explore) and a closed list (already explored nodes)
2. Start with the initial node in the open list
3. While the open list is not empty:
   - Select the node with lowest f(n) = g(n) + h(n) from the open list
   - If this node is the goal, reconstruct and return the path
   - Move the node to the closed list
   - Expand all neighbors not in the closed list
   - For each neighbor, calculate f(n) and add to open list if not already there or if a better path is found

Where:
- g(n) is the actual cost from start to node n
- h(n) is the heuristic estimate of cost from n to goal
- f(n) is the estimated total cost of the path through n

```python
import heapq

def a_star_search(graph, start, goal, heuristic):
    """
    A* search algorithm to find the shortest path from start to goal.
    
    Parameters:
    - graph: A dictionary where keys are nodes and values are dictionaries of neighbors with edge costs
    - start: Starting node
    - goal: Target node
    - heuristic: A function that estimates the cost from a node to the goal
    
    Returns:
    - The shortest path as a list of nodes, or None if no path exists
    """
    # Priority queue for open nodes: (f_score, node)
    open_queue = [(heuristic(start, goal), start)]
    
    # Dictionary to store g_scores (cost from start to node)
    g_score = {start: 0}
    
    # Dictionary to store f_scores (estimated total cost)
    f_score = {start: heuristic(start, goal)}
    
    # Dictionary to reconstruct the path
    came_from = {}
    
    # Set to keep track of nodes in the open queue
    open_set = {start}
    
    # Set to keep track of explored nodes
    closed_set = set()
    
    while open_queue:
        # Get node with lowest f_score
        current_f, current = heapq.heappop(open_queue)
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        open_set.remove(current)
        closed_set.add(current)
        
        # Explore neighbors
        for neighbor, cost in graph[current].items():
            if neighbor in closed_set:
                continue
                
            # Calculate tentative g_score
            tentative_g_score = g_score[current] + cost
            
            if neighbor not in open_set or tentative_g_score < g_score.get(neighbor, float('inf')):
                # Found a better path to neighbor
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                
                if neighbor not in open_set:
                    heapq.heappush(open_queue, (f_score[neighbor], neighbor))
                    open_set.add(neighbor)
    
    # No path found
    return None
```

## Time and Space Complexity

- **Time Complexity**: O(b^d) in the worst case, where b is the branching factor and d is the depth of the goal. With a good heuristic, the time complexity can be significantly better.
- **Space Complexity**: O(b^d) to store the nodes in the open and closed lists

## Advantages and Disadvantages

### Advantages

- Guarantees the shortest path if the heuristic is admissible
- More efficient than uninformed search algorithms
- Combines the benefits of Dijkstra's algorithm and greedy best-first search
- Can be adapted to various problem domains with appropriate heuristics
- Widely used in pathfinding for games and robotics

### Disadvantages

- Performance depends heavily on the quality of the heuristic function
- Memory requirements can be high for large search spaces
- May explore more nodes than necessary if the heuristic is not well-designed
- Implementing an admissible and consistent heuristic can be challenging
- Not suitable for problems where a good heuristic is hard to define

## Use Cases

- Pathfinding in games and robotics
- Route planning in navigation systems
- Puzzle solving (e.g., 8-puzzle, 15-puzzle)
- Motion planning for robots
- Automated planning in artificial intelligence
- Maze solving with shortest path guarantee
- Network routing algorithms

## Implementation Details

### Python Implementation with Grid-Based Example

```python
import heapq

def a_star_grid_search(grid, start, goal):
    """
    A* search for a 2D grid.
    
    Parameters:
    - grid: 2D array where 0 represents free space and 1 represents obstacles
    - start: Tuple (row, col) representing starting position
    - goal: Tuple (row, col) representing target position
    
    Returns:
    - The shortest path as a list of coordinates, or None if no path exists
    """
    rows, cols = len(grid), len(grid[0])
    
    # Define heuristic function (Manhattan distance)
    def heuristic(pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    # Priority queue for open nodes: (f_score, position)
    open_queue = [(heuristic(start, goal), start)]
    
    # Dictionary to store g_scores (cost from start to node)
    g_score = {start: 0}
    
    # Dictionary to store f_scores (estimated total cost)
    f_score = {start: heuristic(start, goal)}
    
    # Dictionary to reconstruct the path
    came_from = {}
    
    # Set to keep track of nodes in the open queue
    open_set = {start}
    
    # Possible moves (up, right, down, left)
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    while open_queue:
        # Get position with lowest f_score
        current_f, current = heapq.heappop(open_queue)
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        open_set.remove(current)
        
        # Explore neighbors
        row, col = current
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            neighbor = (new_row, new_col)
            
            # Check if the move is valid
            if (0 <= new_row < rows and 0 <= new_col < cols and grid[new_row][new_col] == 0):
                # All moves cost 1 in this example
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Found a better path to neighbor
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    
                    if neighbor not in open_set:
                        heapq.heappush(open_queue, (f_score[neighbor], neighbor))
                        open_set.add(neighbor)
    
    # No path found
    return None
```

### Java Implementation

```java
import java.util.*;

public class AStarSearch {
    static class Node implements Comparable<Node> {
        int row, col;
        int gScore;  // Cost from start to this node
        int fScore;  // Estimated total cost (g + h)
        
        Node(int row, int col, int gScore, int fScore) {
            this.row = row;
            this.col = col;
            this.gScore = gScore;
            this.fScore = fScore;
        }
        
        @Override
        public int compareTo(Node other) {
            return Integer.compare(this.fScore, other.fScore);
        }
        
        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            Node node = (Node) obj;
            return row == node.row && col == node.col;
        }
        
        @Override
        public int hashCode() {
            return Objects.hash(row, col);
        }
    }
    
    public static List<int[]> aStarSearch(int[][] grid, int[] start, int[] goal) {
        int rows = grid.length;
        int cols = grid[0].length;
        
        // Define heuristic function (Manhattan distance)
        int h = Math.abs(start[0] - goal[0]) + Math.abs(start[1] - goal[1]);
        
        // Priority queue for open nodes
        PriorityQueue<Node> openQueue = new PriorityQueue<>();
        openQueue.add(new Node(start[0], start[1], 0, h));
        
        // Maps to store g_scores and parents
        Map<String, Integer> gScore = new HashMap<>();
        Map<String, int[]> cameFrom = new HashMap<>();
        
        gScore.put(start[0] + "," + start[1], 0);
        
        // Possible moves (up, right, down, left)
        int[][] directions = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        
        while (!openQueue.isEmpty()) {
            Node current = openQueue.poll();
            
            // Check if we reached the goal
            if (current.row == goal[0] && current.col == goal[1]) {
                // Reconstruct path
                List<int[]> path = new ArrayList<>();
                int[] currentPos = new int[]{current.row, current.col};
                path.add(currentPos);
                
                String key = currentPos[0] + "," + currentPos[1];
                while (cameFrom.containsKey(key)) {
                    currentPos = cameFrom.get(key);
                    path.add(currentPos);
                    key = currentPos[0] + "," + currentPos[1];
                }
                
                Collections.reverse(path);
                return path;
            }
            
            // Explore neighbors
            for (int[] dir : directions) {
                int newRow = current.row + dir[0];
                int newCol = current.col + dir[1];
                
                // Check if the move is valid
                if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols && grid[newRow][newCol] == 0) {
                    String neighborKey = newRow + "," + newCol;
                    int tentativeGScore = current.gScore + 1;
                    
                    if (!gScore.containsKey(neighborKey) || tentativeGScore < gScore.get(neighborKey)) {
                        // Found a better path
                        cameFrom.put(neighborKey, new int[]{current.row, current.col});
                        gScore.put(neighborKey, tentativeGScore);
                        
                        int f = tentativeGScore + Math.abs(newRow - goal[0]) + Math.abs(newCol - goal[1]);
                        openQueue.add(new Node(newRow, newCol, tentativeGScore, f));
                    }
                }
            }
        }
        
        // No path found
        return null;
    }
    
    public static void main(String[] args) {
        int[][] grid = {
            {0, 0, 0, 0, 0},
            {0, 1, 1, 1, 0},
            {0, 0, 0, 0, 0},
            {1, 1, 1, 0, 1},
            {0, 0, 0, 0, 0}
        };
        
        int[] start = {0, 0};
        int[] goal = {4, 4};
        
        List<int[]> path = aStarSearch(grid, start, goal);
        
        if (path != null) {
            System.out.println("Path found:");
            for (int[] pos : path) {
                System.out.println("(" + pos[0] + ", " + pos[1] + ")");
            }
        } else {
            System.out.println("No path found");
        }
    }
}
```

## Common Heuristics

The choice of heuristic function is crucial for A* performance. Here are some common heuristics:

### Manhattan Distance

Used for grid-based maps where movement is restricted to four directions (up, right, down, left):

```python
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
```

### Euclidean Distance

Used when movement in any direction is allowed:

```python
import math

def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
```

### Diagonal Distance (Chebyshev Distance)

Used for grid-based maps where diagonal movement is allowed:

```python
def diagonal_distance(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy)
```

### Diagonal Shortcut Distance

A more accurate heuristic when diagonal movement costs sqrt(2):

```python
def diagonal_shortcut_distance(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return dx + dy + (math.sqrt(2) - 2) * min(dx, dy)
```

## Variations

### IDA* (Iterative Deepening A*)

A memory-efficient variant that uses iterative deepening:

```python
def ida_star(start, goal, heuristic, successors):
    """
    Iterative Deepening A* (IDA*) search algorithm.
    """
    def search(path, g, f_limit):
        node = path[-1]
        f = g + heuristic(node, goal)
        
        if f > f_limit:
            return f, None
            
        if node == goal:
            return -1, path
            
        min_cost = float('inf')
        for successor, cost in successors(node):
            if successor not in path:
                path.append(successor)
                new_g = g + cost
                cost, result = search(path, new_g, f_limit)
                
                if cost < 0:  # Found solution
                    return -1, result
                    
                if cost < min_cost:
                    min_cost = cost
                    
                path.pop()
                
        return min_cost, None
    
    # Initial f-limit is the heuristic value of the start node
    f_limit = heuristic(start, goal)
    
    while True:
        cost, result = search([start], 0, f_limit)
        
        if cost < 0:  # Found solution
            return result
            
        if cost == float('inf'):  # No solution
            return None
            
        # Increase f-limit for next iteration
        f_limit = cost
```

### Bidirectional A*

A variant that searches from both start and goal simultaneously:

```python
def bidirectional_a_star(graph, start, goal, heuristic):
    """
    Bidirectional A* search algorithm.
    """
    # Forward search from start
    open_forward = [(heuristic(start, goal), start)]
    g_forward = {start: 0}
    closed_forward = set()
    
    # Backward search from goal
    open_backward = [(heuristic(goal, start), goal)]
    g_backward = {goal: 0}
    closed_backward = set()
    
    # Parents for path reconstruction
    parents_forward = {}
    parents_backward = {}
    
    # Best path so far
    best_path_cost = float('inf')
    best_path_meeting_point = None
    
    while open_forward and open_backward:
        # Forward search step
        _, current_forward = heapq.heappop(open_forward)
        
        # Check if we can improve the best path
        if current_forward in g_backward:
            path_cost = g_forward[current_forward] + g_backward[current_forward]
            if path_cost < best_path_cost:
                best_path_cost = path_cost
                best_path_meeting_point = current_forward
        
        closed_forward.add(current_forward)
        
        # Expand forward
        for neighbor, cost in graph[current_forward].items():
            if neighbor in closed_forward:
                continue
                
            new_g = g_forward[current_forward] + cost
            
            if neighbor not in g_forward or new_g < g_forward[neighbor]:
                parents_forward[neighbor] = current_forward
                g_forward[neighbor] = new_g
                f = new_g + heuristic(neighbor, goal)
                heapq.heappush(open_forward, (f, neighbor))
        
        # Backward search step
        _, current_backward = heapq.heappop(open_backward)
        
        # Check if we can improve the best path
        if current_backward in g_forward:
            path_cost = g_forward[current_backward] + g_backward[current_backward]
            if path_cost < best_path_cost:
                best_path_cost = path_cost
                best_path_meeting_point = current_backward
        
        closed_backward.add(current_backward)
        
        # Expand backward
        for neighbor, cost in graph[current_backward].items():
            if neighbor in closed_backward:
                continue
                
            new_g = g_backward[current_backward] + cost
            
            if neighbor not in g_backward or new_g < g_backward[neighbor]:
                parents_backward[neighbor] = current_backward
                g_backward[neighbor] = new_g
                f = new_g + heuristic(neighbor, start)
                heapq.heappush(open_backward, (f, neighbor))
    
    # Reconstruct path if found
    if best_path_meeting_point is not None:
        # Build path from start to meeting point
        path_forward = []
        node = best_path_meeting_point
        while node in parents_forward:
            path_forward.append(node)
            node = parents_forward[node]
        path_forward.append(start)
        path_forward.reverse()
        
        # Build path from meeting point to goal
        path_backward = []
        node = best_path_meeting_point
        while node in parents_backward:
            node = parents_backward[node]
            path_backward.append(node)
        
        # Combine paths
        return path_forward + path_backward
    
    return None
```

## Interview Tips

- Explain how A* combines the advantages of Dijkstra's algorithm and greedy best-first search
- Discuss the importance of the heuristic function and what makes a good heuristic (admissibility and consistency)
- Highlight real-world applications like GPS navigation and game pathfinding
- Explain the trade-offs between different heuristic functions
- Be prepared to analyze time and space complexity based on the branching factor and search depth
- Discuss optimizations for specific problem domains
- Know how to prove that A* finds the optimal path when using an admissible heuristic

## Practice Problems

1. Implement A* to find the shortest path in a maze with obstacles
2. Solve the 8-puzzle or 15-puzzle using A* with Manhattan distance heuristic
3. Create a pathfinding system for a game using A* with different terrain costs
4. Optimize A* for a grid with diagonal movement allowed
5. Implement bidirectional A* and compare its performance with standard A*
6. Solve the traveling salesman problem using A* with an admissible heuristic
7. Develop a route planner for a map with different transportation options and costs
