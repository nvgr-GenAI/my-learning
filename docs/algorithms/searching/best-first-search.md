# Best-First Search

## Overview

Best-First Search is an informed search algorithm that uses a heuristic function to determine which node to explore next. Unlike BFS or DFS which explore nodes based on their position in the search tree, Best-First Search explores nodes that appear to be most promising according to a specified heuristic function.

## Algorithm

1. Initialize a priority queue (or min-heap) with the starting node
2. While the priority queue is not empty:
   - Remove the node with the lowest heuristic value
   - If this is the goal node, return the solution
   - Otherwise, expand the node and add its children to the priority queue with their heuristic values
3. If the queue becomes empty without finding the goal, return failure

## Implementation

### Python Implementation

```python
import heapq

class Node:
    def __init__(self, state, parent=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic
    
    def __lt__(self, other):
        return self.heuristic < other.heuristic

def best_first_search(start_state, goal_test, successors, heuristic):
    # Create the start node
    start_node = Node(state=start_state, heuristic=heuristic(start_state))
    
    # Initialize the priority queue with the start node
    frontier = []
    heapq.heappush(frontier, start_node)
    
    # Initialize the explored set
    explored = set()
    
    while frontier:
        # Get the node with the lowest heuristic value
        current_node = heapq.heappop(frontier)
        
        # Check if we reached the goal
        if goal_test(current_node.state):
            # Reconstruct the path
            path = []
            while current_node:
                path.append(current_node.state)
                current_node = current_node.parent
            return list(reversed(path))
        
        # Add the current state to explored
        explored.add(current_node.state)
        
        # Expand the current node
        for successor_state, step_cost in successors(current_node.state):
            if successor_state in explored:
                continue
                
            # Create the successor node
            successor_node = Node(
                state=successor_state,
                parent=current_node,
                cost=current_node.cost + step_cost,
                heuristic=heuristic(successor_state)
            )
            
            # Add to the frontier
            heapq.heappush(frontier, successor_node)
    
    # No solution found
    return None

# Example usage (8-puzzle problem)
def manhattan_distance(state):
    """
    Calculate the Manhattan distance heuristic for the 8-puzzle problem.
    State is represented as a flat list where 0 represents the empty space.
    """
    distance = 0
    goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]  # Goal configuration
    
    for i in range(9):
        if state[i] != 0:  # Skip the empty tile
            current_row, current_col = i // 3, i % 3
            goal_idx = goal_state.index(state[i])
            goal_row, goal_col = goal_idx // 3, goal_idx % 3
            distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    
    return distance

def get_successors(state):
    """Generate all possible next states for the 8-puzzle"""
    successors = []
    empty_idx = state.index(0)
    rows, cols = 3, 3
    row, col = empty_idx // cols, empty_idx % cols
    
    # Check all four possible moves (up, down, left, right)
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        new_row, new_col = row + dr, col + dc
        
        if 0 <= new_row < rows and 0 <= new_col < cols:
            new_empty_idx = new_row * cols + new_col
            new_state = state.copy()
            new_state[empty_idx], new_state[new_empty_idx] = new_state[new_empty_idx], new_state[empty_idx]
            successors.append((new_state, 1))
    
    return successors

def goal_test(state):
    return state == [1, 2, 3, 4, 5, 6, 7, 8, 0]

# Example usage
initial_state = [1, 3, 4, 8, 6, 2, 7, 0, 5]
solution = best_first_search(
    start_state=initial_state,
    goal_test=goal_test,
    successors=get_successors,
    heuristic=manhattan_distance
)

if solution:
    print(f"Solution found in {len(solution) - 1} steps:")
    for step, state in enumerate(solution):
        print(f"Step {step}: {[state[i:i+3] for i in range(0, 9, 3)]}")
else:
    print("No solution found")
```

### Java Implementation

```java
import java.util.*;

public class BestFirstSearch {
    static class Node implements Comparable<Node> {
        int[] state;
        Node parent;
        int cost;
        int heuristic;
        
        public Node(int[] state, Node parent, int cost, int heuristic) {
            this.state = state;
            this.parent = parent;
            this.cost = cost;
            this.heuristic = heuristic;
        }
        
        @Override
        public int compareTo(Node other) {
            return Integer.compare(this.heuristic, other.heuristic);
        }
        
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Node node = (Node) o;
            return Arrays.equals(state, node.state);
        }
        
        @Override
        public int hashCode() {
            return Arrays.hashCode(state);
        }
    }
    
    public static List<int[]> bestFirstSearch(
            int[] startState, 
            GoalTest goalTest, 
            SuccessorFunction successorFunction, 
            HeuristicFunction heuristicFunction) {
        
        // Create start node
        Node startNode = new Node(startState, null, 0, heuristicFunction.computeHeuristic(startState));
        
        // Priority queue (frontier)
        PriorityQueue<Node> frontier = new PriorityQueue<>();
        frontier.add(startNode);
        
        // Set of explored states
        Set<String> explored = new HashSet<>();
        
        while (!frontier.isEmpty()) {
            // Get node with lowest heuristic value
            Node currentNode = frontier.poll();
            
            // Check if goal reached
            if (goalTest.isGoal(currentNode.state)) {
                // Reconstruct the path
                List<int[]> path = new ArrayList<>();
                while (currentNode != null) {
                    path.add(currentNode.state);
                    currentNode = currentNode.parent;
                }
                Collections.reverse(path);
                return path;
            }
            
            // Add to explored set
            explored.add(Arrays.toString(currentNode.state));
            
            // Generate successors
            List<SuccessorNode> successors = successorFunction.getSuccessors(currentNode.state);
            for (SuccessorNode successor : successors) {
                int[] successorState = successor.state;
                
                // Skip if already explored
                if (explored.contains(Arrays.toString(successorState))) {
                    continue;
                }
                
                // Create successor node
                Node successorNode = new Node(
                    successorState,
                    currentNode,
                    currentNode.cost + successor.cost,
                    heuristicFunction.computeHeuristic(successorState)
                );
                
                // Add to frontier
                frontier.add(successorNode);
            }
        }
        
        // No solution found
        return null;
    }
    
    // Interfaces for the algorithm components
    interface GoalTest {
        boolean isGoal(int[] state);
    }
    
    interface SuccessorFunction {
        List<SuccessorNode> getSuccessors(int[] state);
    }
    
    interface HeuristicFunction {
        int computeHeuristic(int[] state);
    }
    
    static class SuccessorNode {
        int[] state;
        int cost;
        
        public SuccessorNode(int[] state, int cost) {
            this.state = state;
            this.cost = cost;
        }
    }
    
    // Example usage with 8-puzzle
    public static void main(String[] args) {
        // Initial state and goal state for 8-puzzle
        int[] initialState = {1, 3, 4, 8, 6, 2, 7, 0, 5};
        
        // Define the goal test
        GoalTest goalTest = state -> Arrays.equals(state, new int[]{1, 2, 3, 4, 5, 6, 7, 8, 0});
        
        // Define the heuristic function (Manhattan distance)
        HeuristicFunction manhattanDistance = state -> {
            int distance = 0;
            int[] goalState = {1, 2, 3, 4, 5, 6, 7, 8, 0};
            
            for (int i = 0; i < 9; i++) {
                if (state[i] != 0) {  // Skip empty tile
                    int currentRow = i / 3, currentCol = i % 3;
                    
                    // Find position in goal state
                    int goalIdx = -1;
                    for (int j = 0; j < 9; j++) {
                        if (goalState[j] == state[i]) {
                            goalIdx = j;
                            break;
                        }
                    }
                    
                    int goalRow = goalIdx / 3, goalCol = goalIdx % 3;
                    distance += Math.abs(currentRow - goalRow) + Math.abs(currentCol - goalCol);
                }
            }
            
            return distance;
        };
        
        // Define successor function
        SuccessorFunction successorFunction = state -> {
            List<SuccessorNode> successors = new ArrayList<>();
            int emptyIdx = -1;
            
            // Find the empty tile
            for (int i = 0; i < 9; i++) {
                if (state[i] == 0) {
                    emptyIdx = i;
                    break;
                }
            }
            
            int row = emptyIdx / 3, col = emptyIdx % 3;
            
            // Try all four directions
            int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}}; // Right, Down, Left, Up
            for (int[] dir : directions) {
                int newRow = row + dir[0], newCol = col + dir[1];
                
                if (newRow >= 0 && newRow < 3 && newCol >= 0 && newCol < 3) {
                    int newEmptyIdx = newRow * 3 + newCol;
                    int[] newState = Arrays.copyOf(state, 9);
                    
                    // Swap the tiles
                    newState[emptyIdx] = newState[newEmptyIdx];
                    newState[newEmptyIdx] = 0;
                    
                    successors.add(new SuccessorNode(newState, 1));
                }
            }
            
            return successors;
        };
        
        // Run the best-first search
        List<int[]> solution = bestFirstSearch(initialState, goalTest, successorFunction, manhattanDistance);
        
        if (solution != null) {
            System.out.println("Solution found in " + (solution.size() - 1) + " steps:");
            for (int i = 0; i < solution.size(); i++) {
                System.out.print("Step " + i + ": ");
                int[] state = solution.get(i);
                for (int j = 0; j < 9; j++) {
                    System.out.print(state[j] + " ");
                    if ((j + 1) % 3 == 0) System.out.print("| ");
                }
                System.out.println();
            }
        } else {
            System.out.println("No solution found");
        }
    }
}
```

## Complexity Analysis

- **Time Complexity**: O(b^d), where b is the branching factor and d is the depth of the goal node
- **Space Complexity**: O(b^d) to store all nodes in the priority queue

## Advantages and Disadvantages

### Advantages
- More efficient than uninformed search algorithms like BFS and DFS when a good heuristic is available
- Explores the most promising nodes first, which can lead to finding solutions faster
- Requires less memory than breadth-first search in many cases

### Disadvantages
- Can get stuck in local optima if the heuristic is misleading
- Not guaranteed to find the optimal (shortest) path
- Performance heavily depends on the quality of the heuristic function
- May explore unnecessary nodes if the heuristic isn't admissible

## Use Cases

- Puzzle solving (8-puzzle, 15-puzzle)
- Route finding in maps when approximate distances are known
- Game AI for exploration of game states
- Robot path planning with obstacles
- Any problem where a reasonable estimate of "closeness to goal" can be formulated

## Variations

1. **Greedy Best-First Search**: Uses only the heuristic function to determine which nodes to explore next, completely ignoring path cost
2. **A* Search**: Combines the path cost from the start node and the heuristic cost to the goal node
3. **Beam Search**: Limits the number of nodes that can be explored at each level
4. **Recursive Best-First Search (RBFS)**: Space-efficient version that uses recursive depth-first exploration
5. **Iterative-Deepening A* (IDA*)**: Combines iterative deepening with A* search for memory efficiency

## Interview Tips

- Understand the difference between Best-First Search and other search algorithms (BFS, DFS, A*)
- Be able to implement the algorithm using a priority queue or min-heap
- Know how to design effective heuristics for different problem domains
- Explain why Best-First Search is not guaranteed to find the optimal solution
- Discuss admissible and consistent (monotonic) heuristics
- Explain how Best-First Search can be converted to A* by including path cost

## Practice Problems

1. [8-Puzzle Problem](https://leetcode.com/problems/sliding-puzzle/) - Solve the 8-puzzle problem using Best-First Search
2. [Word Ladder](https://leetcode.com/problems/word-ladder/) - Find the shortest transformation sequence from one word to another
3. [Shortest Path in a Grid with Obstacles](https://leetcode.com/problems/shortest-path-in-binary-matrix/) - Find the shortest path from top-left to bottom-right in a grid with obstacles
4. [Robot Room Cleaner](https://leetcode.com/problems/robot-room-cleaner/) - Navigate a robot to clean an entire room
5. [Bus Routes](https://leetcode.com/problems/bus-routes/) - Find the least number of buses to reach the destination

## References

1. Russell, Stuart J., and Peter Norvig. "Artificial Intelligence: A Modern Approach." Pearson Education, 2016.
2. Cormen, Thomas H., et al. "Introduction to Algorithms." MIT Press, 2009.
3. Pearl, Judea. "Heuristics: Intelligent Search Strategies for Computer Problem Solving." Addison-Wesley, 1984.
