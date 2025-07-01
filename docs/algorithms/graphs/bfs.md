# Breadth-First Search (BFS)

## Overview

BFS is a graph traversal algorithm that explores all vertices at the current depth before moving to vertices at the next depth level.

## Implementation

### Basic BFS

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        
        if node not in visited:
            visited.add(node)
            print(node)  # Process node
            
            # Add neighbors to queue
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
    
    return visited
```

### BFS with Levels

```python
def bfs_levels(graph, start):
    visited = set([start])
    queue = deque([(start, 0)])  # (node, level)
    levels = {}
    
    while queue:
        node, level = queue.popleft()
        levels[node] = level
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, level + 1))
    
    return levels
```

## Applications

### Shortest Path (Unweighted)

```python
def shortest_path_bfs(graph, start, target):
    if start == target:
        return [start]
    
    visited = set([start])
    queue = deque([(start, [start])])
    
    while queue:
        node, path = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor == target:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None  # No path found
```

### Level Order Traversal

```python
def level_order_traversal(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

## BFS vs DFS

| Aspect | BFS | DFS |
|--------|-----|-----|
| Data Structure | Queue | Stack |
| Memory Usage | O(w) where w is max width | O(h) where h is max height |
| Shortest Path | Yes (unweighted) | No |
| Space Complexity | Higher | Lower |

## Time and Space Complexity

- **Time Complexity**: O(V + E) where V is vertices and E is edges
- **Space Complexity**: O(V) for visited set and queue

## Common BFS Problems

1. **Binary Tree Level Order Traversal**
2. **Rotting Oranges**
3. **Word Ladder**
4. **Minimum Number of Steps to Reach Target**
5. **Open the Lock**

## Practice Problems

- [ ] Binary Tree Level Order Traversal
- [ ] Binary Tree Zigzag Level Order Traversal
- [ ] Minimum Depth of Binary Tree
- [ ] Rotting Oranges
- [ ] Word Ladder
