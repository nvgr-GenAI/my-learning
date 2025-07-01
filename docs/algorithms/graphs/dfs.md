# Depth-First Search (DFS)

## Overview

DFS is a graph traversal algorithm that explores as far as possible along each branch before backtracking.

## Implementation

### Recursive DFS

```python
def dfs_recursive(graph, node, visited):
    if node in visited:
        return
    
    visited.add(node)
    print(node)  # Process node
    
    for neighbor in graph[node]:
        dfs_recursive(graph, neighbor, visited)

# Usage
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

visited = set()
dfs_recursive(graph, 'A', visited)
```

### Iterative DFS

```python
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        
        if node not in visited:
            visited.add(node)
            print(node)  # Process node
            
            # Add neighbors to stack
            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return visited
```

## Applications

### Path Finding

```python
def has_path_dfs(graph, start, target):
    if start == target:
        return True
    
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        
        if node == target:
            return True
        
        if node not in visited:
            visited.add(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return False
```

### Connected Components

```python
def count_connected_components(graph):
    visited = set()
    count = 0
    
    for node in graph:
        if node not in visited:
            dfs_recursive(graph, node, visited)
            count += 1
    
    return count
```

## Time and Space Complexity

- **Time Complexity**: O(V + E) where V is vertices and E is edges
- **Space Complexity**: O(V) for visited set and recursion stack

## Common DFS Problems

1. **Number of Islands**
2. **Clone Graph**
3. **Path Sum in Binary Tree** 
4. **Validate Binary Search Tree**
5. **Course Schedule**

## Practice Problems

- [ ] Number of Islands
- [ ] Max Area of Island
- [ ] Surrounded Regions
- [ ] Pacific Atlantic Water Flow
- [ ] Word Search
