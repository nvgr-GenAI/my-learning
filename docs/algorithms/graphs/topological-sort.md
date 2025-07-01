# Topological Sort

## Overview

Topological Sort is a linear ordering of vertices in a Directed Acyclic Graph (DAG) such that for every directed edge (u, v), vertex u comes before v in the ordering.

## Prerequisites

- Graph must be a **Directed Acyclic Graph (DAG)**
- If graph has cycles, topological sort is not possible

## Algorithms

### Kahn's Algorithm (BFS-based)

```python
from collections import deque, defaultdict

def topological_sort_kahn(graph):
    # Calculate in-degrees
    in_degree = defaultdict(int)
    
    # Initialize in-degrees
    for node in graph:
        in_degree[node] = 0
    
    # Calculate in-degrees
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    
    # Find nodes with no incoming edges
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        # Remove edges from current node
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check for cycles
    if len(result) != len(graph):
        return []  # Cycle detected
    
    return result
```

### DFS-based Approach

```python
def topological_sort_dfs(graph):
    visited = set()
    rec_stack = set()
    result = []
    
    def dfs(node):
        if node in rec_stack:
            return False  # Cycle detected
        
        if node in visited:
            return True
        
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        
        rec_stack.remove(node)
        result.append(node)
        return True
    
    # Visit all nodes
    for node in graph:
        if node not in visited:
            if not dfs(node):
                return []  # Cycle detected
    
    return result[::-1]  # Reverse to get correct order
```

## Applications

### Course Schedule

```python
def can_finish_courses(num_courses, prerequisites):
    # Build graph
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    # Add isolated nodes
    for i in range(num_courses):
        if i not in graph:
            graph[i] = []
    
    # Perform topological sort
    topo_order = topological_sort_kahn(graph)
    
    return len(topo_order) == num_courses

def find_order(num_courses, prerequisites):
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    for i in range(num_courses):
        if i not in graph:
            graph[i] = []
    
    return topological_sort_kahn(graph)
```

### Build Dependencies

```python
def build_order(projects, dependencies):
    # Build graph
    graph = defaultdict(list)
    for dep in dependencies:
        first, second = dep
        graph[first].append(second)
    
    # Add isolated projects
    for project in projects:
        if project not in graph:
            graph[project] = []
    
    return topological_sort_kahn(graph)
```

## Example Usage

```python
# Example: Course prerequisites
graph = {
    0: [1, 2],    # Course 0 is prerequisite for courses 1 and 2
    1: [3],       # Course 1 is prerequisite for course 3
    2: [3],       # Course 2 is prerequisite for course 3
    3: []         # Course 3 has no dependencies
}

print(topological_sort_kahn(graph))  # [0, 1, 2, 3] or [0, 2, 1, 3]
print(topological_sort_dfs(graph))   # [0, 2, 1, 3] or [0, 1, 2, 3]
```

## Time and Space Complexity

- **Time Complexity**: O(V + E) where V is vertices and E is edges
- **Space Complexity**: O(V) for storing in-degrees and result

## Common Problems

1. **Course Schedule**
2. **Course Schedule II**
3. **Alien Dictionary**
4. **Minimum Height Trees**
5. **Parallel Courses**

## Practice Problems

- [ ] Course Schedule
- [ ] Course Schedule II
- [ ] Find Eventual Safe States
- [ ] Sort Items by Groups Respecting Dependencies
