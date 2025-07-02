# Greedy Algorithms Fundamentals

## What are Greedy Algorithms?

Greedy algorithms build solutions by making the locally optimal choice at each step, hoping to find a global optimum. They make decisions based on the current situation without considering future consequences.

## Core Principles

### The Greedy Choice Property

1. **Local Optimality**: Choose the best option available at each step
2. **Irrevocable Decisions**: Once a choice is made, it cannot be undone
3. **Progressive Construction**: Build solution incrementally
4. **No Backtracking**: Never reconsider previous decisions

### When Greedy Works

Greedy algorithms work when the problem has:
- **Optimal Substructure**: Optimal solution contains optimal solutions to subproblems
- **Greedy Choice Property**: Global optimum can be reached by local optimal choices
- **No Dependencies**: Current choice doesn't affect future choices negatively

## Generic Template

```python
def greedy_algorithm(problem_input):
    # Initialize solution
    solution = []
    
    # Sort or organize input based on greedy criteria
    candidates = sort_by_greedy_criteria(problem_input)
    
    # Make greedy choices
    for candidate in candidates:
        if is_feasible(candidate, solution):
            solution.append(candidate)
            
            # Check if solution is complete
            if is_complete(solution):
                break
    
    return solution
```

## Classic Greedy Algorithms

### 1. Activity Selection Problem

**Problem**: Select maximum number of non-overlapping activities.

```python
def activity_selection(activities):
    """
    Select maximum non-overlapping activities.
    
    Time Complexity: O(n log n) - due to sorting
    Space Complexity: O(1)
    """
    # Sort by finish time (greedy criteria)
    activities.sort(key=lambda x: x[1])
    
    selected = [activities[0]]
    last_finish_time = activities[0][1]
    
    for i in range(1, len(activities)):
        start_time, finish_time = activities[i]
        
        # Greedy choice: select if it doesn't overlap
        if start_time >= last_finish_time:
            selected.append(activities[i])
            last_finish_time = finish_time
    
    return selected

# Test
activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]
print(activity_selection(activities))
# Output: [(1, 4), (5, 7), (8, 11), (12, 16)]
```

### 2. Fractional Knapsack

**Problem**: Maximize value in knapsack where items can be broken.

```python
def fractional_knapsack(capacity, items):
    """
    Solve fractional knapsack using greedy approach.
    
    Time Complexity: O(n log n)
    Space Complexity: O(1)
    """
    # Sort by value-to-weight ratio (greedy criteria)
    items.sort(key=lambda x: x[1]/x[0], reverse=True)
    
    total_value = 0
    remaining_capacity = capacity
    selected_items = []
    
    for weight, value in items:
        if remaining_capacity >= weight:
            # Take entire item
            selected_items.append((weight, value, 1.0))
            total_value += value
            remaining_capacity -= weight
        elif remaining_capacity > 0:
            # Take fraction of item
            fraction = remaining_capacity / weight
            selected_items.append((weight, value, fraction))
            total_value += value * fraction
            remaining_capacity = 0
            break
    
    return total_value, selected_items

# Test
items = [(10, 60), (20, 100), (30, 120)]  # (weight, value)
capacity = 50
value, selected = fractional_knapsack(capacity, items)
print(f"Maximum value: {value}")
print(f"Selected items: {selected}")
```

### 3. Huffman Coding

**Problem**: Create optimal prefix-free binary codes for characters.

```python
import heapq
from collections import defaultdict, Counter

class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

def huffman_coding(text):
    """
    Generate Huffman codes for given text.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    # Count character frequencies
    freq_map = Counter(text)
    
    # Create priority queue with leaf nodes
    heap = []
    for char, freq in freq_map.items():
        heapq.heappush(heap, HuffmanNode(char, freq))
    
    # Build Huffman tree
    while len(heap) > 1:
        # Greedy choice: combine two nodes with smallest frequencies
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    
    # Generate codes
    root = heap[0]
    codes = {}
    
    def generate_codes(node, code=""):
        if node.char:  # Leaf node
            codes[node.char] = code if code else "0"  # Handle single character case
        else:
            generate_codes(node.left, code + "0")
            generate_codes(node.right, code + "1")
    
    generate_codes(root)
    return codes

# Test
text = "hello world"
codes = huffman_coding(text)
print("Huffman Codes:")
for char, code in codes.items():
    print(f"'{char}': {code}")
```

## Algorithm Design Patterns

### 1. Scheduling Problems

```python
def earliest_deadline_first(jobs):
    """
    Schedule jobs to minimize maximum lateness.
    
    Greedy Strategy: Process jobs in order of earliest deadline
    """
    # Sort by deadline (greedy criteria)
    jobs.sort(key=lambda x: x[1])
    
    schedule = []
    current_time = 0
    
    for duration, deadline in jobs:
        start_time = current_time
        finish_time = current_time + duration
        lateness = max(0, finish_time - deadline)
        
        schedule.append({
            'job': (duration, deadline),
            'start': start_time,
            'finish': finish_time,
            'lateness': lateness
        })
        
        current_time = finish_time
    
    return schedule
```

### 2. Graph Problems

```python
def dijkstra_shortest_path(graph, start):
    """
    Find shortest paths using Dijkstra's algorithm (greedy).
    
    Greedy Strategy: Always expand the closest unvisited vertex
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    visited = set()
    pq = [(0, start)]
    
    while pq:
        # Greedy choice: select minimum distance node
        current_dist, current_node = heapq.heappop(pq)
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        # Update distances to neighbors
        for neighbor, weight in graph[current_node].items():
            new_dist = current_dist + weight
            
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    
    return distances
```

### 3. Minimum Spanning Tree

```python
def kruskal_mst(edges, num_vertices):
    """
    Find Minimum Spanning Tree using Kruskal's algorithm.
    
    Greedy Strategy: Always select the minimum weight edge that doesn't create a cycle
    """
    # Sort edges by weight (greedy criteria)
    edges.sort(key=lambda x: x[2])
    
    # Union-Find data structure
    parent = list(range(num_vertices))
    rank = [0] * num_vertices
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        
        if rank[px] < rank[py]:
            px, py = py, px
        
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        
        return True
    
    mst = []
    total_weight = 0
    
    for u, v, weight in edges:
        # Greedy choice: add edge if it doesn't create cycle
        if union(u, v):
            mst.append((u, v, weight))
            total_weight += weight
            
            if len(mst) == num_vertices - 1:
                break
    
    return mst, total_weight
```

## Optimization Techniques

### 1. Priority Queue Optimization

```python
def optimized_greedy_with_priority_queue(tasks):
    """
    Use priority queue for efficient greedy selection.
    """
    import heapq
    
    # Transform tasks into priority queue format
    pq = [(priority(task), task) for task in tasks]
    heapq.heapify(pq)
    
    solution = []
    
    while pq:
        # Greedy choice: always select highest priority task
        priority_val, task = heapq.heappop(pq)
        
        if is_feasible(task, solution):
            solution.append(task)
            
            # Update priorities of remaining tasks if needed
            update_priorities(pq, task)
    
    return solution
```

### 2. Sorting Optimization

```python
def multi_criteria_greedy(items):
    """
    Handle multiple greedy criteria with custom sorting.
    """
    # Sort by multiple criteria
    items.sort(key=lambda x: (
        -x.primary_metric,    # Primary criterion (descending)
        x.secondary_metric,   # Secondary criterion (ascending)
        -x.tertiary_metric    # Tertiary criterion (descending)
    ))
    
    solution = []
    for item in items:
        if satisfies_constraints(item, solution):
            solution.append(item)
    
    return solution
```

## Proving Greedy Correctness

### 1. Exchange Argument

```python
"""
Proof Template for Exchange Argument:

1. Assume there exists an optimal solution OPT different from greedy solution GREEDY
2. Find the first position where OPT and GREEDY differ
3. Show that exchanging OPT's choice with GREEDY's choice doesn't worsen the solution
4. Repeat until OPT becomes GREEDY, proving GREEDY is optimal
"""

def prove_exchange_argument():
    """
    Example: Activity Selection Problem
    
    1. Let OPT be an optimal solution different from our greedy solution
    2. Let a1 be the first activity in greedy solution
    3. Let b1 be the first activity in OPT
    4. Since we chose a1 greedily (earliest finish time), finish(a1) ≤ finish(b1)
    5. We can replace b1 with a1 in OPT without reducing the number of activities
    6. Repeat this process to transform OPT into our greedy solution
    """
    pass
```

### 2. Greedy Stays Ahead

```python
"""
Proof Template for Greedy Stays Ahead:

1. Show that after each step, greedy solution is "ahead" of optimal solution
2. Use induction to prove this invariant holds throughout
3. Conclude that greedy solution is optimal
"""

def prove_stays_ahead():
    """
    Example: Earliest Deadline First Scheduling
    
    1. After scheduling k jobs, greedy finishes no later than any optimal schedule
    2. This means greedy has more flexibility for remaining jobs
    3. Induction proves greedy produces optimal schedule
    """
    pass
```

## Common Pitfalls and Solutions

### 1. Non-Optimal Substructure

**Problem**: 0/1 Knapsack
```python
# WRONG: Greedy doesn't work for 0/1 knapsack
def wrong_01_knapsack(capacity, items):
    items.sort(key=lambda x: x[1]/x[0], reverse=True)  # Sort by value/weight ratio
    
    total_value = 0
    for weight, value in items:
        if capacity >= weight:
            capacity -= weight
            total_value += value
    
    return total_value  # This may not be optimal!

# CORRECT: Use Dynamic Programming
def correct_01_knapsack(capacity, items):
    n = len(items)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        weight, value = items[i-1]
        for w in range(capacity + 1):
            if weight <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight] + value)
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]
```

### 2. Multiple Optimal Criteria

**Problem**: When multiple greedy choices seem valid
**Solution**: Prove which criterion leads to optimality

```python
def interval_scheduling_multiple_criteria(intervals):
    """
    Different greedy strategies for interval scheduling:
    1. Shortest intervals first - NOT optimal
    2. Earliest start time first - NOT optimal
    3. Fewest conflicts first - NOT optimal
    4. Earliest finish time first - OPTIMAL
    """
    
    # WRONG strategies
    shortest_first = sorted(intervals, key=lambda x: x[1] - x[0])
    earliest_start = sorted(intervals, key=lambda x: x[0])
    
    # CORRECT strategy
    earliest_finish = sorted(intervals, key=lambda x: x[1])
    
    return earliest_finish
```

## Best Practices

### 1. Problem Analysis

```python
def analyze_greedy_applicability(problem):
    """
    Checklist for greedy algorithm applicability:
    """
    questions = [
        "Does the problem have optimal substructure?",
        "Can local optimal choices lead to global optimum?",
        "Are there dependencies between choices?",
        "Can we prove greedy correctness?",
        "Is there a clear greedy criterion?"
    ]
    return questions
```

### 2. Implementation Guidelines

```python
def greedy_implementation_template():
    """
    Best practices for implementing greedy algorithms:
    """
    guidelines = [
        "1. Identify the greedy criterion clearly",
        "2. Sort input according to greedy criterion",
        "3. Make irrevocable choices sequentially",
        "4. Check feasibility before adding to solution",
        "5. Prove correctness using exchange argument or stays-ahead",
        "6. Analyze time complexity (usually dominated by sorting)",
        "7. Consider edge cases and boundary conditions"
    ]
    return guidelines
```

### 3. Testing Strategy

```python
def test_greedy_algorithm():
    """
    Testing approach for greedy algorithms:
    """
    test_cases = [
        "Empty input",
        "Single element",
        "All elements identical",
        "Sorted input (best case)",
        "Reverse sorted input (worst case)",
        "Random input",
        "Large input for performance testing"
    ]
    return test_cases
```

## Applications by Domain

### 1. Scheduling
- Job scheduling with deadlines
- CPU scheduling algorithms
- Task assignment problems

### 2. Graph Algorithms
- Shortest path (Dijkstra)
- Minimum spanning tree (Kruskal, Prim)
- Network flow problems

### 3. Data Compression
- Huffman coding
- LZ77 compression
- Run-length encoding

### 4. Resource Allocation
- Fractional knapsack
- Load balancing
- Bandwidth allocation

### 5. Approximation Algorithms
- Set cover approximation
- Vertex cover approximation
- Traveling salesman approximations

Greedy algorithms are powerful tools when the problem structure allows for locally optimal choices to lead to globally optimal solutions. Understanding when and how to apply them is crucial for efficient algorithm design.
