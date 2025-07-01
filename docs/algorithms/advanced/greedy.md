# Greedy Algorithms

## Overview

Greedy algorithms make locally optimal choices at each step, hoping to find a global optimum. They work by selecting the best available option at each decision point without considering future consequences.

## Characteristics

1. **Greedy Choice Property**: Local optimal choice leads to global optimal solution
2. **Optimal Substructure**: Optimal solution contains optimal solutions to subproblems
3. **No Backtracking**: Once a choice is made, it's never reconsidered

## Classic Examples

### Activity Selection Problem

```python
def activity_selection(activities):
    """
    Select maximum number of non-overlapping activities
    activities: list of (start, end) tuples
    """
    # Sort by end time
    activities.sort(key=lambda x: x[1])
    
    selected = [activities[0]]
    last_end_time = activities[0][1]
    
    for i in range(1, len(activities)):
        start, end = activities[i]
        if start >= last_end_time:
            selected.append((start, end))
            last_end_time = end
    
    return selected

# Example usage
activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]
print(activity_selection(activities))
# Output: [(1, 4), (5, 7), (8, 11), (12, 16)]
```

### Fractional Knapsack

```python
def fractional_knapsack(items, capacity):
    """
    items: list of (weight, value) tuples
    capacity: knapsack capacity
    """
    # Calculate value-to-weight ratio and sort
    items_with_ratio = [(value/weight, weight, value) for weight, value in items]
    items_with_ratio.sort(reverse=True)
    
    total_value = 0
    remaining_capacity = capacity
    result = []
    
    for ratio, weight, value in items_with_ratio:
        if weight <= remaining_capacity:
            # Take whole item
            total_value += value
            remaining_capacity -= weight
            result.append((weight, value, 1.0))  # (weight, value, fraction)
        else:
            # Take fraction of item
            fraction = remaining_capacity / weight
            total_value += value * fraction
            result.append((weight, value, fraction))
            break
    
    return total_value, result

# Example usage
items = [(10, 60), (20, 100), (30, 120)]
capacity = 50
value, solution = fractional_knapsack(items, capacity)
print(f"Maximum value: {value}")
```

### Huffman Coding

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
    """Generate Huffman codes for given text"""
    # Count frequency of each character
    freq = Counter(text)
    
    # Create priority queue of nodes
    heap = [HuffmanNode(char, freq) for char, freq in freq.items()]
    heapq.heapify(heap)
    
    # Build Huffman tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    
    # Generate codes
    root = heap[0]
    codes = {}
    
    def generate_codes(node, code=""):
        if node.char:  # Leaf node
            codes[node.char] = code if code else "0"  # Handle single character
        else:
            generate_codes(node.left, code + "0")
            generate_codes(node.right, code + "1")
    
    generate_codes(root)
    return codes

# Example usage
text = "hello world"
codes = huffman_coding(text)
print("Huffman codes:", codes)

# Encode text
encoded = ''.join(codes[char] for char in text)
print("Encoded:", encoded)
```

### Dijkstra's Algorithm (Greedy Approach)

```python
import heapq

def dijkstra(graph, start):
    """Find shortest paths from start to all other vertices"""
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        
        if current_vertex in visited:
            continue
        
        visited.add(current_vertex)
        
        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Example usage
graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('C', 1), ('D', 5)],
    'C': [('D', 8), ('E', 10)],
    'D': [('E', 2)],
    'E': []
}
print(dijkstra(graph, 'A'))
```

## More Greedy Examples

### Coin Change (Greedy - works for certain coin systems)

```python
def coin_change_greedy(coins, amount):
    """
    Works for canonical coin systems (like US coins)
    coins: list of coin denominations (sorted in descending order)
    """
    coins.sort(reverse=True)
    result = []
    
    for coin in coins:
        count = amount // coin
        if count > 0:
            result.extend([coin] * count)
            amount -= coin * count
    
    return result if amount == 0 else None

# Example usage
coins = [25, 10, 5, 1]  # US coins: quarters, dimes, nickels, pennies
amount = 67
print(coin_change_greedy(coins, amount))  # [25, 25, 10, 5, 1, 1]
```

### Job Scheduling

```python
def job_scheduling(jobs):
    """
    Schedule jobs to minimize total completion time
    jobs: list of job durations
    """
    jobs.sort()  # Shortest Job First
    
    completion_time = 0
    total_waiting_time = 0
    schedule = []
    
    for job_duration in jobs:
        completion_time += job_duration
        total_waiting_time += completion_time
        schedule.append((job_duration, completion_time))
    
    return schedule, total_waiting_time

# Example usage
jobs = [6, 8, 3, 4, 2]
schedule, total_time = job_scheduling(jobs)
print("Schedule:", schedule)
print("Total waiting time:", total_time)
```

### Minimum Spanning Tree (Kruskal's Algorithm)

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        return True

def kruskal_mst(vertices, edges):
    """
    Find Minimum Spanning Tree using Kruskal's algorithm
    edges: list of (weight, u, v) tuples
    """
    edges.sort()  # Sort by weight
    uf = UnionFind(vertices)
    mst = []
    total_weight = 0
    
    for weight, u, v in edges:
        if uf.union(u, v):
            mst.append((u, v, weight))
            total_weight += weight
            
            if len(mst) == vertices - 1:
                break
    
    return mst, total_weight

# Example usage
vertices = 4
edges = [(1, 0, 1), (3, 0, 2), (2, 1, 2), (4, 1, 3), (5, 2, 3)]
mst, weight = kruskal_mst(vertices, edges)
print("MST edges:", mst)
print("Total weight:", weight)
```

## When Greedy Works

Greedy algorithms work when the problem has:

1. **Optimal Substructure**
2. **Greedy Choice Property**
3. **Matroid Structure** (for advanced cases)

## When Greedy Fails

```python
# Example: 0/1 Knapsack
def greedy_knapsack_fails():
    """
    Greedy doesn't work for 0/1 knapsack
    """
    items = [(1, 1), (4, 3), (5, 4)]  # (weight, value)
    capacity = 5
    
    # Greedy by value/weight ratio: (5, 4) -> ratio = 0.8
    # But optimal is (1, 1) + (4, 3) = value 4 vs greedy value 4
    # This example shows greedy can match optimal, but not always
    
    # Better counterexample:
    items = [(1, 1), (3, 4), (4, 5)]
    capacity = 4
    # Greedy: (4, 5) -> value 5
    # Optimal: (1, 1) + (3, 4) -> value 5
    # Still matches! Need better example:
    
    items = [(2, 1), (3, 4), (4, 5), (5, 7)]
    capacity = 5
    # Greedy by ratio: (5, 7) -> ratio 1.4, value 7
    # Optimal: (2, 1) + (3, 4) -> value 5, not better
    # Actually (5, 7) is optimal here too
    
    # True counterexample:
    items = [(1, 4), (2, 6), (5, 12)]  # ratios: 4, 3, 2.4
    capacity = 6
    # Greedy: (1, 4) + (2, 6) = value 10, weight 3, can add nothing more
    # Wait, still space for weight 3, but no item fits exactly
    # Greedy: (1, 4) + (5, 12) won't fit, so just (1, 4) + (2, 6) = 10
    # Optimal: (5, 12) = value 12
    # This shows greedy fails!
    
    pass
```

## Proving Greedy Correctness

1. **Greedy Choice Property**: Show local optimal choice leads to global optimal
2. **Optimal Substructure**: Show optimal solution contains optimal subsolutions
3. **Exchange Argument**: Show any optimal solution can be transformed to greedy solution

## Practice Problems

- [ ] Jump Game
- [ ] Jump Game II
- [ ] Gas Station
- [ ] Candy
- [ ] Queue Reconstruction by Height
- [ ] Partition Labels
- [ ] Non-overlapping Intervals
- [ ] Minimum Number of Arrows to Burst Balloons
