# Heaps

## Overview

A Heap is a complete binary tree that satisfies the heap property.

## Types of Heaps

### Max Heap

Parent node is greater than or equal to its children.

### Min Heap

Parent node is less than or equal to its children.

## Implementation

### Min Heap

```python
import heapq

class MinHeap:
    def __init__(self):
        self.heap = []
    
    def push(self, val):
        heapq.heappush(self.heap, val)
    
    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)
        return None
    
    def peek(self):
        if self.heap:
            return self.heap[0]
        return None
    
    def size(self):
        return len(self.heap)
```

### Max Heap (using negation)

```python
class MaxHeap:
    def __init__(self):
        self.heap = []
    
    def push(self, val):
        heapq.heappush(self.heap, -val)
    
    def pop(self):
        if self.heap:
            return -heapq.heappop(self.heap)
        return None
    
    def peek(self):
        if self.heap:
            return -self.heap[0]
        return None
```

## Common Heap Problems

1. **Kth Largest Element in an Array**
2. **Top K Frequent Elements**
3. **Merge k Sorted Lists**
4. **Find Median from Data Stream**
5. **Task Scheduler**

## Applications

### Priority Queue

```python
import heapq

def process_tasks(tasks):
    # tasks = [(priority, task_name), ...]
    heap = []
    
    for priority, task in tasks:
        heapq.heappush(heap, (priority, task))
    
    while heap:
        priority, task = heapq.heappop(heap)
        print(f"Processing: {task} (Priority: {priority})")
```

### K Largest Elements

```python
def k_largest(nums, k):
    return heapq.nlargest(k, nums)

def k_smallest(nums, k):
    return heapq.nsmallest(k, nums)
```

## Time Complexities

| Operation | Time Complexity |
|-----------|----------------|
| Insert    | O(log n)       |
| Extract   | O(log n)       |
| Peek      | O(1)           |
| Build     | O(n)           |

## Practice Problems

- [ ] Kth Largest Element in an Array
- [ ] Last Stone Weight
- [ ] K Closest Points to Origin
- [ ] Top K Frequent Elements
- [ ] Merge k Sorted Lists
