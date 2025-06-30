# Algorithms & Data Structures

Master fundamental algorithms and data structures essential for problem-solving and technical interviews.

## Learning Path

<div class="grid cards" markdown>

-   :material-code-array: **Data Structures**
    
    ---
    
    Arrays, linked lists, stacks, queues, trees, graphs
    
    [Explore structures →](data-structures.md)

-   :material-sort: **Sorting Algorithms**
    
    ---
    
    Bubble, merge, quick, heap sort and analysis
    
    [Sort it out →](sorting.md)

-   :material-graph: **Graph Algorithms**
    
    ---
    
    BFS, DFS, shortest path, minimum spanning tree
    
    [Navigate graphs →](graphs.md)

-   :material-chart-gantt: **Dynamic Programming**
    
    ---
    
    Optimization problems and recursive solutions
    
    [Optimize solutions →](dp.md)

</div>

## Complexity Analysis

Understanding time and space complexity is crucial for algorithm analysis.

### Big O Notation

| Complexity | Name | Example |
|------------|------|---------|
| O(1) | Constant | Array access |
| O(log n) | Logarithmic | Binary search |
| O(n) | Linear | Linear search |
| O(n log n) | Linearithmic | Merge sort |
| O(n²) | Quadratic | Bubble sort |
| O(2ⁿ) | Exponential | Recursive fibonacci |

### Quick Reference

```python
# Time Complexity Examples

# O(1) - Constant
def get_first(arr):
    return arr[0] if arr else None

# O(n) - Linear  
def linear_search(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

# O(log n) - Logarithmic
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# O(n²) - Quadratic
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
```

## Essential Data Structures

### Arrays & Strings

```python
# Dynamic Array Implementation
class DynamicArray:
    def __init__(self):
        self.capacity = 2
        self.size = 0
        self.data = [None] * self.capacity
    
    def resize(self):
        self.capacity *= 2
        new_data = [None] * self.capacity
        for i in range(self.size):
            new_data[i] = self.data[i]
        self.data = new_data
    
    def append(self, item):
        if self.size >= self.capacity:
            self.resize()
        self.data[self.size] = item
        self.size += 1
    
    def get(self, index):
        if 0 <= index < self.size:
            return self.data[index]
        raise IndexError("Index out of bounds")
```

### Linked Lists

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, val):
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, val):
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
    
    def delete(self, val):
        if not self.head:
            return
        
        if self.head.val == val:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next and current.next.val != val:
            current = current.next
        
        if current.next:
            current.next = current.next.next
```

### Trees

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        self.root = self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, node, val):
        if not node:
            return TreeNode(val)
        
        if val < node.val:
            node.left = self._insert_recursive(node.left, val)
        else:
            node.right = self._insert_recursive(node.right, val)
        
        return node
    
    def search(self, val):
        return self._search_recursive(self.root, val)
    
    def _search_recursive(self, node, val):
        if not node or node.val == val:
            return node
        
        if val < node.val:
            return self._search_recursive(node.left, val)
        return self._search_recursive(node.right, val)
```

## Problem-Solving Patterns

### Two Pointers

```python
def two_sum_sorted(arr, target):
    """Find two numbers that sum to target in sorted array."""
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []
```

### Sliding Window

```python
def max_sum_subarray(arr, k):
    """Find maximum sum of subarray of size k."""
    if len(arr) < k:
        return None
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

## Practice Resources

### Online Judges
- **LeetCode**: Comprehensive problem set
- **HackerRank**: Algorithmic challenges  
- **CodeForces**: Competitive programming
- **AtCoder**: Japanese competitive programming
- **SPOJ**: Sphere Online Judge

### Books
- **Introduction to Algorithms** (CLRS)
- **Algorithm Design Manual** by Steven Skiena
- **Cracking the Coding Interview** by Gayle McDowell
- **Elements of Programming Interviews**

### Problem Categories
- Array and String manipulation
- Two pointers and sliding window
- Hash tables and sets
- Recursion and backtracking
- Dynamic programming
- Graph traversal and shortest paths
- Tree traversal and manipulation
- Sorting and searching

## Interview Preparation

### Common Topics
1. **Arrays & Strings** (30%)
2. **Trees & Graphs** (25%)
3. **Dynamic Programming** (20%)
4. **System Design** (15%)
5. **Others** (10%)

### Study Plan
- **Week 1-2**: Arrays, strings, two pointers
- **Week 3-4**: Linked lists, stacks, queues
- **Week 5-6**: Trees and tree traversals
- **Week 7-8**: Graphs and graph algorithms
- **Week 9-10**: Dynamic programming
- **Week 11-12**: System design and review

---

*Practice makes perfect! Keep coding! 💻*
