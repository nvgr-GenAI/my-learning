# Heap Sort

Heap Sort is an efficient comparison-based sorting algorithm that uses a binary heap data structure to sort elements.

## Algorithm Overview

**Time Complexity:**
- Best/Average/Worst Case: O(n log n)

**Space Complexity:** O(1)

**Properties:**
- ❌ Not stable (relative order of equal elements may change)
- ✅ In-place (requires only O(1) extra space)
- ❌ Not adaptive (doesn't benefit from partially sorted data)

## How It Works

1. **Build a max heap** from the input array
2. **Extract the maximum** element (root) and place it at the end
3. **Restore the heap property** by heapifying the remaining elements
4. **Repeat** until all elements are sorted

## Binary Heap Properties

A binary heap is a complete binary tree where:
- **Max Heap**: Parent ≥ Children (for sorting in ascending order)
- **Min Heap**: Parent ≤ Children (for sorting in descending order)

**Array Representation:**
- Parent of node at index `i`: `(i-1)//2`
- Left child of node at index `i`: `2*i + 1`
- Right child of node at index `i`: `2*i + 2`

## Implementation

### Basic Heap Sort

```python
def heap_sort(arr):
    """
    Heap Sort implementation
    Time: O(n log n), Space: O(1)
    """
    n = len(arr)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # Extract elements from heap one by one
    for i in range(n - 1, 0, -1):
        # Move current root (maximum) to end
        arr[0], arr[i] = arr[i], arr[0]
        
        # Call heapify on the reduced heap
        heapify(arr, i, 0)
    
    return arr

def heapify(arr, n, i):
    """
    Heapify a subtree rooted at index i
    n is the size of the heap
    """
    largest = i  # Initialize largest as root
    left = 2 * i + 1
    right = 2 * i + 2
    
    # Check if left child exists and is greater than root
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    # Check if right child exists and is greater than largest so far
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    # If largest is not root, swap and continue heapifying
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

# Example usage
data = [12, 11, 13, 5, 6, 7]
print(f"Original: {data}")
print(f"Sorted: {heap_sort(data.copy())}")
```

### Iterative Heapify

```python
def heapify_iterative(arr, n, i):
    """
    Iterative version of heapify to avoid recursion overhead
    """
    while True:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        # Check left child
        if left < n and arr[left] > arr[largest]:
            largest = left
        
        # Check right child
        if right < n and arr[right] > arr[largest]:
            largest = right
        
        # If largest is root, heap property is satisfied
        if largest == i:
            break
        
        # Swap and continue with the affected subtree
        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest

def heap_sort_iterative(arr):
    """Heap sort with iterative heapify"""
    n = len(arr)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify_iterative(arr, n, i)
    
    # Extract elements
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify_iterative(arr, i, 0)
    
    return arr
```

### Min Heap Sort (Descending Order)

```python
def min_heapify(arr, n, i):
    """
    Min heapify for sorting in descending order
    """
    smallest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] < arr[smallest]:
        smallest = left
    
    if right < n and arr[right] < arr[smallest]:
        smallest = right
    
    if smallest != i:
        arr[i], arr[smallest] = arr[smallest], arr[i]
        min_heapify(arr, n, smallest)

def heap_sort_descending(arr):
    """Sort in descending order using min heap"""
    n = len(arr)
    
    # Build min heap
    for i in range(n // 2 - 1, -1, -1):
        min_heapify(arr, n, i)
    
    # Extract elements
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        min_heapify(arr, i, 0)
    
    return arr
```

## Advanced Heap Operations

### Heap Class Implementation

```python
class MaxHeap:
    """
    Max Heap implementation with various operations
    """
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def insert(self, value):
        """Insert a value into the heap"""
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)
    
    def _heapify_up(self, i):
        """Restore heap property upwards"""
        while i > 0:
            parent_idx = self.parent(i)
            if self.heap[i] <= self.heap[parent_idx]:
                break
            self.swap(i, parent_idx)
            i = parent_idx
    
    def extract_max(self):
        """Remove and return the maximum element"""
        if not self.heap:
            return None
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        max_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        return max_val
    
    def _heapify_down(self, i):
        """Restore heap property downwards"""
        size = len(self.heap)
        
        while True:
            largest = i
            left = self.left_child(i)
            right = self.right_child(i)
            
            if left < size and self.heap[left] > self.heap[largest]:
                largest = left
            
            if right < size and self.heap[right] > self.heap[largest]:
                largest = right
            
            if largest == i:
                break
            
            self.swap(i, largest)
            i = largest
    
    def peek(self):
        """Return maximum without removing it"""
        return self.heap[0] if self.heap else None
    
    def size(self):
        return len(self.heap)
    
    def is_empty(self):
        return len(self.heap) == 0

def heap_sort_using_class(arr):
    """Heap sort using MaxHeap class"""
    heap = MaxHeap()
    
    # Insert all elements
    for num in arr:
        heap.insert(num)
    
    # Extract all elements in sorted order
    result = []
    while not heap.is_empty():
        result.append(heap.extract_max())
    
    return result[::-1]  # Reverse for ascending order
```

### K Largest/Smallest Elements

```python
def k_largest_elements(arr, k):
    """
    Find k largest elements using heap sort approach
    Time: O(n log n), Space: O(1)
    """
    # Build max heap
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    result = []
    
    # Extract k largest elements
    for i in range(min(k, n)):
        # Extract max and add to result
        result.append(arr[0])
        
        # Move last element to root and heapify
        arr[0] = arr[n - 1 - i]
        heapify(arr, n - 1 - i, 0)
    
    return result

def k_smallest_elements_efficient(arr, k):
    """
    Find k smallest elements using min heap
    More efficient approach using heapq
    """
    import heapq
    return heapq.nsmallest(k, arr)

def k_largest_elements_efficient(arr, k):
    """
    Find k largest elements using max heap
    More efficient approach using heapq
    """
    import heapq
    return heapq.nlargest(k, arr)
```

## Applications

### Priority Queue Implementation

```python
class PriorityQueue:
    """
    Priority Queue using max heap
    Higher values have higher priority
    """
    def __init__(self):
        self.heap = []
    
    def enqueue(self, item, priority):
        """Add item with given priority"""
        heapq.heappush(self.heap, (-priority, item))  # Use negative for max heap
    
    def dequeue(self):
        """Remove and return highest priority item"""
        if self.heap:
            priority, item = heapq.heappop(self.heap)
            return item, -priority
        return None
    
    def peek(self):
        """Return highest priority item without removing"""
        if self.heap:
            priority, item = self.heap[0]
            return item, -priority
        return None
    
    def is_empty(self):
        return len(self.heap) == 0

# Example usage
pq = PriorityQueue()
pq.enqueue("Task A", 3)
pq.enqueue("Task B", 1)
pq.enqueue("Task C", 5)

while not pq.is_empty():
    task, priority = pq.dequeue()
    print(f"Processing {task} with priority {priority}")
```

### Heap Sort for Custom Objects

```python
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    
    def __str__(self):
        return f"{self.name}: {self.grade}"
    
    def __repr__(self):
        return self.__str__()

def heap_sort_students(students, key=lambda s: s.grade, reverse=False):
    """
    Sort students using heap sort with custom key
    """
    def heapify_students(arr, n, i):
        if reverse:
            extreme = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n and key(arr[left]) < key(arr[extreme]):
                extreme = left
            
            if right < n and key(arr[right]) < key(arr[extreme]):
                extreme = right
            
            if extreme != i:
                arr[i], arr[extreme] = arr[extreme], arr[i]
                heapify_students(arr, n, extreme)
        else:
            extreme = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n and key(arr[left]) > key(arr[extreme]):
                extreme = left
            
            if right < n and key(arr[right]) > key(arr[extreme]):
                extreme = right
            
            if extreme != i:
                arr[i], arr[extreme] = arr[extreme], arr[i]
                heapify_students(arr, n, extreme)
    
    n = len(students)
    
    # Build heap
    for i in range(n // 2 - 1, -1, -1):
        heapify_students(students, n, i)
    
    # Extract elements
    for i in range(n - 1, 0, -1):
        students[0], students[i] = students[i], students[0]
        heapify_students(students, i, 0)
    
    return students

# Example usage
students = [
    Student("Alice", 85),
    Student("Bob", 92),
    Student("Charlie", 78),
    Student("Diana", 96)
]

print("Original:", students)
heap_sort_students(students)
print("Sorted by grade:", students)
```

## When to Use Heap Sort

### ✅ Good For:
- **Guaranteed O(n log n)** - No worst-case degradation like Quick Sort
- **Memory-constrained environments** - O(1) space complexity
- **Priority queue operations** - Natural fit for heap-based structures
- **Finding k largest/smallest** - Efficient for partial sorting

### ❌ Avoid When:
- **Stability is required** - Heap Sort is not stable
- **Small arrays** - Overhead may not be worth it
- **Cache performance matters** - Poor cache locality due to jumping around

## Complexity Analysis

- **Time Complexity: O(n log n)** in all cases
  - Building heap: O(n)
  - Extracting n elements: O(n log n)
- **Space Complexity: O(1)** - In-place sorting
- **Not adaptive** - Doesn't perform better on partially sorted data

## Visual Example

```
Initial array: [4, 10, 3, 5, 1]

Build Max Heap:
       10
      /  \
     5    3
    / \
   4   1

Array representation: [10, 5, 3, 4, 1]

Extract maximum (10): [1, 5, 3, 4] | [10]
Heapify: [5, 4, 3, 1] | [10]

Extract maximum (5): [1, 4, 3] | [5, 10]
Heapify: [4, 1, 3] | [5, 10]

Continue until sorted: [1, 3, 4, 5, 10]
```

## Interview Questions

1. **What is the difference between a heap and a binary search tree?**
2. **How do you build a heap from an unsorted array?**
3. **Why is Heap Sort not stable?**
4. **How would you find the kth largest element using a heap?**
5. **Implement a priority queue using a heap**

---

*Heap Sort provides guaranteed O(n log n) performance with O(1) space - perfect when memory is limited!*
