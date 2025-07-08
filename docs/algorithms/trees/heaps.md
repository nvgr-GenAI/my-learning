# Heaps 🏔️

## Introduction

Heaps are complete binary trees that satisfy the heap property. They're essential for priority queues, sorting algorithms, and many graph algorithms like Dijkstra's shortest path.

=== "Overview"
    **Core Concept**:
    
    - Special binary tree-based data structure that satisfies the heap property
    - Complete binary tree (all levels filled except possibly the last)
    - Allows efficient access to the minimum or maximum element
    
    **When to Use**:
    
    - When you need quick access to the minimum/maximum element
    - For implementing priority queues
    - When you need to efficiently extract elements in priority order
    - In algorithms requiring partial sorting
    
    **Time Complexity**:
    
    - Insert: O(log n)
    - Extract Min/Max: O(log n)
    - Peek Min/Max: O(1)
    - Heapify: O(n)
    
    **Real-World Applications**:
    
    - Priority scheduling in operating systems
    - Dijkstra's and Prim's algorithms for graphs
    - Event-driven simulation
    - Task scheduling in distributed systems

=== "Heap Properties"
    **Basic Properties**:
    
    - **Max Heap**: For each node, parent key ≥ children's keys (root is maximum)
    - **Min Heap**: For each node, parent key ≤ children's keys (root is minimum)
- **Min Heap**: Parent ≤ Children (root is minimum)
- **Complete Binary Tree**: All levels filled except possibly the last
- **Array Representation**: Efficient storage using indices

### Index Relationships
- **Parent**: `(i-1) // 2`
- **Left Child**: `2*i + 1`  
- **Right Child**: `2*i + 2`

## 🔧 Implementation

### Basic Min Heap

```python
class MinHeap:
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def insert(self, val):
        """Insert element and maintain heap property"""
        self.heap.append(val)
        self._heapify_up(len(self.heap) - 1)
    
    def extract_min(self):
        """Remove and return minimum element"""
        if not self.heap:
            return None
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()  # Move last to root
        self._heapify_down(0)
        return min_val
    
    def peek(self):
        """Return minimum without removing"""
        return self.heap[0] if self.heap else None
    
    def size(self):
        return len(self.heap)
    
    def is_empty(self):
        return len(self.heap) == 0
    
    def _heapify_up(self, i):
        """Restore heap property upwards"""
        if i == 0:
            return
        
        parent_idx = self.parent(i)
        if self.heap[i] < self.heap[parent_idx]:
            self.heap[i], self.heap[parent_idx] = self.heap[parent_idx], self.heap[i]
            self._heapify_up(parent_idx)
    
    def _heapify_down(self, i):
        """Restore heap property downwards"""
        left = self.left_child(i)
        right = self.right_child(i)
        smallest = i
        
        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left
        
        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right
        
        if smallest != i:
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            self._heapify_down(smallest)
```

### Max Heap Implementation

```python
class MaxHeap:
    def __init__(self):
        self.heap = []
    
    def insert(self, val):
        self.heap.append(val)
        self._heapify_up(len(self.heap) - 1)
    
    def extract_max(self):
        if not self.heap:
            return None
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        max_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        return max_val
    
    def _heapify_up(self, i):
        if i == 0:
            return
        
        parent_idx = (i - 1) // 2
        if self.heap[i] > self.heap[parent_idx]:
            self.heap[i], self.heap[parent_idx] = self.heap[parent_idx], self.heap[i]
            self._heapify_up(parent_idx)
    
    def _heapify_down(self, i):
        left = 2 * i + 1
        right = 2 * i + 2
        largest = i
        
        if left < len(self.heap) and self.heap[left] > self.heap[largest]:
            largest = left
        
        if right < len(self.heap) and self.heap[right] > self.heap[largest]:
            largest = right
        
        if largest != i:
            self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
            self._heapify_down(largest)
```

## 🎨 Common Heap Problems

### Problem 1: Kth Largest Element

```python
import heapq

def findKthLargest(nums, k):
    """
    Find kth largest element using min heap
    Time: O(n log k), Space: O(k)
    """
    # Maintain min heap of size k
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heap[0]  # Root of min heap is kth largest

def findKthLargestQuickSelect(nums, k):
    """
    Quick Select approach - Average O(n), Worst O(n²)
    """
    def partition(left, right, pivot_idx):
        pivot = nums[pivot_idx]
        nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
        
        store_idx = left
        for i in range(left, right):
            if nums[i] < pivot:
                nums[store_idx], nums[i] = nums[i], nums[store_idx]
                store_idx += 1
        
        nums[right], nums[store_idx] = nums[store_idx], nums[right]
        return store_idx
    
    def select(left, right, k_smallest):
        if left == right:
            return nums[left]
        
        pivot_idx = left + (right - left) // 2
        pivot_idx = partition(left, right, pivot_idx)
        
        if k_smallest == pivot_idx:
            return nums[k_smallest]
        elif k_smallest < pivot_idx:
            return select(left, pivot_idx - 1, k_smallest)
        else:
            return select(pivot_idx + 1, right, k_smallest)
    
    return select(0, len(nums) - 1, len(nums) - k)

# Test
nums = [3,2,1,5,6,4]
k = 2
print(findKthLargest(nums, k))  # 5
```

### Problem 2: Merge K Sorted Lists

```python
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __lt__(self, other):
        return self.val < other.val

def mergeKLists(lists):
    """
    Merge k sorted linked lists using min heap
    Time: O(N log k), Space: O(k) where N = total nodes
    """
    heap = []
    
    # Add first node of each list to heap
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, lst)
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        # Get minimum node
        node = heapq.heappop(heap)
        current.next = node
        current = current.next
        
        # Add next node from same list
        if node.next:
            heapq.heappush(heap, node.next)
    
    return dummy.next

def mergeKListsDivideConquer(lists):
    """
    Divide and conquer approach
    Time: O(N log k), Space: O(log k)
    """
    if not lists:
        return None
    
    def mergeTwoLists(l1, l2):
        dummy = ListNode(0)
        current = dummy
        
        while l1 and l2:
            if l1.val <= l2.val:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next
            current = current.next
        
        current.next = l1 or l2
        return dummy.next
    
    while len(lists) > 1:
        merged_lists = []
        
        for i in range(0, len(lists), 2):
            l1 = lists[i]
            l2 = lists[i + 1] if i + 1 < len(lists) else None
            merged_lists.append(mergeTwoLists(l1, l2))
        
        lists = merged_lists
    
    return lists[0]
```

### Problem 3: Top K Frequent Elements

```python
import heapq
from collections import Counter

def topKFrequent(nums, k):
    """
    Find k most frequent elements using heap
    Time: O(n log k), Space: O(n)
    """
    # Count frequencies
    count = Counter(nums)
    
    # Use min heap to keep k most frequent
    heap = []
    
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for freq, num in heap]

def topKFrequentBucketSort(nums, k):
    """
    Bucket sort approach - O(n) time
    """
    count = Counter(nums)
    
    # Create buckets for each frequency
    buckets = [[] for _ in range(len(nums) + 1)]
    
    for num, freq in count.items():
        buckets[freq].append(num)
    
    result = []
    # Traverse from highest frequency
    for i in range(len(buckets) - 1, 0, -1):
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result
    
    return result

# Test
nums = [1,1,1,2,2,3]
k = 2
print(topKFrequent(nums, k))  # [1, 2]
```

### Problem 4: Sliding Window Maximum

```python
import heapq
from collections import deque

def maxSlidingWindow(nums, k):
    """
    Find maximum in each sliding window using max heap
    Time: O(n log k), Space: O(k)
    """
    if not nums or k == 0:
        return []
    
    heap = []  # (-value, index)
    result = []
    
    for i, num in enumerate(nums):
        # Add current element
        heapq.heappush(heap, (-num, i))
        
        # Remove elements outside window
        while heap and heap[0][1] <= i - k:
            heapq.heappop(heap)
        
        # Add to result when window is full
        if i >= k - 1:
            result.append(-heap[0][0])
    
    return result

def maxSlidingWindowDeque(nums, k):
    """
    Optimized solution using deque - O(n) time
    """
    if not nums or k == 0:
        return []
    
    dq = deque()  # Store indices
    result = []
    
    for i, num in enumerate(nums):
        # Remove indices outside window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove smaller elements from back
        while dq and nums[dq[-1]] < num:
            dq.pop()
        
        dq.append(i)
        
        # Add to result when window is full
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

# Test
nums = [1,3,-1,-3,5,3,6,7]
k = 3
print(maxSlidingWindow(nums, k))  # [3,3,5,5,6,7]
```

### Problem 5: Find Median from Data Stream

```python
import heapq

class MedianFinder:
    """
    Find median using two heaps
    Time: O(log n) insert, O(1) find median
    """
    
    def __init__(self):
        self.small = []  # Max heap (negative values)
        self.large = []  # Min heap
    
    def addNum(self, num):
        # Add to appropriate heap
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)
        else:
            heapq.heappush(self.large, num)
        
        # Balance heaps
        if len(self.small) > len(self.large) + 1:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        elif len(self.large) > len(self.small) + 1:
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)
    
    def findMedian(self):
        if len(self.small) == len(self.large):
            return (-self.small[0] + self.large[0]) / 2.0
        elif len(self.small) > len(self.large):
            return float(-self.small[0])
        else:
            return float(self.large[0])

# Usage
mf = MedianFinder()
mf.addNum(1)
mf.addNum(2)
print(mf.findMedian())  # 1.5
mf.addNum(3)
print(mf.findMedian())  # 2.0
```

### Problem 6: Meeting Rooms II

```python
import heapq

def minMeetingRooms(intervals):
    """
    Find minimum meeting rooms needed using heap
    Time: O(n log n), Space: O(n)
    """
    if not intervals:
        return 0
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    # Min heap to track end times
    heap = []
    
    for start, end in intervals:
        # If room is available (earliest end time <= current start)
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        
        # Add current meeting's end time
        heapq.heappush(heap, end)
    
    return len(heap)

# Test
intervals = [[0,30],[5,10],[15,20]]
print(minMeetingRooms(intervals))  # 2
```

### Problem 7: Task Scheduler

```python
import heapq
from collections import Counter, deque

def leastInterval(tasks, n):
    """
    Schedule tasks with cooldown using heap
    Time: O(m log m), Space: O(m) where m = unique tasks
    """
    # Count task frequencies
    task_counts = Counter(tasks)
    
    # Max heap (use negative values)
    heap = [-count for count in task_counts.values()]
    heapq.heapify(heap)
    
    # Queue to track cooling down tasks
    queue = deque()  # (count, available_time)
    
    time = 0
    
    while heap or queue:
        time += 1
        
        # Add back tasks that finished cooling
        if queue and queue[0][1] == time:
            count = queue.popleft()[0]
            heapq.heappush(heap, count)
        
        # Execute most frequent available task
        if heap:
            count = heapq.heappop(heap)
            count += 1  # Decrease count (it was negative)
            
            if count < 0:  # Still has remaining executions
                queue.append((count, time + n + 1))
    
    return time

# Test
tasks = ["A","A","A","B","B","B"]
n = 2
print(leastInterval(tasks, n))  # 8
```

## 🎯 Graph Algorithm Applications

### Dijkstra's Shortest Path

```python
import heapq

def dijkstra(graph, start):
    """
    Shortest path using min heap
    Time: O((V + E) log V), Space: O(V)
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    
    heap = [(0, start)]  # (distance, node)
    visited = set()
    
    while heap:
        current_dist, current = heapq.heappop(heap)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        for neighbor, weight in graph[current]:
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))
    
    return distances

# Test graph
graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('C', 1), ('D', 5)],
    'C': [('D', 8), ('E', 10)],
    'D': [('E', 2)],
    'E': []
}
print(dijkstra(graph, 'A'))
```

### A* Pathfinding

```python
import heapq

def astar(grid, start, goal):
    """
    A* pathfinding algorithm using heap
    """
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance
    
    heap = [(0, start)]  # (f_score, position)
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    came_from = {}
    
    while heap:
        current_f, current = heapq.heappop(heap)
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        # Check neighbors
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            neighbor = (current[0] + dr, current[1] + dc)
            
            # Check bounds and obstacles
            if (0 <= neighbor[0] < len(grid) and 
                0 <= neighbor[1] < len(grid[0]) and 
                grid[neighbor[0]][neighbor[1]] == 0):
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(heap, (f_score[neighbor], neighbor))
    
    return []  # No path found
```

## 📊 Complexity Analysis

| **Operation** | **Time** | **Space** | **Notes** |
|---------------|----------|-----------|-----------|
| **Insert** | O(log n) | O(1) | Heapify up |
| **Extract Min/Max** | O(log n) | O(1) | Heapify down |
| **Peek** | O(1) | O(1) | Access root |
| **Build Heap** | O(n) | O(1) | From array |
| **Heap Sort** | O(n log n) | O(1) | In-place |

## 🏆 Practice Problems

### Easy
- [Kth Largest Element in Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)
- [Last Stone Weight](https://leetcode.com/problems/last-stone-weight/)
- [Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs/)

### Medium  
- [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)
- [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)
- [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)
- [Task Scheduler](https://leetcode.com/problems/task-scheduler/)
- [Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)

### Hard
- [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)
- [IPO](https://leetcode.com/problems/ipo/)
- [Minimum Cost to Hire K Workers](https://leetcode.com/problems/minimum-cost-to-hire-k-workers/)

## 📚 Key Applications

### Priority Queues
- **Task Scheduling**: Operating systems
- **Event Simulation**: Discrete event systems
- **Bandwidth Management**: Network QoS

### Graph Algorithms
- **Dijkstra's Algorithm**: Shortest paths
- **Prim's Algorithm**: Minimum spanning trees
- **A* Search**: Pathfinding with heuristics

### Data Processing
- **Top-K Problems**: Finding most/least frequent
- **Streaming Data**: Online algorithms
- **Memory Management**: LRU caches

## 🎯 Key Takeaways

1. **Choose the right heap type** - Min heap vs Max heap
2. **Python's heapq is min heap** - Use negative values for max heap
3. **Heap = Priority Queue** - Essential for many algorithms
4. **Two heap technique** - Powerful for median/percentile problems
5. **Heap + Hash Map** - Common pattern for sliding window problems
