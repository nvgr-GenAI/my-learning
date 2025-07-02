# Queues: Fundamentals & Theory

## 📚 What is a Queue?

A **Queue** is a linear data structure that follows the **First In, First Out (FIFO)** principle. Elements are added at one end (called the **rear** or **back**) and removed from the other end (called the **front**). Think of it like a line at a grocery store - the first person in line is the first person to be served.

## 🔄 FIFO Principle

**First In, First Out** means:

- The **first element** added to the queue is the **first one** to be removed
- Elements are added at the **rear** and removed from the **front**
- You cannot access elements in the middle without removing elements before them

```text
Enqueue (Add):           Dequeue (Remove):
                        
Rear  →  |   |   |   |  ← Front
         | C | B | A |
         +---+---+---+
         
         C added last   A removed first
```

## 🏗️ Basic Operations

### 1. Enqueue (Insert)

Add an element to the rear of the queue.

```python
def enqueue(queue, element):
    """Add element to rear of queue."""
    queue.append(element)
    
# Time: O(1), Space: O(1)
```

### 2. Dequeue (Remove)

Remove and return the front element from the queue.

```python
def dequeue(queue):
    """Remove and return front element."""
    if is_empty(queue):
        raise IndexError("Dequeue from empty queue")
    return queue.pop(0)  # Note: O(n) for list

# Time: O(n) for list, O(1) with proper implementation
```

### 3. Front (Access)

Return the front element without removing it.

```python
def front(queue):
    """Return front element without removing."""
    if is_empty(queue):
        raise IndexError("Front from empty queue")
    return queue[0]

# Time: O(1), Space: O(1)
```

### 4. Rear (Access)

Return the rear element without removing it.

```python
def rear(queue):
    """Return rear element without removing."""
    if is_empty(queue):
        raise IndexError("Rear from empty queue")
    return queue[-1]

# Time: O(1), Space: O(1)
```

### 5. isEmpty (Check)

Check if the queue is empty.

```python
def is_empty(queue):
    """Check if queue is empty."""
    return len(queue) == 0

# Time: O(1), Space: O(1)
```

## 💻 Implementations

### 1. Array-Based Implementation (Simple)

```python
class SimpleQueue:
    def __init__(self, capacity=10):
        """Initialize queue with given capacity."""
        self.items = []
        self.capacity = capacity
    
    def enqueue(self, item):
        """Add item to rear of queue."""
        if len(self.items) >= self.capacity:
            raise OverflowError("Queue overflow")
        self.items.append(item)
    
    def dequeue(self):
        """Remove item from front of queue."""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self.items.pop(0)  # O(n) operation!
    
    def front(self):
        """Get front item without removing."""
        if self.is_empty():
            raise IndexError("Front from empty queue")
        return self.items[0]
    
    def rear(self):
        """Get rear item without removing."""
        if self.is_empty():
            raise IndexError("Rear from empty queue")
        return self.items[-1]
    
    def is_empty(self):
        """Check if queue is empty."""
        return len(self.items) == 0
    
    def size(self):
        """Get queue size."""
        return len(self.items)
    
    def __str__(self):
        """String representation."""
        return f"Queue(front {self.items} rear)"
```

### 2. Circular Array Implementation (Efficient)

```python
class CircularQueue:
    def __init__(self, capacity):
        """Initialize circular queue with fixed capacity."""
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front_idx = 0
        self.rear_idx = -1
        self.size = 0
    
    def enqueue(self, item):
        """Add item to rear of queue."""
        if self.is_full():
            raise OverflowError("Queue overflow")
        
        self.rear_idx = (self.rear_idx + 1) % self.capacity
        self.queue[self.rear_idx] = item
        self.size += 1
    
    def dequeue(self):
        """Remove item from front of queue."""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        
        item = self.queue[self.front_idx]
        self.queue[self.front_idx] = None  # Optional: clear reference
        self.front_idx = (self.front_idx + 1) % self.capacity
        self.size -= 1
        return item
    
    def front(self):
        """Get front item without removing."""
        if self.is_empty():
            raise IndexError("Front from empty queue")
        return self.queue[self.front_idx]
    
    def rear(self):
        """Get rear item without removing."""
        if self.is_empty():
            raise IndexError("Rear from empty queue")
        return self.queue[self.rear_idx]
    
    def is_empty(self):
        """Check if queue is empty."""
        return self.size == 0
    
    def is_full(self):
        """Check if queue is full."""
        return self.size == self.capacity
    
    def get_size(self):
        """Get current queue size."""
        return self.size
    
    def __str__(self):
        """String representation."""
        if self.is_empty():
            return "CircularQueue([])"
        
        items = []
        idx = self.front_idx
        for _ in range(self.size):
            items.append(self.queue[idx])
            idx = (idx + 1) % self.capacity
        
        return f"CircularQueue({items})"

# Usage
cq = CircularQueue(5)
cq.enqueue(1)
cq.enqueue(2)
cq.enqueue(3)
print(cq)           # CircularQueue([1, 2, 3])
print(cq.dequeue()) # 1
cq.enqueue(4)
print(cq)           # CircularQueue([2, 3, 4])
```

### 3. Linked List Implementation

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedListQueue:
    def __init__(self):
        """Initialize empty queue."""
        self.front_node = None
        self.rear_node = None
        self._size = 0
    
    def enqueue(self, item):
        """Add item to rear of queue."""
        new_node = ListNode(item)
        
        if self.is_empty():
            self.front_node = self.rear_node = new_node
        else:
            self.rear_node.next = new_node
            self.rear_node = new_node
        
        self._size += 1
    
    def dequeue(self):
        """Remove item from front of queue."""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        
        item = self.front_node.val
        self.front_node = self.front_node.next
        
        if self.front_node is None:  # Queue became empty
            self.rear_node = None
        
        self._size -= 1
        return item
    
    def front(self):
        """Get front item without removing."""
        if self.is_empty():
            raise IndexError("Front from empty queue")
        return self.front_node.val
    
    def rear(self):
        """Get rear item without removing."""
        if self.is_empty():
            raise IndexError("Rear from empty queue")
        return self.rear_node.val
    
    def is_empty(self):
        """Check if queue is empty."""
        return self.front_node is None
    
    def size(self):
        """Get queue size."""
        return self._size
    
    def __str__(self):
        """String representation."""
        items = []
        current = self.front_node
        while current:
            items.append(current.val)
            current = current.next
        return f"LinkedQueue(front {items} rear)"
```

### 4. Using Python's Collections.deque

```python
from collections import deque

class DequeQueue:
    def __init__(self):
        """Initialize queue using deque."""
        self.queue = deque()
    
    def enqueue(self, item):
        """Add item to rear of queue."""
        self.queue.append(item)
    
    def dequeue(self):
        """Remove item from front of queue."""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self.queue.popleft()
    
    def front(self):
        """Get front item without removing."""
        if self.is_empty():
            raise IndexError("Front from empty queue")
        return self.queue[0]
    
    def rear(self):
        """Get rear item without removing."""
        if self.is_empty():
            raise IndexError("Rear from empty queue")
        return self.queue[-1]
    
    def is_empty(self):
        """Check if queue is empty."""
        return len(self.queue) == 0
    
    def size(self):
        """Get queue size."""
        return len(self.queue)
    
    def __str__(self):
        """String representation."""
        return f"DequeQueue({list(self.queue)})"

# Usage (Recommended for most use cases)
queue = DequeQueue()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(queue)           # DequeQueue([1, 2, 3])
print(queue.dequeue()) # 1
```

## 📊 Complexity Analysis

| **Implementation** | **Operation** | **Time** | **Space** |
|-------------------|---------------|----------|-----------|
| **Simple Array** | Enqueue | O(1) | O(1) |
| | Dequeue | O(n) | O(1) |
| | Front/Rear | O(1) | O(1) |
| **Circular Array** | Enqueue | O(1) | O(1) |
| | Dequeue | O(1) | O(1) |
| | Front/Rear | O(1) | O(1) |
| **Linked List** | Enqueue | O(1) | O(1) |
| | Dequeue | O(1) | O(1) |
| | Front/Rear | O(1) | O(1) |
| **Deque** | Enqueue | O(1) | O(1) |
| | Dequeue | O(1) | O(1) |
| | Front/Rear | O(1) | O(1) |

## ⚖️ Implementation Comparison

| **Feature** | **Simple Array** | **Circular Array** | **Linked List** | **Deque** |
|------------|------------------|-------------------|-----------------|-----------|
| **Space Efficiency** | Poor (unused space) | Excellent | Good | Excellent |
| **Time Complexity** | Poor (O(n) dequeue) | Excellent | Excellent | Excellent |
| **Memory Overhead** | Low | Low | High (pointers) | Low |
| **Implementation** | Simple | Moderate | Complex | Simple |
| **Dynamic Size** | Yes | No | Yes | Yes |

## 🎯 When to Use Each Implementation

### ✅ Use Simple Array When
- **Learning purposes**: Understanding basic queue concepts
- **Small queues**: Size is very small and performance isn't critical

### ✅ Use Circular Array When
- **Fixed size**: Maximum queue size is known
- **Memory efficiency**: Want to avoid wasted space
- **Embedded systems**: Memory is limited

### ✅ Use Linked List When
- **Dynamic size**: Queue size varies significantly
- **Memory is not a concern**: Extra pointer overhead is acceptable
- **Educational purposes**: Learning pointer manipulation

### ✅ Use Deque When
- **Production code**: Most practical choice
- **High performance**: Need O(1) operations
- **Python development**: Built-in and optimized

## 🔧 Advanced Queue Types

### 1. Priority Queue

Elements are served based on priority, not insertion order.

```python
import heapq

class PriorityQueue:
    def __init__(self):
        """Initialize priority queue."""
        self.heap = []
        self.index = 0
    
    def enqueue(self, item, priority):
        """Add item with given priority."""
        heapq.heappush(self.heap, (priority, self.index, item))
        self.index += 1
    
    def dequeue(self):
        """Remove item with highest priority."""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return heapq.heappop(self.heap)[2]
    
    def is_empty(self):
        """Check if queue is empty."""
        return len(self.heap) == 0

# Usage
pq = PriorityQueue()
pq.enqueue("Low priority", 3)
pq.enqueue("High priority", 1)
pq.enqueue("Medium priority", 2)

print(pq.dequeue())  # "High priority"
print(pq.dequeue())  # "Medium priority"
print(pq.dequeue())  # "Low priority"
```

### 2. Double-Ended Queue (Deque)

Allows insertion and deletion at both ends.

```python
from collections import deque

class Deque:
    def __init__(self):
        """Initialize double-ended queue."""
        self.deque = deque()
    
    def add_front(self, item):
        """Add item to front."""
        self.deque.appendleft(item)
    
    def add_rear(self, item):
        """Add item to rear."""
        self.deque.append(item)
    
    def remove_front(self):
        """Remove item from front."""
        if self.is_empty():
            raise IndexError("Remove from empty deque")
        return self.deque.popleft()
    
    def remove_rear(self):
        """Remove item from rear."""
        if self.is_empty():
            raise IndexError("Remove from empty deque")
        return self.deque.pop()
    
    def is_empty(self):
        """Check if deque is empty."""
        return len(self.deque) == 0
```

## 🎨 Common Patterns

### 1. Level-Order Traversal (BFS)

```python
def level_order_traversal(root):
    """Traverse tree level by level using queue."""
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

### 2. Sliding Window Maximum

```python
def sliding_window_maximum(nums, k):
    """Find maximum in each sliding window of size k."""
    from collections import deque
    
    dq = deque()  # Store indices
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove indices with smaller values
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add maximum to result
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

### 3. Circular Queue for Buffer

```python
class CircularBuffer:
    def __init__(self, capacity):
        """Initialize circular buffer."""
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0
    
    def put(self, item):
        """Add item to buffer."""
        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.capacity
        
        if self.size < self.capacity:
            self.size += 1
        else:
            # Buffer is full, overwrite oldest
            self.head = (self.head + 1) % self.capacity
    
    def get(self):
        """Remove oldest item from buffer."""
        if self.size == 0:
            raise IndexError("Buffer is empty")
        
        item = self.buffer[self.head]
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return item
```

## 🚀 Applications

### 1. Process Scheduling

```python
class ProcessScheduler:
    def __init__(self):
        """Initialize process scheduler."""
        self.ready_queue = deque()
        self.current_process = None
    
    def add_process(self, process):
        """Add process to ready queue."""
        self.ready_queue.append(process)
    
    def schedule_next(self):
        """Schedule next process."""
        if self.ready_queue:
            self.current_process = self.ready_queue.popleft()
            return self.current_process
        return None
    
    def preempt(self, process):
        """Preempt current process and add to queue."""
        if self.current_process:
            self.ready_queue.append(self.current_process)
        self.current_process = process
```

### 2. Request Rate Limiting

```python
class RateLimiter:
    def __init__(self, max_requests, time_window):
        """Initialize rate limiter."""
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
    
    def allow_request(self, timestamp):
        """Check if request is allowed."""
        # Remove old requests outside time window
        while self.requests and self.requests[0] <= timestamp - self.time_window:
            self.requests.popleft()
        
        # Check if under limit
        if len(self.requests) < self.max_requests:
            self.requests.append(timestamp)
            return True
        
        return False
```

### 3. Cache with LRU

```python
class LRUCache:
    def __init__(self, capacity):
        """Initialize LRU cache."""
        self.capacity = capacity
        self.cache = {}
        self.access_order = deque()
    
    def get(self, key):
        """Get value and mark as recently used."""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return -1
    
    def put(self, key, value):
        """Put key-value pair."""
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                lru_key = self.access_order.popleft()
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)
```

## 💡 Pro Tips

!!! tip "Memory Management"
    Use circular queues for fixed-size scenarios to avoid memory waste and improve cache performance.

!!! warning "Common Mistakes"
    - **Simple array dequeue**: O(n) time complexity kills performance
    - **Circular queue bounds**: Off-by-one errors in index calculations
    - **Empty queue operations**: Always check before dequeue/front/rear
    - **Thread safety**: Most implementations are not thread-safe

!!! success "Best Practices"
    - Use `collections.deque` for most Python applications
    - Implement circular queues for embedded/resource-constrained systems
    - Consider priority queues for scheduling problems
    - Use queues for BFS and level-order processing

## 🚀 Next Steps

Now that you understand queue fundamentals, practice with:

- **[Easy Problems](easy-problems.md)** - Build confidence with basic queue operations
- **[Medium Problems](medium-problems.md)** - Learn advanced patterns like sliding window
- **[Hard Problems](hard-problems.md)** - Master complex queue applications

---

*Ready to start practicing? Begin with the [Easy Problems](easy-problems.md) section!*
