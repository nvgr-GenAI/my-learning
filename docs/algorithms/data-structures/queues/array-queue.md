# Array-Based Queue

## 🔍 Overview

Array-based queues use a dynamic array (like Python lists) as the underlying data structure. This implementation offers simplicity but requires careful handling of the front and rear pointers to maintain efficiency.

---

## 📊 Characteristics

### Key Properties

- **Fixed or Dynamic Size**: Can use fixed-size arrays or dynamic arrays
- **Front and Rear Pointers**: Track the beginning and end of the queue
- **Sequential Memory**: Elements stored in consecutive memory locations
- **Simple Concept**: Easy to understand conceptually
- **Potential Inefficiency**: Basic implementation can be O(n) for dequeue

### Memory Layout

```text
Array-Based Queue:
Index:  0   1   2   3   4   5
Array: [ ] [A] [B] [C] [ ] [ ]
        ↑   ↑           ↑
      unused front     rear
```

---

## ⏱️ Time Complexities

### Simple Array Implementation

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| **Enqueue** | O(1) amortized | Add to end of array |
| **Dequeue** | O(n) | Must shift all elements |
| **Front** | O(1) | Direct access to first element |
| **Rear** | O(1) | Direct access to last element |
| **isEmpty** | O(1) | Check if array is empty |

### Optimized with Front/Rear Pointers

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| **Enqueue** | O(1) | Add at rear position |
| **Dequeue** | O(1) | Remove at front position |
| **Front** | O(1) | Access element at front index |
| **Rear** | O(1) | Access element at rear index |
| **isEmpty** | O(1) | Check if front > rear |

---

## 💻 Implementation

### Simple Array Queue (Inefficient)

```python
class SimpleArrayQueue:
    """Simple array-based queue (inefficient dequeue)."""
    
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
        """Remove item from front of queue (O(n))."""
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
        return f"Queue(front -> {self.items} <- rear)"

# Usage Example
queue = SimpleArrayQueue()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(queue)            # Queue(front -> [1, 2, 3] <- rear)
print(queue.dequeue())  # 1 (but this is O(n)!)
print(queue.front())    # 2
```

### Optimized Array Queue with Pointers

```python
class OptimizedArrayQueue:
    """Array-based queue with front/rear pointers."""
    
    def __init__(self, capacity=10):
        """Initialize queue with given capacity."""
        self.items = [None] * capacity
        self.capacity = capacity
        self.front_idx = 0
        self.rear_idx = -1
        self.count = 0
    
    def enqueue(self, item):
        """Add item to rear of queue."""
        if self.count >= self.capacity:
            raise OverflowError("Queue overflow")
        
        self.rear_idx = (self.rear_idx + 1) % self.capacity
        self.items[self.rear_idx] = item
        self.count += 1
    
    def dequeue(self):
        """Remove item from front of queue (O(1))."""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        
        item = self.items[self.front_idx]
        self.items[self.front_idx] = None  # Clear reference
        self.front_idx = (self.front_idx + 1) % self.capacity
        self.count -= 1
        return item
    
    def front(self):
        """Get front item without removing."""
        if self.is_empty():
            raise IndexError("Front from empty queue")
        return self.items[self.front_idx]
    
    def rear(self):
        """Get rear item without removing."""
        if self.is_empty():
            raise IndexError("Rear from empty queue")
        return self.items[self.rear_idx]
    
    def is_empty(self):
        """Check if queue is empty."""
        return self.count == 0
    
    def is_full(self):
        """Check if queue is full."""
        return self.count == self.capacity
    
    def size(self):
        """Get queue size."""
        return self.count
    
    def __str__(self):
        """String representation."""
        if self.is_empty():
            return "Queue(empty)"
        
        items = []
        idx = self.front_idx
        for _ in range(self.count):
            items.append(self.items[idx])
            idx = (idx + 1) % self.capacity
        
        return f"Queue(front -> {items} <- rear)"

# Usage Example
queue = OptimizedArrayQueue(5)
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(queue)            # Queue(front -> [1, 2, 3] <- rear)
print(queue.dequeue())  # 1 (O(1) operation!)
print(queue.front())    # 2
```

---

## ⚖️ Advantages & Disadvantages

### ✅ Advantages

- **Simple Concept**: Easy to understand and implement
- **Memory Efficient**: No extra pointer overhead per element
- **Cache Friendly**: Good memory locality for sequential access
- **Bounded Size**: Prevents unlimited memory usage

### ❌ Disadvantages

- **Fixed Capacity**: Must specify maximum size in advance
- **Wasted Space**: May have unused array positions
- **No Dynamic Resizing**: Cannot grow beyond initial capacity
- **Complex Index Management**: Requires careful circular array handling

---

## 🎯 When to Use Array-Based Queues

### ✅ Use Array-Based Queue When

- **Known Maximum Size**: Queue size is predictable and bounded
- **Memory Constraints**: Want to avoid pointer overhead
- **Performance Critical**: Need fast enqueue/dequeue operations
- **Simple Requirements**: Don't need dynamic resizing

### ❌ Avoid Array-Based Queue When

- **Unknown Size**: Queue size varies greatly
- **Dynamic Requirements**: Need unlimited growth
- **Memory Sensitive**: Cannot waste space on unused capacity
- **Frequent Resizing**: Size changes often

---

## 🔧 Advanced Techniques

### Dynamic Resizing

```python
class DynamicArrayQueue:
    """Array queue that can resize when needed."""
    
    def __init__(self, initial_capacity=4):
        """Initialize with small capacity that can grow."""
        self.items = [None] * initial_capacity
        self.capacity = initial_capacity
        self.front_idx = 0
        self.rear_idx = -1
        self.count = 0
    
    def _resize(self, new_capacity):
        """Resize the underlying array."""
        new_items = [None] * new_capacity
        idx = self.front_idx
        
        # Copy existing items to new array
        for i in range(self.count):
            new_items[i] = self.items[idx]
            idx = (idx + 1) % self.capacity
        
        self.items = new_items
        self.capacity = new_capacity
        self.front_idx = 0
        self.rear_idx = self.count - 1 if self.count > 0 else -1
    
    def enqueue(self, item):
        """Add item, resize if necessary."""
        if self.count >= self.capacity:
            self._resize(self.capacity * 2)  # Double capacity
        
        self.rear_idx = (self.rear_idx + 1) % self.capacity
        self.items[self.rear_idx] = item
        self.count += 1
    
    def dequeue(self):
        """Remove item, shrink if necessary."""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        
        item = self.items[self.front_idx]
        self.items[self.front_idx] = None
        self.front_idx = (self.front_idx + 1) % self.capacity
        self.count -= 1
        
        # Shrink if using less than 1/4 of capacity
        if self.count > 0 and self.count == self.capacity // 4:
            self._resize(self.capacity // 2)
        
        return item
```

### Memory-Efficient Implementation

```python
from collections import deque

class EfficientQueue:
    """Memory-efficient queue using collections.deque."""
    
    def __init__(self):
        """Initialize using deque for O(1) operations."""
        self._items = deque()
    
    def enqueue(self, item):
        """Add item to rear (O(1))."""
        self._items.append(item)
    
    def dequeue(self):
        """Remove item from front (O(1))."""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self._items.popleft()
    
    def front(self):
        """Get front item."""
        if self.is_empty():
            raise IndexError("Front from empty queue")
        return self._items[0]
    
    def rear(self):
        """Get rear item."""
        if self.is_empty():
            raise IndexError("Rear from empty queue")
        return self._items[-1]
    
    def is_empty(self):
        """Check if empty."""
        return len(self._items) == 0
    
    def size(self):
        """Get size."""
        return len(self._items)
```

---

## 🚀 Next Steps

- **[Linked List Queue](linked-list-queue.md)**: Learn dynamic implementation
- **[Circular Queue](circular-queue.md)**: Master space-efficient circular queues
- **[Priority Queue](priority-queue.md)**: Understand priority-based ordering
- **[Easy Problems](easy-problems.md)**: Practice basic queue operations

---

*Master the fundamentals before moving to more complex implementations!*
