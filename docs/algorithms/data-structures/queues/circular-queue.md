# Circular Queue

## 🔍 Overview

A Circular Queue (also known as Ring Buffer) is an advanced array-based queue implementation that overcomes the space wastage problem of linear array queues. It treats the array as circular by connecting the end back to the beginning, allowing efficient reuse of empty spaces.

---

## 📊 Characteristics

### Key Properties

- **Circular Array**: Last position connects back to the first position
- **Space Efficient**: Reuses empty positions left by dequeued elements
- **Fixed Size**: Predetermined maximum capacity
- **Full Detection**: Requires careful logic to distinguish full vs empty states
- **Optimal Performance**: O(1) for all basic operations

### Memory Layout

```text
Circular Queue (Capacity = 6):
Indices: 0   1   2   3   4   5
Array:  [D] [ ] [ ] [A] [B] [C]
         ↑               ↑
       rear            front

After enqueue(E): rear moves to index 1
Array:  [D] [E] [ ] [A] [B] [C]
             ↑       ↑
           rear    front
```

---

## ⏱️ Time Complexities

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| **Enqueue** | O(1) | Move rear pointer circularly |
| **Dequeue** | O(1) | Move front pointer circularly |
| **Front** | O(1) | Direct access at front index |
| **Rear** | O(1) | Direct access at rear index |
| **isEmpty** | O(1) | Compare pointers or check count |
| **isFull** | O(1) | Compare pointers or check count |

---

## 💻 Implementation

### Method 1: Using Count Variable

```python
class CircularQueue:
    """Circular queue implementation using count variable."""
    
    def __init__(self, capacity):
        """Initialize circular queue with given capacity."""
        self.capacity = capacity
        self.items = [None] * capacity
        self.front = 0
        self.rear = -1
        self.count = 0
    
    def enqueue(self, item):
        """Add item to rear of queue."""
        if self.is_full():
            raise OverflowError("Queue is full")
        
        # Move rear circularly
        self.rear = (self.rear + 1) % self.capacity
        self.items[self.rear] = item
        self.count += 1
    
    def dequeue(self):
        """Remove item from front of queue."""
        if self.is_empty():
            raise IndexError("Queue is empty")
        
        item = self.items[self.front]
        self.items[self.front] = None  # Clear reference
        
        # Move front circularly
        self.front = (self.front + 1) % self.capacity
        self.count -= 1
        
        return item
    
    def front_item(self):
        """Get front item without removing."""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[self.front]
    
    def rear_item(self):
        """Get rear item without removing."""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[self.rear]
    
    def is_empty(self):
        """Check if queue is empty."""
        return self.count == 0
    
    def is_full(self):
        """Check if queue is full."""
        return self.count == self.capacity
    
    def size(self):
        """Get current size."""
        return self.count
    
    def __str__(self):
        """String representation."""
        if self.is_empty():
            return "CircularQueue(empty)"
        
        items = []
        idx = self.front
        for _ in range(self.count):
            items.append(str(self.items[idx]))
            idx = (idx + 1) % self.capacity
        
        return f"CircularQueue([{', '.join(items)}])"

# Usage Example
queue = CircularQueue(5)
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(queue)            # CircularQueue([1, 2, 3])
print(queue.dequeue())  # 1
queue.enqueue(4)
queue.enqueue(5)
queue.enqueue(6)        # Queue is now full
print(queue)            # CircularQueue([2, 3, 4, 5, 6])
```

### Method 2: Sacrificing One Position

```python
class CircularQueueSacrificial:
    """Circular queue that sacrifices one position to detect full state."""
    
    def __init__(self, capacity):
        """Initialize circular queue."""
        self.capacity = capacity + 1  # Add 1 for sacrificial position
        self.items = [None] * self.capacity
        self.front = 0
        self.rear = 0
    
    def enqueue(self, item):
        """Add item to rear of queue."""
        if self.is_full():
            raise OverflowError("Queue is full")
        
        self.items[self.rear] = item
        self.rear = (self.rear + 1) % self.capacity
    
    def dequeue(self):
        """Remove item from front of queue."""
        if self.is_empty():
            raise IndexError("Queue is empty")
        
        item = self.items[self.front]
        self.items[self.front] = None
        self.front = (self.front + 1) % self.capacity
        
        return item
    
    def is_empty(self):
        """Check if queue is empty."""
        return self.front == self.rear
    
    def is_full(self):
        """Check if queue is full."""
        return (self.rear + 1) % self.capacity == self.front
    
    def size(self):
        """Get current size."""
        return (self.rear - self.front + self.capacity) % self.capacity
    
    def actual_capacity(self):
        """Get actual usable capacity."""
        return self.capacity - 1

# Usage Example
queue = CircularQueueSacrificial(5)  # Can actually hold 5 items
for i in range(5):
    queue.enqueue(i)
print(f"Size: {queue.size()}")       # Size: 5
print(f"Is full: {queue.is_full()}")  # Is full: True
```

### Method 3: Using Flag Variable

```python
class CircularQueueFlag:
    """Circular queue using flag to distinguish full vs empty."""
    
    def __init__(self, capacity):
        """Initialize circular queue."""
        self.capacity = capacity
        self.items = [None] * capacity
        self.front = 0
        self.rear = 0
        self.is_queue_full = False
    
    def enqueue(self, item):
        """Add item to rear of queue."""
        if self.is_full():
            raise OverflowError("Queue is full")
        
        self.items[self.rear] = item
        self.rear = (self.rear + 1) % self.capacity
        
        # Check if queue becomes full
        if self.rear == self.front:
            self.is_queue_full = True
    
    def dequeue(self):
        """Remove item from front of queue."""
        if self.is_empty():
            raise IndexError("Queue is empty")
        
        item = self.items[self.front]
        self.items[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.is_queue_full = False  # Can't be full after dequeue
        
        return item
    
    def is_empty(self):
        """Check if queue is empty."""
        return self.front == self.rear and not self.is_queue_full
    
    def is_full(self):
        """Check if queue is full."""
        return self.is_queue_full
    
    def size(self):
        """Get current size."""
        if self.is_queue_full:
            return self.capacity
        return (self.rear - self.front + self.capacity) % self.capacity
```

---

## 🔧 Advanced Features

### Dynamic Circular Queue

```python
class DynamicCircularQueue:
    """Circular queue that can resize when needed."""
    
    def __init__(self, initial_capacity=4):
        """Initialize with small capacity that can grow."""
        self.capacity = initial_capacity
        self.items = [None] * self.capacity
        self.front = 0
        self.rear = -1
        self.count = 0
    
    def _resize(self, new_capacity):
        """Resize the circular queue."""
        new_items = [None] * new_capacity
        
        # Copy existing items to new array
        idx = self.front
        for i in range(self.count):
            new_items[i] = self.items[idx]
            idx = (idx + 1) % self.capacity
        
        # Update queue parameters
        self.items = new_items
        self.capacity = new_capacity
        self.front = 0
        self.rear = self.count - 1 if self.count > 0 else -1
    
    def enqueue(self, item):
        """Add item, resize if necessary."""
        if self.is_full():
            self._resize(self.capacity * 2)
        
        self.rear = (self.rear + 1) % self.capacity
        self.items[self.rear] = item
        self.count += 1
    
    def dequeue(self):
        """Remove item, shrink if necessary."""
        if self.is_empty():
            raise IndexError("Queue is empty")
        
        item = self.items[self.front]
        self.items[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.count -= 1
        
        # Shrink if using less than 1/4 of capacity
        if self.count > 0 and self.count <= self.capacity // 4:
            self._resize(max(4, self.capacity // 2))
        
        return item
    
    def is_empty(self):
        return self.count == 0
    
    def is_full(self):
        return self.count == self.capacity
```

### Thread-Safe Circular Queue

```python
import threading

class ThreadSafeCircularQueue:
    """Thread-safe circular queue using locks."""
    
    def __init__(self, capacity):
        """Initialize thread-safe circular queue."""
        self.capacity = capacity
        self.items = [None] * capacity
        self.front = 0
        self.rear = -1
        self.count = 0
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)
    
    def enqueue(self, item, timeout=None):
        """Add item with optional timeout."""
        with self.not_full:
            # Wait until not full
            if not self.not_full.wait_for(lambda: not self.is_full(), timeout):
                raise TimeoutError("Enqueue timeout")
            
            self.rear = (self.rear + 1) % self.capacity
            self.items[self.rear] = item
            self.count += 1
            
            # Notify waiting dequeue operations
            self.not_empty.notify()
    
    def dequeue(self, timeout=None):
        """Remove item with optional timeout."""
        with self.not_empty:
            # Wait until not empty
            if not self.not_empty.wait_for(lambda: not self.is_empty(), timeout):
                raise TimeoutError("Dequeue timeout")
            
            item = self.items[self.front]
            self.items[self.front] = None
            self.front = (self.front + 1) % self.capacity
            self.count -= 1
            
            # Notify waiting enqueue operations
            self.not_full.notify()
            return item
    
    def is_empty(self):
        """Check if empty (should be called with lock held)."""
        return self.count == 0
    
    def is_full(self):
        """Check if full (should be called with lock held)."""
        return self.count == self.capacity
    
    def size(self):
        """Get size in thread-safe manner."""
        with self.lock:
            return self.count
```

---

## ⚖️ Advantages & Disadvantages

### ✅ Advantages

- **Space Efficient**: No wasted array positions
- **O(1) Operations**: All basic operations are constant time
- **Memory Locality**: Good cache performance
- **Predictable Performance**: No dynamic allocation overhead
- **Bounded Size**: Prevents unlimited memory usage

### ❌ Disadvantages

- **Fixed Capacity**: Must specify maximum size in advance
- **Complex Logic**: Full vs empty detection requires careful handling
- **No Dynamic Growth**: Cannot expand beyond initial capacity
- **Implementation Complexity**: More complex than simple array queue

---

## 🎯 When to Use Circular Queues

### ✅ Use Circular Queue When

- **Fixed Buffer Size**: Known maximum capacity requirements
- **High Performance Needed**: Want O(1) operations with minimal overhead
- **Memory Efficiency Critical**: Cannot waste space
- **Producer-Consumer Scenarios**: Bounded buffer between threads
- **Real-time Systems**: Predictable performance requirements

### ❌ Avoid Circular Queue When

- **Unknown Size Requirements**: Queue size varies greatly
- **Dynamic Growth Needed**: Capacity requirements change
- **Simple Applications**: Linear queue is sufficient
- **Memory Not a Constraint**: Simplicity preferred over efficiency

---

## 🔄 Real-World Applications

### 1. **Operating Systems**

```python
class CPUScheduler:
    """Round-robin CPU scheduler using circular queue."""
    
    def __init__(self, max_processes):
        self.ready_queue = CircularQueue(max_processes)
        self.time_quantum = 10
    
    def add_process(self, process):
        """Add process to ready queue."""
        self.ready_queue.enqueue(process)
    
    def schedule_next(self):
        """Schedule next process."""
        if not self.ready_queue.is_empty():
            return self.ready_queue.dequeue()
        return None
    
    def preempt_process(self, process):
        """Preempt current process and add back to queue."""
        if not self.ready_queue.is_full():
            self.ready_queue.enqueue(process)
```

### 2. **Streaming and Buffering**

```python
class StreamBuffer:
    """Circular buffer for streaming data."""
    
    def __init__(self, buffer_size):
        self.buffer = CircularQueue(buffer_size)
        self.overflow_count = 0
    
    def write(self, data):
        """Write data to buffer."""
        try:
            self.buffer.enqueue(data)
        except OverflowError:
            # Handle overflow - could overwrite or drop
            self.overflow_count += 1
            if not self.buffer.is_empty():
                self.buffer.dequeue()  # Remove oldest
            self.buffer.enqueue(data)   # Add new
    
    def read(self):
        """Read data from buffer."""
        if not self.buffer.is_empty():
            return self.buffer.dequeue()
        return None
```

### 3. **Game Development**

```python
class InputBuffer:
    """Game input buffer using circular queue."""
    
    def __init__(self, buffer_size=100):
        self.inputs = CircularQueue(buffer_size)
        self.frame_counter = 0
    
    def record_input(self, input_event):
        """Record player input."""
        try:
            self.inputs.enqueue({
                'event': input_event,
                'frame': self.frame_counter
            })
        except OverflowError:
            # Oldest input gets overwritten
            self.inputs.dequeue()
            self.inputs.enqueue({
                'event': input_event,
                'frame': self.frame_counter
            })
    
    def get_next_input(self):
        """Get next input to process."""
        if not self.inputs.is_empty():
            return self.inputs.dequeue()
        return None
    
    def update_frame(self):
        """Update frame counter."""
        self.frame_counter += 1
```

---

## 🚀 Next Steps

- **[Array Queue](array-queue.md)**: Compare with simple array implementation
- **[Linked List Queue](linked-list-queue.md)**: Understand dynamic alternative
- **[Priority Queue](priority-queue.md)**: Learn priority-based ordering
- **[Medium Problems](medium-problems.md)**: Practice with sliding window problems

---

*Circular queues are essential for efficient buffer management and real-time systems!*
