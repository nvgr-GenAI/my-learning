# Linked List Queue

## 🔍 Overview

Linked list-based queues use nodes connected via pointers to implement the queue data structure. This approach provides true dynamic sizing without the need for pre-allocated arrays and naturally supports the FIFO principle with separate front and rear pointers.

---

## 📊 Characteristics

### Key Properties

- **Dynamic Size**: Grows and shrinks as needed without pre-allocation
- **Node-Based**: Each element is stored in a separate node with pointers
- **Front and Rear Pointers**: Maintain references to both ends of the queue
- **Memory Efficient**: Uses exactly the memory needed for current elements
- **Pointer Overhead**: Each node requires additional memory for the pointer

### Memory Layout

```text
Linked List Queue:
Front → [A|•] → [B|•] → [C|•] ← Rear
        data     data     data
        next     next     NULL
```

---

## ⏱️ Time Complexities

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| **Enqueue** | O(1) | Add at rear with rear pointer |
| **Dequeue** | O(1) | Remove at front with front pointer |
| **Front** | O(1) | Direct access to front node |
| **Rear** | O(1) | Direct access to rear node |
| **isEmpty** | O(1) | Check if front is None |
| **Size** | O(1) or O(n) | Depends on whether size is tracked |

---

## 💻 Implementation

### Basic Node Structure

```python
class QueueNode:
    """Node for linked list queue."""
    
    def __init__(self, data):
        """Initialize node with data."""
        self.data = data
        self.next = None
    
    def __str__(self):
        """String representation of node."""
        return str(self.data)
```

### Linked List Queue Implementation

```python
class LinkedListQueue:
    """Queue implementation using linked list."""
    
    def __init__(self):
        """Initialize empty queue."""
        self.front = None  # Points to front node
        self.rear = None   # Points to rear node
        self._size = 0     # Track size for O(1) size operation
    
    def enqueue(self, item):
        """Add item to rear of queue (O(1))."""
        new_node = QueueNode(item)
        
        if self.is_empty():
            # First element - both front and rear point to it
            self.front = new_node
            self.rear = new_node
        else:
            # Add to rear and update rear pointer
            self.rear.next = new_node
            self.rear = new_node
        
        self._size += 1
    
    def dequeue(self):
        """Remove item from front of queue (O(1))."""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        
        # Get data from front node
        item = self.front.data
        
        # Move front pointer to next node
        self.front = self.front.next
        
        # If queue becomes empty, update rear pointer
        if self.front is None:
            self.rear = None
        
        self._size -= 1
        return item
    
    def front_item(self):
        """Get front item without removing (O(1))."""
        if self.is_empty():
            raise IndexError("Front from empty queue")
        return self.front.data
    
    def rear_item(self):
        """Get rear item without removing (O(1))."""
        if self.is_empty():
            raise IndexError("Rear from empty queue")
        return self.rear.data
    
    def is_empty(self):
        """Check if queue is empty (O(1))."""
        return self.front is None
    
    def size(self):
        """Get queue size (O(1))."""
        return self._size
    
    def __str__(self):
        """String representation of queue."""
        if self.is_empty():
            return "Queue(empty)"
        
        items = []
        current = self.front
        while current:
            items.append(str(current.data))
            current = current.next
        
        return f"Queue(front -> {' -> '.join(items)} <- rear)"
    
    def __len__(self):
        """Return queue size."""
        return self._size

# Usage Example
queue = LinkedListQueue()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(queue)              # Queue(front -> 1 -> 2 -> 3 <- rear)
print(queue.dequeue())    # 1
print(queue.front_item()) # 2
print(f"Size: {queue.size()}")  # Size: 2
```

### Advanced Features

```python
class AdvancedLinkedListQueue:
    """Enhanced linked list queue with additional features."""
    
    def __init__(self):
        """Initialize empty queue."""
        self.front = None
        self.rear = None
        self._size = 0
    
    def enqueue(self, item):
        """Add item to rear of queue."""
        new_node = QueueNode(item)
        
        if self.is_empty():
            self.front = new_node
            self.rear = new_node
        else:
            self.rear.next = new_node
            self.rear = new_node
        
        self._size += 1
    
    def dequeue(self):
        """Remove item from front of queue."""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        
        item = self.front.data
        self.front = self.front.next
        
        if self.front is None:
            self.rear = None
        
        self._size -= 1
        return item
    
    def peek_front(self):
        """Peek at front item."""
        if self.is_empty():
            raise IndexError("Peek front from empty queue")
        return self.front.data
    
    def peek_rear(self):
        """Peek at rear item."""
        if self.is_empty():
            raise IndexError("Peek rear from empty queue")
        return self.rear.data
    
    def contains(self, item):
        """Check if item exists in queue (O(n))."""
        current = self.front
        while current:
            if current.data == item:
                return True
            current = current.next
        return False
    
    def to_list(self):
        """Convert queue to list (O(n))."""
        result = []
        current = self.front
        while current:
            result.append(current.data)
            current = current.next
        return result
    
    def clear(self):
        """Clear all elements from queue."""
        self.front = None
        self.rear = None
        self._size = 0
    
    def is_empty(self):
        """Check if queue is empty."""
        return self.front is None
    
    def size(self):
        """Get queue size."""
        return self._size
    
    def __iter__(self):
        """Make queue iterable."""
        current = self.front
        while current:
            yield current.data
            current = current.next
    
    def __str__(self):
        """String representation."""
        if self.is_empty():
            return "Queue(empty)"
        return f"Queue({' -> '.join(str(item) for item in self)})"

# Usage Example
queue = AdvancedLinkedListQueue()
queue.enqueue("first")
queue.enqueue("second")
queue.enqueue("third")

print(queue)  # Queue(first -> second -> third)
print(f"Contains 'second': {queue.contains('second')}")  # True
print(f"As list: {queue.to_list()}")  # ['first', 'second', 'third']

# Iterate through queue
for item in queue:
    print(f"Item: {item}")
```

---

## ⚖️ Advantages & Disadvantages

### ✅ Advantages

- **True Dynamic Size**: No need to specify capacity in advance
- **Memory Efficient**: Only allocates what's needed
- **No Wasted Space**: Every allocated node is used
- **Unlimited Growth**: Can grow as large as available memory
- **Simple Logic**: FIFO naturally fits linked list structure

### ❌ Disadvantages

- **Memory Overhead**: Each node requires extra memory for pointers
- **No Random Access**: Cannot directly access middle elements
- **Cache Performance**: Poor memory locality due to scattered nodes
- **Pointer Management**: More complex than array-based implementation

---

## 🎯 When to Use Linked List Queues

### ✅ Use Linked List Queue When

- **Unknown Size Requirements**: Queue size varies greatly
- **Memory Constraints**: Cannot pre-allocate large arrays
- **Dynamic Applications**: Frequent size changes
- **Unlimited Growth**: Need potential for very large queues

### ❌ Avoid Linked List Queue When

- **High Performance Required**: Need maximum speed with predictable size
- **Memory Overhead Matters**: Pointer overhead is significant
- **Cache Locality Important**: Need better memory access patterns
- **Simple Fixed-Size Requirements**: Array implementation is simpler

---

## 🔧 Memory Management Considerations

### Python's Automatic Garbage Collection

```python
class GCFriendlyQueue:
    """Queue that helps with garbage collection."""
    
    def __init__(self):
        self.front = None
        self.rear = None
        self._size = 0
    
    def dequeue(self):
        """Dequeue with explicit cleanup."""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        
        item = self.front.data
        old_front = self.front
        self.front = self.front.next
        
        # Explicitly clear reference to help GC
        old_front.next = None
        old_front.data = None
        
        if self.front is None:
            self.rear = None
        
        self._size -= 1
        return item
```

### Manual Memory Management (C++ Style)

```python
class ManualMemoryQueue:
    """Example showing memory management concepts."""
    
    def __init__(self):
        self.front = None
        self.rear = None
        self._size = 0
        self._node_pool = []  # Simulate memory pool
    
    def _get_node(self, data):
        """Get node from pool or create new one."""
        if self._node_pool:
            node = self._node_pool.pop()
            node.data = data
            node.next = None
            return node
        return QueueNode(data)
    
    def _return_node(self, node):
        """Return node to pool for reuse."""
        node.data = None
        node.next = None
        self._node_pool.append(node)
    
    def dequeue(self):
        """Dequeue with node reuse."""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        
        item = self.front.data
        old_front = self.front
        self.front = self.front.next
        
        # Return node to pool
        self._return_node(old_front)
        
        if self.front is None:
            self.rear = None
        
        self._size -= 1
        return item
```

---

## 🚀 Variations and Extensions

### Double-Ended Queue (Deque)

```python
class LinkedListDeque:
    """Double-ended queue using doubly linked list."""
    
    class Node:
        def __init__(self, data):
            self.data = data
            self.next = None
            self.prev = None
    
    def __init__(self):
        self.front = None
        self.rear = None
        self._size = 0
    
    def add_front(self, item):
        """Add item to front."""
        new_node = self.Node(item)
        
        if self.is_empty():
            self.front = new_node
            self.rear = new_node
        else:
            new_node.next = self.front
            self.front.prev = new_node
            self.front = new_node
        
        self._size += 1
    
    def add_rear(self, item):
        """Add item to rear."""
        new_node = self.Node(item)
        
        if self.is_empty():
            self.front = new_node
            self.rear = new_node
        else:
            self.rear.next = new_node
            new_node.prev = self.rear
            self.rear = new_node
        
        self._size += 1
    
    def remove_front(self):
        """Remove item from front."""
        if self.is_empty():
            raise IndexError("Remove from empty deque")
        
        item = self.front.data
        self.front = self.front.next
        
        if self.front:
            self.front.prev = None
        else:
            self.rear = None
        
        self._size -= 1
        return item
    
    def remove_rear(self):
        """Remove item from rear."""
        if self.is_empty():
            raise IndexError("Remove from empty deque")
        
        item = self.rear.data
        self.rear = self.rear.prev
        
        if self.rear:
            self.rear.next = None
        else:
            self.front = None
        
        self._size -= 1
        return item
```

---

## 🚀 Next Steps

- **[Array Queue](array-queue.md)**: Compare with array-based implementation
- **[Circular Queue](circular-queue.md)**: Learn space-efficient circular approach
- **[Priority Queue](priority-queue.md)**: Understand priority-based ordering
- **[Easy Problems](easy-problems.md)**: Practice queue operations

---

*Understanding linked list queues prepares you for more advanced data structures!*
