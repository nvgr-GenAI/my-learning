# Data Structures 📊

Master the fundamental building blocks of computer science - from basic arrays to advanced tree structures.

## Overview

Data structures are specialized formats for organizing, processing, retrieving and storing data. Choosing the right data structure for your problem is crucial for writing efficient algorithms.

## 📋 Quick Reference

| Data Structure | Access | Search | Insertion | Deletion | Space | Use Cases |
|----------------|--------|--------|-----------|----------|-------|-----------|
| **Array** | O(1) | O(n) | O(n) | O(n) | O(n) | Random access, cache-friendly |
| **Dynamic Array** | O(1) | O(n) | O(1)* | O(n) | O(n) | Resizable arrays, lists |
| **Linked List** | O(n) | O(n) | O(1) | O(1) | O(n) | Frequent insertions/deletions |
| **Stack** | O(n) | O(n) | O(1) | O(1) | O(n) | LIFO operations, recursion |
| **Queue** | O(n) | O(n) | O(1) | O(1) | O(n) | FIFO operations, BFS |
| **Hash Table** | N/A | O(1)* | O(1)* | O(1)* | O(n) | Fast lookups, caching |
| **Binary Search Tree** | O(log n)* | O(log n)* | O(log n)* | O(log n)* | O(n) | Ordered data, range queries |
| **Heap** | O(1) | O(n) | O(log n) | O(log n) | O(n) | Priority queues, sorting |

*Amortized or average case

## 🔗 Linear Data Structures

### Arrays

Fixed-size, contiguous memory allocation with O(1) random access.

```python
# Static Array Operations
class StaticArray:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = [None] * capacity
        self.size = 0
    
    def get(self, index):
        """Get element at index. O(1) time."""
        if 0 <= index < self.size:
            return self.data[index]
        raise IndexError("Index out of bounds")
    
    def set(self, index, value):
        """Set element at index. O(1) time."""
        if 0 <= index < self.size:
            self.data[index] = value
        else:
            raise IndexError("Index out of bounds")
    
    def append(self, value):
        """Add element to end. O(1) time if space available."""
        if self.size < self.capacity:
            self.data[self.size] = value
            self.size += 1
        else:
            raise OverflowError("Array is full")
    
    def insert(self, index, value):
        """Insert element at index. O(n) time."""
        if self.size >= self.capacity:
            raise OverflowError("Array is full")
        
        if not 0 <= index <= self.size:
            raise IndexError("Index out of bounds")
        
        # Shift elements to the right
        for i in range(self.size, index, -1):
            self.data[i] = self.data[i - 1]
        
        self.data[index] = value
        self.size += 1
    
    def delete(self, index):
        """Delete element at index. O(n) time."""
        if not 0 <= index < self.size:
            raise IndexError("Index out of bounds")
        
        value = self.data[index]
        
        # Shift elements to the left
        for i in range(index, self.size - 1):
            self.data[i] = self.data[i + 1]
        
        self.size -= 1
        return value
```

### Dynamic Arrays

Resizable arrays that grow automatically when needed.

```python
class DynamicArray:
    """
    A resizable array implementation with amortized O(1) append.
    Demonstrates capacity doubling strategy.
    """
    def __init__(self, capacity=2):
        self.capacity = capacity
        self.size = 0
        self.data = [None] * self.capacity
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError("Index out of range")
        return self.data[index]
    
    def __setitem__(self, index, value):
        if not 0 <= index < self.size:
            raise IndexError("Index out of range")
        self.data[index] = value
    
    def _resize(self, new_capacity):
        """Double the capacity when needed."""
        old_data = self.data
        self.capacity = new_capacity
        self.data = [None] * self.capacity
        
        for i in range(self.size):
            self.data[i] = old_data[i]
    
    def append(self, item):
        """Add item to end. Amortized O(1) time."""
        if self.size >= self.capacity:
            self._resize(2 * self.capacity)
        
        self.data[self.size] = item
        self.size += 1
    
    def insert(self, index, item):
        """Insert item at index. O(n) time."""
        if not 0 <= index <= self.size:
            raise IndexError("Index out of range")
        
        if self.size >= self.capacity:
            self._resize(2 * self.capacity)
        
        # Shift elements to the right
        for i in range(self.size, index, -1):
            self.data[i] = self.data[i-1]
        
        self.data[index] = item
        self.size += 1
    
    def pop(self, index=-1):
        """Remove and return item at index. O(n) time."""
        if self.size == 0:
            raise IndexError("Pop from empty array")
        
        if index < 0:
            index += self.size
        
        if not 0 <= index < self.size:
            raise IndexError("Index out of range")
        
        item = self.data[index]
        
        # Shift elements to the left
        for i in range(index, self.size - 1):
            self.data[i] = self.data[i + 1]
        
        self.size -= 1
        
        # Shrink if necessary (optional optimization)
        if self.size <= self.capacity // 4:
            self._resize(max(2, self.capacity // 2))
        
        return item

# Usage example
arr = DynamicArray()
for i in range(10):
    arr.append(i)

print(f"Array: {[arr[i] for i in range(len(arr))]}")
print(f"Length: {len(arr)}, Capacity: {arr.capacity}")
```

### Linked Lists

Dynamic data structures with nodes connected via pointers.

```python
class ListNode:
    """Node for singly linked list."""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class SinglyLinkedList:
    """Basic singly linked list implementation."""
    def __init__(self):
        self.head = None
        self.size = 0
    
    def prepend(self, val):
        """Add to front. O(1) time."""
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
        self.size += 1
    
    def append(self, val):
        """Add to back. O(n) time."""
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self.size += 1
    
    def insert(self, index, val):
        """Insert at index. O(n) time."""
        if index < 0 or index > self.size:
            raise IndexError("Index out of bounds")
        
        if index == 0:
            self.prepend(val)
            return
        
        new_node = ListNode(val)
        current = self.head
        for _ in range(index - 1):
            current = current.next
        
        new_node.next = current.next
        current.next = new_node
        self.size += 1
    
    def delete(self, val):
        """Delete first occurrence. O(n) time."""
        if not self.head:
            raise ValueError("List is empty")
        
        if self.head.val == val:
            self.head = self.head.next
            self.size -= 1
            return
        
        current = self.head
        while current.next and current.next.val != val:
            current = current.next
        
        if current.next:
            current.next = current.next.next
            self.size -= 1
        else:
            raise ValueError("Value not found")
    
    def find(self, val):
        """Find first occurrence. O(n) time."""
        current = self.head
        index = 0
        while current:
            if current.val == val:
                return index
            current = current.next
            index += 1
        return -1
    
    def to_list(self):
        """Convert to Python list."""
        result = []
        current = self.head
        while current:
            result.append(current.val)
            current = current.next
        return result
    
    def __len__(self):
        return self.size

class DoublyLinkedListNode:
    """Node for doubly linked list."""
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

class DoublyLinkedList:
    """
    Doubly linked list with O(1) insertion/deletion at both ends.
    Useful for implementing deques, LRU cache, etc.
    """
    def __init__(self):
        # Sentinel nodes to simplify edge cases
        self.head = DoublyLinkedListNode()
        self.tail = DoublyLinkedListNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
    
    def _add_node(self, node, prev_node):
        """Add node after prev_node."""
        next_node = prev_node.next
        
        prev_node.next = node
        node.prev = prev_node
        node.next = next_node
        next_node.prev = node
        
        self.size += 1
    
    def _remove_node(self, node):
        """Remove given node."""
        prev_node = node.prev
        next_node = node.next
        
        prev_node.next = next_node
        next_node.prev = prev_node
        
        self.size -= 1
    
    def add_first(self, val):
        """Add to front. O(1) time."""
        new_node = DoublyLinkedListNode(val)
        self._add_node(new_node, self.head)
    
    def add_last(self, val):
        """Add to back. O(1) time."""
        new_node = DoublyLinkedListNode(val)
        self._add_node(new_node, self.tail.prev)
    
    def remove_first(self):
        """Remove from front. O(1) time."""
        if self.size == 0:
            raise IndexError("Remove from empty list")
        
        first_node = self.head.next
        self._remove_node(first_node)
        return first_node.val
    
    def remove_last(self):
        """Remove from back. O(1) time."""
        if self.size == 0:
            raise IndexError("Remove from empty list")
        
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node.val
```

### Stacks

LIFO (Last In, First Out) data structure.

```python
class Stack:
    """Stack implementation using dynamic array."""
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """Add item to top. O(1) time."""
        self.items.append(item)
    
    def pop(self):
        """Remove and return top item. O(1) time."""
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self.items.pop()
    
    def peek(self):
        """Return top item without removing. O(1) time."""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self.items[-1]
    
    def is_empty(self):
        """Check if stack is empty. O(1) time."""
        return len(self.items) == 0
    
    def size(self):
        """Return number of items. O(1) time."""
        return len(self.items)

# Stack using linked list
class LinkedStack:
    """Stack implementation using linked list."""
    def __init__(self):
        self.head = None
        self._size = 0
    
    def push(self, item):
        """Add item to top. O(1) time."""
        new_node = ListNode(item)
        new_node.next = self.head
        self.head = new_node
        self._size += 1
    
    def pop(self):
        """Remove and return top item. O(1) time."""
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        
        item = self.head.val
        self.head = self.head.next
        self._size -= 1
        return item
    
    def peek(self):
        """Return top item without removing. O(1) time."""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self.head.val
    
    def is_empty(self):
        return self.head is None
    
    def size(self):
        return self._size
```

### Queues

FIFO (First In, First Out) data structure.

```python
from collections import deque

class Queue:
    """Queue implementation using deque for O(1) operations."""
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        """Add item to rear. O(1) time."""
        self.items.append(item)
    
    def dequeue(self):
        """Remove and return front item. O(1) time."""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self.items.popleft()
    
    def front(self):
        """Return front item without removing. O(1) time."""
        if self.is_empty():
            raise IndexError("Front from empty queue")
        return self.items[0]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

class CircularQueue:
    """Circular queue with fixed capacity."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front_idx = 0
        self.rear_idx = -1
        self.size = 0
    
    def enqueue(self, item):
        """Add item to rear. O(1) time."""
        if self.is_full():
            raise OverflowError("Queue is full")
        
        self.rear_idx = (self.rear_idx + 1) % self.capacity
        self.queue[self.rear_idx] = item
        self.size += 1
    
    def dequeue(self):
        """Remove and return front item. O(1) time."""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        
        item = self.queue[self.front_idx]
        self.queue[self.front_idx] = None
        self.front_idx = (self.front_idx + 1) % self.capacity
        self.size -= 1
        return item
    
    def is_empty(self):
        return self.size == 0
    
    def is_full(self):
        return self.size == self.capacity
    
    def get_size(self):
        return self.size
```

## 🏗️ Non-Linear Data Structures

### Hash Tables

Fast key-value storage with average O(1) operations.

```python
class HashTable:
    """Hash table with separate chaining for collision resolution."""
    def __init__(self, capacity=8):
        self.capacity = capacity
        self.size = 0
        self.buckets = [[] for _ in range(capacity)]
    
    def _hash(self, key):
        """Simple hash function."""
        return hash(key) % self.capacity
    
    def _resize(self):
        """Resize when load factor exceeds threshold."""
        old_buckets = self.buckets
        self.capacity *= 2
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        
        # Rehash all items
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)
    
    def put(self, key, value):
        """Insert or update key-value pair. O(1) average."""
        if self.size >= self.capacity * 0.75:  # Load factor threshold
            self._resize()
        
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        # Update existing key
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        # Add new key-value pair
        bucket.append((key, value))
        self.size += 1
    
    def get(self, key):
        """Get value by key. O(1) average."""
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        raise KeyError(key)
    
    def delete(self, key):
        """Delete key-value pair. O(1) average."""
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.size -= 1
                return v
        
        raise KeyError(key)
    
    def contains(self, key):
        """Check if key exists. O(1) average."""
        try:
            self.get(key)
            return True
        except KeyError:
            return False
    
    def keys(self):
        """Return all keys."""
        result = []
        for bucket in self.buckets:
            for key, _ in bucket:
                result.append(key)
        return result
    
    def values(self):
        """Return all values."""
        result = []
        for bucket in self.buckets:
            for _, value in bucket:
                result.append(value)
        return result
```

## 🌳 Tree Data Structures

See detailed implementations in:
- [Binary Trees](trees.md)
- [Binary Search Trees](trees.md#binary-search-trees)
- [AVL Trees](trees.md#avl-trees)
- [Heaps](trees.md#heaps)

## 📈 Performance Comparison

### Space-Time Tradeoffs

| Operation | Array | Linked List | Hash Table | BST | Heap |
|-----------|-------|-------------|------------|-----|------|
| **Access** | O(1) | O(n) | N/A | O(log n) | O(1) peek |
| **Search** | O(n) | O(n) | O(1)* | O(log n) | O(n) |
| **Insert** | O(n) | O(1) | O(1)* | O(log n) | O(log n) |
| **Delete** | O(n) | O(1) | O(1)* | O(log n) | O(log n) |
| **Memory** | Compact | Extra pointers | Extra buckets | Extra pointers | Compact |

### When to Use Each Structure

**Use Arrays when:**
- You need random access to elements
- Memory usage is critical
- Cache performance matters
- Size is relatively fixed

**Use Linked Lists when:**
- Frequent insertions/deletions at arbitrary positions
- Size varies significantly
- You don't need random access

**Use Hash Tables when:**
- You need fast lookups by key
- Key-value relationships
- Caching and memoization

**Use Trees when:**
- You need sorted/ordered data
- Range queries are common
- Hierarchical relationships exist

---

**Master the fundamentals, build efficiently! 🏗️**
