# Circular Linked List 🔗🔄

## 🎯 Overview

A **Circular Linked List** is a variation of linked list where the last node points back to the first node, forming a circle. This creates a continuous loop where you can traverse the entire list starting from any node.

## 🔄 Types of Circular Linked Lists

### 1. Circular Singly Linked List

- Each node has one pointer to the next node
- Last node points to the first node

### 2. Circular Doubly Linked List

- Each node has two pointers (next and previous)
- Last node's next points to first node
- First node's previous points to last node

## 🏗️ Structure

```text
Circular Singly Linked List:
┌─────┐    ┌─────┐    ┌─────┐
│  A  │───→│  B  │───→│  C  │
└─────┘    └─────┘    └─────┘
   ↑________________________│
   └─────────────────────────┘

Circular Doubly Linked List:
┌─────┐ ↔ ┌─────┐ ↔ ┌─────┐
│  A  │   │  B  │   │  C  │
└─────┘   └─────┘   └─────┘
   ↑_____________________│
   └──────────────────────┘
```

## 💻 Implementation

### Circular Singly Linked List

```python
class CircularNode:
    def __init__(self, data=0):
        self.data = data
        self.next = None
    
    def __str__(self):
        return str(self.data)

class CircularSinglyLinkedList:
    def __init__(self):
        self.head = None
        self.size = 0
    
    def is_empty(self):
        """Check if list is empty"""
        return self.head is None
    
    def get_size(self):
        """Get the size of the list"""
        return self.size
    
    def insert_at_beginning(self, data):
        """Insert node at the beginning"""
        new_node = CircularNode(data)
        
        if self.is_empty():
            new_node.next = new_node  # Point to itself
            self.head = new_node
        else:
            # Find the last node
            current = self.head
            while current.next != self.head:
                current = current.next
            
            # Insert new node
            new_node.next = self.head
            current.next = new_node
            self.head = new_node
        
        self.size += 1
    
    def insert_at_end(self, data):
        """Insert node at the end"""
        new_node = CircularNode(data)
        
        if self.is_empty():
            new_node.next = new_node
            self.head = new_node
        else:
            # Find the last node
            current = self.head
            while current.next != self.head:
                current = current.next
            
            # Insert new node
            new_node.next = self.head
            current.next = new_node
        
        self.size += 1
    
    def insert_at_position(self, data, position):
        """Insert node at specific position"""
        if position < 0 or position > self.size:
            raise IndexError("Position out of bounds")
        
        if position == 0:
            self.insert_at_beginning(data)
            return
        
        new_node = CircularNode(data)
        current = self.head
        
        # Navigate to position - 1
        for i in range(position - 1):
            current = current.next
        
        # Insert new node
        new_node.next = current.next
        current.next = new_node
        
        self.size += 1
    
    def delete_from_beginning(self):
        """Delete node from beginning"""
        if self.is_empty():
            raise IndexError("Delete from empty list")
        
        deleted_data = self.head.data
        
        if self.size == 1:
            self.head = None
        else:
            # Find the last node
            current = self.head
            while current.next != self.head:
                current = current.next
            
            # Update pointers
            current.next = self.head.next
            self.head = self.head.next
        
        self.size -= 1
        return deleted_data
    
    def delete_from_end(self):
        """Delete node from end"""
        if self.is_empty():
            raise IndexError("Delete from empty list")
        
        if self.size == 1:
            deleted_data = self.head.data
            self.head = None
            self.size -= 1
            return deleted_data
        
        # Find the second last node
        current = self.head
        while current.next.next != self.head:
            current = current.next
        
        deleted_data = current.next.data
        current.next = self.head
        
        self.size -= 1
        return deleted_data
    
    def delete_by_value(self, data):
        """Delete first node with given value"""
        if self.is_empty():
            raise ValueError("List is empty")
        
        # If head contains the data
        if self.head.data == data:
            return self.delete_from_beginning()
        
        current = self.head
        while current.next != self.head:
            if current.next.data == data:
                deleted_data = current.next.data
                current.next = current.next.next
                self.size -= 1
                return deleted_data
            current = current.next
        
        raise ValueError(f"Value {data} not found")
    
    def search(self, data):
        """Search for a value and return its position"""
        if self.is_empty():
            return -1
        
        current = self.head
        position = 0
        
        while True:
            if current.data == data:
                return position
            current = current.next
            position += 1
            
            if current == self.head:  # Completed one full circle
                break
        
        return -1
    
    def get_at_position(self, position):
        """Get data at specific position"""
        if position < 0 or position >= self.size:
            raise IndexError("Position out of bounds")
        
        current = self.head
        for i in range(position):
            current = current.next
        
        return current.data
    
    def display(self, max_rounds=1):
        """Display the circular list"""
        if self.is_empty():
            return "Empty list"
        
        result = []
        current = self.head
        count = 0
        
        while count < self.size * max_rounds:
            result.append(str(current.data))
            current = current.next
            count += 1
            
            if count < self.size * max_rounds:
                result.append(" → ")
        
        return "".join(result) + " → (circular)"
    
    def to_list(self):
        """Convert to Python list"""
        if self.is_empty():
            return []
        
        result = []
        current = self.head
        
        while True:
            result.append(current.data)
            current = current.next
            if current == self.head:
                break
        
        return result
    
    def __str__(self):
        return self.display()
    
    def __len__(self):
        return self.size
```

### Circular Doubly Linked List

```python
class CircularDoublyNode:
    def __init__(self, data=0):
        self.data = data
        self.next = None
        self.prev = None
    
    def __str__(self):
        return str(self.data)

class CircularDoublyLinkedList:
    def __init__(self):
        self.head = None
        self.size = 0
    
    def is_empty(self):
        """Check if list is empty"""
        return self.head is None
    
    def get_size(self):
        """Get the size of the list"""
        return self.size
    
    def insert_at_beginning(self, data):
        """Insert node at the beginning"""
        new_node = CircularDoublyNode(data)
        
        if self.is_empty():
            new_node.next = new_node
            new_node.prev = new_node
            self.head = new_node
        else:
            tail = self.head.prev
            
            new_node.next = self.head
            new_node.prev = tail
            self.head.prev = new_node
            tail.next = new_node
            self.head = new_node
        
        self.size += 1
    
    def insert_at_end(self, data):
        """Insert node at the end"""
        new_node = CircularDoublyNode(data)
        
        if self.is_empty():
            new_node.next = new_node
            new_node.prev = new_node
            self.head = new_node
        else:
            tail = self.head.prev
            
            new_node.next = self.head
            new_node.prev = tail
            tail.next = new_node
            self.head.prev = new_node
        
        self.size += 1
    
    def delete_from_beginning(self):
        """Delete node from beginning"""
        if self.is_empty():
            raise IndexError("Delete from empty list")
        
        deleted_data = self.head.data
        
        if self.size == 1:
            self.head = None
        else:
            tail = self.head.prev
            new_head = self.head.next
            
            tail.next = new_head
            new_head.prev = tail
            self.head = new_head
        
        self.size -= 1
        return deleted_data
    
    def delete_from_end(self):
        """Delete node from end"""
        if self.is_empty():
            raise IndexError("Delete from empty list")
        
        if self.size == 1:
            deleted_data = self.head.data
            self.head = None
            self.size -= 1
            return deleted_data
        
        tail = self.head.prev
        deleted_data = tail.data
        new_tail = tail.prev
        
        new_tail.next = self.head
        self.head.prev = new_tail
        
        self.size -= 1
        return deleted_data
    
    def display_forward(self):
        """Display list forward"""
        if self.is_empty():
            return "Empty list"
        
        result = []
        current = self.head
        
        while True:
            result.append(str(current.data))
            current = current.next
            if current == self.head:
                break
        
        return " ↔ ".join(result) + " ↔ (circular)"
    
    def display_backward(self):
        """Display list backward"""
        if self.is_empty():
            return "Empty list"
        
        result = []
        current = self.head.prev  # Start from tail
        
        while True:
            result.append(str(current.data))
            current = current.prev
            if current == self.head.prev:
                break
        
        return " ↔ ".join(result) + " ↔ (circular)"
    
    def __str__(self):
        return self.display_forward()
```

## 🧪 Usage Examples

```python
# Circular Singly Linked List
csll = CircularSinglyLinkedList()

# Insert elements
csll.insert_at_end(1)
csll.insert_at_end(2)
csll.insert_at_end(3)
csll.insert_at_beginning(0)
print(csll)  # 0 → 1 → 2 → 3 → (circular)

# Search and access
print(f"Search for 2: position {csll.search(2)}")  # 2
print(f"Element at position 1: {csll.get_at_position(1)}")  # 1

# Delete elements
csll.delete_from_beginning()
csll.delete_by_value(2)
print(csll)  # 1 → 3 → (circular)

# Circular Doubly Linked List
cdll = CircularDoublyLinkedList()
cdll.insert_at_end(1)
cdll.insert_at_end(2)
cdll.insert_at_end(3)
print(f"Forward: {cdll.display_forward()}")
print(f"Backward: {cdll.display_backward()}")
```

## ⚡ Advantages

1. **Continuous Traversal**: Can traverse infinitely without null checks
2. **Efficient Round-Robin**: Perfect for round-robin scheduling
3. **Memory Efficient**: No null pointers
4. **Symmetric Operations**: Insert/delete at both ends are similar

## ⚠️ Disadvantages

1. **Infinite Loops**: Risk of infinite loops if not handled properly
2. **Complex Implementation**: More complex than linear linked lists
3. **Debugging Difficulty**: Harder to debug circular references

## 📊 Time Complexity

| **Operation** | **Singly Circular** | **Doubly Circular** |
|---------------|-------------------|-------------------|
| **Access** | O(n) | O(n) |
| **Search** | O(n) | O(n) |
| **Insert at Beginning** | O(n) | O(1) |
| **Insert at End** | O(n) | O(1) |
| **Delete from Beginning** | O(n) | O(1) |
| **Delete from End** | O(n) | O(1) |

## 🎯 Common Use Cases

### 1. Round-Robin Scheduling

```python
class RoundRobinScheduler:
    def __init__(self):
        self.processes = CircularSinglyLinkedList()
        self.current = None
    
    def add_process(self, process_id):
        """Add a process to the scheduler"""
        self.processes.insert_at_end(process_id)
        if self.current is None:
            self.current = self.processes.head
    
    def next_process(self):
        """Get next process in round-robin fashion"""
        if self.current:
            process = self.current.data
            self.current = self.current.next
            return process
        return None
    
    def remove_process(self, process_id):
        """Remove a process from scheduler"""
        try:
            # Update current pointer if needed
            if self.current and self.current.data == process_id:
                self.current = self.current.next if self.processes.size > 1 else None
            
            self.processes.delete_by_value(process_id)
        except ValueError:
            pass  # Process not found
```

### 2. Circular Buffer

```python
class CircularBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = CircularSinglyLinkedList()
        self.current_size = 0
    
    def add(self, data):
        """Add data to circular buffer"""
        if self.current_size < self.capacity:
            self.buffer.insert_at_end(data)
            self.current_size += 1
        else:
            # Overwrite oldest data
            self.buffer.delete_from_beginning()
            self.buffer.insert_at_end(data)
    
    def get_all(self):
        """Get all data in buffer"""
        return self.buffer.to_list()
```

### 3. Josephus Problem

```python
def josephus_problem(n, k):
    """
    Solve Josephus problem using circular linked list
    n: number of people
    k: every k-th person is eliminated
    """
    # Create circular list with n people
    people = CircularSinglyLinkedList()
    for i in range(1, n + 1):
        people.insert_at_end(i)
    
    current = people.head
    
    # Eliminate people until only one remains
    while people.size > 1:
        # Move k-1 steps
        for _ in range(k - 1):
            current = current.next
        
        # Eliminate current person
        next_person = current.next
        people.delete_by_value(current.data)
        current = next_person
    
    return people.head.data  # Last person standing
```

## 🔄 Circular vs Linear Comparison

| **Feature** | **Linear Linked List** | **Circular Linked List** |
|-------------|----------------------|-------------------------|
| **Last Node** | Points to NULL | Points to first node |
| **Traversal** | Stops at NULL | Can continue infinitely |
| **Memory** | Slightly less | Slightly more (no NULL) |
| **Use Cases** | General purpose | Cyclic applications |
| **Complexity** | Simpler | More complex |

## 🎓 Interview Problems

### Problem 1: Detect if Linked List is Circular

```python
def is_circular(head):
    """Check if linked list is circular using Floyd's cycle detection"""
    if not head:
        return False
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    
    return False
```

### Problem 2: Find Length of Circular Linked List

```python
def circular_list_length(head):
    """Find length of circular linked list"""
    if not head:
        return 0
    
    count = 1
    current = head.next
    
    while current != head:
        count += 1
        current = current.next
    
    return count
```

### Problem 3: Split Circular Linked List

```python
def split_circular_list(head):
    """Split circular linked list into two halves"""
    if not head or head.next == head:
        return head, None
    
    slow = fast = head
    
    # Find middle using Floyd's algorithm
    while fast.next != head and fast.next.next != head:
        slow = slow.next
        fast = fast.next.next
    
    # Split the list
    head2 = slow.next
    slow.next = head
    
    # Find the end of second half
    current = head2
    while current.next != head:
        current = current.next
    current.next = head2
    
    return head, head2
```

## 🎯 Key Takeaways

1. **Circular Nature**: Last node connects to first, forming a circle
2. **Infinite Traversal**: Can traverse continuously without null checks
3. **Round-Robin Applications**: Perfect for cyclic scheduling
4. **Loop Detection**: Always check for termination conditions
5. **Memory Efficiency**: No null pointers, but risk of infinite loops
6. **Complexity Trade-off**: More complex implementation for specific use cases

Circular linked lists are specialized data structures perfect for applications requiring cyclic behavior and continuous traversal!
