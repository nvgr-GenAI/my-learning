# Linked Lists: Fundamentals & Operations

## 📚 What are Linked Lists?

A **Linked List** is a linear data structure where elements are not stored in contiguous memory locations. Instead, each element (called a **node**) contains data and a reference (or pointer) to the next node in the sequence. This allows for dynamic memory allocation and efficient insertion/deletion operations.

## 🔗 Types of Linked Lists

### 1. Singly Linked List

Each node points to the next node, and the last node points to `null`.

```text
[Data|Next] -> [Data|Next] -> [Data|Next] -> null
```

**Structure:**

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __str__(self):
        return f"Node({self.val})"
```

### 2. Doubly Linked List

Each node has pointers to both the next and previous nodes.

```text
null <- [Prev|Data|Next] <-> [Prev|Data|Next] <-> [Prev|Data|Next] -> null
```

**Structure:**

```python
class DoublyListNode:
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next
```

### 3. Circular Linked List

The last node points back to the first node, forming a circle.

```text
[Data|Next] -> [Data|Next] -> [Data|Next] 
      ^                              |
      |______________________________|
```

## 🏗️ Basic Operations

### 1. Traversal

**Purpose:** Visit each node in the list.

```python
def traverse(head):
    """Print all values in the linked list."""
    current = head
    while current:
        print(current.val, end=" -> " if current.next else " -> null\n")
        current = current.next

# Time: O(n), Space: O(1)
```

### 2. Insertion

#### Insert at Beginning
```python
def insert_at_beginning(head, val):
    """Insert a new node at the beginning of the list."""
    new_node = ListNode(val)
    new_node.next = head
    return new_node  # New head

# Time: O(1), Space: O(1)
```

#### Insert at End
```python
def insert_at_end(head, val):
    """Insert a new node at the end of the list."""
    new_node = ListNode(val)
    
    if not head:
        return new_node
    
    current = head
    while current.next:
        current = current.next
    
    current.next = new_node
    return head

# Time: O(n), Space: O(1)
```

#### Insert at Position
```python
def insert_at_position(head, val, pos):
    """Insert a new node at the given position (0-indexed)."""
    if pos == 0:
        return insert_at_beginning(head, val)
    
    new_node = ListNode(val)
    current = head
    
    # Traverse to position-1
    for i in range(pos - 1):
        if not current:
            raise IndexError("Position out of bounds")
        current = current.next
    
    new_node.next = current.next
    current.next = new_node
    return head

# Time: O(n), Space: O(1)
```

### 3. Deletion

#### Delete by Value
```python
def delete_by_value(head, val):
    """Delete the first node with the given value."""
    if not head:
        return None
    
    # If head needs to be deleted
    if head.val == val:
        return head.next
    
    current = head
    while current.next:
        if current.next.val == val:
            current.next = current.next.next
            return head
        current = current.next
    
    return head  # Value not found

# Time: O(n), Space: O(1)
```

#### Delete at Position
```python
def delete_at_position(head, pos):
    """Delete node at the given position (0-indexed)."""
    if not head:
        return None
    
    # Delete head
    if pos == 0:
        return head.next
    
    current = head
    # Traverse to position-1
    for i in range(pos - 1):
        if not current or not current.next:
            raise IndexError("Position out of bounds")
        current = current.next
    
    current.next = current.next.next
    return head

# Time: O(n), Space: O(1)
```

### 4. Search

```python
def search(head, val):
    """Search for a value in the linked list."""
    current = head
    position = 0
    
    while current:
        if current.val == val:
            return position
        current = current.next
        position += 1
    
    return -1  # Not found

# Time: O(n), Space: O(1)
```

### 5. Length

```python
def get_length(head):
    """Get the length of the linked list."""
    length = 0
    current = head
    
    while current:
        length += 1
        current = current.next
    
    return length

# Time: O(n), Space: O(1)
```

## 📊 Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| **Access** | O(n) | O(1) |
| **Search** | O(n) | O(1) |
| **Insert at beginning** | O(1) | O(1) |
| **Insert at end** | O(n) | O(1) |
| **Insert at position** | O(n) | O(1) |
| **Delete at beginning** | O(1) | O(1) |
| **Delete at end** | O(n) | O(1) |
| **Delete at position** | O(n) | O(1) |

## 🎯 Complete Implementation

Here's a complete linked list class with all operations:

```python
class LinkedList:
    def __init__(self):
        self.head = None
        self.size = 0
    
    def append(self, val):
        """Add element to the end."""
        if not self.head:
            self.head = ListNode(val)
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = ListNode(val)
        self.size += 1
    
    def prepend(self, val):
        """Add element to the beginning."""
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
        self.size += 1
    
    def delete(self, val):
        """Delete first occurrence of value."""
        if not self.head:
            return False
        
        if self.head.val == val:
            self.head = self.head.next
            self.size -= 1
            return True
        
        current = self.head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
                self.size -= 1
                return True
            current = current.next
        
        return False
    
    def find(self, val):
        """Find index of value."""
        current = self.head
        index = 0
        
        while current:
            if current.val == val:
                return index
            current = current.next
            index += 1
        
        return -1
    
    def get(self, index):
        """Get value at index."""
        if index < 0 or index >= self.size:
            raise IndexError("Index out of bounds")
        
        current = self.head
        for _ in range(index):
            current = current.next
        
        return current.val
    
    def is_empty(self):
        """Check if list is empty."""
        return self.head is None
    
    def __len__(self):
        """Get size of list."""
        return self.size
    
    def __str__(self):
        """String representation."""
        if not self.head:
            return "[]"
        
        result = []
        current = self.head
        while current:
            result.append(str(current.val))
            current = current.next
        
        return " -> ".join(result) + " -> null"

# Usage example
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.prepend(0)
print(ll)  # 0 -> 1 -> 2 -> null
print(f"Length: {len(ll)}")  # Length: 3
print(f"Find 1: {ll.find(1)}")  # Find 1: 1
ll.delete(1)
print(ll)  # 0 -> 2 -> null
```

## ⚖️ Linked Lists vs Arrays

| **Feature** | **Linked List** | **Array** |
|------------|-----------------|-----------|
| **Memory Layout** | Scattered (non-contiguous) | Contiguous |
| **Access Time** | O(n) - Sequential | O(1) - Random |
| **Insert/Delete** | O(1) at known position | O(n) - Need shifting |
| **Memory Overhead** | Extra pointer storage | Minimal overhead |
| **Cache Performance** | Poor (scattered memory) | Excellent (locality) |
| **Memory Allocation** | Dynamic | Static/Dynamic |
| **Memory Usage** | Higher (pointers) | Lower |

## 🎯 When to Use Linked Lists

### ✅ Use Linked Lists When

- **Frequent insertions/deletions** at the beginning
- **Unknown or highly variable size** requirements
- **Memory is scattered** or fragmented
- **Implementing other data structures** (stacks, queues)
- **Sequential access** is sufficient

### ❌ Avoid Linked Lists When

- **Random access** to elements is needed
- **Memory usage** is critical
- **Cache performance** is important
- Working with **small, fixed-size** collections
- **Binary search** or similar algorithms are needed

## 🔧 Common Techniques

### 1. Dummy Node

Use a dummy node to simplify edge cases:

```python
def remove_elements(head, val):
    """Remove all nodes with given value using dummy node."""
    dummy = ListNode(0)
    dummy.next = head
    current = dummy
    
    while current.next:
        if current.next.val == val:
            current.next = current.next.next
        else:
            current = current.next
    
    return dummy.next
```

### 2. Two Pointers (Fast & Slow)

Useful for finding middle, detecting cycles:

```python
def find_middle(head):
    """Find middle node using two pointers."""
    if not head:
        return None
    
    slow = fast = head
    
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow
```

### 3. Reversing Links

```python
def reverse_list(head):
    """Reverse a linked list iteratively."""
    prev = None
    current = head
    
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    
    return prev
```

## 💡 Pro Tips

!!! tip "Memory Management"
    In languages like C++, always `delete` nodes to avoid memory leaks. Python and Java handle this automatically with garbage collection.

!!! warning "Common Pitfalls"
    - **Null pointer exceptions**: Always check if nodes exist before accessing
    - **Losing references**: Keep track of nodes during modifications
    - **Off-by-one errors**: Be careful with position-based operations

!!! success "Best Practices"
    - Use dummy nodes to handle edge cases elegantly
    - Draw diagrams when solving complex problems
    - Practice pointer manipulation thoroughly
    - Test with empty lists and single-node lists

## 🚀 Next Steps

Now that you understand the fundamentals, practice with:

- **[Easy Problems](easy-problems.md)** - Build confidence with basic operations
- **[Medium Problems](medium-problems.md)** - Learn advanced techniques
- **[Hard Problems](hard-problems.md)** - Master complex scenarios

---

*Ready to start practicing? Begin with the [Easy Problems](easy-problems.md) section!*
