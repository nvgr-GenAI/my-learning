# Singly Linked Lists 🔗

## 🎯 Overview

A **Singly Linked List** is the simplest form of linked list where each node points to the next node in the sequence. The last node points to `null`, indicating the end of the list.

## 🏗️ Structure

```text
[Data|Next] -> [Data|Next] -> [Data|Next] -> null
     Head                           Tail
```

### Node Definition

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __str__(self):
        return f"Node({self.val})"

class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.size = 0
    
    def __len__(self):
        return self.size
    
    def is_empty(self):
        return self.head is None
```

## 🔧 Core Operations

### 1. Traversal

```python
def traverse(self):
    """Print all values in the linked list."""
    current = self.head
    values = []
    
    while current:
        values.append(str(current.val))
        current = current.next
    
    return " -> ".join(values) + " -> null"

def traverse_with_callback(self, callback):
    """Apply callback to each node."""
    current = self.head
    while current:
        callback(current.val)
        current = current.next

# Time: O(n), Space: O(1)
```

### 2. Insertion Operations

#### Insert at Beginning

```python
def insert_at_beginning(self, val):
    """Insert a new node at the beginning of the list."""
    new_node = ListNode(val)
    new_node.next = self.head
    self.head = new_node
    self.size += 1

# Time: O(1), Space: O(1)
```

#### Insert at End

```python
def insert_at_end(self, val):
    """Insert a new node at the end of the list."""
    new_node = ListNode(val)
    
    if not self.head:
        self.head = new_node
    else:
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    self.size += 1

# Time: O(n), Space: O(1)
```

#### Insert at Position

```python
def insert_at_position(self, pos, val):
    """Insert a new node at the specified position."""
    if pos < 0 or pos > self.size:
        raise IndexError("Position out of bounds")
    
    if pos == 0:
        self.insert_at_beginning(val)
        return
    
    new_node = ListNode(val)
    current = self.head
    
    # Traverse to position - 1
    for _ in range(pos - 1):
        current = current.next
    
    new_node.next = current.next
    current.next = new_node
    self.size += 1

# Time: O(n), Space: O(1)
```

### 3. Deletion Operations

#### Delete at Beginning

```python
def delete_at_beginning(self):
    """Delete the first node."""
    if not self.head:
        raise IndexError("Cannot delete from empty list")
    
    deleted_val = self.head.val
    self.head = self.head.next
    self.size -= 1
    return deleted_val

# Time: O(1), Space: O(1)
```

#### Delete at End

```python
def delete_at_end(self):
    """Delete the last node."""
    if not self.head:
        raise IndexError("Cannot delete from empty list")
    
    if not self.head.next:
        deleted_val = self.head.val
        self.head = None
        self.size -= 1
        return deleted_val
    
    current = self.head
    while current.next.next:
        current = current.next
    
    deleted_val = current.next.val
    current.next = None
    self.size -= 1
    return deleted_val

# Time: O(n), Space: O(1)
```

#### Delete by Value

```python
def delete_by_value(self, val):
    """Delete the first node with the specified value."""
    if not self.head:
        return False
    
    # If head needs to be deleted
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

# Time: O(n), Space: O(1)
```

### 4. Search Operations

```python
def search(self, val):
    """Search for a value and return its position."""
    current = self.head
    position = 0
    
    while current:
        if current.val == val:
            return position
        current = current.next
        position += 1
    
    return -1

def get_at_position(self, pos):
    """Get value at specified position."""
    if pos < 0 or pos >= self.size:
        raise IndexError("Position out of bounds")
    
    current = self.head
    for _ in range(pos):
        current = current.next
    
    return current.val

# Time: O(n), Space: O(1)
```

## 🎯 Complete Implementation

```python
class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.size = 0
    
    # Insert operations
    def insert_at_beginning(self, val):
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
        self.size += 1
    
    def insert_at_end(self, val):
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self.size += 1
    
    def insert_at_position(self, pos, val):
        if pos < 0 or pos > self.size:
            raise IndexError("Position out of bounds")
        
        if pos == 0:
            self.insert_at_beginning(val)
            return
        
        new_node = ListNode(val)
        current = self.head
        for _ in range(pos - 1):
            current = current.next
        
        new_node.next = current.next
        current.next = new_node
        self.size += 1
    
    # Delete operations
    def delete_at_beginning(self):
        if not self.head:
            raise IndexError("Cannot delete from empty list")
        
        deleted_val = self.head.val
        self.head = self.head.next
        self.size -= 1
        return deleted_val
    
    def delete_at_end(self):
        if not self.head:
            raise IndexError("Cannot delete from empty list")
        
        if not self.head.next:
            deleted_val = self.head.val
            self.head = None
            self.size -= 1
            return deleted_val
        
        current = self.head
        while current.next.next:
            current = current.next
        
        deleted_val = current.next.val
        current.next = None
        self.size -= 1
        return deleted_val
    
    def delete_by_value(self, val):
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
    
    # Search operations
    def search(self, val):
        current = self.head
        position = 0
        while current:
            if current.val == val:
                return position
            current = current.next
            position += 1
        return -1
    
    def get_at_position(self, pos):
        if pos < 0 or pos >= self.size:
            raise IndexError("Position out of bounds")
        
        current = self.head
        for _ in range(pos):
            current = current.next
        return current.val
    
    # Utility methods
    def __len__(self):
        return self.size
    
    def is_empty(self):
        return self.head is None
    
    def __str__(self):
        if not self.head:
            return "Empty list"
        
        current = self.head
        values = []
        while current:
            values.append(str(current.val))
            current = current.next
        
        return " -> ".join(values) + " -> null"

# Example usage
if __name__ == "__main__":
    sll = SinglyLinkedList()
    
    # Test insertions
    sll.insert_at_beginning(1)
    sll.insert_at_end(2)
    sll.insert_at_end(3)
    sll.insert_at_position(1, 5)
    print(sll)  # 1 -> 5 -> 2 -> 3 -> null
    
    # Test searches
    print(f"Search for 5: {sll.search(5)}")  # 1
    print(f"Get at position 2: {sll.get_at_position(2)}")  # 2
    
    # Test deletions
    sll.delete_at_beginning()
    print(sll)  # 5 -> 2 -> 3 -> null
    
    sll.delete_by_value(2)
    print(sll)  # 5 -> 3 -> null
```

## 📊 Complexity Analysis

| **Operation** | **Time Complexity** | **Space Complexity** |
|---------------|-------------------|---------------------|
| **Insert at Beginning** | O(1) | O(1) |
| **Insert at End** | O(n) | O(1) |
| **Insert at Position** | O(n) | O(1) |
| **Delete at Beginning** | O(1) | O(1) |
| **Delete at End** | O(n) | O(1) |
| **Delete by Value** | O(n) | O(1) |
| **Search** | O(n) | O(1) |
| **Access by Position** | O(n) | O(1) |
| **Traversal** | O(n) | O(1) |

## ⚡ Advantages & Disadvantages

### ✅ Advantages

1. **Dynamic Size**: Can grow and shrink during runtime
2. **Memory Efficient**: Only allocates memory as needed
3. **Fast Insertion/Deletion**: O(1) at the beginning
4. **No Memory Waste**: No unused allocated memory

### ❌ Disadvantages

1. **No Random Access**: Must traverse to reach elements
2. **Extra Memory**: Each node requires additional pointer storage
3. **Poor Cache Performance**: Nodes may not be contiguous in memory
4. **No Reverse Traversal**: Can only move forward

## 🎯 When to Use Singly Linked Lists

### ✅ Use When

- **Frequent insertions/deletions** at the beginning
- **Unknown or variable size** of data
- **Memory is limited** and you need precise allocation
- **Don't need random access** to elements

### ❌ Avoid When

- **Need random access** to elements by index
- **Frequent searches** are required
- **Cache performance** is critical
- **Need to traverse backwards**

## 🔗 Related Concepts

- **Doubly Linked Lists**: For bidirectional traversal
- **Circular Linked Lists**: For cyclic data structures
- **Dynamic Arrays**: Alternative for variable-size collections
- **Stacks/Queues**: Can be implemented using singly linked lists
