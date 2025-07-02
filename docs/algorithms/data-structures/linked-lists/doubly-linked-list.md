# Doubly Linked List 🔗↔️

## 🎯 Overview

A **Doubly Linked List** is a linear data structure where each node contains data and two pointers: one pointing to the next node and another pointing to the previous node. This bidirectional linking allows efficient traversal in both directions.

## 🏗️ Structure

```text
NULL ← [prev|data|next] ↔ [prev|data|next] ↔ [prev|data|next] → NULL
       └─────head─────┘    └─────────────┘    └─────tail─────┘
```

## 💻 Implementation

### Node Structure

```python
class DoublyListNode:
    def __init__(self, data=0):
        self.data = data
        self.next = None
        self.prev = None
    
    def __str__(self):
        return str(self.data)
```

### Doubly Linked List Class

```python
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def is_empty(self):
        """Check if list is empty"""
        return self.head is None
    
    def get_size(self):
        """Get the size of the list"""
        return self.size
    
    def insert_at_beginning(self, data):
        """Insert node at the beginning"""
        new_node = DoublyListNode(data)
        
        if self.is_empty():
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        
        self.size += 1
    
    def insert_at_end(self, data):
        """Insert node at the end"""
        new_node = DoublyListNode(data)
        
        if self.is_empty():
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        
        self.size += 1
    
    def insert_at_position(self, data, position):
        """Insert node at specific position (0-indexed)"""
        if position < 0 or position > self.size:
            raise IndexError("Position out of bounds")
        
        if position == 0:
            self.insert_at_beginning(data)
            return
        
        if position == self.size:
            self.insert_at_end(data)
            return
        
        new_node = DoublyListNode(data)
        current = self.head
        
        # Navigate to position
        for i in range(position):
            current = current.next
        
        # Insert new node
        new_node.next = current
        new_node.prev = current.prev
        current.prev.next = new_node
        current.prev = new_node
        
        self.size += 1
    
    def delete_from_beginning(self):
        """Delete node from beginning"""
        if self.is_empty():
            raise IndexError("Delete from empty list")
        
        deleted_data = self.head.data
        
        if self.size == 1:
            self.head = self.tail = None
        else:
            self.head = self.head.next
            self.head.prev = None
        
        self.size -= 1
        return deleted_data
    
    def delete_from_end(self):
        """Delete node from end"""
        if self.is_empty():
            raise IndexError("Delete from empty list")
        
        deleted_data = self.tail.data
        
        if self.size == 1:
            self.head = self.tail = None
        else:
            self.tail = self.tail.prev
            self.tail.next = None
        
        self.size -= 1
        return deleted_data
    
    def delete_at_position(self, position):
        """Delete node at specific position"""
        if position < 0 or position >= self.size:
            raise IndexError("Position out of bounds")
        
        if position == 0:
            return self.delete_from_beginning()
        
        if position == self.size - 1:
            return self.delete_from_end()
        
        current = self.head
        
        # Navigate to position
        for i in range(position):
            current = current.next
        
        # Delete current node
        current.prev.next = current.next
        current.next.prev = current.prev
        
        self.size -= 1
        return current.data
    
    def delete_by_value(self, data):
        """Delete first node with given value"""
        current = self.head
        
        while current:
            if current.data == data:
                if current == self.head:
                    return self.delete_from_beginning()
                elif current == self.tail:
                    return self.delete_from_end()
                else:
                    current.prev.next = current.next
                    current.next.prev = current.prev
                    self.size -= 1
                    return data
            current = current.next
        
        raise ValueError(f"Value {data} not found")
    
    def search(self, data):
        """Search for a value and return its position"""
        current = self.head
        position = 0
        
        while current:
            if current.data == data:
                return position
            current = current.next
            position += 1
        
        return -1
    
    def get_at_position(self, position):
        """Get data at specific position"""
        if position < 0 or position >= self.size:
            raise IndexError("Position out of bounds")
        
        # Optimize: start from head or tail based on position
        if position <= self.size // 2:
            # Start from head
            current = self.head
            for i in range(position):
                current = current.next
        else:
            # Start from tail
            current = self.tail
            for i in range(self.size - 1 - position):
                current = current.prev
        
        return current.data
    
    def reverse(self):
        """Reverse the doubly linked list"""
        if self.is_empty() or self.size == 1:
            return
        
        current = self.head
        
        # Swap next and prev for all nodes
        while current:
            current.next, current.prev = current.prev, current.next
            current = current.prev  # Move to next node (which is now prev)
        
        # Swap head and tail
        self.head, self.tail = self.tail, self.head
    
    def display_forward(self):
        """Display list from head to tail"""
        if self.is_empty():
            return "Empty list"
        
        result = []
        current = self.head
        while current:
            result.append(str(current.data))
            current = current.next
        
        return " ↔ ".join(result)
    
    def display_backward(self):
        """Display list from tail to head"""
        if self.is_empty():
            return "Empty list"
        
        result = []
        current = self.tail
        while current:
            result.append(str(current.data))
            current = current.prev
        
        return " ↔ ".join(result)
    
    def to_list(self):
        """Convert to Python list"""
        result = []
        current = self.head
        while current:
            result.append(current.data)
            current = current.next
        return result
    
    def __str__(self):
        return self.display_forward()
    
    def __len__(self):
        return self.size
```

## 🧪 Usage Examples

```python
# Create doubly linked list
dll = DoublyLinkedList()

# Insert elements
dll.insert_at_end(1)
dll.insert_at_end(2)
dll.insert_at_end(3)
dll.insert_at_beginning(0)
print(dll)  # 0 ↔ 1 ↔ 2 ↔ 3

# Insert at specific position
dll.insert_at_position(1.5, 2)
print(dll)  # 0 ↔ 1 ↔ 1.5 ↔ 2 ↔ 3

# Access elements
print(f"Element at position 2: {dll.get_at_position(2)}")  # 1.5
print(f"Search for 2: position {dll.search(2)}")  # 3

# Display in both directions
print(f"Forward: {dll.display_forward()}")
print(f"Backward: {dll.display_backward()}")

# Delete elements
dll.delete_from_beginning()
dll.delete_from_end()
dll.delete_by_value(1.5)
print(dll)  # 1 ↔ 2

# Reverse the list
dll.reverse()
print(dll)  # 2 ↔ 1
```

## ⚡ Advantages

1. **Bidirectional Traversal**: Can traverse in both directions
2. **Efficient Deletion**: O(1) deletion when node reference is available
3. **Flexible Insertion**: Easy insertion at any position
4. **Better Navigation**: Can move backward from any node

## ⚠️ Disadvantages

1. **Extra Memory**: Additional pointer per node
2. **Complex Implementation**: More complex than singly linked list
3. **Cache Performance**: Poor cache locality due to non-contiguous memory

## 📊 Time Complexity

| **Operation** | **Time Complexity** | **Space Complexity** |
|---------------|-------------------|---------------------|
| **Access** | O(n) | O(1) |
| **Search** | O(n) | O(1) |
| **Insert at Beginning** | O(1) | O(1) |
| **Insert at End** | O(1) | O(1) |
| **Insert at Position** | O(n) | O(1) |
| **Delete from Beginning** | O(1) | O(1) |
| **Delete from End** | O(1) | O(1) |
| **Delete at Position** | O(n) | O(1) |
| **Delete by Value** | O(n) | O(1) |

## 🎯 Common Use Cases

### 1. Browser History

```python
class BrowserHistory:
    def __init__(self):
        self.dll = DoublyLinkedList()
        self.current_pos = -1
    
    def visit(self, url):
        """Visit a new URL"""
        # Remove forward history
        while self.current_pos < self.dll.size - 1:
            self.dll.delete_from_end()
        
        # Add new URL
        self.dll.insert_at_end(url)
        self.current_pos = self.dll.size - 1
    
    def back(self):
        """Go back in history"""
        if self.current_pos > 0:
            self.current_pos -= 1
            return self.dll.get_at_position(self.current_pos)
        return None
    
    def forward(self):
        """Go forward in history"""
        if self.current_pos < self.dll.size - 1:
            self.current_pos += 1
            return self.dll.get_at_position(self.current_pos)
        return None
```

### 2. Music Playlist

```python
class MusicPlaylist:
    def __init__(self):
        self.dll = DoublyLinkedList()
        self.current = None
    
    def add_song(self, song):
        """Add song to playlist"""
        self.dll.insert_at_end(song)
        if self.current is None:
            self.current = self.dll.head
    
    def next_song(self):
        """Play next song"""
        if self.current and self.current.next:
            self.current = self.current.next
        return self.current.data if self.current else None
    
    def previous_song(self):
        """Play previous song"""
        if self.current and self.current.prev:
            self.current = self.current.prev
        return self.current.data if self.current else None
```

## 🔄 Comparison with Singly Linked List

| **Feature** | **Singly Linked** | **Doubly Linked** |
|-------------|------------------|------------------|
| **Memory per Node** | 1 pointer | 2 pointers |
| **Backward Traversal** | No | Yes |
| **Deletion Complexity** | O(n) | O(1) with node reference |
| **Implementation** | Simpler | More complex |
| **Use Cases** | Forward-only operations | Bidirectional operations |

## 🎓 Interview Problems

### Problem 1: Remove Duplicates from Sorted Doubly Linked List

```python
def remove_duplicates_sorted_dll(head):
    """Remove duplicates from sorted doubly linked list"""
    if not head:
        return head
    
    current = head
    
    while current.next:
        if current.data == current.next.data:
            # Remove duplicate
            duplicate = current.next
            current.next = duplicate.next
            if duplicate.next:
                duplicate.next.prev = current
        else:
            current = current.next
    
    return head
```

### Problem 2: Clone Doubly Linked List with Random Pointer

```python
def clone_dll_with_random(head):
    """Clone doubly linked list with random pointers"""
    if not head:
        return None
    
    # Create mapping of original to clone nodes
    node_map = {}
    
    # First pass: create all nodes
    current = head
    while current:
        node_map[current] = DoublyListNode(current.data)
        current = current.next
    
    # Second pass: set pointers
    current = head
    while current:
        clone = node_map[current]
        if current.next:
            clone.next = node_map[current.next]
        if current.prev:
            clone.prev = node_map[current.prev]
        if hasattr(current, 'random') and current.random:
            clone.random = node_map[current.random]
        current = current.next
    
    return node_map[head]
```

## 🎯 Key Takeaways

1. **Bidirectional Navigation**: Major advantage over singly linked lists
2. **Memory Trade-off**: Extra pointer per node for functionality
3. **Implementation Complexity**: More edge cases to handle
4. **Performance**: Better for applications requiring backward traversal
5. **Real-world Usage**: Browsers, media players, undo/redo systems

Doubly linked lists are perfect when you need efficient bidirectional navigation and don't mind the extra memory overhead!
