# Linked Lists

## Overview

Linked Lists are linear data structures where elements are stored in nodes, and each node contains data and a reference to the next node.

## Types of Linked Lists

### Singly Linked List
- Each node points to the next node
- Last node points to null

### Doubly Linked List
- Each node has pointers to both next and previous nodes
- Allows bidirectional traversal

### Circular Linked List
- Last node points back to the first node
- Forms a circular structure

## Implementation

### Node Structure
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

### Basic Operations
```python
class LinkedList:
    def __init__(self):
        self.head = None
    
    def insert_at_beginning(self, val):
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
    
    def insert_at_end(self, val):
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def delete_node(self, val):
        if not self.head:
            return
        
        if self.head.val == val:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next and current.next.val != val:
            current = current.next
        
        if current.next:
            current.next = current.next.next
```

## Common Problems

1. **Reverse Linked List**
2. **Detect Cycle in Linked List**
3. **Merge Two Sorted Lists**
4. **Remove Nth Node From End**
5. **Intersection of Two Linked Lists**

## Time Complexities

| Operation | Time Complexity |
|-----------|----------------|
| Access    | O(n)           |
| Search    | O(n)           |
| Insertion | O(1) at head, O(n) at tail |
| Deletion  | O(1) if node given, O(n) to find |

## Practice Problems

- [ ] Reverse Linked List
- [ ] Linked List Cycle
- [ ] Merge Two Sorted Lists
- [ ] Remove Duplicates from Sorted List
- [ ] Palindrome Linked List
