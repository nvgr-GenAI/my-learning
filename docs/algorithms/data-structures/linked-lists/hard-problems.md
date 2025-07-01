# Linked Lists: Hard Problems

These advanced problems will test your mastery of linked list concepts and push you to combine multiple techniques. Success here means you've truly mastered linked lists!

## 🎯 Learning Objectives

By completing these problems, you'll achieve:

- Mastery of complex pointer manipulation
- Ability to combine multiple algorithms  
- Advanced optimization techniques
- Real-world problem-solving skills
- Preparation for system design interviews

---

## Problem 1: Merge k Sorted Lists

**LeetCode 23** | **Difficulty: Hard**

### Problem Statement

You are given an array of k linked-lists, each sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.

**Example:**

```text
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
```

### Solution 1: Divide and Conquer

```python
def mergeKLists(lists):
    """
    Merge k sorted lists using divide and conquer.
    
    Time: O(N log k) where N is total number of nodes
    Space: O(log k) for recursion stack
    """
    if not lists:
        return None
    
    def merge_two_lists(l1, l2):
        """Helper function to merge two sorted lists."""
        dummy = ListNode(0)
        current = dummy
        
        while l1 and l2:
            if l1.val <= l2.val:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next
            current = current.next
        
        current.next = l1 if l1 else l2
        return dummy.next
    
    def merge_lists(lists, start, end):
        """Divide and conquer approach."""
        if start == end:
            return lists[start]
        
        if start > end:
            return None
        
        mid = (start + end) // 2
        left = merge_lists(lists, start, mid)
        right = merge_lists(lists, mid + 1, end)
        
        return merge_two_lists(left, right)
    
    return merge_lists(lists, 0, len(lists) - 1)
```

### Solution 2: Priority Queue (Heap)

```python
import heapq

def mergeKListsHeap(lists):
    """
    Merge k sorted lists using min heap.
    
    Time: O(N log k)
    Space: O(k) for heap
    """
    if not lists:
        return None
    
    # Create min heap with first node from each list
    heap = []
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        val, list_idx, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        
        # Add next node from same list
        if node.next:
            heapq.heappush(heap, (node.next.val, list_idx, node.next))
    
    return dummy.next
```

### 🔍 Analysis

**Divide and Conquer vs Heap:**
- **Divide & Conquer**: Better space complexity O(log k) vs O(k)
- **Heap**: More intuitive, handles dynamic list addition easily
- **Both**: Same time complexity O(N log k)

---

## Problem 2: Reverse Nodes in k-Group

**LeetCode 25** | **Difficulty: Hard**

### Problem Statement

Given the head of a linked list, reverse the nodes of the list k at a time, and return the modified list.

**Example:**

```text
Input: head = 1->2->3->4->5, k = 3
Output: 3->2->1->4->5
```

### Solution

```python
def reverseKGroup(head, k):
    """
    Reverse nodes in k-group.
    
    Time: O(n) where n is number of nodes
    Space: O(1) - iterative approach
    """
    def get_length(node):
        """Get length of remaining list."""
        length = 0
        while node:
            length += 1
            node = node.next
        return length
    
    def reverse_list(head, k):
        """Reverse first k nodes and return new head and tail."""
        prev = None
        current = head
        
        for _ in range(k):
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        
        return prev, head  # new_head, new_tail
    
    # Check if we have at least k nodes
    if get_length(head) < k:
        return head
    
    # Reverse first k nodes
    new_head, new_tail = reverse_list(head, k)
    
    # Recursively handle remaining nodes
    remaining = head
    for _ in range(k):
        remaining = remaining.next
    
    new_tail.next = reverseKGroup(remaining, k)
    
    return new_head
```

### Iterative Solution

```python
def reverseKGroupIterative(head, k):
    """
    Iterative version of reverse k-group.
    
    Time: O(n)
    Space: O(1)
    """
    def reverse_k_nodes(start, k):
        """Reverse k nodes starting from start."""
        prev = None
        current = start
        
        for _ in range(k):
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        
        return prev  # new head of reversed group
    
    # Count total nodes
    count = 0
    current = head
    while current:
        count += 1
        current = current.next
    
    dummy = ListNode(0)
    dummy.next = head
    group_prev = dummy
    
    while count >= k:
        group_start = group_prev.next
        group_end = group_start
        
        # Find end of current group
        for _ in range(k - 1):
            group_end = group_end.next
        
        # Save next group start
        next_group_start = group_end.next
        
        # Reverse current group
        new_head = reverse_k_nodes(group_start, k)
        
        # Connect with previous group
        group_prev.next = new_head
        group_start.next = next_group_start
        
        # Move to next group
        group_prev = group_start
        count -= k
    
    return dummy.next
```

---

## Problem 3: Copy List with Random Pointer

**LeetCode 138** | **Difficulty: Hard**

### Problem Statement

A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null. Construct a deep copy of the list.

**Example:**

```text
Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]
```

### Node Definition

```python
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
```

### Solution 1: HashMap Approach

```python
def copyRandomList(head):
    """
    Copy list with random pointers using HashMap.
    
    Time: O(n)
    Space: O(n) for hashmap
    """
    if not head:
        return None
    
    # Map original nodes to new nodes
    node_map = {}
    
    # First pass: create all new nodes
    current = head
    while current:
        node_map[current] = Node(current.val)
        current = current.next
    
    # Second pass: set next and random pointers
    current = head
    while current:
        if current.next:
            node_map[current].next = node_map[current.next]
        if current.random:
            node_map[current].random = node_map[current.random]
        current = current.next
    
    return node_map[head]
```

### Solution 2: Interweaving Approach (O(1) Space)

```python
def copyRandomListOptimal(head):
    """
    Copy list with random pointers using interweaving.
    
    Time: O(n)
    Space: O(1) - only using constant extra space
    """
    if not head:
        return None
    
    # Step 1: Create new nodes and interweave with original
    current = head
    while current:
        new_node = Node(current.val)
        new_node.next = current.next
        current.next = new_node
        current = new_node.next
    
    # Step 2: Set random pointers for new nodes
    current = head
    while current:
        if current.random:
            current.next.random = current.random.next
        current = current.next.next
    
    # Step 3: Separate the two lists
    dummy = Node(0)
    new_current = dummy
    current = head
    
    while current:
        new_current.next = current.next
        current.next = current.next.next
        new_current = new_current.next
        current = current.next
    
    return dummy.next
```

### 🔍 Interweaving Technique

The optimal solution uses a clever interweaving approach:

```text
Original:  A -> B -> C -> null
           |    |    |
           v    v    v
          B    C    A  (random pointers)

After Step 1:  A -> A' -> B -> B' -> C -> C' -> null

After Step 2:  A -> A' -> B -> B' -> C -> C' -> null
               |    |    |    |    |    |
               v    |    v    |    v    |
              B     B'  C     C'   A    A'

After Step 3:  A -> B -> C -> null
               A' -> B' -> C' -> null (result)
```

---

## Problem 4: LRU Cache

**LeetCode 146** | **Difficulty: Hard**

### Problem Statement

Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:
- `LRUCache(int capacity)` Initialize with positive size capacity
- `int get(int key)` Return value of key if exists, otherwise -1
- `void put(int key, int value)` Update value if key exists, otherwise add key-value pair

Both operations should run in O(1) average time complexity.

### Solution: HashMap + Doubly Linked List

```python
class DLLNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        """
        Initialize LRU Cache with given capacity.
        
        Uses HashMap + Doubly Linked List for O(1) operations.
        """
        self.capacity = capacity
        self.cache = {}  # key -> node mapping
        
        # Create dummy head and tail nodes
        self.head = DLLNode()
        self.tail = DLLNode()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node):
        """Add node right after head."""
        node.prev = self.head
        node.next = self.head.next
        
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node):
        """Remove an existing node from linked list."""
        prev_node = node.prev
        next_node = node.next
        
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _move_to_head(self, node):
        """Move node to head (mark as recently used)."""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self):
        """Remove last node before tail."""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node
    
    def get(self, key: int) -> int:
        """
        Get value for key, mark as recently used.
        
        Time: O(1)
        """
        node = self.cache.get(key)
        
        if not node:
            return -1
        
        # Move to head (mark as recently used)
        self._move_to_head(node)
        
        return node.value
    
    def put(self, key: int, value: int) -> None:
        """
        Put key-value pair, handle capacity overflow.
        
        Time: O(1)
        """
        node = self.cache.get(key)
        
        if not node:
            new_node = DLLNode(key, value)
            
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                tail = self._pop_tail()
                del self.cache[tail.key]
            
            # Add new node
            self.cache[key] = new_node
            self._add_node(new_node)
        else:
            # Update existing node
            node.value = value
            self._move_to_head(node)

# Usage
lru = LRUCache(2)
lru.put(1, 1)
lru.put(2, 2)
print(lru.get(1))  # 1
lru.put(3, 3)      # evicts key 2
print(lru.get(2))  # -1 (not found)
lru.put(4, 4)      # evicts key 1
print(lru.get(1))  # -1 (not found)
print(lru.get(3))  # 3
print(lru.get(4))  # 4
```

### 🔍 Design Insights

**Why HashMap + Doubly Linked List?**
- **HashMap**: O(1) key lookup
- **Doubly Linked List**: O(1) insertion/deletion at any position
- **Combination**: Achieves O(1) for both get and put operations

**Key Operations:**
1. **Get**: Move node to head (recently used)
2. **Put (new)**: Add to head, evict from tail if needed
3. **Put (existing)**: Update value, move to head

---

## Problem 5: Design Linked List

**LeetCode 707** | **Difficulty: Hard** (Design)

### Problem Statement

Design your implementation of the linked list with the following operations:
- `get(index)`: Get the value of the index-th node
- `addAtHead(val)`: Add a node of value val before the first element
- `addAtTail(val)`: Append a node of value val to the last element
- `addAtIndex(index, val)`: Add node before the index-th node
- `deleteAtIndex(index)`: Delete the index-th node if valid

### Solution: Doubly Linked List Implementation

```python
class ListNode:
    def __init__(self, val=0):
        self.val = val
        self.next = None
        self.prev = None

class MyLinkedList:
    def __init__(self):
        """Initialize with dummy head and tail."""
        self.head = ListNode()
        self.tail = ListNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
    
    def get(self, index: int) -> int:
        """Get value at index."""
        if index < 0 or index >= self.size:
            return -1
        
        node = self._get_node(index)
        return node.val
    
    def addAtHead(self, val: int) -> None:
        """Add node at head."""
        self._add_after(self.head, val)
    
    def addAtTail(self, val: int) -> None:
        """Add node at tail."""
        self._add_before(self.tail, val)
    
    def addAtIndex(self, index: int, val: int) -> None:
        """Add node at index."""
        if index < 0 or index > self.size:
            return
        
        if index == self.size:
            self.addAtTail(val)
        else:
            node = self._get_node(index)
            self._add_before(node, val)
    
    def deleteAtIndex(self, index: int) -> None:
        """Delete node at index."""
        if index < 0 or index >= self.size:
            return
        
        node = self._get_node(index)
        self._remove_node(node)
    
    def _get_node(self, index):
        """Get node at index (optimized for closer end)."""
        if index < self.size // 2:
            # Start from head
            current = self.head.next
            for _ in range(index):
                current = current.next
        else:
            # Start from tail
            current = self.tail.prev
            for _ in range(self.size - index - 1):
                current = current.prev
        
        return current
    
    def _add_after(self, node, val):
        """Add new node after given node."""
        new_node = ListNode(val)
        new_node.next = node.next
        new_node.prev = node
        node.next.prev = new_node
        node.next = new_node
        self.size += 1
    
    def _add_before(self, node, val):
        """Add new node before given node."""
        new_node = ListNode(val)
        new_node.next = node
        new_node.prev = node.prev
        node.prev.next = new_node
        node.prev = new_node
        self.size += 1
    
    def _remove_node(self, node):
        """Remove given node."""
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1
```

---

## 🎯 Advanced Patterns Mastered

### 1. Divide and Conquer
- **Merge k Lists**: Split problem into subproblems
- **Time Complexity**: Reduces from O(k*N) to O(N log k)

### 2. Interweaving Technique
- **Copy with Random Pointer**: Clever space optimization
- **Pattern**: Temporarily modify structure, then restore

### 3. Dual Data Structure Design
- **LRU Cache**: HashMap + Doubly Linked List
- **Benefit**: O(1) operations for complex requirements

### 4. Optimized Traversal
- **Design Linked List**: Bidirectional traversal optimization
- **Technique**: Choose starting point based on position

## 🧠 Problem-Solving Strategies

### 1. Break Down Complex Problems
```python
# Instead of solving everything at once:
def complex_operation(head):
    # Step 1: Handle edge cases
    if not head:
        return None
    
    # Step 2: Preprocess if needed
    length = get_length(head)
    
    # Step 3: Main algorithm
    result = main_logic(head, length)
    
    # Step 4: Post-process if needed
    return finalize(result)
```

### 2. Use Helper Functions
```python
def main_function(head):
    def helper_function(node, param):
        # Focused logic
        pass
    
    return helper_function(head, initial_param)
```

### 3. Consider Multiple Approaches
- **Recursive vs Iterative**: Space vs simplicity trade-off
- **One-pass vs Multi-pass**: Time vs space considerations
- **Extra space vs In-place**: Space optimization opportunities

## 🏆 Mastery Checklist

### Core Concepts Mastered
- [ ] **Complex Pointer Manipulation** - K-group reversal
- [ ] **Multiple Data Structure Integration** - LRU Cache
- [ ] **Advanced Optimization Techniques** - Interweaving
- [ ] **Divide and Conquer** - Merge k lists
- [ ] **System Design Principles** - Design linked list

### Problem-Solving Skills
- [ ] **Pattern Recognition** - Identify optimal approach quickly
- [ ] **Trade-off Analysis** - Time vs space considerations
- [ ] **Edge Case Handling** - Robust solution design
- [ ] **Code Organization** - Clean, maintainable implementations

### Interview Readiness
- [ ] **Explain Approach** - Clear problem-solving communication
- [ ] **Optimize Solutions** - Improve time/space complexity
- [ ] **Handle Follow-ups** - Adapt to requirement changes
- [ ] **Design Discussions** - Architectural considerations

## 🚀 Congratulations!

You've mastered the hardest linked list problems! You can now:

✅ **Solve complex linked list problems** with confidence
✅ **Optimize solutions** for time and space complexity  
✅ **Design data structures** that use linked lists effectively
✅ **Handle interview questions** at any difficulty level
✅ **Apply techniques** to real-world problems

### What's Next?

1. **Practice Implementation**: Implement data structures from scratch
2. **System Design**: Apply linked lists in larger system contexts
3. **Competitive Programming**: Tackle contest-level problems
4. **Teach Others**: Solidify understanding by helping peers

### Real-World Applications

Your linked list mastery applies to:
- **Database Systems**: Record linking and indexing
- **Operating Systems**: Process scheduling and memory management
- **Web Development**: Undo/redo functionality, caching systems
- **Game Development**: Object management and state tracking
- **Network Programming**: Packet queuing and routing

---

*Congratulations on completing the comprehensive linked list mastery journey! 🎉*
