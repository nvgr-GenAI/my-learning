# Linked Lists: Easy Problems

Welcome to the easy problems section! These problems are perfect for beginners to build confidence and understand fundamental linked list operations. Each problem includes multiple solutions with detailed explanations.

## 🎯 Learning Objectives

By completing these problems, you'll master:

- Basic linked list traversal
- Simple pointer manipulation
- Edge case handling (empty lists, single nodes)
- Common patterns (two pointers, dummy nodes)

---

## Problem 1: Reverse Linked List

**LeetCode 206** | **Difficulty: Easy**

### Problem Statement

Given the head of a singly linked list, reverse the list and return the new head.

**Example:**
```
Input:  1 -> 2 -> 3 -> 4 -> 5 -> null
Output: 5 -> 4 -> 3 -> 2 -> 1 -> null
```

### Solution 1: Iterative Approach

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseList(head):
    """
    Reverse a linked list iteratively.
    
    Time: O(n) - visit each node once
    Space: O(1) - only use three pointers
    """
    prev = None
    current = head
    
    while current:
        # Store next node before breaking the link
        next_temp = current.next
        
        # Reverse the link
        current.next = prev
        
        # Move pointers forward
        prev = current
        current = next_temp
    
    return prev  # prev is now the new head

# Test
head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)

result = reverseList(head)
# Result: 3 -> 2 -> 1 -> null
```

### Solution 2: Recursive Approach

```python
def reverseListRecursive(head):
    """
    Reverse a linked list recursively.
    
    Time: O(n) - visit each node once
    Space: O(n) - recursion stack
    """
    # Base case: empty list or single node
    if not head or not head.next:
        return head
    
    # Recursively reverse the rest of the list
    new_head = reverseListRecursive(head.next)
    
    # Reverse the current connection
    head.next.next = head
    head.next = None
    
    return new_head
```

### 🔍 Step-by-Step Trace

**Iterative approach:**
```
Initial: prev=null, current=1->2->3->null

Step 1: next_temp=2->3->null, current.next=null, prev=1, current=2
        Result: null<-1  2->3->null

Step 2: next_temp=3->null, current.next=1, prev=2, current=3  
        Result: null<-1<-2  3->null

Step 3: next_temp=null, current.next=2, prev=3, current=null
        Result: null<-1<-2<-3
```

---

## Problem 2: Merge Two Sorted Lists

**LeetCode 21** | **Difficulty: Easy**

### Problem Statement

Merge two sorted linked lists and return it as a sorted list.

**Example:**
```
Input: list1 = 1->2->4, list2 = 1->3->4
Output: 1->1->2->3->4->4
```

### Solution 1: Iterative with Dummy Node

```python
def mergeTwoLists(list1, list2):
    """
    Merge two sorted lists using dummy node.
    
    Time: O(n + m) - visit all nodes once
    Space: O(1) - only use constant extra space
    """
    # Create dummy node to simplify logic
    dummy = ListNode(0)
    current = dummy
    
    # Compare and merge while both lists have nodes
    while list1 and list2:
        if list1.val <= list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next
    
    # Attach remaining nodes
    current.next = list1 if list1 else list2
    
    return dummy.next  # Skip dummy node

# Test
list1 = ListNode(1)
list1.next = ListNode(2)
list1.next.next = ListNode(4)

list2 = ListNode(1)
list2.next = ListNode(3)
list2.next.next = ListNode(4)

result = mergeTwoLists(list1, list2)
# Result: 1->1->2->3->4->4
```

### Solution 2: Recursive Approach

```python
def mergeTwoListsRecursive(list1, list2):
    """
    Merge two sorted lists recursively.
    
    Time: O(n + m)
    Space: O(n + m) - recursion stack
    """
    # Base cases
    if not list1:
        return list2
    if not list2:
        return list1
    
    # Choose smaller node and recurse
    if list1.val <= list2.val:
        list1.next = mergeTwoListsRecursive(list1.next, list2)
        return list1
    else:
        list2.next = mergeTwoListsRecursive(list1, list2.next)
        return list2
```

---

## Problem 3: Remove Duplicates from Sorted List

**LeetCode 83** | **Difficulty: Easy**

### Problem Statement

Given the head of a sorted linked list, delete all duplicates such that each element appears only once.

**Example:**
```
Input:  1->1->2->3->3
Output: 1->2->3
```

### Solution

```python
def deleteDuplicates(head):
    """
    Remove duplicates from sorted linked list.
    
    Time: O(n) - visit each node once
    Space: O(1) - only use constant extra space
    """
    if not head:
        return head
    
    current = head
    
    while current.next:
        if current.val == current.next.val:
            # Skip duplicate node
            current.next = current.next.next
        else:
            # Move to next different node
            current = current.next
    
    return head

# Test
head = ListNode(1)
head.next = ListNode(1)
head.next.next = ListNode(2)
head.next.next.next = ListNode(3)
head.next.next.next.next = ListNode(3)

result = deleteDuplicates(head)
# Result: 1->2->3
```

### 🔍 Key Insight

Since the list is sorted, duplicates are always adjacent. We only need to compare each node with its immediate next node.

---

## Problem 4: Linked List Cycle

**LeetCode 141** | **Difficulty: Easy**

### Problem Statement

Given head, determine if the linked list has a cycle in it.

**Example:**
```
Input: 3->2->0->-4 (where -4 points back to 2)
Output: true
```

### Solution 1: Two Pointers (Floyd's Algorithm)

```python
def hasCycle(head):
    """
    Detect cycle using Floyd's tortoise and hare algorithm.
    
    Time: O(n) - at most visit each node twice
    Space: O(1) - only use two pointers
    """
    if not head or not head.next:
        return False
    
    slow = head  # Tortoise (moves 1 step)
    fast = head  # Hare (moves 2 steps)
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        # If they meet, there's a cycle
        if slow == fast:
            return True
    
    return False

# Test
head = ListNode(3)
head.next = ListNode(2)
head.next.next = ListNode(0)
head.next.next.next = ListNode(-4)
head.next.next.next.next = head.next  # Create cycle

result = hasCycle(head)  # True
```

### Solution 2: Hash Set

```python
def hasCycleHashSet(head):
    """
    Detect cycle using hash set to track visited nodes.
    
    Time: O(n)
    Space: O(n) - store all nodes in worst case
    """
    visited = set()
    current = head
    
    while current:
        if current in visited:
            return True
        visited.add(current)
        current = current.next
    
    return False
```

### 🔍 Why Floyd's Algorithm Works

- **Slow pointer** moves 1 step at a time
- **Fast pointer** moves 2 steps at a time
- If there's a cycle, fast pointer will eventually "lap" the slow pointer
- If there's no cycle, fast pointer reaches the end

---

## Problem 5: Middle of the Linked List

**LeetCode 876** | **Difficulty: Easy**

### Problem Statement

Given the head of a singly linked list, return the middle node. If there are two middle nodes, return the second middle node.

**Example:**
```
Input:  1->2->3->4->5
Output: 3->4->5 (node 3 is the middle)

Input:  1->2->3->4->5->6
Output: 4->5->6 (node 4 is the second middle)
```

### Solution 1: Two Pointers

```python
def middleNode(head):
    """
    Find middle node using two pointers.
    
    Time: O(n) - visit each node once
    Space: O(1) - only use two pointers
    """
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow

# Test
head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)
head.next.next.next.next = ListNode(5)

result = middleNode(head)  # Returns node 3
```

### Solution 2: Count and Traverse

```python
def middleNodeCount(head):
    """
    Find middle by counting nodes first.
    
    Time: O(n) - two passes through the list
    Space: O(1)
    """
    # Count total nodes
    count = 0
    current = head
    while current:
        count += 1
        current = current.next
    
    # Find middle position
    middle_pos = count // 2
    
    # Traverse to middle
    current = head
    for _ in range(middle_pos):
        current = current.next
    
    return current
```

---

## Problem 6: Remove Linked List Elements

**LeetCode 203** | **Difficulty: Easy**

### Problem Statement

Given the head of a linked list and an integer val, remove all nodes with Node.val == val.

**Example:**
```
Input: head = 1->2->6->3->4->5->6, val = 6
Output: 1->2->3->4->5
```

### Solution

```python
def removeElements(head, val):
    """
    Remove all nodes with given value using dummy node.
    
    Time: O(n) - visit each node once
    Space: O(1) - only use constant extra space
    """
    # Use dummy node to handle edge cases
    dummy = ListNode(0)
    dummy.next = head
    current = dummy
    
    while current.next:
        if current.next.val == val:
            # Remove the node
            current.next = current.next.next
        else:
            # Move to next node
            current = current.next
    
    return dummy.next

# Test
head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(6)
head.next.next.next = ListNode(3)
head.next.next.next.next = ListNode(4)
head.next.next.next.next.next = ListNode(5)
head.next.next.next.next.next.next = ListNode(6)

result = removeElements(head, 6)
# Result: 1->2->3->4->5
```

---

## 🎯 Practice Tips

### 1. Always Handle Edge Cases

```python
# Check these cases for every problem:
if not head:           # Empty list
    return None
if not head.next:      # Single node
    return head
```

### 2. Use Dummy Nodes

Dummy nodes simplify code by eliminating special cases for head operations:

```python
dummy = ListNode(0)
dummy.next = head
# Now you can safely modify head without special handling
```

### 3. Draw Diagrams

Visual representation helps understand pointer movements:

```
Before: A -> B -> C -> D
After:  A -> C -> D (removed B)
```

### 4. Two Pointers Pattern

Many problems use two pointers with different speeds:

- **Same speed**: Find differences, merge operations
- **Different speed**: Find middle, detect cycles

## 📚 Summary

### Key Patterns Learned

1. **Iterative vs Recursive**: Most problems have both solutions
2. **Dummy Nodes**: Simplify edge cases in modification problems
3. **Two Pointers**: Essential for cycle detection and finding middle
4. **Pointer Manipulation**: Core skill for all linked list problems

### Time Complexities

- **Traversal**: O(n)
- **Search**: O(n)
- **Insertion/Deletion**: O(1) at known position, O(n) to find position

### Common Mistakes to Avoid

- Forgetting to handle null pointers
- Losing references to nodes during operations
- Not considering empty lists or single-node lists

---

**Ready for more challenges?** Move on to **[Medium Problems](medium-problems.md)** to learn advanced techniques!

## 🏆 Progress Checklist

- [ ] **Reverse Linked List** - Master iterative and recursive approaches
- [ ] **Merge Two Sorted Lists** - Understand dummy node pattern
- [ ] **Remove Duplicates** - Practice simple pointer manipulation
- [ ] **Linked List Cycle** - Learn Floyd's algorithm
- [ ] **Middle of List** - Master two-pointer technique
- [ ] **Remove Elements** - Combine dummy nodes with traversal

Once you've completed all problems, you're ready for the next level!
