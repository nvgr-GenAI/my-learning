# Linked Lists: Medium Problems

These intermediate problems will challenge you to combine basic operations with more advanced techniques. You'll learn to handle complex scenarios and optimize your solutions.

## 🎯 Learning Objectives

By completing these problems, you'll master:

- Advanced pointer manipulation techniques
- Multi-pass algorithms
- Handling complex edge cases
- Optimizing time and space complexity
- Problem-solving patterns for real-world scenarios

---

## Problem 1: Add Two Numbers

**LeetCode 2** | **Difficulty: Medium**

### Problem Description

You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

**Example:**

```text
Input: l1 = 2->4->3, l2 = 5->6->4
Output: 7->0->8
Explanation: 342 + 465 = 807
```

### Solution

```python
def addTwoNumbers(l1, l2):
    """
    Add two numbers represented as linked lists.
    
    Time: O(max(n, m)) where n, m are lengths of lists
    Space: O(max(n, m)) for the result list
    """
    dummy = ListNode(0)
    current = dummy
    carry = 0
    
    while l1 or l2 or carry:
        # Get values (0 if node is None)
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        # Calculate sum and carry
        total = val1 + val2 + carry
        carry = total // 10
        digit = total % 10
        
        # Create new node
        current.next = ListNode(digit)
        current = current.next
        
        # Move to next nodes
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    
    return dummy.next

# Test
l1 = ListNode(2)
l1.next = ListNode(4)
l1.next.next = ListNode(3)

l2 = ListNode(5)
l2.next = ListNode(6)
l2.next.next = ListNode(4)

result = addTwoNumbers(l1, l2)
# Result: 7->0->8
```

### 🔍 Key Insights

1. **Reverse order**: Digits are stored in reverse, making addition easier (start from least significant digit)
2. **Carry handling**: Don't forget to handle carry for the final digit
3. **Unequal lengths**: Handle cases where one list is longer than the other

---

## Problem 2: Remove Nth Node From End of List

**LeetCode 19** | **Difficulty: Medium**

### Problem Description

Given the head of a linked list, remove the nth node from the end of the list and return its head.

**Example:**

```text
Input: head = 1->2->3->4->5, n = 2
Output: 1->2->3->5
```

### Solution 1: Two-Pass Approach

```python
def removeNthFromEnd(head, n):
    """
    Remove nth node from end using two passes.
    
    Time: O(L) where L is length of list
    Space: O(1)
    """
    # First pass: count total nodes
    length = 0
    current = head
    while current:
        length += 1
        current = current.next
    
    # Special case: remove head
    if n == length:
        return head.next
    
    # Second pass: find (length - n)th node
    current = head
    for _ in range(length - n - 1):
        current = current.next
    
    # Remove the node
    current.next = current.next.next
    return head
```

### Solution 2: One-Pass with Two Pointers

```python
def removeNthFromEndOnePass(head, n):
    """
    Remove nth node from end using one pass with two pointers.
    
    Time: O(L)
    Space: O(1)
    """
    dummy = ListNode(0)
    dummy.next = head
    
    # Create two pointers n+1 nodes apart
    fast = slow = dummy
    
    # Move fast pointer n+1 steps ahead
    for _ in range(n + 1):
        fast = fast.next
    
    # Move both pointers until fast reaches end
    while fast:
        fast = fast.next
        slow = slow.next
    
    # slow is now at the node before the one to remove
    slow.next = slow.next.next
    
    return dummy.next
```

### 🔍 Two Pointers Technique

The key insight is maintaining a gap of `n+1` nodes between pointers:

```text
Initial: dummy->1->2->3->4->5->null, n=2
After setup: 
         slow  fast
         dummy->1->2->3->4->5->null
                   ^     ^
                   |     |
                   gap = n+1 = 3

After moving:
               slow     fast
         dummy->1->2->3->4->5->null
                     ^        ^
                     |        |
                     gap = 3
```

---

## Problem 3: Intersection of Two Linked Lists

**LeetCode 160** | **Difficulty: Medium**

### Problem Description

Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect. If they do not intersect, return null.

**Example:**

```text
Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5]
Output: Reference to node with value = 8
```

### Solution 1: Hash Set

```python
def getIntersectionNode(headA, headB):
    """
    Find intersection using hash set.
    
    Time: O(m + n) where m, n are lengths of lists
    Space: O(m) to store nodes from first list
    """
    if not headA or not headB:
        return None
    
    # Store all nodes from list A
    visited = set()
    current = headA
    while current:
        visited.add(current)
        current = current.next
    
    # Check if any node from list B is in visited set
    current = headB
    while current:
        if current in visited:
            return current
        current = current.next
    
    return None
```

### Solution 2: Two Pointers (Optimal)

```python
def getIntersectionNodeTwoPointers(headA, headB):
    """
    Find intersection using two pointers.
    
    Time: O(m + n)
    Space: O(1)
    """
    if not headA or not headB:
        return None
    
    pointerA = headA
    pointerB = headB
    
    # Traverse both lists
    while pointerA != pointerB:
        # If reached end, switch to other list
        pointerA = pointerA.next if pointerA else headB
        pointerB = pointerB.next if pointerB else headA
    
    return pointerA  # Either intersection point or None
```

### 🔍 Two Pointers Logic

The elegant insight: if lists intersect, pointers will meet at intersection after traversing both lists.

```text
List A: 4->1->8->4->5
List B: 5->6->1->8->4->5
                 ^
                 intersection

Pointer A path: 4->1->8->4->5->5->6->1->8 (meets here)
Pointer B path: 5->6->1->8->4->5->4->1->8 (meets here)
```

---

## Problem 4: Palindrome Linked List

**LeetCode 234** | **Difficulty: Medium**

### Problem Description

Given the head of a singly linked list, return true if it is a palindrome.

**Example:**

```text
Input: head = 1->2->2->1
Output: true

Input: head = 1->2
Output: false
```

### Solution 1: Array Conversion

```python
def isPalindrome(head):
    """
    Check palindrome by converting to array.
    
    Time: O(n)
    Space: O(n) for array storage
    """
    values = []
    current = head
    
    # Store all values in array
    while current:
        values.append(current.val)
        current = current.next
    
    # Check if array is palindrome
    return values == values[::-1]
```

### Solution 2: Reverse Second Half

```python
def isPalindromeOptimal(head):
    """
    Check palindrome by reversing second half.
    
    Time: O(n)
    Space: O(1)
    """
    if not head or not head.next:
        return True
    
    # Find middle using two pointers
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    second_half = reverseList(slow.next)
    
    # Compare first and second halves
    first_half = head
    while second_half:
        if first_half.val != second_half.val:
            return False
        first_half = first_half.next
        second_half = second_half.next
    
    return True

def reverseList(head):
    """Helper function to reverse a linked list."""
    prev = None
    current = head
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    return prev
```

### 🔍 Optimal Approach Breakdown

1. **Find middle**: Use fast/slow pointers
2. **Reverse second half**: Reverse from middle to end
3. **Compare**: Walk through both halves simultaneously
4. **Space optimization**: Only uses O(1) extra space

---

## Problem 5: Odd Even Linked List

**LeetCode 328** | **Difficulty: Medium**

### Problem Description

Given the head of a singly linked list, group all the nodes with odd indices together followed by the nodes with even indices, and return the reordered list.

**Example:**

```text
Input: head = 1->2->3->4->5
Output: 1->3->5->2->4
```

### Solution

```python
def oddEvenList(head):
    """
    Group odd and even positioned nodes.
    
    Time: O(n)
    Space: O(1)
    """
    if not head or not head.next:
        return head
    
    # Initialize pointers
    odd = head
    even = head.next
    even_head = even  # Keep reference to even list start
    
    # Rearrange nodes
    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next
    
    # Connect odd list to even list
    odd.next = even_head
    
    return head

# Test
head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)
head.next.next.next.next = ListNode(5)

result = oddEvenList(head)
# Result: 1->3->5->2->4
```

### 🔍 Pattern Recognition

This problem demonstrates the **two-list manipulation** pattern:

1. **Separate**: Create two separate chains (odd and even)
2. **Maintain**: Keep track of both chains simultaneously
3. **Connect**: Join the chains at the end

---

## Problem 6: Rotate List

**LeetCode 61** | **Difficulty: Medium**

### Problem Description

Given the head of a linked list, rotate the list to the right by k places.

**Example:**

```text
Input: head = 1->2->3->4->5, k = 2
Output: 4->5->1->2->3
```

### Solution

```python
def rotateRight(head, k):
    """
    Rotate linked list to the right by k places.
    
    Time: O(n)
    Space: O(1)
    """
    if not head or not head.next or k == 0:
        return head
    
    # Find length and make it circular
    length = 1
    current = head
    while current.next:
        current = current.next
        length += 1
    
    # Connect tail to head (make circular)
    current.next = head
    
    # Find new tail (length - k % length - 1 steps from head)
    k = k % length
    steps_to_new_tail = length - k
    new_tail = head
    
    for _ in range(steps_to_new_tail - 1):
        new_tail = new_tail.next
    
    # New head is next to new tail
    new_head = new_tail.next
    
    # Break the circle
    new_tail.next = None
    
    return new_head
```

### 🔍 Key Insights

1. **Circular approach**: Temporarily make list circular for easier rotation
2. **Modulo operation**: Handle cases where k > length
3. **New head/tail**: Calculate where to break the circle

---

## Problem 7: Swap Nodes in Pairs

**LeetCode 24** | **Difficulty: Medium**

### Problem Description

Given a linked list, swap every two adjacent nodes and return its head.

**Example:**

```text
Input: head = 1->2->3->4
Output: 2->1->4->3
```

### Solution 1: Iterative

```python
def swapPairs(head):
    """
    Swap every two adjacent nodes iteratively.
    
    Time: O(n)
    Space: O(1)
    """
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    
    while prev.next and prev.next.next:
        # Nodes to be swapped
        first = prev.next
        second = prev.next.next
        
        # Swapping
        prev.next = second
        first.next = second.next
        second.next = first
        
        # Move prev to end of swapped pair
        prev = first
    
    return dummy.next
```

### Solution 2: Recursive

```python
def swapPairsRecursive(head):
    """
    Swap every two adjacent nodes recursively.
    
    Time: O(n)
    Space: O(n) for recursion stack
    """
    # Base case
    if not head or not head.next:
        return head
    
    # Save second node
    second = head.next
    
    # Recursively swap the rest
    head.next = swapPairsRecursive(second.next)
    
    # Swap current pair
    second.next = head
    
    return second
```

---

## 🎯 Key Patterns in Medium Problems

### 1. Two Pointers Technique

**Applications:**
- Finding nth node from end
- Detecting cycles
- Finding intersection
- Finding middle

**Pattern:**
```python
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
```

### 2. Dummy Node Pattern

**Applications:**
- Simplifying edge cases
- Node removal operations
- List manipulation

**Pattern:**
```python
dummy = ListNode(0)
dummy.next = head
# Work with dummy.next
return dummy.next
```

### 3. List Reversal

**Applications:**
- Palindrome checking
- K-group reversal
- General list manipulation

**Pattern:**
```python
prev = None
current = head
while current:
    next_temp = current.next
    current.next = prev
    prev = current
    current = next_temp
```

### 4. Multiple Pass Approach

**Applications:**
- When you need list length
- Complex operations requiring preprocessing

**Pattern:**
```python
# First pass: gather information
length = get_length(head)
# Second pass: actual operation
```

## 📚 Practice Tips

### 1. Visualize the Problem

Always draw the linked list and trace through your algorithm:

```text
Before: 1 -> 2 -> 3 -> 4 -> 5
After:  1 -> 3 -> 5 -> 2 -> 4
```

### 2. Handle Edge Cases

Common edge cases to consider:
- Empty list (`head = None`)
- Single node list
- Two node list
- Operations at boundaries

### 3. Think About Space Complexity

- Can you solve it in O(1) space?
- Is the recursive solution worth the O(n) space?
- When is extra space justified?

### 4. Practice Pattern Recognition

Identify which pattern applies:
- Need to find middle? → Two pointers
- Need to remove nodes? → Dummy node
- Need to reverse? → Three pointers
- Need list info first? → Two passes

## 🏆 Progress Checklist

- [ ] **Add Two Numbers** - Master carry handling and unequal lengths
- [ ] **Remove Nth from End** - Learn two-pointer gap technique
- [ ] **Intersection of Lists** - Understand path switching approach
- [ ] **Palindrome List** - Practice list reversal in context
- [ ] **Odd Even List** - Master two-list manipulation
- [ ] **Rotate List** - Handle circular operations
- [ ] **Swap Pairs** - Practice node swapping patterns

## 🚀 Next Steps

Ready for the ultimate challenge? Advance to **[Hard Problems](hard-problems.md)** where you'll tackle:

- Complex multi-pointer algorithms
- Advanced data structure combinations
- Optimization challenges
- Real-world system design patterns

---

*Congratulations on mastering medium-level linked list problems! You're well-prepared for advanced challenges.*
