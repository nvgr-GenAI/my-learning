# Linked Lists - Easy Problems

## 🎯 Learning Objectives

Master fundamental linked list operations and patterns:

- Basic traversal and pointer manipulation
- Two-pointer techniques (slow/fast, dummy nodes)
- List reversal and merging operations
- Cycle detection and basic modifications
- Edge case handling (empty lists, single nodes)

---

## Problem 1: Reverse Linked List

**Difficulty**: 🟢 Easy  
**Pattern**: Pointer Manipulation  
**Time**: O(n), **Space**: O(1)

=== "Problem"

    Given the head of a singly linked list, reverse the list and return the new head.

    **Example 1:**
    ```
    Input:  1 -> 2 -> 3 -> 4 -> 5 -> null
    Output: 5 -> 4 -> 3 -> 2 -> 1 -> null
    ```

    **Example 2:**
    ```
    Input:  1 -> 2 -> null
    Output: 2 -> 1 -> null
    ```

=== "Solution"

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
    
    # Recursive approach
    def reverseList_recursive(head):
        """
        Reverse using recursion.
        
        Time: O(n) - visit each node once
        Space: O(n) - recursion stack
        """
        # Base case
        if not head or not head.next:
            return head
        
        # Reverse the rest of the list
        new_head = reverseList_recursive(head.next)
        
        # Reverse current connection
        head.next.next = head
        head.next = None
        
        return new_head
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Three Pointers**: Track previous, current, and next nodes
    - **Link Reversal**: Change `current.next` to point to `prev`
    - **Pointer Movement**: Move all pointers one step forward
    - **Base Cases**: Handle empty list and single node
    
    **Common Mistakes:**
    - Forgetting to store `next` before breaking the link
    - Not handling empty list case
    - Returning wrong pointer (should return `prev`, not `current`)

---

## Problem 2: Merge Two Sorted Lists

**Difficulty**: 🟢 Easy  
**Pattern**: Two Pointers + Dummy Node  
**Time**: O(n + m), **Space**: O(1)

=== "Problem"

    You are given the heads of two sorted linked lists `list1` and `list2`.

    Merge the two lists in a sorted manner and return the head of the merged linked list.

    **Example 1:**
    ```
    Input: list1 = [1,2,4], list2 = [1,3,4]
    Output: [1,1,2,3,4,4]
    ```

    **Example 2:**
    ```
    Input: list1 = [], list2 = []
    Output: []
    ```

=== "Solution"

    ```python
    def mergeTwoLists(list1, list2):
        """
        Merge two sorted linked lists.
        
        Time: O(n + m) where n, m are lengths of lists
        Space: O(1) - only using pointers
        """
        # Create dummy node to simplify edge cases
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
        
        return dummy.next
    
    # Recursive approach
    def mergeTwoLists_recursive(list1, list2):
        """Recursive merge approach"""
        # Base cases
        if not list1:
            return list2
        if not list2:
            return list1
        
        # Choose smaller value and recurse
        if list1.val <= list2.val:
            list1.next = mergeTwoLists_recursive(list1.next, list2)
            return list1
        else:
            list2.next = mergeTwoLists_recursive(list1, list2.next)
            return list2
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Dummy Node**: Simplifies edge cases and result construction
    - **Two Pointers**: Track current position in both lists
    - **Comparison Logic**: Always choose smaller value
    - **Remaining Nodes**: Attach leftover nodes at the end
    
    **Dummy Node Pattern:**
    ```python
    dummy = ListNode(0)
    current = dummy
    # ... build list using current.next ...
    return dummy.next  # Skip dummy node
    ```

---

## Problem 3: Remove Duplicates from Sorted List

**Difficulty**: 🟢 Easy  
**Pattern**: Single Pointer Traversal  
**Time**: O(n), **Space**: O(1)

=== "Problem"

    Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.

    **Example 1:**
    ```
    Input: head = [1,1,2]
    Output: [1,2]
    ```

    **Example 2:**
    ```
    Input: head = [1,1,2,3,3]
    Output: [1,2,3]
    ```

=== "Solution"

    ```python
    def deleteDuplicates(head):
        """
        Remove duplicates from sorted linked list.
        
        Time: O(n) - single pass through list
        Space: O(1) - only using one pointer
        """
        if not head:
            return head
        
        current = head
        
        while current and current.next:
            if current.val == current.next.val:
                # Skip the duplicate node
                current.next = current.next.next
            else:
                # Move to next unique value
                current = current.next
        
        return head
    
    # Alternative with explicit duplicate removal
    def deleteDuplicates_explicit(head):
        """More explicit duplicate removal"""
        if not head or not head.next:
            return head
        
        current = head
        
        while current.next:
            if current.val == current.next.val:
                duplicate = current.next
                current.next = duplicate.next
                # Optional: explicitly delete duplicate node
                duplicate.next = None
            else:
                current = current.next
        
        return head
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Skip vs Move**: Skip duplicate nodes, move to different values
    - **Pointer Adjustment**: Change `next` pointer to bypass duplicates
    - **Current Position**: Don't move current when removing duplicates
    - **Sorted Property**: Only need to check adjacent nodes
    
    **Critical Logic:**
    ```python
    if current.val == current.next.val:
        current.next = current.next.next  # Skip duplicate
    else:
        current = current.next            # Move forward
    ```

---

## Problem 4: Linked List Cycle

**Difficulty**: 🟢 Easy  
**Pattern**: Floyd's Cycle Detection (Tortoise and Hare)  
**Time**: O(n), **Space**: O(1)

=== "Problem"

    Given `head`, the head of a linked list, determine if the linked list has a cycle in it.

    There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer.

    **Example 1:**
    ```
    Input: head = [3,2,0,-4], pos = 1 (cycle back to node 1)
    Output: true
    ```

    **Example 2:**
    ```
    Input: head = [1,2], pos = -1 (no cycle)
    Output: false
    ```

=== "Solution"

    ```python
    def hasCycle(head):
        """
        Detect cycle using Floyd's algorithm (two pointers).
        
        Time: O(n) - at most 2n steps before meeting
        Space: O(1) - only two pointers
        """
        if not head or not head.next:
            return False
        
        slow = head      # Moves 1 step at a time
        fast = head.next # Moves 2 steps at a time
        
        while fast and fast.next:
            if slow == fast:
                return True  # Cycle detected
            
            slow = slow.next
            fast = fast.next.next
        
        return False  # No cycle
    
    # Alternative starting both pointers at head
    def hasCycle_alternative(head):
        """Both pointers start at head"""
        if not head:
            return False
        
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                return True
        
        return False
    
    # Using set for comparison (less optimal)
    def hasCycle_set(head):
        """
        Using set to track visited nodes.
        
        Time: O(n), Space: O(n)
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

=== "Insights"

    **Key Concepts:**
    
    - **Floyd's Algorithm**: Slow and fast pointers will meet if cycle exists
    - **Speed Difference**: Fast pointer gains 1 position per iteration
    - **Meeting Point**: They will meet within n steps if cycle exists
    - **No Cycle**: Fast pointer reaches end (null)
    
    **Why It Works:**
    - If there's a cycle, fast pointer will eventually "lap" slow pointer
    - Distance between them decreases by 1 each iteration
    - They must meet within cycle length iterations
    
    **Memory Optimization:**
    Floyd's algorithm uses O(1) space vs O(n) for set-based approach.

---

## Problem 5: Middle of the Linked List

**Difficulty**: 🟢 Easy  
**Pattern**: Two Pointers (Slow/Fast)  
**Time**: O(n), **Space**: O(1)

=== "Problem"

    Given the head of a singly linked list, return the middle node of the linked list.

    If there are two middle nodes, return the **second middle** node.

    **Example 1:**
    ```
    Input: head = [1,2,3,4,5]
    Output: [3,4,5] (node 3 is the middle)
    ```

    **Example 2:**
    ```
    Input: head = [1,2,3,4,5,6]
    Output: [4,5,6] (node 4 is the second middle)
    ```

=== "Solution"

    ```python
    def middleNode(head):
        """
        Find middle node using two pointers.
        
        Time: O(n) - single pass through list
        Space: O(1) - only two pointers
        """
        slow = fast = head
        
        # Fast moves 2 steps, slow moves 1 step
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow  # slow is at middle when fast reaches end
    
    # Alternative: count nodes first
    def middleNode_count(head):
        """
        Two-pass approach: count then find middle.
        
        Time: O(n), Space: O(1)
        """
        # First pass: count nodes
        count = 0
        current = head
        while current:
            count += 1
            current = current.next
        
        # Second pass: find middle
        middle_index = count // 2
        current = head
        for _ in range(middle_index):
            current = current.next
        
        return current
    
    # For returning first middle in even-length lists
    def middleNode_first_middle(head):
        """Return first middle for even-length lists"""
        if not head or not head.next:
            return head
        
        slow = head
        fast = head.next
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Two-Pointer Technique**: Fast pointer moves twice as fast as slow
    - **Termination Condition**: When fast reaches end, slow is at middle
    - **Even vs Odd Length**: Fast pointer determines which middle to return
    - **Edge Cases**: Handle single node and empty list
    
    **Pointer Movement Pattern:**
    ```
    Odd length (5 nodes):   1-2-3-4-5
    slow:                   1   2   3     (middle)
    fast:                   1     3     5 (end)
    
    Even length (6 nodes):  1-2-3-4-5-6
    slow:                   1   2   3   4 (second middle)
    fast:                   1     3     5   (next is null)
    ```

---

## Problem 6: Remove Linked List Elements

**Difficulty**: 🟢 Easy  
**Pattern**: Dummy Node + Pointer Manipulation  
**Time**: O(n), **Space**: O(1)

=== "Problem"

    Given the head of a linked list and an integer `val`, remove all the nodes of the linked list that has `Node.val == val`, and return the new head.

    **Example 1:**
    ```
    Input: head = [1,2,6,3,4,5,6], val = 6
    Output: [1,2,3,4,5]
    ```

    **Example 2:**
    ```
    Input: head = [7,7,7,7], val = 7
    Output: []
    ```

=== "Solution"

    ```python
    def removeElements(head, val):
        """
        Remove all nodes with given value.
        
        Time: O(n) - single pass through list
        Space: O(1) - only using pointers
        """
        # Use dummy node to handle head removal
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        current = head
        
        while current:
            if current.val == val:
                # Remove current node
                prev.next = current.next
            else:
                # Move prev pointer only when not removing
                prev = current
            current = current.next
        
        return dummy.next
    
    # Alternative without dummy node
    def removeElements_no_dummy(head, val):
        """Remove without dummy node (handle head separately)"""
        # Remove from beginning while head matches
        while head and head.val == val:
            head = head.next
        
        if not head:
            return None
        
        current = head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
            else:
                current = current.next
        
        return head
    
    # Recursive approach
    def removeElements_recursive(head, val):
        """Recursive removal"""
        if not head:
            return None
        
        # Process rest of list first
        head.next = removeElements_recursive(head.next, val)
        
        # Return current node if it should be kept
        return head.next if head.val == val else head
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Dummy Node**: Simplifies removal when head needs to be removed
    - **Previous Pointer**: Track node before current for removal
    - **Conditional Movement**: Only move `prev` when not removing
    - **Edge Cases**: All nodes removed, no nodes match value
    
    **Removal Pattern:**
    ```python
    if should_remove:
        prev.next = current.next  # Skip current
    else:
        prev = current           # Move prev forward
    current = current.next       # Always move current
    ```

---

## Problem 7: Palindrome Linked List

**Difficulty**: 🟢 Easy  
**Pattern**: Two Pointers + List Reversal  
**Time**: O(n), **Space**: O(1)

=== "Problem"

    Given the head of a singly linked list, return `true` if it is a palindrome or `false` otherwise.

    **Example 1:**
    ```
    Input: head = [1,2,2,1]
    Output: true
    ```

    **Example 2:**
    ```
    Input: head = [1,2]
    Output: false
    ```

=== "Solution"

    ```python
    def isPalindrome(head):
        """
        Check if linked list is palindrome.
        
        Time: O(n) - traverse list twice
        Space: O(1) - only pointers used
        """
        if not head or not head.next:
            return True
        
        # Step 1: Find middle using two pointers
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        # Step 2: Reverse second half
        second_half = reverse_list(slow.next)
        
        # Step 3: Compare first half with reversed second half
        first_half = head
        while second_half:
            if first_half.val != second_half.val:
                return False
            first_half = first_half.next
            second_half = second_half.next
        
        return True
    
    def reverse_list(head):
        """Helper function to reverse a linked list"""
        prev = None
        current = head
        
        while current:
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        
        return prev
    
    # Using extra space (simpler but O(n) space)
    def isPalindrome_array(head):
        """
        Convert to array and check palindrome.
        
        Time: O(n), Space: O(n)
        """
        values = []
        current = head
        
        # Store all values
        while current:
            values.append(current.val)
            current = current.next
        
        # Check if array is palindrome
        return values == values[::-1]
    
    # Recursive approach
    def isPalindrome_recursive(head):
        """Recursive approach with early termination"""
        def check_palindrome(node):
            if not node:
                return True, head
            
            is_pal, left = check_palindrome(node.next)
            if not is_pal:
                return False, left
            
            is_equal = (left.val == node.val)
            left_next = left.next
            
            return is_equal, left_next
        
        result, _ = check_palindrome(head)
        return result
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Three-Step Process**: Find middle → Reverse half → Compare
    - **Middle Finding**: Use slow/fast pointers to locate center
    - **Comparison Strategy**: Compare first half with reversed second half
    - **Space Optimization**: Reverse in-place vs storing in array
    
    **Algorithm Steps:**
    1. Find middle of list using two pointers
    2. Reverse the second half of the list
    3. Compare first half with reversed second half
    4. Return result of comparison
    
    **Edge Cases:**
    - Empty list: palindrome by definition
    - Single node: always palindrome
    - Two nodes: palindrome if values are equal

---

## Problem 8: Intersection of Two Linked Lists

**Difficulty**: 🟢 Easy  
**Pattern**: Two Pointers with Length Alignment  
**Time**: O(n + m), **Space**: O(1)

=== "Problem"

    Given the heads of two singly linked-lists `headA` and `headB`, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return `null`.

    **Example 1:**
    ```
    Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
    Output: Intersected at '8'
    ```

=== "Solution"

    ```python
    def getIntersectionNode(headA, headB):
        """
        Find intersection using two pointers.
        
        Time: O(n + m) where n, m are list lengths
        Space: O(1) - only two pointers
        """
        if not headA or not headB:
            return None
        
        # Two pointers approach
        ptrA, ptrB = headA, headB
        
        # Continue until they meet or both reach end
        while ptrA != ptrB:
            # Switch to other list when reaching end
            ptrA = headB if ptrA is None else ptrA.next
            ptrB = headA if ptrB is None else ptrB.next
        
        return ptrA  # Either intersection node or None
    
    # Alternative: calculate lengths and align
    def getIntersectionNode_align(headA, headB):
        """Align lists by length difference"""
        def get_length(head):
            length = 0
            while head:
                length += 1
                head = head.next
            return length
        
        lenA = get_length(headA)
        lenB = get_length(headB)
        
        # Align starting positions
        while lenA > lenB:
            headA = headA.next
            lenA -= 1
        
        while lenB > lenA:
            headB = headB.next
            lenB -= 1
        
        # Find intersection
        while headA and headB:
            if headA == headB:
                return headA
            headA = headA.next
            headB = headB.next
        
        return None
    
    # Using set (less optimal)
    def getIntersectionNode_set(headA, headB):
        """
        Using set to track visited nodes.
        
        Time: O(n + m), Space: O(n)
        """
        visited = set()
        
        # Add all nodes from listA to set
        current = headA
        while current:
            visited.add(current)
            current = current.next
        
        # Check if any node from listB is in set
        current = headB
        while current:
            if current in visited:
                return current
            current = current.next
        
        return None
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Pointer Switching**: When one pointer reaches end, switch to other list
    - **Equal Distance**: After switching, both pointers travel same total distance
    - **Meeting Point**: Pointers meet at intersection or both reach None
    - **Length Alignment**: Alternative approach aligns pointers by length difference
    
    **Two-Pointer Intuition:**
    ```
    If intersection exists:
    ptrA travels: lenA + lenB - common
    ptrB travels: lenB + lenA - common
    They meet at intersection after same distance
    
    If no intersection:
    Both reach None after traveling lenA + lenB
    ```

---

## Problem 9: Convert Binary Number to Integer

**Difficulty**: 🟢 Easy  
**Pattern**: Single Pass Traversal  
**Time**: O(n), **Space**: O(1)

=== "Problem"

    Given `head` which is a reference node to a singly-linked list. The value of each node in the linked list is either `0` or `1`. The linked list holds the binary representation of a number.

    Return the decimal value of the number in the linked list.

    **Example 1:**
    ```
    Input: head = [1,0,1]
    Output: 5
    Explanation: (101) in base 2 = (5) in base 10
    ```

    **Example 2:**
    ```
    Input: head = [0]
    Output: 0
    ```

=== "Solution"

    ```python
    def getDecimalValue(head):
        """
        Convert binary linked list to decimal.
        
        Time: O(n) - single pass through list
        Space: O(1) - only one variable
        """
        result = 0
        current = head
        
        while current:
            # Shift left and add current bit
            result = result * 2 + current.val
            current = current.next
        
        return result
    
    # Using bit operations
    def getDecimalValue_bitwise(head):
        """Using bit shift operations"""
        result = 0
        current = head
        
        while current:
            # Left shift and OR with current bit
            result = (result << 1) | current.val
            current = current.next
        
        return result
    
    # Two-pass approach (find length first)
    def getDecimalValue_two_pass(head):
        """Two-pass: count bits then calculate"""
        # First pass: count bits
        length = 0
        current = head
        while current:
            length += 1
            current = current.next
        
        # Second pass: calculate decimal value
        result = 0
        current = head
        power = length - 1
        
        while current:
            if current.val == 1:
                result += 2 ** power
            power -= 1
            current = current.next
        
        return result
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Binary to Decimal**: Each position represents power of 2
    - **Left-to-Right Processing**: Most significant bit first
    - **Accumulation**: `result = result * 2 + current_bit`
    - **Bit Operations**: Left shift is equivalent to multiplication by 2
    
    **Mathematical Approach:**
    ```
    For binary 101:
    result = 0
    result = 0 * 2 + 1 = 1
    result = 1 * 2 + 0 = 2  
    result = 2 * 2 + 1 = 5
    ```

---

## Problem 10: Delete Node in Linked List

**Difficulty**: 🟢 Easy  
**Pattern**: Value Copying  
**Time**: O(1), **Space**: O(1)

=== "Problem"

    Write a function to delete a node in a singly-linked list. You will **not** be given access to the `head` of the list, instead you will be given access to **the node to be deleted** directly.

    It is **guaranteed** that the node to be deleted is **not a tail node** in the list.

    **Example 1:**
    ```
    Input: head = [4,5,1,9], node = 5
    Output: [4,1,9]
    Explanation: After deleting node with value 5
    ```

=== "Solution"

    ```python
    def deleteNode(node):
        """
        Delete node without access to head.
        
        Time: O(1) - constant time operation
        Space: O(1) - no extra space needed
        """
        # Copy value from next node
        node.val = node.next.val
        
        # Skip the next node (effectively deleting it)
        node.next = node.next.next
    
    # Alternative with explicit next node handling
    def deleteNode_explicit(node):
        """More explicit version showing the swap"""
        if not node or not node.next:
            return  # Cannot delete tail or None node
        
        next_node = node.next
        
        # Copy data from next node
        node.val = next_node.val
        node.next = next_node.next
        
        # Optional: clean up next_node
        next_node.next = None
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Cannot Delete Directly**: No access to previous node
    - **Value Copying**: Copy next node's value to current node
    - **Link Adjustment**: Skip the next node to "delete" it
    - **Guaranteed Constraint**: Node is not tail (always has next)
    
    **Algorithm Logic:**
    1. Copy value from next node to current node
    2. Point current node to next.next (skip next node)
    3. The "next node" is effectively deleted
    
    **Why This Works:**
    We're not actually deleting the given node, we're transforming it to look like the next node and then removing the next node.

---

## Problem 11: Merge Nodes in Between Zeros

**Difficulty**: 🟢 Easy  
**Pattern**: Segmented Processing  
**Time**: O(n), **Space**: O(1)

=== "Problem"

    You are given the head of a linked list, which contains a series of integers **separated** by `0`'s. The **beginning** and **end** of the linked list will have `Node.val == 0`.

    For **every** two consecutive `0`'s, **merge** all the nodes lying in between them into a **single** node whose value is the **sum** of all the merged nodes. The modified list should not contain any `0`'s.

    Return the head of the modified linked list.

    **Example 1:**
    ```
    Input: head = [0,3,1,0,4,5,2,0]
    Output: [4,11]
    Explanation: 
    - Sum between first two zeros: 3 + 1 = 4
    - Sum between next two zeros: 4 + 5 + 2 = 11
    ```

=== "Solution"

    ```python
    def mergeNodes(head):
        """
        Merge nodes between zeros.
        
        Time: O(n) - single pass through list
        Space: O(1) - modify list in place
        """
        dummy = ListNode(0)
        result_tail = dummy
        
        current = head.next  # Skip first zero
        
        while current:
            if current.val == 0:
                # Found zero, move to next segment
                current = current.next
            else:
                # Start new sum segment
                sum_val = 0
                
                # Sum all values until next zero
                while current and current.val != 0:
                    sum_val += current.val
                    current = current.next
                
                # Create new node with sum
                result_tail.next = ListNode(sum_val)
                result_tail = result_tail.next
        
        return dummy.next
    
    # In-place modification approach
    def mergeNodes_inplace(head):
        """Modify existing nodes in place"""
        current = head.next  # Skip first zero
        
        while current:
            sum_val = 0
            
            # Sum until next zero
            while current.val != 0:
                sum_val += current.val
                current = current.next
            
            # Reuse current zero node for sum
            current.val = sum_val
            
            # Skip to next non-zero or end
            next_segment = current.next
            while next_segment and next_segment.val == 0:
                next_segment = next_segment.next
            
            current.next = next_segment
            current = next_segment
        
        return head.next  # Skip original first zero
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Segment Processing**: Process nodes between zero markers
    - **Accumulation**: Sum values within each segment
    - **List Construction**: Build new list or modify in place
    - **Zero Handling**: Zeros act as delimiters, not data
    
    **Processing Pattern:**
    1. Skip initial zero
    2. For each segment: accumulate sum until next zero
    3. Create node with sum value
    4. Continue until end of list

---

## Problem 12: Remove Duplicates from Sorted List II (Easy Version)

**Difficulty**: 🟢 Easy  
**Pattern**: Conditional Node Removal  
**Time**: O(n), **Space**: O(1)

=== "Problem"

    Given the head of a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list. Return the linked list sorted as well.

    **Note**: This is a simplified version focusing on basic duplicate removal patterns.

    **Example 1:**
    ```
    Input: head = [1,2,3,3,4,4,5]
    Output: [1,2,5]
    ```

    **Example 2:**
    ```
    Input: head = [1,1,1,2,3]
    Output: [2,3]
    ```

=== "Solution"

    ```python
    def deleteDuplicates(head):
        """
        Remove all nodes with duplicate values.
        
        Time: O(n) - single pass through list
        Space: O(1) - only pointers used
        """
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        current = head
        
        while current:
            # Check if current node has duplicates
            if current.next and current.val == current.next.val:
                # Skip all nodes with same value
                val = current.val
                while current and current.val == val:
                    current = current.next
                prev.next = current
            else:
                # No duplicates, move prev pointer
                prev = current
                current = current.next
        
        return dummy.next
    
    # Two-pass approach for clarity
    def deleteDuplicates_two_pass(head):
        """Two-pass: mark duplicates then remove"""
        if not head:
            return None
        
        # First pass: mark duplicate values
        duplicate_vals = set()
        current = head
        
        while current and current.next:
            if current.val == current.next.val:
                duplicate_vals.add(current.val)
            current = current.next
        
        # Second pass: remove nodes with duplicate values
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        current = head
        
        while current:
            if current.val in duplicate_vals:
                prev.next = current.next
            else:
                prev = current
            current = current.next
        
        return dummy.next
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Complete Removal**: Remove ALL instances of duplicate values
    - **Dummy Node**: Handles case where head needs removal
    - **Value Tracking**: Identify which values appear multiple times
    - **Conditional Advancement**: Only advance prev when keeping node
    
    **Duplicate Detection Logic:**
    ```python
    if current.next and current.val == current.next.val:
        # Found duplicate, skip all instances
        val = current.val
        while current and current.val == val:
            current = current.next
    ```

---

## Problem 13: Find All Numbers Disappeared in Array (Linked List Version)

**Difficulty**: 🟢 Easy  
**Pattern**: Cycle Detection Variant  
**Time**: O(n), **Space**: O(1)

=== "Problem"

    Given a linked list where each node contains an integer from 1 to n (where n is the length of the list), and each integer appears exactly once except for one missing number, find the missing number.

    **Example 1:**
    ```
    Input: head = [4,3,2,7,8,2,3,1] (missing 5,6)
    Output: [5,6]
    ```

=== "Solution"

    ```python
    def findDisappearedNumbers(head):
        """
        Find missing numbers using linked list traversal.
        
        Time: O(n) - single pass
        Space: O(1) - if modifying in place, O(n) for result
        """
        if not head:
            return []
        
        # Convert to array for easier processing
        nums = []
        current = head
        while current:
            nums.append(current.val)
            current = current.next
        
        n = len(nums)
        
        # Mark present numbers by making them negative
        for num in nums:
            index = abs(num) - 1
            if nums[index] > 0:
                nums[index] = -nums[index]
        
        # Find missing numbers
        missing = []
        for i in range(n):
            if nums[i] > 0:
                missing.append(i + 1)
        
        return missing
    
    # Using set approach
    def findDisappearedNumbers_set(head):
        """Using set to track present numbers"""
        if not head:
            return []
        
        # Collect all numbers and count length
        present = set()
        length = 0
        current = head
        
        while current:
            present.add(current.val)
            length += 1
            current = current.next
        
        # Find missing numbers
        missing = []
        for i in range(1, length + 1):
            if i not in present:
                missing.append(i)
        
        return missing
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Range Property**: Numbers are in range [1, n]
    - **Array Indexing**: Use values as indices to mark presence
    - **Sign Marking**: Use negative values to mark visited indices
    - **Missing Detection**: Positive values indicate missing numbers
    
    **Algorithm Steps:**
    1. Convert linked list to array
    2. For each number, mark its index as visited (negative)
    3. Indices with positive values correspond to missing numbers

---

## Problem 14: Split Linked List in Parts

**Difficulty**: 🟢 Easy  
**Pattern**: List Division  
**Time**: O(n), **Space**: O(1)

=== "Problem"

    Given the head of a singly linked list and an integer `k`, split the linked list into `k` consecutive linked list parts.

    The length of each part should be as equal as possible: no two parts should have a size differing by more than one. This may lead to some parts being null.

    **Example 1:**
    ```
    Input: head = [1,2,3], k = 5
    Output: [[1],[2],[3],[],[]]
    ```

    **Example 2:**
    ```
    Input: head = [1,2,3,4,5,6,7,8,9,10], k = 3
    Output: [[1,2,3,4],[5,6,7],[8,9,10]]
    ```

=== "Solution"

    ```python
    def splitListToParts(head, k):
        """
        Split linked list into k parts.
        
        Time: O(n) - traverse list twice
        Space: O(1) - only pointers (result array not counted)
        """
        # Count total length
        length = 0
        current = head
        while current:
            length += 1
            current = current.next
        
        # Calculate part sizes
        part_size = length // k
        extra_parts = length % k
        
        result = []
        current = head
        
        for i in range(k):
            # Determine size of current part
            current_part_size = part_size + (1 if i < extra_parts else 0)
            
            if current_part_size == 0:
                result.append(None)
                continue
            
            # Mark start of current part
            part_head = current
            
            # Move to end of current part
            for j in range(current_part_size - 1):
                if current:
                    current = current.next
            
            # Split the list
            if current:
                next_part = current.next
                current.next = None
                current = next_part
            
            result.append(part_head)
        
        return result
    
    # Alternative with helper function
    def splitListToParts_helper(head, k):
        """Using helper function for cleaner code"""
        def get_length(node):
            length = 0
            while node:
                length += 1
                node = node.next
            return length
        
        def split_at(node, size):
            """Split list after 'size' nodes, return head and remaining"""
            if size <= 0:
                return None, node
            
            head = node
            for _ in range(size - 1):
                if node:
                    node = node.next
            
            if node:
                remaining = node.next
                node.next = None
                return head, remaining
            
            return head, None
        
        length = get_length(head)
        part_size = length // k
        extra = length % k
        
        result = []
        current = head
        
        for i in range(k):
            size = part_size + (1 if i < extra else 0)
            part_head, current = split_at(current, size)
            result.append(part_head)
        
        return result
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Even Distribution**: Distribute nodes as evenly as possible
    - **Extra Nodes**: First `length % k` parts get one extra node
    - **List Splitting**: Break connections to separate parts
    - **Null Parts**: When k > length, some parts will be null
    
    **Size Calculation:**
    ```python
    base_size = length // k
    extra_parts = length % k
    
    # First 'extra_parts' get size: base_size + 1
    # Remaining parts get size: base_size
    ```

---

## Problem 15: Next Greater Node in Linked List

**Difficulty**: 🟢 Easy  
**Pattern**: Stack-based Next Greater Element  
**Time**: O(n), **Space**: O(n)

=== "Problem"

    You are given the head of a linked list with integer values. For each node in the linked list, find the value of the next greater node. That is, for each node, find the value of the first node that is greater than this node's value and comes after it in the list.

    Return an array of integers answer, where `answer[i]` is the next greater element for the ith node (**1-indexed**). If the ith node does not have a next greater element, set `answer[i] = 0`.

    **Example 1:**
    ```
    Input: head = [2,1,5]
    Output: [5,5,0]
    ```

    **Example 2:**
    ```
    Input: head = [1,7,5,1,9,2,5,1]
    Output: [7,9,9,9,0,5,0,0]
    ```

=== "Solution"

    ```python
    def nextLargerNodes(head):
        """
        Find next greater element for each node.
        
        Time: O(n) - each node pushed and popped at most once
        Space: O(n) - stack and result array
        """
        # Convert to array first
        values = []
        current = head
        while current:
            values.append(current.val)
            current = current.next
        
        n = len(values)
        result = [0] * n
        stack = []  # Stack stores indices
        
        for i in range(n):
            # Pop elements smaller than current
            while stack and values[stack[-1]] < values[i]:
                index = stack.pop()
                result[index] = values[i]
            
            stack.append(i)
        
        return result
    
    # Single-pass approach without converting to array
    def nextLargerNodes_single_pass(head):
        """Single pass without array conversion"""
        result = []
        stack = []  # Stack stores (value, index) pairs
        index = 0
        current = head
        
        while current:
            # Extend result array
            result.append(0)
            
            # Pop smaller elements and update their results
            while stack and stack[-1][0] < current.val:
                _, idx = stack.pop()
                result[idx] = current.val
            
            stack.append((current.val, index))
            index += 1
            current = current.next
        
        return result
    
    # Brute force approach for comparison
    def nextLargerNodes_brute(head):
        """
        Brute force: for each node, scan rest of list.
        
        Time: O(n²), Space: O(n)
        """
        values = []
        current = head
        while current:
            values.append(current.val)
            current = current.next
        
        result = []
        n = len(values)
        
        for i in range(n):
            next_greater = 0
            for j in range(i + 1, n):
                if values[j] > values[i]:
                    next_greater = values[j]
                    break
            result.append(next_greater)
        
        return result
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Monotonic Stack**: Stack maintains decreasing order of values
    - **Next Greater Element**: Classic stack-based pattern
    - **Index Tracking**: Stack stores indices to update result array
    - **Single Pass**: Each element pushed and popped at most once
    
    **Stack Algorithm:**
    1. For each element, pop smaller elements from stack
    2. Set popped elements' next greater to current element
    3. Push current element onto stack
    4. Elements remaining in stack have no next greater element
    
    **Time Complexity Analysis:**
    Although there's a nested while loop, each element is pushed and popped at most once, making the overall complexity O(n).

---

## 📝 Summary

### Core Linked List Patterns

| **Pattern** | **Technique** | **Use Cases** |
|-------------|---------------|---------------|
| **Two Pointers** | Slow/Fast pointers | Middle finding, cycle detection |
| **Dummy Node** | Extra node before head | Simplifies head removal cases |
| **Pointer Manipulation** | Prev/Current tracking | Reversal, removal operations |
| **Value Copying** | Copy from next node | Delete without head access |
| **Stack-based** | Monotonic stack | Next greater element problems |

### Common Two-Pointer Techniques

```python
# Find middle
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next

# Detect cycle  
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:
        return True

# Find intersection
ptrA, ptrB = headA, headB
while ptrA != ptrB:
    ptrA = headB if ptrA is None else ptrA.next
    ptrB = headA if ptrB is None else ptrB.next
```

### Essential Operations

1. **Traversal**: Visit each node sequentially
2. **Reversal**: Change direction of links
3. **Merging**: Combine two sorted lists
4. **Splitting**: Divide list into parts
5. **Cycle Detection**: Find loops in list
6. **Removal**: Delete specific nodes or values

### Time Complexities

- **Single Pass**: O(n) - most operations
- **Two Pass**: O(n) - count then process
- **Nested Operations**: O(n²) - brute force approaches
- **Stack-based**: O(n) - amortized for monotonic stack

### Problem-Solving Strategy

1. **Identify Pattern**: What type of operation is needed?
2. **Choose Technique**: Two pointers, dummy node, or direct manipulation?
3. **Handle Edge Cases**: Empty list, single node, all same values
4. **Optimize Space**: Can we modify in-place?
5. **Verify Logic**: Test with small examples

---

## 🎯 Next Steps

- **[Medium Linked List Problems](medium-problems.md)** - More complex pointer manipulations
- **[Hard Linked List Problems](hard-problems.md)** - Advanced algorithms and optimizations
- **[Trees](../../trees/index.md)** - Apply similar concepts to tree structures

Master these fundamental patterns before progressing to more complex linked list algorithms!
