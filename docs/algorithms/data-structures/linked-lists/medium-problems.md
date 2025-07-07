# Linked Lists: Medium Problems

## 🚀 Intermediate Linked List Challenges

Build upon basic linked list operations to solve more complex problems involving multiple pointers, list manipulation, and advanced patterns.

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Add Two Numbers | Linked List Traversal | Medium | O(max(m,n)) | O(max(m,n)) |
    | 2 | Remove Nth Node From End | Two Pointers | Medium | O(n) | O(1) |
    | 3 | Swap Nodes in Pairs | Iterative/Recursive | Medium | O(n) | O(1) |
    | 4 | Rotate List | Two Pointers | Medium | O(n) | O(1) |
    | 5 | Reverse Nodes in k-Group | Iterative Reversal | Medium | O(n) | O(1) |
    | 6 | Partition List | Two Pointers | Medium | O(n) | O(1) |
    | 7 | Remove Duplicates from Sorted List II | Two Pointers | Medium | O(n) | O(1) |
    | 8 | Reorder List | Fast/Slow + Reverse | Medium | O(n) | O(1) |
    | 9 | Odd Even Linked List | Two Pointers | Medium | O(n) | O(1) |
    | 10 | Split Linked List in Parts | Traversal | Medium | O(n) | O(k) |
    | 11 | Next Greater Node | Monotonic Stack | Medium | O(n) | O(n) |
    | 12 | Delete Node in Linked List | Direct Manipulation | Medium | O(1) | O(1) |
    | 13 | Sort List | Merge Sort | Medium | O(n log n) | O(log n) |
    | 14 | Insertion Sort List | Insertion Sort | Medium | O(n²) | O(1) |
    | 15 | Copy List with Random Pointer | HashMap/Weaving | Medium | O(n) | O(n) |

=== "🎯 Interview Tips"

    **📝 Key Patterns:**
    
    - **Two Pointers**: Fast/slow for cycle detection, nth node removal
    - **Dummy Head**: Simplifies edge cases for list modifications
    - **Reversal**: In-place reversal for reordering problems
    - **Merge Patterns**: Combining multiple lists or parts
    - **Stack/Queue**: For maintaining order in complex operations
    
    **⚡ Problem-Solving Strategies:**
    
    - Draw the linked list structure before coding
    - Use dummy nodes to handle edge cases
    - Practice reversal patterns thoroughly
    - Consider both iterative and recursive approaches
    
    **🚫 Common Pitfalls:**
    
    - Not handling null pointers properly
    - Losing references during manipulation
    - Off-by-one errors in two-pointer problems
    - Not updating next pointers correctly

=== "📚 Study Plan"

    **Week 1: Basic Manipulation (Problems 1-5)**
    - Master two-pointer techniques
    - Practice list reversal patterns
    
    **Week 2: Advanced Patterns (Problems 6-10)**
    - Learn partitioning and reordering
    - Focus on in-place algorithms
    
    **Week 3: Complex Operations (Problems 11-15)**
    - Stack-based solutions
    - Sorting algorithms on linked lists

=== "Add Two Numbers"

    **Problem Statement:**
    You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

    **Example:**
    ```text
    Input: l1 = [2,4,3], l2 = [5,6,4]
    Output: [7,0,8]
    Explanation: 342 + 465 = 807
    ```

    **Solution:**
    ```python
    def addTwoNumbers(l1, l2):
        """
        Simulate addition with carry propagation.
        
        Time: O(max(m,n)) - traverse both lists
        Space: O(max(m,n)) - result list
        """
        dummy = ListNode(0)
        current = dummy
        carry = 0
        
        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            
            total = val1 + val2 + carry
            carry = total // 10
            digit = total % 10
            
            current.next = ListNode(digit)
            current = current.next
            
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        
        return dummy.next
    ```

    **Key Insights:**
    - Handle different length lists gracefully
    - Carry propagation is crucial for correctness
    - Dummy head simplifies list construction
    - Single pass through both lists

=== "Remove Nth Node From End"

    **Problem Statement:**
    Given the head of a linked list, remove the nth node from the end of the list and return its head.

    **Example:**
    ```text
    Input: head = [1,2,3,4,5], n = 2
    Output: [1,2,3,5]
    Explanation: Remove the 4th node from the end
    ```

    **Solution:**
    ```python
    def removeNthFromEnd(head, n):
        """
        Two-pointer technique with n+1 gap.
        
        Time: O(n) - single pass
        Space: O(1) - constant space
        """
        dummy = ListNode(0)
        dummy.next = head
        
        # Initialize pointers
        fast = slow = dummy
        
        # Move fast pointer n+1 steps ahead
        for _ in range(n + 1):
            fast = fast.next
        
        # Move both pointers until fast reaches end
        while fast:
            fast = fast.next
            slow = slow.next
        
        # Remove the nth node
        slow.next = slow.next.next
        
        return dummy.next
    ```

    **Key Insights:**
    - Two-pointer maintains exactly n+1 gap
    - Dummy head handles edge case of removing first node
    - Fast pointer reaches end when slow is at target
    - We need slow to point to node BEFORE target

=== "Swap Nodes in Pairs"

    **Problem Statement:**
    Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes.

    **Example:**
    ```text
    Input: head = [1,2,3,4]
    Output: [2,1,4,3]
    ```

    **Solution:**
    ```python
    def swapPairs(head):
        """
        Iterative approach with dummy head.
        
        Time: O(n) - single pass
        Space: O(1) - constant space
        """
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        
        while prev.next and prev.next.next:
            # Nodes to swap
            first = prev.next
            second = prev.next.next
            
            # Perform swap
            prev.next = second
            first.next = second.next
            second.next = first
            
            # Move to next pair
            prev = first
        
        return dummy.next
    ```

    **Alternative (Recursive):**
    ```python
    def swapPairs(head):
        if not head or not head.next:
            return head
        
        second = head.next
        head.next = swapPairs(second.next)
        second.next = head
        
        return second
    ```

    **Key Insights:**
    - Maintain three pointers: prev, first, second
    - Dummy head simplifies edge cases
    - Recursive approach is more intuitive
    - Be careful with the order of assignments

=== "Rotate List"

    **Problem Statement:**
    Given the head of a linked list, rotate the list to the right by k places.

    **Example:**
    ```text
    Input: head = [1,2,3,4,5], k = 2
    Output: [4,5,1,2,3]
    ```

    **Solution:**
    ```python
    def rotateRight(head, k):
        """
        Convert to circular, then break at correct position.
        
        Time: O(n) - single pass to find length + rotation
        Space: O(1) - constant space
        """
        if not head or not head.next or k == 0:
            return head
        
        # Find length and make circular
        length = 1
        tail = head
        while tail.next:
            tail = tail.next
            length += 1
        
        # Calculate actual rotation needed
        k = k % length
        if k == 0:
            return head
        
        # Find new tail (length - k - 1 steps from head)
        new_tail = head
        for _ in range(length - k - 1):
            new_tail = new_tail.next
        
        # Break the circle
        new_head = new_tail.next
        new_tail.next = None
        
        return new_head
    ```

    **Key Insights:**
    - Rotation is circular: k > length wraps around
    - Find the break point: (length - k) from start
    - Circular approach is more intuitive
    - Use k % length to handle large k values

=== "Reverse Nodes in k-Group"

    **Problem Statement:**
    Given the head of a linked list, reverse the nodes of the list k at a time, and return the modified list. If the number of nodes is not a multiple of k, the remaining nodes should stay as is.

    **Example:**
    ```text
    Input: head = [1,2,3,4,5], k = 3
    Output: [3,2,1,4,5]
    ```

    **Solution:**
    ```python
    def reverseKGroup(head, k):
        """
        Recursive solution with cleaner logic.
        
        Time: O(n)
        Space: O(n/k) - recursion stack
        """
        # Check if we have k nodes
        current = head
        for _ in range(k):
            if not current:
                return head
            current = current.next
        
        # Reverse current group
        prev = None
        curr = head
        for _ in range(k):
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        
        # Recursively reverse remaining groups
        head.next = reverseKGroup(curr, k)
        
        return prev
    ```

    **Key Insights:**
    - Only reverse if we have exactly k nodes
    - Use dummy head for consistent edge case handling
    - Recursive approach is more elegant
    - Each node processed exactly once

=== "Partition List"

    **Problem Statement:**
    Given the head of a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.

    **Example:**
    ```text
    Input: head = [1,4,3,2,5,2], x = 3
    Output: [1,2,2,4,3,5]
    ```

    **Solution:**
    ```python
    def partition(head, x):
        """
        Two separate lists approach.
        
        Time: O(n) - single pass
        Space: O(1) - constant space
        """
        # Create two dummy heads
        before_head = ListNode(0)
        after_head = ListNode(0)
        
        before = before_head
        after = after_head
        
        # Traverse original list
        current = head
        while current:
            if current.val < x:
                before.next = current
                before = before.next
            else:
                after.next = current
                after = after.next
            current = current.next
        
        # Connect the two parts
        after.next = None  # Important: terminate the list
        before.next = after_head.next
        
        return before_head.next
    ```

    **Key Insights:**
    - Maintain relative order within each partition
    - Two separate lists is cleaner than in-place
    - Don't forget to terminate the second list
    - Preserves original order within partitions

=== "Remove Duplicates from Sorted List II"

    **Problem Statement:**
    Given the head of a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list.

    **Example:**
    ```text
    Input: head = [1,2,3,3,4,4,5]
    Output: [1,2,5]
    ```

    **Solution:**
    ```python
    def deleteDuplicates(head):
        """
        Two-pointer approach with dummy head.
        
        Time: O(n) - single pass
        Space: O(1) - constant space
        """
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        
        while head:
            # Check if current node has duplicates
            if head.next and head.val == head.next.val:
                # Skip all nodes with same value
                while head.next and head.val == head.next.val:
                    head = head.next
                
                # Connect prev to node after duplicates
                prev.next = head.next
            else:
                # No duplicates, move prev pointer
                prev = prev.next
            
            head = head.next
        
        return dummy.next
    ```

    **Key Insights:**
    - Use dummy head to handle edge cases
    - Skip entire groups of duplicates
    - Maintain prev pointer to non-duplicate node
    - Only advance prev when no duplicates found

=== "Reorder List"

    **Problem Statement:**
    You are given the head of a singly linked-list. Reorder the list to be in the form: L₀ → Lₙ → L₁ → Lₙ₋₁ → L₂ → Lₙ₋₂ → ...

    **Example:**
    ```text
    Input: head = [1,2,3,4,5]
    Output: [1,5,2,4,3]
    ```

    **Solution:**
    ```python
    def reorderList(head):
        """
        Three-step process: find middle, reverse second half, merge.
        
        Time: O(n) - three passes
        Space: O(1) - constant space
        """
        if not head or not head.next:
            return
        
        # Step 1: Find the middle using fast/slow pointers
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        # Step 2: Reverse the second half
        def reverse_list(head):
            prev = None
            while head:
                next_node = head.next
                head.next = prev
                prev = head
                head = next_node
            return prev
        
        # Split the list and reverse second half
        second_half = slow.next
        slow.next = None
        second_half = reverse_list(second_half)
        
        # Step 3: Merge the two halves
        first_half = head
        while second_half:
            # Store next nodes
            first_next = first_half.next
            second_next = second_half.next
            
            # Reorder connections
            first_half.next = second_half
            second_half.next = first_next
            
            # Move to next nodes
            first_half = first_next
            second_half = second_next
    ```

    **Key Insights:**
    - Combine multiple linked list techniques
    - Three-step process: find middle, reverse, merge
    - In-place solution requires careful pointer management
    - Algorithm breakdown ensures correctness

=== "Odd Even Linked List"

    **Problem Statement:**
    Given the head of a singly linked list, group all the nodes with odd indices together followed by the nodes with even indices.

    **Example:**
    ```text
    Input: head = [1,2,3,4,5]
    Output: [1,3,5,2,4]
    ```

    **Solution:**
    ```python
    def oddEvenList(head):
        """
        Two-pointer approach maintaining odd and even chains.
        
        Time: O(n) - single pass
        Space: O(1) - constant space
        """
        if not head or not head.next:
            return head
        
        # Initialize pointers
        odd = head
        even = head.next
        even_head = even
        
        # Traverse and separate odd/even positioned nodes
        while even and even.next:
            # Link odd nodes
            odd.next = even.next
            odd = odd.next
            
            # Link even nodes
            even.next = odd.next
            even = even.next
        
        # Connect odd chain to even chain
        odd.next = even_head
        
        return head
    ```

    **Key Insights:**
    - Position-based grouping (1-indexed)
    - Maintain two separate chains
    - Connect odd chain to even chain at the end
    - Preserve original relative order within each group

=== "Split Linked List in Parts"

    **Problem Statement:**
    Given the head of a singly linked list and an integer k, split the linked list into k consecutive linked list parts.

    **Example:**
    ```text
    Input: head = [1,2,3], k = 5
    Output: [[1],[2],[3],[],[]]
    ```

    **Solution:**
    ```python
    def splitListToParts(head, k):
        """
        Calculate sizes and split accordingly.
        
        Time: O(n) - two passes
        Space: O(k) - result array
        """
        # Find length
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
            # Size of current part
            current_size = part_size + (1 if i < extra_parts else 0)
            
            if current_size == 0:
                result.append(None)
            else:
                result.append(current)
                # Move to end of current part
                for _ in range(current_size - 1):
                    current = current.next
                
                # Break the connection
                if current:
                    next_node = current.next
                    current.next = None
                    current = next_node
        
        return result
    ```

    **Key Insights:**
    - Calculate optimal part sizes
    - Distribute extra nodes evenly
    - Break connections properly
    - Handle empty parts for k > length

=== "Next Greater Node"

    **Problem Statement:**
    You are given the head of a linked list with integer values. For each node, find the value of the next greater node. If there is no such node, the answer is 0.

    **Example:**
    ```text
    Input: head = [2,1,5]
    Output: [5,5,0]
    ```

    **Solution:**
    ```python
    def nextLargerNodes(head):
        """
        Monotonic stack approach.
        
        Time: O(n) - each node processed once
        Space: O(n) - stack and result array
        """
        # Convert to array for easier indexing
        values = []
        current = head
        while current:
            values.append(current.val)
            current = current.next
        
        n = len(values)
        result = [0] * n
        stack = []  # Stack of indices
        
        for i in range(n):
            # Pop elements that are smaller than current
            while stack and values[stack[-1]] < values[i]:
                idx = stack.pop()
                result[idx] = values[i]
            
            stack.append(i)
        
        return result
    ```

    **Key Insights:**
    - Monotonic stack maintains decreasing order
    - Stack stores indices, not values
    - Each element pushed and popped at most once
    - Perfect for "next greater" type problems

=== "Delete Node in Linked List"

    **Problem Statement:**
    Delete a node in a linked list. You will not be given access to the head of the list, instead you will be given access to the node to be deleted directly.

    **Example:**
    ```text
    Input: head = [4,5,1,9], node = 5
    Output: [4,1,9]
    ```

    **Solution:**
    ```python
    def deleteNode(node):
        """
        Copy next node's value and delete next node.
        
        Time: O(1) - constant time
        Space: O(1) - constant space
        """
        # Copy the value of the next node
        node.val = node.next.val
        
        # Delete the next node
        node.next = node.next.next
    ```

    **Key Insights:**
    - Cannot actually delete the given node
    - Instead, overwrite with next node's data
    - Delete the next node to maintain list integrity
    - Only works if node is not the last node

=== "Sort List"

    **Problem Statement:**
    Given the head of a linked list, return the list after sorting it in ascending order.

    **Example:**
    ```text
    Input: head = [4,2,1,3]
    Output: [1,2,3,4]
    ```

    **Solution:**
    ```python
    def sortList(head):
        """
        Merge sort implementation for linked lists.
        
        Time: O(n log n) - optimal for comparison-based sorting
        Space: O(log n) - recursion stack
        """
        if not head or not head.next:
            return head
        
        # Find middle using fast/slow pointers
        def find_middle(head):
            slow = head
            fast = head
            prev = None
            
            while fast and fast.next:
                prev = slow
                slow = slow.next
                fast = fast.next.next
            
            # Split the list
            prev.next = None
            return slow
        
        # Merge two sorted lists
        def merge(l1, l2):
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
            
            # Append remaining nodes
            current.next = l1 if l1 else l2
            return dummy.next
        
        # Split the list
        middle = find_middle(head)
        
        # Recursively sort both halves
        left = sortList(head)
        right = sortList(middle)
        
        # Merge sorted halves
        return merge(left, right)
    ```

    **Key Insights:**
    - Merge sort is optimal for linked lists
    - Divide using fast/slow pointer technique
    - Bottom-up approach can achieve O(1) space
    - Stable sorting algorithm

=== "Insertion Sort List"

    **Problem Statement:**
    Given the head of a singly linked list, sort the list using insertion sort, and return the sorted list's head.

    **Example:**
    ```text
    Input: head = [4,2,1,3]
    Output: [1,2,3,4]
    ```

    **Solution:**
    ```python
    def insertionSortList(head):
        """
        Insertion sort implementation for linked lists.
        
        Time: O(n²) - worst case
        Space: O(1) - constant space
        """
        if not head or not head.next:
            return head
        
        dummy = ListNode(0)
        current = head
        
        while current:
            next_node = current.next
            
            # Find insertion position
            prev = dummy
            while prev.next and prev.next.val < current.val:
                prev = prev.next
            
            # Insert current node
            current.next = prev.next
            prev.next = current
            
            current = next_node
        
        return dummy.next
    ```

    **Key Insights:**
    - Build sorted portion from left to right
    - Find correct insertion position for each node
    - Use dummy head for cleaner implementation
    - O(n²) time but O(1) space

=== "Copy List with Random Pointer"

    **Problem Statement:**
    A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null. Return a deep copy of the list.

    **Example:**
    ```text
    Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
    Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]
    ```

    **Solution:**
    ```python
    def copyRandomList(head):
        """
        Two-pass approach using interweaving.
        
        Time: O(n) - three passes
        Space: O(1) - no extra space for mapping
        """
        if not head:
            return None
        
        # Step 1: Create copied nodes and interweave
        current = head
        while current:
            copied = Node(current.val)
            copied.next = current.next
            current.next = copied
            current = copied.next
        
        # Step 2: Assign random pointers
        current = head
        while current:
            if current.random:
                current.next.random = current.random.next
            current = current.next.next
        
        # Step 3: Separate the two lists
        dummy = Node(0)
        copied_current = dummy
        current = head
        
        while current:
            copied_current.next = current.next
            current.next = current.next.next
            
            copied_current = copied_current.next
            current = current.next
        
        return dummy.next
    ```

    **Alternative (HashMap):**
    ```python
    def copyRandomList(head):
        if not head:
            return None
        
        # Map original nodes to copied nodes
        node_map = {}
        
        # First pass: create all nodes
        current = head
        while current:
            node_map[current] = Node(current.val)
            current = current.next
        
        # Second pass: assign pointers
        current = head
        while current:
            if current.next:
                node_map[current].next = node_map[current.next]
            if current.random:
                node_map[current].random = node_map[current.random]
            current = current.next
        
        return node_map[head]
    ```

    **Key Insights:**
    - Deep copy requires new nodes with same structure
    - Random pointers create dependency challenges
    - Interweaving technique avoids extra space
    - HashMap approach is cleaner but uses O(n) space

## 📝 Summary

Medium linked list problems require:

- **Advanced Pointer Techniques**: Two pointers, fast/slow patterns
- **List Manipulation**: Reversal, partitioning, merging
- **Complex Algorithms**: Merge sort, cycle detection
- **Edge Case Handling**: Null pointers, empty lists
- **Space Optimization**: In-place algorithms when possible

These problems are crucial for:

- **Technical Interviews**: Common in FAANG companies
- **Algorithm Design**: Building blocks for complex systems
- **Data Structure Mastery**: Understanding linked list versatility
- **Problem-Solving Skills**: Breaking down complex operations

Practice these patterns to master linked list manipulations! 🚀
