# Linked Lists - Medium Problems

## 🎯 Learning Objectives

Master intermediate linked list techniques and patterns:

- Advanced pointer manipulation and traversal
- Multi-pass algorithms and optimization techniques
- Complex node operations and reconstructions
- Pattern recognition for linked list problems
- Time/space complexity optimization strategies

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Add Two Numbers | Digit Addition | Medium | O(max(m,n)) | O(max(m,n)) |
    | 2 | Remove Nth Node From End | Two Pointers | Medium | O(n) | O(1) |
    | 3 | Swap Nodes in Pairs | Pointer Manipulation | Medium | O(n) | O(1) |
    | 4 | Rotate List | Two Pointers + Cycle | Medium | O(n) | O(1) |
    | 5 | Reverse Nodes in k-Group | Recursive Reversal | Medium | O(n) | O(k) |
    | 6 | Partition List | Two Pointers | Medium | O(n) | O(1) |
    | 7 | Remove Duplicates from Sorted List II | Two Pointers | Medium | O(n) | O(1) |
    | 8 | Reorder List | Find Middle + Reverse | Medium | O(n) | O(1) |
    | 9 | Linked List Cycle II | Floyd's Algorithm | Medium | O(n) | O(1) |
    | 10 | Copy List with Random Pointer | Hash Map/Weaving | Medium | O(n) | O(n) |
    | 11 | Sort List | Merge Sort | Medium | O(n log n) | O(log n) |
    | 12 | Insertion Sort List | Insertion Sort | Medium | O(n²) | O(1) |
    | 13 | Split Linked List in Parts | Array Division | Medium | O(n+k) | O(k) |
    | 14 | Odd Even Linked List | Two Pointers | Medium | O(n) | O(1) |
    | 15 | Add Two Numbers II | Stack/Reverse | Medium | O(max(m,n)) | O(max(m,n)) |

=== "🎯 Core Patterns"

    **🔗 Advanced Pointer Techniques:**
    - Two-pointer traversal with different speeds
    - Multi-pass algorithms for optimization
    - Dummy node usage for edge case handling
    
    **🔄 List Transformation:**
    - In-place reversal of sublists
    - Partitioning and rearrangement
    - Merging and splitting operations
    
    **⚡ Optimization Strategies:**
    - Single-pass solutions where possible
    - Space-efficient algorithms
    - Recursive vs iterative approaches
    
    **🎯 Problem Recognition:**
    - Cycle detection and handling
    - Arithmetic operations on lists
    - Structural modifications and reconstructions

=== "💡 Solutions"

    === "Add Two Numbers"
        ```python
        def addTwoNumbers(l1, l2):
            """Add two numbers represented as linked lists."""
            dummy = ListNode(0)
            current = dummy
            carry = 0
            
            while l1 or l2 or carry:
                val1 = l1.val if l1 else 0
                val2 = l2.val if l2 else 0
                
                total = val1 + val2 + carry
                carry = total // 10
                current.next = ListNode(total % 10)
                current = current.next
                
                if l1: l1 = l1.next
                if l2: l2 = l2.next
            
            return dummy.next
        ```
    
    === "Remove Nth Node From End"
        ```python
        def removeNthFromEnd(head, n):
            """Remove nth node from end using two pointers."""
            dummy = ListNode(0)
            dummy.next = head
            first = second = dummy
            
            # Move first pointer n+1 steps ahead
            for _ in range(n + 1):
                first = first.next
            
            # Move both pointers until first reaches end
            while first:
                first = first.next
                second = second.next
            
            # Remove the nth node
            second.next = second.next.next
            return dummy.next
        ```
    
    === "Swap Nodes in Pairs"
        ```python
        def swapPairs(head):
            """Swap every two adjacent nodes."""
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
                
                # Move to next pair
                prev = first
            
            return dummy.next
        ```
    
    === "Rotate List"
        ```python
        def rotateRight(head, k):
            """Rotate list to the right by k places."""
            if not head or not head.next or k == 0:
                return head
            
            # Find length and make it circular
            length = 1
            current = head
            while current.next:
                current = current.next
                length += 1
            
            current.next = head  # Make circular
            
            # Find new tail (length - k % length - 1)
            k = k % length
            steps = length - k
            
            new_tail = head
            for _ in range(steps - 1):
                new_tail = new_tail.next
            
            new_head = new_tail.next
            new_tail.next = None
            
            return new_head
        ```
    
    === "Reverse Nodes in k-Group"
        ```python
        def reverseKGroup(head, k):
            """Reverse every k nodes in the list."""
            # Check if we have k nodes
            current = head
            for _ in range(k):
                if not current:
                    return head
                current = current.next
            
            # Reverse k nodes
            prev = None
            current = head
            for _ in range(k):
                next_node = current.next
                current.next = prev
                prev = current
                current = next_node
            
            # Connect with the rest
            head.next = self.reverseKGroup(current, k)
            return prev
        ```
    
    === "Partition List"
        ```python
        def partition(head, x):
            """Partition list around value x."""
            less_head = ListNode(0)
            greater_head = ListNode(0)
            less = less_head
            greater = greater_head
            
            current = head
            while current:
                if current.val < x:
                    less.next = current
                    less = less.next
                else:
                    greater.next = current
                    greater = greater.next
                current = current.next
            
            # Connect the two parts
            greater.next = None
            less.next = greater_head.next
            
            return less_head.next
        ```
    
    === "Remove Duplicates from Sorted List II"
        ```python
        def deleteDuplicates(head):
            """Remove all nodes that have duplicates."""
            dummy = ListNode(0)
            dummy.next = head
            prev = dummy
            
            while head:
                if head.next and head.val == head.next.val:
                    # Skip all duplicates
                    while head.next and head.val == head.next.val:
                        head = head.next
                    prev.next = head.next
                else:
                    prev = prev.next
                head = head.next
            
            return dummy.next
        ```
    
    === "Reorder List"
        ```python
        def reorderList(head):
            """Reorder list: L0→L1→…→Ln-1→Ln to L0→Ln→L1→Ln-1→…"""
            if not head or not head.next:
                return
            
            # Find middle
            slow = fast = head
            while fast.next and fast.next.next:
                slow = slow.next
                fast = fast.next.next
            
            # Reverse second half
            second = slow.next
            slow.next = None
            
            prev = None
            while second:
                next_node = second.next
                second.next = prev
                prev = second
                second = next_node
            
            # Merge two halves
            first = head
            second = prev
            while second:
                first_next = first.next
                second_next = second.next
                
                first.next = second
                second.next = first_next
                
                first = first_next
                second = second_next
        ```
    
    === "Linked List Cycle II"
        ```python
        def detectCycle(head):
            """Find the start of the cycle."""
            if not head or not head.next:
                return None
            
            # Phase 1: Detect cycle
            slow = fast = head
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
                if slow == fast:
                    break
            else:
                return None
            
            # Phase 2: Find cycle start
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            
            return slow
        ```
    
    === "Copy List with Random Pointer"
        ```python
        def copyRandomList(head):
            """Deep copy list with random pointers."""
            if not head:
                return None
            
            # Create copy nodes
            current = head
            while current:
                copy = Node(current.val)
                copy.next = current.next
                current.next = copy
                current = copy.next
            
            # Set random pointers
            current = head
            while current:
                if current.random:
                    current.next.random = current.random.next
                current = current.next.next
            
            # Separate lists
            dummy = Node(0)
            copy_current = dummy
            current = head
            
            while current:
                copy_current.next = current.next
                current.next = current.next.next
                copy_current = copy_current.next
                current = current.next
            
            return dummy.next
        ```
    
    === "Sort List"
        ```python
        def sortList(head):
            """Sort linked list using merge sort."""
            if not head or not head.next:
                return head
            
            # Split list in half
            mid = self.getMid(head)
            left = head
            right = mid.next
            mid.next = None
            
            # Recursively sort both halves
            left = self.sortList(left)
            right = self.sortList(right)
            
            # Merge sorted halves
            return self.merge(left, right)
        
        def getMid(self, head):
            slow = fast = head
            while fast.next and fast.next.next:
                slow = slow.next
                fast = fast.next.next
            return slow
        
        def merge(self, left, right):
            dummy = ListNode(0)
            current = dummy
            
            while left and right:
                if left.val <= right.val:
                    current.next = left
                    left = left.next
                else:
                    current.next = right
                    right = right.next
                current = current.next
            
            current.next = left or right
            return dummy.next
        ```
    
    === "Insertion Sort List"
        ```python
        def insertionSortList(head):
            """Sort list using insertion sort."""
            if not head or not head.next:
                return head
            
            dummy = ListNode(0)
            current = head
            
            while current:
                next_node = current.next
                
                # Find position to insert
                prev = dummy
                while prev.next and prev.next.val < current.val:
                    prev = prev.next
                
                # Insert current
                current.next = prev.next
                prev.next = current
                
                current = next_node
            
            return dummy.next
        ```
    
    === "Split Linked List in Parts"
        ```python
        def splitListToParts(head, k):
            """Split list into k parts."""
            # Find length
            length = 0
            current = head
            while current:
                length += 1
                current = current.next
            
            # Calculate part sizes
            part_size = length // k
            extra = length % k
            
            result = []
            current = head
            
            for i in range(k):
                part_head = current
                current_size = part_size + (1 if i < extra else 0)
                
                # Move to end of current part
                for j in range(current_size - 1):
                    if current:
                        current = current.next
                
                # Break connection
                if current:
                    next_part = current.next
                    current.next = None
                    current = next_part
                
                result.append(part_head)
            
            return result
        ```
    
    === "Odd Even Linked List"
        ```python
        def oddEvenList(head):
            """Group odd and even positioned nodes."""
            if not head or not head.next:
                return head
            
            odd = head
            even = head.next
            even_head = even
            
            while even and even.next:
                odd.next = even.next
                odd = odd.next
                even.next = odd.next
                even = even.next
            
            odd.next = even_head
            return head
        ```
    
    === "Add Two Numbers II"
        ```python
        def addTwoNumbers(l1, l2):
            """Add two numbers with most significant digit first."""
            stack1, stack2 = [], []
            
            # Push all numbers to stacks
            while l1:
                stack1.append(l1.val)
                l1 = l1.next
            
            while l2:
                stack2.append(l2.val)
                l2 = l2.next
            
            # Add numbers
            carry = 0
            result = None
            
            while stack1 or stack2 or carry:
                val1 = stack1.pop() if stack1 else 0
                val2 = stack2.pop() if stack2 else 0
                
                total = val1 + val2 + carry
                carry = total // 10
                
                # Create new node at front
                node = ListNode(total % 10)
                node.next = result
                result = node
            
            return result
        ```

=== "📊 Complexity Analysis"

    **⚡ Time Complexity Patterns:**
    - **Single Pass**: O(n) for most traversal problems
    - **Two Pass**: O(n) for problems requiring length calculation
    - **Sorting**: O(n log n) for merge sort, O(n²) for insertion sort
    - **Nested Operations**: O(n²) for problems with inner loops
    
    **🔧 Space Complexity Patterns:**
    - **In-place**: O(1) for pointer manipulation
    - **Auxiliary**: O(n) for hash maps or stacks
    - **Recursive**: O(depth) for recursive solutions
    - **Output**: O(k) where k is the number of parts/groups
    
    **🎯 Optimization Strategies:**
    - Use dummy nodes to handle edge cases
    - Prefer iterative over recursive when possible
    - Combine operations in single pass when feasible
    - Use two-pointer technique for efficiency

=== "🚀 Advanced Tips"

    **💡 Problem-Solving Approach:**
    1. **Identify Pattern**: Recognize the core operation needed
    2. **Choose Technique**: Select appropriate pointer strategy
    3. **Handle Edge Cases**: Consider empty lists, single nodes
    4. **Optimize**: Look for single-pass solutions
    5. **Test**: Verify with various input sizes and patterns
    
    **🔍 Common Pitfalls:**
    - Forgetting to handle null pointers
    - Off-by-one errors in counting
    - Memory leaks in languages without garbage collection
    - Incorrect cycle detection logic
    
    **🏆 Best Practices:**
    - Use dummy nodes for simpler code
    - Draw diagrams for complex pointer operations
    - Test with edge cases (empty, single node, cycles)
    - Consider both iterative and recursive approaches

## 📝 Summary

Medium linked list problems focus on:

- **Advanced Pointer Manipulation** for complex operations
- **Two-Pointer Techniques** for efficient traversal
- **List Transformation** through reversal and rearrangement
- **Cycle Detection** and handling
- **Optimization Strategies** for time and space efficiency

These problems prepare you for senior-level interviews and complex system implementations requiring sophisticated linked list operations.
