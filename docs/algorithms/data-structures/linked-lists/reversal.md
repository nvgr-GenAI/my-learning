# Linked List Reversal

Reversing a linked list is a fundamental operation that transforms the list so that the last node becomes the first node, the second-last becomes the second, and so on. This operation showcases pointer manipulation techniques and is frequently asked in coding interviews.

## Overview

Linked list reversal can be accomplished through iterative or recursive approaches. Both methods involve manipulating the next pointers to reverse the direction of the links.

## Iterative Reversal

The iterative approach uses three pointers to track the previous, current, and next nodes while traversing the list:

```python
def reverse_iterative(self):
    if self.head is None or self.head.next is None:
        return  # Empty or single node list
    
    prev = None
    current = self.head
    
    # Update tail pointer
    self.tail = self.head
    
    while current:
        # Store next before we overwrite current.next
        next_node = current.next
        
        # Reverse the link
        current.next = prev
        
        # Move pointers forward
        prev = current
        current = next_node
    
    # Update head to the new front (which was the last node)
    self.head = prev
```

**Time Complexity**: O(n) - We traverse each node exactly once
**Space Complexity**: O(1) - We only use a constant amount of extra space

## Recursive Reversal

The recursive approach uses the call stack to remember the order of nodes:

```python
def reverse_recursive(self):
    if self.head is None:
        return
    
    self.tail = self.head
    self.head = self._reverse_recursive_helper(self.head)

def _reverse_recursive_helper(self, node):
    # Base case: if this is the last node or an empty list
    if node is None or node.next is None:
        return node
    
    # Recursively reverse the rest of the list
    new_head = self._reverse_recursive_helper(node.next)
    
    # Make the next node point to current node
    node.next.next = node
    
    # Break the link from current to next node
    node.next = None
    
    # Return the new head
    return new_head
```

**Time Complexity**: O(n) - Each node is processed once
**Space Complexity**: O(n) - The recursion stack can grow to n levels deep

## Reversal in Specific Ranges

Sometimes we need to reverse only a section of the linked list, from position m to n:

```python
def reverse_between(self, m, n):
    if self.head is None:
        return
    
    # Create a dummy node to handle edge cases
    dummy = Node(0)
    dummy.next = self.head
    prev = dummy
    
    # Move prev to the node just before position m
    for _ in range(m - 1):
        if prev.next is None:
            return  # m is out of bounds
        prev = prev.next
    
    # Current will point to the node at position m
    current = prev.next
    
    # Reverse the sublist from position m to n
    for _ in range(n - m):
        if current.next is None:
            break  # n is out of bounds
        
        temp = current.next
        current.next = temp.next
        temp.next = prev.next
        prev.next = temp
    
    # Update head if m was 1
    self.head = dummy.next
    
    # Update tail if n was the last position
    if current.next is None:
        self.tail = current
```

**Time Complexity**: O(n) - We might need to traverse most of the list
**Space Complexity**: O(1) - We only use a constant amount of extra space

## Reversing a Doubly Linked List

For doubly linked lists, we need to update both next and prev pointers:

```python
def reverse_doubly_linked_list(self):
    if self.head is None:
        return
    
    current = self.head
    temp = None
    
    # Swap head and tail
    self.tail = self.head
    
    while current:
        # Store the previous pointer
        temp = current.prev
        
        # Swap next and prev pointers
        current.prev = current.next
        current.next = temp
        
        # Move to the next node (which is now at prev)
        current = current.prev
    
    # Update head to the last node
    if temp:
        self.head = temp.prev
```

**Time Complexity**: O(n) - We traverse each node exactly once
**Space Complexity**: O(1) - We only use a constant amount of extra space

## Applications of Linked List Reversal

1. **Palindrome Detection**: Reverse the second half of a linked list and compare with the first half to check for palindromes.

2. **K-Group Reversal**: In problems like "Reverse Nodes in k-Group" (a common interview question), you need to reverse every k nodes in the list.

3. **Alternate Merge**: When merging two lists in an alternate fashion, reversing one list can simplify the implementation.

4. **Cycle Detection**: Reversal techniques are sometimes used in cycle detection algorithms.

## Common Mistakes

1. **Not Updating Head/Tail**: After reversal, remember to update the head and tail pointers.

2. **Forgetting Edge Cases**: Empty lists, single-node lists, or reversing portions that go out of bounds need special handling.

3. **Breaking the List**: Incorrectly updating pointers can lead to lost nodes or cycles in the list.

## Testing Strategies

Always test your reversal function with:

1. Empty list
2. Single-node list
3. Two-node list
4. Longer lists with odd and even number of nodes
5. When reversing portions, test boundary conditions

## Interview Tips

1. **Think Aloud**: Explain your approach before coding - iterative solutions are often preferred for their space efficiency.

2. **Draw It Out**: Use diagrams to visualize pointer changes.

3. **Test Your Solution**: Walk through an example to verify correctness.

## Conclusion

Linked list reversal is not just a common interview question but a fundamental technique that demonstrates your understanding of pointer manipulation. Mastering both iterative and recursive approaches will strengthen your linked list operations toolkit and prepare you for more complex linked list problems.
