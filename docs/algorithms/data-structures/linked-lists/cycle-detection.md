# Cycle Detection in Linked Lists

Cycle detection is a crucial operation in linked lists, identifying if a linked list contains a cycle or loop. This is an important problem in computer science with applications in memory leak detection, deadlock detection, and graph algorithms.

## Overview

A cycle in a linked list occurs when a node's next pointer points back to a previously visited node, creating an infinite loop when traversing the list. Detecting such cycles is essential for preventing infinite loops in algorithms that process linked lists.

## Floyd's Cycle-Finding Algorithm (Tortoise and Hare)

The most efficient and widely used cycle detection algorithm is Floyd's Cycle-Finding Algorithm, also known as the "Tortoise and Hare" algorithm:

```python
def has_cycle(self):
    if self.head is None:
        return False
    
    # Initialize slow and fast pointers
    slow = self.head  # Tortoise
    fast = self.head  # Hare
    
    # Move slow by one step and fast by two steps
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        
        # If slow and fast meet, there's a cycle
        if slow == fast:
            return True
    
    # If we reach the end of the list, there's no cycle
    return False
```

**Time Complexity**: O(n) - In the worst case, slow pointer will traverse at most n nodes
**Space Complexity**: O(1) - We only use two pointers regardless of list size

## Finding the Start of the Cycle

Once we detect a cycle, we often need to find where it begins:

```python
def find_cycle_start(self):
    if self.head is None:
        return None
    
    # First, detect if there's a cycle
    slow = fast = self.head
    has_cycle = False
    
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            has_cycle = True
            break
    
    # If no cycle, return None
    if not has_cycle:
        return None
    
    # Find the start of the cycle
    # Reset slow to head, keep fast at meeting point
    slow = self.head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    # Both pointers now point to the start of the cycle
    return slow
```

**Time Complexity**: O(n) - We might need to traverse the entire list
**Space Complexity**: O(1) - We only use a constant amount of extra space

## Measuring Cycle Length

We can also determine the length of a cycle:

```python
def cycle_length(self):
    if self.head is None:
        return 0
    
    # First, detect if there's a cycle
    slow = fast = self.head
    has_cycle = False
    
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            has_cycle = True
            break
    
    # If no cycle, return 0
    if not has_cycle:
        return 0
    
    # Measure cycle length
    length = 1
    fast = fast.next
    
    while fast != slow:
        fast = fast.next
        length += 1
    
    return length
```

**Time Complexity**: O(n) - We might need to traverse the entire list
**Space Complexity**: O(1) - We only use a constant amount of extra space

## Brent's Algorithm

An alternative to Floyd's algorithm is Brent's cycle-finding algorithm, which can be faster in practice:

```python
def has_cycle_brent(self):
    if self.head is None:
        return False
    
    # Initialize pointers and values
    hare = tortoise = self.head
    power = lambda_length = 1
    
    # Main loop
    while True:
        # Move hare ahead
        for _ in range(lambda_length):
            if hare.next is None:
                return False  # No cycle
            hare = hare.next
            
            # Check if hare meets tortoise
            if tortoise == hare:
                return True  # Cycle detected
        
        # Increase power of 2
        power *= 2
        lambda_length = 0
        tortoise = hare
        
        # Move hare ahead by at most 'power' steps or until it meets tortoise
        for _ in range(power):
            if hare.next is None:
                return False  # No cycle
            hare = hare.next
            lambda_length += 1
            
            # Check if hare meets tortoise
            if tortoise == hare:
                return True  # Cycle detected
```

**Time Complexity**: O(μ + λ) where μ is the start of the cycle and λ is the length of the cycle
**Space Complexity**: O(1) - We only use a constant amount of extra space

## Hashmap-Based Detection

A simpler but less space-efficient approach uses a hashmap to track visited nodes:

```python
def has_cycle_hashmap(self):
    if self.head is None:
        return False
    
    # Set to store visited nodes
    visited = set()
    current = self.head
    
    while current:
        # If node already in set, we found a cycle
        if current in visited:
            return True
        
        # Add current node to visited set
        visited.add(current)
        current = current.next
    
    # If we reach the end, no cycle
    return False
```

**Time Complexity**: O(n) - We traverse each node once
**Space Complexity**: O(n) - We might need to store all nodes in the hash set

## Applications of Cycle Detection

1. **Memory Leak Detection**: Detecting cyclic references in garbage collection.

2. **Deadlock Detection**: Identifying circular wait conditions in resource allocation.

3. **Functional Program Evaluation**: Detecting infinite recursion.

4. **Computer Networks**: Finding routing loops in network topologies.

5. **Compiler Optimization**: Detecting infinite loops in programs.

## Common Interview Questions

1. **Detect if a linked list has a cycle**
2. **Find the starting point of a cycle**
3. **Calculate the length of a cycle**
4. **Remove a cycle from a linked list**

## Removing a Cycle

Once we detect a cycle, we might need to remove it:

```python
def remove_cycle(self):
    if self.head is None:
        return
    
    # Find the start of the cycle
    cycle_start = self.find_cycle_start()
    
    if cycle_start is None:
        return  # No cycle found
    
    # Find the last node in the cycle
    current = cycle_start
    while current.next != cycle_start:
        current = current.next
    
    # Break the cycle
    current.next = None
    
    # Update tail
    self.tail = current
```

**Time Complexity**: O(n) - We might need to traverse the entire list
**Space Complexity**: O(1) - We only use a constant amount of extra space

## Detecting Cycles in Memory-Constrained Environments

In extremely memory-constrained environments, we might use destructive methods temporarily:

```python
def has_cycle_destructive(self):
    if self.head is None:
        return False
    
    current = self.head
    
    while current:
        if current.visited:  # We assume a 'visited' flag exists
            return True
        
        current.visited = True
        current = current.next
    
    # Reset visited flags
    current = self.head
    while current:
        current.visited = False
        current = current.next
    
    return False
```

**Time Complexity**: O(n) - We traverse each node at most twice
**Space Complexity**: O(1) - We only use a constant amount of extra space, but modify the nodes

## Conclusion

Cycle detection is a fundamental linked list operation that showcases the power of pointer manipulation algorithms. Floyd's algorithm, with its elegant use of slow and fast pointers, demonstrates how sophisticated algorithms can solve complex problems efficiently without requiring additional space. Understanding cycle detection is essential for any developer working with linked data structures and is a common focus in technical interviews.
