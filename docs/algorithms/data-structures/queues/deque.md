# Double-ended Queue (Deque)

A double-ended queue, often abbreviated as "deque" (pronounced "deck"), is a linear collection that supports element insertion and removal at both ends. The deque is a versatile data structure that combines the features of stacks and queues, making it ideal for a wide range of applications.

## Overview

A deque allows elements to be added or removed from either the front or the back, providing maximum flexibility for insertion and removal operations. This makes deques suitable for implementing both LIFO (stack) and FIFO (queue) behaviors, as well as more complex algorithms that require insertion and removal from both ends.

## Basic Operations

A deque typically supports the following core operations:

1. **insertFront()**: Add an element to the front of the deque
2. **insertRear()**: Add an element to the back of the deque
3. **deleteFront()**: Remove an element from the front of the deque
4. **deleteRear()**: Remove an element from the back of the deque
5. **getFront()**: Get the element at the front without removing it
6. **getRear()**: Get the element at the rear without removing it
7. **isEmpty()**: Check if the deque is empty
8. **isFull()** (for bounded implementations): Check if the deque is full
9. **size()**: Get the number of elements in the deque

## Implementation Approaches

### 1. Array-based Implementation

A circular array provides an efficient array-based implementation of a deque:

```python
class ArrayDeque:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.array = [None] * capacity
        self.front = 0  # Index of the front element
        self.rear = 0   # Index just past the last element
        self.size = 0   # Current size of the deque
    
    def is_empty(self):
        return self.size == 0
    
    def is_full(self):
        return self.size == self.capacity
    
    def insert_front(self, item):
        if self.is_full():
            raise Exception("Deque is full")
        
        # Move front pointer backward (circular)
        self.front = (self.front - 1) % self.capacity
        self.array[self.front] = item
        self.size += 1
    
    def insert_rear(self, item):
        if self.is_full():
            raise Exception("Deque is full")
        
        self.array[self.rear] = item
        # Move rear pointer forward (circular)
        self.rear = (self.rear + 1) % self.capacity
        self.size += 1
    
    def delete_front(self):
        if self.is_empty():
            raise Exception("Deque is empty")
        
        item = self.array[self.front]
        # Move front pointer forward (circular)
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return item
    
    def delete_rear(self):
        if self.is_empty():
            raise Exception("Deque is empty")
        
        # Move rear pointer backward (circular)
        self.rear = (self.rear - 1) % self.capacity
        item = self.array[self.rear]
        self.size -= 1
        return item
    
    def get_front(self):
        if self.is_empty():
            raise Exception("Deque is empty")
        return self.array[self.front]
    
    def get_rear(self):
        if self.is_empty():
            raise Exception("Deque is empty")
        return self.array[(self.rear - 1) % self.capacity]
```

**Time Complexity**: All operations are O(1)

**Space Complexity**: O(n) where n is the capacity

### 2. Doubly Linked List Implementation

A doubly linked list naturally supports insertions and deletions at both ends:

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class LinkedListDeque:
    def __init__(self):
        self.front = None
        self.rear = None
        self.size = 0
    
    def is_empty(self):
        return self.front is None
    
    def insert_front(self, item):
        new_node = Node(item)
        
        if self.is_empty():
            self.front = self.rear = new_node
        else:
            new_node.next = self.front
            self.front.prev = new_node
            self.front = new_node
        
        self.size += 1
    
    def insert_rear(self, item):
        new_node = Node(item)
        
        if self.is_empty():
            self.front = self.rear = new_node
        else:
            new_node.prev = self.rear
            self.rear.next = new_node
            self.rear = new_node
        
        self.size += 1
    
    def delete_front(self):
        if self.is_empty():
            raise Exception("Deque is empty")
        
        item = self.front.data
        
        if self.front == self.rear:
            self.front = self.rear = None
        else:
            self.front = self.front.next
            self.front.prev = None
        
        self.size -= 1
        return item
    
    def delete_rear(self):
        if self.is_empty():
            raise Exception("Deque is empty")
        
        item = self.rear.data
        
        if self.front == self.rear:
            self.front = self.rear = None
        else:
            self.rear = self.rear.prev
            self.rear.next = None
        
        self.size -= 1
        return item
    
    def get_front(self):
        if self.is_empty():
            raise Exception("Deque is empty")
        return self.front.data
    
    def get_rear(self):
        if self.is_empty():
            raise Exception("Deque is empty")
        return self.rear.data
```

**Time Complexity**: All operations are O(1)

**Space Complexity**: O(n) where n is the number of elements

### 3. Dynamic Array Implementation

A dynamic array implementation that resizes when necessary:

```python
class DynamicArrayDeque:
    def __init__(self):
        self.array = []
    
    def is_empty(self):
        return len(self.array) == 0
    
    def size(self):
        return len(self.array)
    
    def insert_front(self, item):
        self.array.insert(0, item)  # Insert at beginning
    
    def insert_rear(self, item):
        self.array.append(item)  # Insert at end
    
    def delete_front(self):
        if self.is_empty():
            raise Exception("Deque is empty")
        return self.array.pop(0)  # Remove from beginning
    
    def delete_rear(self):
        if self.is_empty():
            raise Exception("Deque is empty")
        return self.array.pop()  # Remove from end
    
    def get_front(self):
        if self.is_empty():
            raise Exception("Deque is empty")
        return self.array[0]
    
    def get_rear(self):
        if self.is_empty():
            raise Exception("Deque is empty")
        return self.array[-1]
```

**Time Complexity**: 
- insert_rear(), delete_rear(), get_front(), get_rear() are O(1)
- insert_front() and delete_front() are O(n) due to shifting elements

**Space Complexity**: O(n) where n is the number of elements

## Applications of Deques

### 1. Sliding Window Problems

Deques are excellent for sliding window algorithm implementations:

```python
def max_sliding_window(nums, k):
    result = []
    deque = []  # Will store indices of elements
    
    for i, num in enumerate(nums):
        # Remove elements outside the window
        while deque and deque[0] < i - k + 1:
            deque.pop(0)
        
        # Remove smaller elements that won't be maximum
        while deque and nums[deque[-1]] < num:
            deque.pop()
        
        deque.append(i)
        
        # Add the maximum to result if we've processed k elements
        if i >= k - 1:
            result.append(nums[deque[0]])
    
    return result
```

### 2. Palindrome Checking

Deques make palindrome checking straightforward:

```python
def is_palindrome(s):
    deque = []
    
    # Clean the string and add to deque
    for char in s.lower():
        if char.isalnum():
            deque.append(char)
    
    while len(deque) > 1:
        if deque.pop(0) != deque.pop():  # Compare front and rear
            return False
    
    return True
```

### 3. Work Stealing Schedulers

Many parallel processing frameworks use deques for task scheduling:

```python
class WorkStealingScheduler:
    def __init__(self, num_threads):
        self.thread_deques = [[] for _ in range(num_threads)]
    
    def add_task(self, thread_id, task):
        self.thread_deques[thread_id].append(task)
    
    def get_task(self, thread_id):
        # Try to get task from own deque
        if self.thread_deques[thread_id]:
            return self.thread_deques[thread_id].pop()
        
        # Try to steal task from another thread
        for i in range(len(self.thread_deques)):
            if i != thread_id and self.thread_deques[i]:
                return self.thread_deques[i].pop(0)  # Steal from front
        
        return None  # No tasks available
```

### 4. Undo/Redo Functionality

Deques can be used to implement undo/redo stacks:

```python
class TextEditor:
    def __init__(self):
        self.text = ""
        self.undo_stack = []
        self.redo_stack = []
    
    def insert(self, char):
        self.undo_stack.append(self.text)
        self.text += char
        self.redo_stack = []  # Clear redo stack on new action
    
    def delete(self):
        if not self.text:
            return
        
        self.undo_stack.append(self.text)
        self.text = self.text[:-1]
        self.redo_stack = []  # Clear redo stack on new action
    
    def undo(self):
        if not self.undo_stack:
            return
        
        self.redo_stack.append(self.text)
        self.text = self.undo_stack.pop()
    
    def redo(self):
        if not self.redo_stack:
            return
        
        self.undo_stack.append(self.text)
        self.text = self.redo_stack.pop()
```

## Language-Specific Implementations

### Python

Python's `collections.deque` is a built-in implementation:

```python
from collections import deque

# Create a new deque
d = deque()

# Add elements to both ends
d.appendleft(1)  # [1]
d.append(2)      # [1, 2]
d.appendleft(0)  # [0, 1, 2]
d.append(3)      # [0, 1, 2, 3]

# Remove elements from both ends
front = d.popleft()  # front = 0, d = [1, 2, 3]
rear = d.pop()       # rear = 3, d = [1, 2]

# Other operations
d.extend([3, 4, 5])     # [1, 2, 3, 4, 5]
d.extendleft([0, -1])   # [-1, 0, 1, 2, 3, 4, 5]
d.rotate(1)             # [5, -1, 0, 1, 2, 3, 4]
d.rotate(-2)            # [0, 1, 2, 3, 4, 5, -1]
```

### Java

Java's `Deque` interface has several implementations:

```java
import java.util.ArrayDeque;
import java.util.Deque;

public class DequeExample {
    public static void main(String[] args) {
        // ArrayDeque is a common implementation
        Deque<Integer> deque = new ArrayDeque<>();
        
        // Add elements to both ends
        deque.addFirst(1);   // [1]
        deque.addLast(2);    // [1, 2]
        deque.addFirst(0);   // [0, 1, 2]
        deque.addLast(3);    // [0, 1, 2, 3]
        
        // Remove elements from both ends
        int front = deque.removeFirst();  // front = 0, deque = [1, 2, 3]
        int rear = deque.removeLast();    // rear = 3, deque = [1, 2]
        
        // Peek without removing
        Integer firstElem = deque.peekFirst();  // 1
        Integer lastElem = deque.peekLast();    // 2
        
        // Other operations
        deque.push(5);   // Add to front, [5, 1, 2]
        int top = deque.pop();  // Remove from front, top = 5, deque = [1, 2]
    }
}
```

## Performance Considerations

1. **Array-based vs. Linked List**: Array-based implementations offer better cache locality, while linked lists avoid the need for resizing.

2. **Dynamic Resizing**: When using array-based implementations, choose an initial capacity that minimizes resizing operations.

3. **Front Operations in Arrays**: Be cautious with front insertions and deletions in array-based deques, as they may be O(n) if not implemented with circular arrays.

4. **Memory Overhead**: Linked list implementations have higher memory overhead due to pointer storage.

## Common Pitfalls

1. **Confusing Methods**: In some languages, deque implementations have multiple method names for the same operation (e.g., Java's `addFirst()` vs. `offerFirst()`).

2. **Array Index Calculations**: When implementing circular array deques, be careful with index calculations to avoid off-by-one errors.

3. **Empty/Full Conditions**: With circular array implementations, distinguishing between empty and full states requires careful design.

## Comparison with Other Data Structures

| Operation | Deque | Queue | Stack | Vector/ArrayList |
|-----------|-------|-------|-------|-----------------|
| Insert at front | O(1) | O(n) | N/A | O(n) |
| Insert at back | O(1) | O(1) | O(1) | Amortized O(1) |
| Delete from front | O(1) | O(1) | N/A | O(n) |
| Delete from back | O(1) | O(n) | O(1) | O(1) |
| Access front | O(1) | O(1) | O(1) | O(1) |
| Access back | O(1) | O(n) | O(1) | O(1) |

## Conclusion

The deque is one of the most versatile data structures, combining the capabilities of stacks and queues while offering efficient operations at both ends. Its flexibility makes it suitable for a wide range of applications, from algorithm implementation to system design. Understanding deques and their implementations provides an excellent foundation for solving many complex programming problems efficiently.
