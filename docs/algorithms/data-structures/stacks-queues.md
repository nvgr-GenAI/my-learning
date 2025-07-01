# Stacks and Queues

## Stack

### Overview

A Stack is a Last-In-First-Out (LIFO) data structure.

### Implementation

```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        return None
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
```

### Common Stack Problems

1. **Valid Parentheses**
2. **Min Stack**
3. **Evaluate Reverse Polish Notation**
4. **Daily Temperatures**
5. **Largest Rectangle in Histogram**

## Queue

### Overview

A Queue is a First-In-First-Out (FIFO) data structure.

### Implementation

```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()
        return None
    
    def front(self):
        if not self.is_empty():
            return self.items[0]
        return None
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
```

### Common Queue Problems

1. **Implement Queue using Stacks**
2. **Sliding Window Maximum**
3. **Level Order Traversal**
4. **Rotting Oranges**
5. **Design Circular Queue**

## Time Complexities

| Operation | Stack | Queue |
|-----------|-------|-------|
| Push/Enqueue | O(1) | O(1) |
| Pop/Dequeue  | O(1) | O(1) |
| Peek/Front   | O(1) | O(1) |
| Search       | O(n) | O(n) |

## Practice Problems

### Stack Problems

- [ ] Valid Parentheses
- [ ] Min Stack
- [ ] Baseball Game
- [ ] Next Greater Element I

### Queue Problems

- [ ] Implement Stack using Queues
- [ ] Design Circular Queue
- [ ] Recent Counter
