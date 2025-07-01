# Stacks and Queues

Two fundamental linear data structures that serve as building blocks for many algorithms and applications.

## Stack

### Overview

A **Stack** is a Last-In-First-Out (LIFO) data structure where elements are added and removed from the same end, called the "top". Think of it like a stack of plates - you can only add or remove plates from the top.

### Key Characteristics

- **LIFO Principle**: Last element added is the first to be removed
- **Single-ended**: Operations occur at one end (top)
- **Dynamic Size**: Can grow and shrink during runtime
- **Memory Efficient**: Only stores elements, no extra pointers needed for basic implementation

### Real-World Applications

- **Function Calls**: Call stack in programming languages
- **Undo Operations**: Text editors, browsers (back button)
- **Expression Evaluation**: Parsing mathematical expressions
- **Backtracking**: Maze solving, game AI
- **Memory Management**: Stack frames in program execution

### Stack Implementation

**Array-based Implementation:**

```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """Add item to top of stack - O(1)"""
        self.items.append(item)
    
    def pop(self):
        """Remove and return top item - O(1)"""
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("pop from empty stack")
    
    def peek(self):
        """Return top item without removing - O(1)"""
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("peek from empty stack")
    
    def is_empty(self):
        """Check if stack is empty - O(1)"""
        return len(self.items) == 0
    
    def size(self):
        """Return number of items - O(1)"""
        return len(self.items)
    
    def __str__(self):
        return f"Stack({self.items})"
```

**Linked List Implementation:**

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedStack:
    def __init__(self):
        self.head = None
        self._size = 0
    
    def push(self, item):
        """Add item to top - O(1)"""
        new_node = ListNode(item)
        new_node.next = self.head
        self.head = new_node
        self._size += 1
    
    def pop(self):
        """Remove and return top item - O(1)"""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        val = self.head.val
        self.head = self.head.next
        self._size -= 1
        return val
    
    def peek(self):
        """Return top item without removing - O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self.head.val
    
    def is_empty(self):
        return self.head is None
    
    def size(self):
        return self._size
```

### Stack Algorithms & Patterns

#### 1. Monotonic Stack
Used for finding next/previous greater/smaller elements:

```python
def next_greater_element(nums):
    """Find next greater element for each number"""
    result = [-1] * len(nums)
    stack = []  # stores indices
    
    for i, num in enumerate(nums):
        while stack and nums[stack[-1]] < num:
            result[stack.pop()] = num
        stack.append(i)
    
    return result

# Example: [2,1,2,4,3,1] -> [4,2,4,-1,-1,-1]
```

#### 2. Expression Evaluation

```python
def evaluate_postfix(expression):
    """Evaluate postfix notation expression"""
    stack = []
    operators = {'+', '-', '*', '/'}
    
    for token in expression.split():
        if token in operators:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
            elif token == '*':
                result = a * b
            elif token == '/':
                result = a / b
            stack.append(result)
        else:
            stack.append(int(token))
    
    return stack[0]

# Example: "2 1 + 3 *" -> 9
```

#### 3. Balanced Parentheses

```python
def is_valid_parentheses(s):
    """Check if parentheses are balanced"""
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return not stack

# Example: "({[]})" -> True, "({[})" -> False
```

### Common Stack Problems

| Problem | Difficulty | Key Concept |
|---------|-----------|-------------|
| Valid Parentheses | Easy | Pattern matching |
| Min Stack | Easy | Auxiliary stack |
| Evaluate RPN | Medium | Postfix evaluation |
| Daily Temperatures | Medium | Monotonic stack |
| Largest Rectangle in Histogram | Hard | Monotonic stack |
| Trapping Rain Water | Hard | Two pointers + stack |
| Basic Calculator | Hard | Expression parsing |
| Remove K Digits | Medium | Greedy + monotonic stack |

---

## Queue

### Queue Overview

A **Queue** is a First-In-First-Out (FIFO) data structure where elements are added at one end (rear/back) and removed from the other end (front). Think of it like a line of people waiting - first person in line is the first to be served.

### Queue Key Characteristics

- **FIFO Principle**: First element added is the first to be removed
- **Two-ended**: Elements added at rear, removed from front
- **Dynamic Size**: Can grow and shrink during runtime
- **Fair Processing**: Maintains order of arrival

### Queue Applications

- **Task Scheduling**: Operating system process queues
- **Breadth-First Search**: Graph and tree traversal
- **Buffer Management**: IO operations, network requests
- **Print Queues**: Document printing systems
- **Call Centers**: Customer service wait times

### Queue Implementation

**Array-based Implementation (Circular Queue):**

```python
class Queue:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.items = [None] * capacity
        self.front = 0
        self.rear = -1
        self.size = 0
    
    def enqueue(self, item):
        """Add item to rear of queue - O(1)"""
        if self.size == self.capacity:
            raise OverflowError("Queue is full")
        self.rear = (self.rear + 1) % self.capacity
        self.items[self.rear] = item
        self.size += 1
    
    def dequeue(self):
        """Remove and return front item - O(1)"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        item = self.items[self.front]
        self.items[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return item
    
    def peek(self):
        """Return front item without removing - O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self.items[self.front]
    
    def is_empty(self):
        return self.size == 0
    
    def is_full(self):
        return self.size == self.capacity
    
    def __len__(self):
        return self.size
```

**Deque Implementation (Recommended):**

```python
from collections import deque

class SimpleQueue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        """Add item to rear - O(1)"""
        self.items.append(item)
    
    def dequeue(self):
        """Remove and return front item - O(1)"""
        if not self.is_empty():
            return self.items.popleft()
        raise IndexError("dequeue from empty queue")
    
    def peek(self):
        """Return front item without removing - O(1)"""
        if not self.is_empty():
            return self.items[0]
        raise IndexError("peek from empty queue")
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
```

**Linked List Implementation:**

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedQueue:
    def __init__(self):
        self.front = None
        self.rear = None
        self._size = 0
    
    def enqueue(self, item):
        """Add item to rear - O(1)"""
        new_node = ListNode(item)
        if self.rear:
            self.rear.next = new_node
        else:
            self.front = new_node
        self.rear = new_node
        self._size += 1
    
    def dequeue(self):
        """Remove and return front item - O(1)"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        val = self.front.val
        self.front = self.front.next
        if not self.front:
            self.rear = None
        self._size -= 1
        return val
    
    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self.front.val
    
    def is_empty(self):
        return self.front is None
    
    def size(self):
        return self._size
```

### Queue Variants

#### 1. Priority Queue

```python
import heapq

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0
    
    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1
    
    def pop(self):
        return heapq.heappop(self._queue)[-1]
    
    def is_empty(self):
        return len(self._queue) == 0
```

#### 2. Deque (Double-ended Queue)

```python
from collections import deque

# Built-in deque supports both stack and queue operations
dq = deque()

# Queue operations
dq.append(1)      # enqueue to rear
dq.popleft()      # dequeue from front

# Stack operations  
dq.append(1)      # push to top
dq.pop()          # pop from top

# Additional operations
dq.appendleft(0)  # add to front
dq.pop()          # remove from rear
```

### Queue Algorithms & Patterns

#### 1. Breadth-First Search (BFS)

```python
def bfs_tree(root):
    """Level-order traversal of binary tree"""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        result.append(node.val)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return result
```

#### 2. Sliding Window Maximum

```python
def sliding_window_maximum(nums, k):
    """Find maximum in each sliding window"""
    from collections import deque
    
    dq = deque()  # stores indices
    result = []
    
    for i, num in enumerate(nums):
        # Remove indices outside window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove smaller elements (not useful)
        while dq and nums[dq[-1]] < num:
            dq.pop()
        
        dq.append(i)
        
        # Add to result if window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

#### 3. Level Order Traversal

```python
def level_order(root):
    """Return level-by-level traversal"""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

### Common Queue Problems

| Problem | Difficulty | Key Concept |
|---------|-----------|-------------|
| Implement Queue using Stacks | Easy | Stack simulation |
| Design Circular Queue | Medium | Circular array |
| Sliding Window Maximum | Hard | Monotonic deque |
| Level Order Traversal | Medium | BFS with queue |
| Rotting Oranges | Medium | Multi-source BFS |
| Design Hit Counter | Medium | Sliding window |
| Moving Average | Easy | Sliding window |
| Shortest Path in Binary Matrix | Medium | BFS pathfinding |

---

## Comparison: Stack vs Queue

| Aspect | Stack | Queue |
|--------|-------|-------|
| **Principle** | LIFO (Last-In-First-Out) | FIFO (First-In-First-Out) |
| **Access Points** | One end (top) | Two ends (front & rear) |
| **Main Operations** | push(), pop(), peek() | enqueue(), dequeue(), peek() |
| **Use Cases** | Recursion, undo, parsing | Scheduling, BFS, buffering |
| **Memory Layout** | Contiguous (array) or linked | Circular array or linked |

## Time & Space Complexities

### Stack Operations

| Operation | Array Implementation | Linked List Implementation |
|-----------|---------------------|---------------------------|
| Push | O(1) amortized | O(1) |
| Pop | O(1) | O(1) |
| Peek/Top | O(1) | O(1) |
| Search | O(n) | O(n) |
| Space | O(n) | O(n) |

### Queue Operations

| Operation | Array (Circular) | Linked List | Deque |
|-----------|------------------|-------------|-------|
| Enqueue | O(1) | O(1) | O(1) |
| Dequeue | O(1) | O(1) | O(1) |
| Peek/Front | O(1) | O(1) | O(1) |
| Search | O(n) | O(n) | O(n) |
| Space | O(n) | O(n) | O(n) |

## Advanced Topics

### Memory Management

**Stack Overflow:**

```python
def prevent_stack_overflow(depth_limit=1000):
    """Iterative approach to prevent stack overflow"""
    stack = [(initial_state, 0)]  # (state, depth)
    
    while stack:
        state, depth = stack.pop()
        
        if depth > depth_limit:
            raise RecursionError("Maximum depth exceeded")
        
        # Process state...
        for next_state in get_next_states(state):
            stack.append((next_state, depth + 1))
```

**Queue Buffer Management:**

```python
class BoundedQueue:
    """Queue with maximum capacity"""
    def __init__(self, max_size):
        self.max_size = max_size
        self.items = deque()
    
    def enqueue(self, item):
        if len(self.items) >= self.max_size:
            self.items.popleft()  # Remove oldest
        self.items.append(item)
    
    def dequeue(self):
        if self.items:
            return self.items.popleft()
        return None
```

## Practice Problems

### Beginner Level

- [ ] Implement Stack using Arrays
- [ ] Implement Queue using Stacks  
- [ ] Valid Parentheses
- [ ] Baseball Game
- [ ] Design Circular Queue

### Intermediate Level

- [ ] Min Stack / Max Stack
- [ ] Daily Temperatures
- [ ] Next Greater Element
- [ ] Level Order Traversal
- [ ] Sliding Window Maximum

### Advanced Level

- [ ] Largest Rectangle in Histogram
- [ ] Trapping Rain Water
- [ ] Basic Calculator I & II
- [ ] Shortest Subarray with Sum at Least K
- [ ] Constrained Subsequence Sum

## Tips for Interviews

### Stack Problems

1. **Look for LIFO patterns** - nested structures, backtracking
2. **Monotonic stacks** - finding next/previous greater/smaller elements
3. **Expression evaluation** - infix to postfix conversion
4. **Matching problems** - balanced parentheses, tag matching

### Queue Problems

1. **Look for FIFO patterns** - level-by-level processing
2. **BFS applications** - shortest path, level traversal
3. **Sliding window** - maintain window state with deque
4. **Producer-consumer** - buffering, rate limiting

### Common Patterns

- **Stack for DFS** - depth-first exploration
- **Queue for BFS** - breadth-first exploration  
- **Deque for sliding window** - efficient window operations
- **Priority queue** - when order matters based on priority

---

## Resources for Further Learning

- [LeetCode Stack Problems](https://leetcode.com/tag/stack/)
- [LeetCode Queue Problems](https://leetcode.com/tag/queue/)
- [Visualizing Stack and Queue Operations](https://visualgo.net/en/list)
- [Stack Overflow: When to use Stack vs Queue](https://stackoverflow.com/questions/2074970/)

Remember: **Practice is key!** Start with basic implementations, then move to problem-solving patterns.
