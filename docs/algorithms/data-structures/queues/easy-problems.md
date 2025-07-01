# Queues: Easy Problems

## 🚀 Foundation Queue Challenges

Perfect for building your queue intuition and understanding core patterns.

---

## Problem 1: Implement Queue Using Stacks

**Difficulty:** Easy  
**Pattern:** Stack Simulation  
**Time:** O(1) amortized | **Space:** O(n)

### Problem Statement

Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (`push`, `peek`, `pop`, and `empty`).

### Solution

```python
class MyQueue:
    """
    Queue implementation using two stacks.
    
    Stack1: For enqueue operations
    Stack2: For dequeue operations (reversed order)
    """
    
    def __init__(self):
        self.stack1 = []  # For enqueue
        self.stack2 = []  # For dequeue
    
    def push(self, x):
        """Add element to back of queue."""
        self.stack1.append(x)
    
    def pop(self):
        """Remove element from front of queue."""
        self._move_to_stack2()
        return self.stack2.pop()
    
    def peek(self):
        """Get front element without removing."""
        self._move_to_stack2()
        return self.stack2[-1]
    
    def empty(self):
        """Check if queue is empty."""
        return len(self.stack1) == 0 and len(self.stack2) == 0
    
    def _move_to_stack2(self):
        """Move elements from stack1 to stack2 if stack2 is empty."""
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())

# Test the implementation
queue = MyQueue()
queue.push(1)
queue.push(2)
print(f"Front: {queue.peek()}")  # Output: 1
print(f"Pop: {queue.pop()}")    # Output: 1
print(f"Empty: {queue.empty()}")  # Output: False
```

### Explanation

1. **Two Stacks Strategy**: Use one stack for input, another for output
2. **Lazy Transfer**: Only move elements when output stack is empty
3. **Amortized O(1)**: Each element is moved at most once

---

## Problem 2: First Unique Character in String

**Difficulty:** Easy  
**Pattern:** Queue + Hash Map  
**Time:** O(n) | **Space:** O(1)

### Problem Statement

Given a string `s`, find the first non-repeating character and return its index. If it doesn't exist, return `-1`.

**Example:**

```
Input: s = "leetcode"
Output: 0

Input: s = "loveleetcode"  
Output: 2
```

### Solution

```python
from collections import Counter, deque

def first_uniq_char(s):
    """
    Find first unique character using counter and queue.
    
    Method 1: Simple counter approach
    """
    count = Counter(s)
    
    for i, char in enumerate(s):
        if count[char] == 1:
            return i
    
    return -1

def first_uniq_char_queue(s):
    """
    Method 2: Using queue for streaming approach.
    Useful when string is very large or streaming.
    """
    count = {}
    queue = deque()
    
    for i, char in enumerate(s):
        count[char] = count.get(char, 0) + 1
        queue.append((char, i))
        
        # Remove duplicates from front of queue
        while queue and count[queue[0][0]] > 1:
            queue.popleft()
    
    return queue[0][1] if queue else -1

# Test both approaches
test_strings = ["leetcode", "loveleetcode", "aabb"]
for s in test_strings:
    print(f"'{s}' -> Method 1: {first_uniq_char(s)}, Method 2: {first_uniq_char_queue(s)}")
```

---

## Problem 3: Number of Recent Calls

**Difficulty:** Easy  
**Pattern:** Sliding Window with Queue  
**Time:** O(1) per call | **Space:** O(n)

### Problem Statement

Write a class `RecentCounter` that counts the number of recent requests within a certain time frame. Implement the `ping(t)` method that adds a new request at time `t` and returns the number of requests in the range `[t-3000, t]`.

### Solution

```python
from collections import deque

class RecentCounter:
    """
    Count recent requests within 3000ms window using queue.
    
    Maintains a sliding window of valid timestamps.
    """
    
    def __init__(self):
        self.requests = deque()
    
    def ping(self, t):
        """
        Add new request and count requests in [t-3000, t] range.
        """
        # Add current request
        self.requests.append(t)
        
        # Remove requests outside the 3000ms window
        while self.requests and self.requests[0] < t - 3000:
            self.requests.popleft()
        
        return len(self.requests)

# Test the implementation
counter = RecentCounter()
print(counter.ping(1))     # 1
print(counter.ping(100))   # 2  
print(counter.ping(3001))  # 3
print(counter.ping(3002))  # 3 (1 is outside window)
```

### Explanation

1. **Sliding Window**: Maintain only requests within valid time range
2. **Queue Properties**: FIFO helps remove oldest invalid requests first
3. **Efficient Cleanup**: Remove expired requests before counting

---

## Problem 4: Design Circular Queue

**Difficulty:** Easy  
**Pattern:** Array-based Queue  
**Time:** O(1) | **Space:** O(k)

### Problem Statement

Design your implementation of a circular queue with fixed size `k`.

### Solution

```python
class MyCircularQueue:
    """
    Circular queue implementation using fixed-size array.
    
    Uses front and rear pointers with size tracking.
    """
    
    def __init__(self, k):
        self.size = 0
        self.max_size = k
        self.data = [0] * k
        self.front = 0
        self.rear = -1
    
    def enQueue(self, value):
        """Add element to rear of queue."""
        if self.isFull():
            return False
        
        self.rear = (self.rear + 1) % self.max_size
        self.data[self.rear] = value
        self.size += 1
        return True
    
    def deQueue(self):
        """Remove element from front of queue."""
        if self.isEmpty():
            return False
        
        self.front = (self.front + 1) % self.max_size
        self.size -= 1
        return True
    
    def Front(self):
        """Get front element."""
        return -1 if self.isEmpty() else self.data[self.front]
    
    def Rear(self):
        """Get rear element."""
        return -1 if self.isEmpty() else self.data[self.rear]
    
    def isEmpty(self):
        """Check if queue is empty."""
        return self.size == 0
    
    def isFull(self):
        """Check if queue is full."""
        return self.size == self.max_size

# Test the circular queue
cq = MyCircularQueue(3)
print(cq.enQueue(1))  # True
print(cq.enQueue(2))  # True  
print(cq.enQueue(3))  # True
print(cq.enQueue(4))  # False (full)
print(cq.Rear())      # 3
print(cq.isFull())    # True
print(cq.deQueue())   # True
print(cq.enQueue(4))  # True
print(cq.Rear())      # 4
```

---

## Problem 5: Moving Average from Data Stream

**Difficulty:** Easy  
**Pattern:** Sliding Window Queue  
**Time:** O(1) | **Space:** O(size)

### Problem Statement

Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.

### Solution

```python
from collections import deque

class MovingAverage:
    """
    Calculate moving average using queue to maintain window.
    """
    
    def __init__(self, size):
        self.size = size
        self.queue = deque()
        self.sum = 0
    
    def next(self, val):
        """Add new value and return current moving average."""
        self.queue.append(val)
        self.sum += val
        
        # Remove oldest value if window exceeds size
        if len(self.queue) > self.size:
            old_val = self.queue.popleft()
            self.sum -= old_val
        
        return self.sum / len(self.queue)

# Alternative: List-based approach (less efficient for large windows)
class MovingAverageList:
    def __init__(self, size):
        self.size = size
        self.values = []
    
    def next(self, val):
        self.values.append(val)
        if len(self.values) > self.size:
            self.values.pop(0)  # O(n) operation
        return sum(self.values) / len(self.values)

# Test both implementations
ma_queue = MovingAverage(3)
ma_list = MovingAverageList(3)

values = [1, 10, 3, 5]
for val in values:
    avg_q = ma_queue.next(val)
    avg_l = ma_list.next(val)
    print(f"Value: {val}, Queue avg: {avg_q:.2f}, List avg: {avg_l:.2f}")
```

---

## Problem 6: Binary Tree Level Order Traversal

**Difficulty:** Easy  
**Pattern:** BFS with Queue  
**Time:** O(n) | **Space:** O(w) where w is max width

### Problem Statement

Given the root of a binary tree, return the level order traversal of its nodes' values (i.e., from left to right, level by level).

### Solution

```python
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def level_order(root):
    """
    Level order traversal using BFS with queue.
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level_values = []
        
        # Process all nodes at current level
        for _ in range(level_size):
            node = queue.popleft()
            level_values.append(node.val)
            
            # Add children for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level_values)
    
    return result

# Alternative: Single list without level separation
def level_order_simple(root):
    """Return all values in level order as single list."""
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

# Create test tree:     3
#                      / \
#                     9   20
#                        /  \
#                       15   7
root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)

print("Level order by levels:", level_order(root))
# Output: [[3], [9, 20], [15, 7]]

print("Level order simple:", level_order_simple(root))
# Output: [3, 9, 20, 15, 7]
```

---

## 🎯 Problem-Solving Patterns

### 1. Queue Simulation
- **Use when:** Need FIFO behavior with different data structures
- **Pattern:** Use auxiliary structures to simulate queue operations
- **Examples:** Implement queue with stacks, circular queue

### 2. Sliding Window with Queue
- **Use when:** Need to track elements in a moving window
- **Pattern:** Add new elements, remove expired ones
- **Examples:** Moving average, recent calls counter

### 3. BFS with Queue
- **Use when:** Need to explore nodes level by level
- **Pattern:** Process all nodes at current level before moving to next
- **Examples:** Tree level order, shortest path in unweighted graph

### 4. Streaming Data Processing
- **Use when:** Processing continuous data stream
- **Pattern:** Maintain relevant data in queue, discard outdated
- **Examples:** First unique character in stream, moving statistics

## 💡 Key Tips

!!! tip "Queue vs Stack Choice"
    Use queues when you need FIFO (first in, first out) behavior. Use stacks when you need LIFO (last in, first out).

!!! note "Deque Efficiency"
    Python's `collections.deque` provides O(1) operations at both ends, making it perfect for queue implementations.

!!! success "Memory Management"
    For streaming problems, always remove outdated elements to prevent memory leaks.

## 🚀 Next Steps

Ready for more challenging problems? Try:
- [Medium Queue Problems](medium-problems.md)
- [Hard Queue Problems](hard-problems.md)
- Practice BFS problems with queues

---

*🎉 Great job! You've mastered easy queue problems. Ready for [Medium Problems](medium-problems.md)?*
