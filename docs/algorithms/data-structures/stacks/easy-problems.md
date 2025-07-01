# Stacks: Easy Problems

Welcome to the easy problems section for stacks! These problems will help you understand the fundamental LIFO (Last In, First Out) principle and build confidence with basic stack operations.

## 🎯 Learning Objectives

By completing these problems, you'll master:

- Basic stack operations (push, pop, peek)
- Understanding LIFO principle in practice
- Handling edge cases (empty stack)
- Pattern recognition for stack-based solutions
- Implementation of stack using different approaches

---

## Problem 1: Valid Parentheses

**LeetCode 20** | **Difficulty: Easy**

### Problem Statement

Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:
- Open brackets must be closed by the same type of brackets
- Open brackets must be closed in the correct order

**Example:**

```text
Input: s = "()"
Output: true

Input: s = "()[]{}"
Output: true

Input: s = "(]"
Output: false
```

### Solution

```python
def isValid(s):
    """
    Check if parentheses are valid using stack.
    
    Time: O(n) - visit each character once
    Space: O(n) - worst case all opening brackets
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            # Closing bracket
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            # Opening bracket
            stack.append(char)
    
    return len(stack) == 0

# Test cases
print(isValid("()"))        # True
print(isValid("()[]{})"))   # True
print(isValid("(]"))        # False
print(isValid("([)]"))      # False
print(isValid("{[]}"))      # True
```

### 🔍 Key Insights

1. **Stack for matching**: Perfect use case for LIFO - last opened bracket should be first closed
2. **Mapping approach**: Use dictionary to quickly check matching pairs
3. **Edge cases**: Empty stack when closing bracket encountered

---

## Problem 2: Min Stack

**LeetCode 155** | **Difficulty: Easy**

### Problem Statement

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:
- `MinStack()` initializes the stack object
- `void push(int val)` pushes the element val onto the stack
- `void pop()` removes the element on the top of the stack
- `int top()` gets the top element of the stack
- `int getMin()` retrieves the minimum element in the stack

### Solution

```python
class MinStack:
    def __init__(self):
        """
        Initialize stack with minimum tracking.
        
        Uses auxiliary stack to track minimums.
        """
        self.stack = []
        self.min_stack = []
    
    def push(self, val):
        """
        Push value and update minimum.
        
        Time: O(1)
        """
        self.stack.append(val)
        
        # Push to min_stack if it's empty or val is new minimum
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        """
        Pop value and update minimum.
        
        Time: O(1)
        """
        if not self.stack:
            return
        
        val = self.stack.pop()
        
        # Remove from min_stack if it was the minimum
        if val == self.min_stack[-1]:
            self.min_stack.pop()
    
    def top(self):
        """
        Get top element.
        
        Time: O(1)
        """
        if not self.stack:
            return None
        return self.stack[-1]
    
    def getMin(self):
        """
        Get minimum element in O(1).
        
        Time: O(1)
        """
        if not self.min_stack:
            return None
        return self.min_stack[-1]

# Usage
min_stack = MinStack()
min_stack.push(-2)
min_stack.push(0)
min_stack.push(-3)
print(min_stack.getMin())  # -3
min_stack.pop()
print(min_stack.top())     # 0
print(min_stack.getMin())  # -2
```

### Alternative Solution: Single Stack

```python
class MinStackSingle:
    def __init__(self):
        """Single stack approach storing (val, min) pairs."""
        self.stack = []
    
    def push(self, val):
        """Push (value, current_minimum) pair."""
        if not self.stack:
            self.stack.append((val, val))
        else:
            current_min = min(val, self.stack[-1][1])
            self.stack.append((val, current_min))
    
    def pop(self):
        """Pop the top pair."""
        if self.stack:
            self.stack.pop()
    
    def top(self):
        """Get top value."""
        if self.stack:
            return self.stack[-1][0]
        return None
    
    def getMin(self):
        """Get minimum from top pair."""
        if self.stack:
            return self.stack[-1][1]
        return None
```

---

## Problem 3: Remove Outermost Parentheses

**LeetCode 1021** | **Difficulty: Easy**

### Problem Statement

A valid parentheses string is either empty, `"(" + A + ")"`, or `A + B`, where A and B are valid parentheses strings, and + represents string concatenation.

Remove the outermost parentheses of every primitive string in the parentheses representation of s.

**Example:**

```text
Input: s = "(()())(())"
Output: "()()()"

Input: s = "(()())(())(()(()))"
Output: "()()()()(())"
```

### Solution

```python
def removeOuterParentheses(s):
    """
    Remove outermost parentheses using stack counter.
    
    Time: O(n) - visit each character once
    Space: O(n) - for result string
    """
    result = []
    depth = 0
    
    for char in s:
        if char == '(':
            # Only add if not outermost opening
            if depth > 0:
                result.append(char)
            depth += 1
        else:  # char == ')'
            depth -= 1
            # Only add if not outermost closing
            if depth > 0:
                result.append(char)
    
    return ''.join(result)

# Test
print(removeOuterParentheses("(()())(())"))  # "()()()"
print(removeOuterParentheses("(()())(())(()(()))"))  # "()()()()(())"
```

### 🔍 Key Insight

Use depth counter instead of actual stack - we only need to track nesting level, not store the characters.

---

## Problem 4: Baseball Game

**LeetCode 682** | **Difficulty: Easy**

### Problem Statement

You are keeping score for a baseball game with strange rules. The game consists of several rounds, where the scores of past rounds may affect future rounds' scores.

At the beginning of the game, you start with an empty record. You are given a list of strings ops, where ops[i] is the ith operation you must apply to the record and is one of the following:

- Integer x: Record a new score of x
- "+": Record a new score that is the sum of the previous two scores
- "D": Record a new score that is double the previous score
- "C": Cancel the previous score, removing it from the record

**Example:**

```text
Input: ops = ["5","2","C","D","+"]
Output: 30
Explanation:
"5" - Add 5 to the record: [5]
"2" - Add 2 to the record: [5, 2]
"C" - Cancel the previous score: [5]
"D" - Add 2 * 5 = 10 to the record: [5, 10]
"+" - Add 5 + 10 = 15 to the record: [5, 10, 15]
Total sum = 5 + 10 + 15 = 30
```

### Solution

```python
def calPoints(ops):
    """
    Calculate baseball game score using stack.
    
    Time: O(n) - process each operation once
    Space: O(n) - store all valid scores
    """
    stack = []
    
    for op in ops:
        if op == "+":
            # Sum of last two scores
            stack.append(stack[-1] + stack[-2])
        elif op == "D":
            # Double the last score
            stack.append(2 * stack[-1])
        elif op == "C":
            # Cancel last score
            stack.pop()
        else:
            # Regular score
            stack.append(int(op))
    
    return sum(stack)

# Test
print(calPoints(["5","2","C","D","+"]))     # 30
print(calPoints(["5","-2","4","C","D","9","+","+"]))  # 27
```

---

## Problem 5: Build Array with Stack Operations

**LeetCode 1441** | **Difficulty: Easy**

### Problem Statement

Given an array target and an integer n. In each iteration, you will read a number from list = {1,2,3..., n}.

Build the target array using the following operations:
- Push: Read a new element from the beginning list, and push it in the array
- Pop: delete the last element of the array

It is guaranteed that the target array is strictly increasing.

**Example:**

```text
Input: target = [1,3], n = 3
Output: ["Push","Pop","Push"]
Explanation: 
Read number 1 and automatically push it to the array → [1]
Read number 2 and automatically push it to the array → [1,2]
Pop the last element of the array → [1]
Read number 3 and automatically push it to the array → [1,3]
```

### Solution

```python
def buildArray(target, n):
    """
    Build operations to construct target array.
    
    Time: O(max(target)) - iterate through numbers
    Space: O(k) where k is number of operations
    """
    operations = []
    target_set = set(target)
    target_index = 0
    
    for num in range(1, n + 1):
        if target_index >= len(target):
            break
            
        operations.append("Push")
        
        if num == target[target_index]:
            # This number is in target, keep it
            target_index += 1
        else:
            # This number is not in target, pop it
            operations.append("Pop")
    
    return operations

# Test
print(buildArray([1,3], 3))      # ["Push","Pop","Push"]
print(buildArray([1,2,3], 3))    # ["Push","Push","Push"]
print(buildArray([1,2], 4))      # ["Push","Push"]
```

---

## Problem 6: Implement Stack using Queues

**LeetCode 225** | **Difficulty: Easy**

### Problem Statement

Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (push, top, pop, and empty).

### Solution 1: Two Queues

```python
from collections import deque

class MyStack:
    def __init__(self):
        """Initialize stack using two queues."""
        self.q1 = deque()
        self.q2 = deque()
    
    def push(self, x):
        """
        Push element to top of stack.
        
        Time: O(1)
        """
        self.q1.append(x)
    
    def pop(self):
        """
        Remove and return top element.
        
        Time: O(n) - need to move n-1 elements
        """
        if not self.q1:
            return None
        
        # Move all elements except last to q2
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        
        # Pop the last element (top of stack)
        result = self.q1.popleft()
        
        # Swap queues
        self.q1, self.q2 = self.q2, self.q1
        
        return result
    
    def top(self):
        """
        Get top element without removing.
        
        Time: O(n)
        """
        if not self.q1:
            return None
        
        # Similar to pop but put the element back
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        
        result = self.q1.popleft()
        self.q2.append(result)  # Put it back
        
        # Swap queues
        self.q1, self.q2 = self.q2, self.q1
        
        return result
    
    def empty(self):
        """Check if stack is empty."""
        return len(self.q1) == 0
```

### Solution 2: One Queue (Optimal)

```python
from collections import deque

class MyStackOptimal:
    def __init__(self):
        """Initialize stack using one queue."""
        self.queue = deque()
    
    def push(self, x):
        """
        Push element and rotate to maintain stack order.
        
        Time: O(n) - rotate all existing elements
        """
        size = len(self.queue)
        self.queue.append(x)
        
        # Rotate to bring new element to front
        for _ in range(size):
            self.queue.append(self.queue.popleft())
    
    def pop(self):
        """
        Remove and return top element.
        
        Time: O(1)
        """
        if self.queue:
            return self.queue.popleft()
        return None
    
    def top(self):
        """
        Get top element.
        
        Time: O(1)
        """
        if self.queue:
            return self.queue[0]
        return None
    
    def empty(self):
        """Check if stack is empty."""
        return len(self.queue) == 0

# Usage
stack = MyStackOptimal()
stack.push(1)
stack.push(2)
print(stack.top())    # 2
print(stack.pop())    # 2
print(stack.empty())  # False
```

---

## 🎯 Common Patterns in Easy Stack Problems

### 1. Matching/Pairing Problems

Use stack when you need to match opening and closing elements:

```python
def is_balanced(s):
    """Generic balanced string checker."""
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}
    
    for char in s:
        if char in pairs:
            stack.append(char)
        elif char in pairs.values():
            if not stack or pairs[stack.pop()] != char:
                return False
    
    return len(stack) == 0
```

### 2. Auxiliary Stack Pattern

When you need to track additional information:

```python
def track_property(operations):
    """Track property using auxiliary stack."""
    main_stack = []
    aux_stack = []  # Tracks some property
    
    for op in operations:
        if op == "push":
            main_stack.append(value)
            # Update auxiliary stack
            if not aux_stack or should_update(value, aux_stack[-1]):
                aux_stack.append(value)
        elif op == "pop":
            val = main_stack.pop()
            if val == aux_stack[-1]:
                aux_stack.pop()
```

### 3. Simulation Pattern

Use stack to simulate processes:

```python
def simulate_process(instructions):
    """Simulate a process using stack."""
    stack = []
    
    for instruction in instructions:
        if instruction.startswith("add"):
            stack.append(parse_value(instruction))
        elif instruction == "undo":
            if stack:
                stack.pop()
        elif instruction.startswith("operation"):
            # Perform operation on stack
            result = perform_operation(stack, instruction)
            stack.append(result)
    
    return stack
```

## 💡 Practice Tips

### 1. Recognize Stack Problems

Look for these keywords:
- **Matching/Pairing**: Parentheses, brackets, tags
- **Undo operations**: Cancel, remove last
- **Nested structures**: Function calls, expressions
- **Most recent**: Last added, latest operation

### 2. Edge Cases to Consider

```python
# Always check these cases:
if not stack:           # Empty stack
    handle_empty_case()

if len(stack) == 1:     # Single element
    handle_single_case()

# Before operations
if stack:               # Non-empty check
    value = stack.pop()
```

### 3. Time Complexity Analysis

- **Push/Pop/Top**: O(1) for proper stack implementation
- **Search**: O(n) - need to pop all elements
- **Space**: O(n) for n elements

## 🏆 Progress Checklist

- [ ] **Valid Parentheses** - Master the matching pattern
- [ ] **Min Stack** - Learn auxiliary stack technique
- [ ] **Remove Outer Parentheses** - Practice depth tracking
- [ ] **Baseball Game** - Handle multiple operations
- [ ] **Build Array** - Understand stack simulation
- [ ] **Stack using Queues** - Cross-data structure implementation

## 🚀 Next Steps

Ready for more challenging problems? Advance to **[Medium Problems](medium-problems.md)** to learn:

- Monotonic stack patterns
- Expression evaluation
- Advanced stack applications
- Optimization techniques

---

*Great job completing the easy stack problems! You now understand the fundamental LIFO principle and basic stack operations. Keep practicing!*
