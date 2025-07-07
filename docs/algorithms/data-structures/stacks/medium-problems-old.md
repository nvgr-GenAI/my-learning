# Stacks: Medium Problems

These intermediate problems will challenge you to use advanced stack techniques and patterns. You'll learn about monotonic stacks, expression evaluation, and complex stack applications.

## 🎯 Learning Objectives

By completing these problems, you'll master:

- Monotonic stack patterns for next greater/smaller elements
- Expression parsing and evaluation
- Stack-based optimization techniques
- Complex string manipulation with stacks
- Advanced problem-solving patterns

---

## Problem 1: Daily Temperatures

**LeetCode 739** | **Difficulty: Medium**

### Problem Description

Given an array of integers temperatures representing daily temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature.

**Example:**

```text
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
```

### Solution: Monotonic Stack

```python
def dailyTemperatures(temperatures):
    """
    Find next warmer temperature using monotonic stack.
    
    Time: O(n) - each element pushed and popped at most once
    Space: O(n) - stack can contain all indices
    """
    n = len(temperatures)
    result = [0] * n
    stack = []  # Store indices
    
    for i in range(n):
        # While stack not empty and current temp is warmer
        while stack and temperatures[i] > temperatures[stack[-1]]:
            prev_index = stack.pop()
            result[prev_index] = i - prev_index
        
        stack.append(i)
    
    return result

# Test
temps = [73,74,75,71,69,72,76,73]
print(dailyTemperatures(temps))  # [1,1,4,2,1,1,0,0]
```

### 🔍 Monotonic Stack Pattern

**Key Insight**: Maintain stack in monotonic (decreasing) order by temperature. When we find a warmer temperature, it's the answer for all cooler temperatures in the stack.

---

## Problem 2: Next Greater Element II

**LeetCode 503** | **Difficulty: Medium**

### Problem Description

Given a circular array, return the next greater number for every element. If it doesn't exist, return -1 for this number.

**Example:**

```text
Input: nums = [1,2,1]
Output: [2,-1,2]
Explanation: 
- For 1: next greater is 2
- For 2: no greater element, -1  
- For 1: next greater is 2 (circular)
```

### Solution

```python
def nextGreaterElements(nums):
    """
    Find next greater element in circular array.
    
    Time: O(n) - each element processed at most twice
    Space: O(n) - stack and result array
    """
    n = len(nums)
    result = [-1] * n
    stack = []
    
    # Process array twice to handle circular nature
    for i in range(2 * n):
        # Use modulo to handle circular indexing
        current_index = i % n
        current_value = nums[current_index]
        
        # Pop elements smaller than current
        while stack and nums[stack[-1]] < current_value:
            index = stack.pop()
            result[index] = current_value
        
        # Only push during first iteration
        if i < n:
            stack.append(current_index)
    
    return result

# Test
print(nextGreaterElements([1,2,1]))     # [2,-1,2]
print(nextGreaterElements([1,2,3,4,3])) # [2,3,4,-1,4]
```

---

## Problem 3: Evaluate Reverse Polish Notation

**LeetCode 150** | **Difficulty: Medium**

### Problem Description

Evaluate the value of an arithmetic expression in Reverse Polish Notation (RPN).

Valid operators are +, -, *, and /. Each operand may be an integer or another expression.

**Example:**

```text
Input: tokens = ["2","1","+","3","*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9

Input: tokens = ["4","13","5","/","+"]
Output: 6
Explanation: (4 + (13 / 5)) = 6
```

### Solution

```python
def evalRPN(tokens):
    """
    Evaluate Reverse Polish Notation expression.
    
    Time: O(n) - process each token once
    Space: O(n) - stack for operands
    """
    stack = []
    operators = {'+', '-', '*', '/'}
    
    for token in tokens:
        if token in operators:
            # Pop two operands (order matters for - and /)
            b = stack.pop()
            a = stack.pop()
            
            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
            elif token == '*':
                result = a * b
            elif token == '/':
                # Use int() for truncation towards zero
                result = int(a / b)
            
            stack.append(result)
        else:
            # Operand
            stack.append(int(token))
    
    return stack[0]

# Test
print(evalRPN(["2","1","+","3","*"]))        # 9
print(evalRPN(["4","13","5","/","+"]))       # 6
print(evalRPN(["10","6","9","3","+","-11","*","/","*","17","+","5","+"]))  # 22
```

---

## Problem 4: Decode String

**LeetCode 394** | **Difficulty: Medium**

### Problem Description

Given an encoded string, return its decoded string. The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times.

**Example:**

```text
Input: s = "3[a]2[bc]"
Output: "aaabcbc"

Input: s = "3[a2[c]]"
Output: "accaccacc"
```

### Solution

```python
def decodeString(s):
    """
    Decode string using stack for nested patterns.
    
    Time: O(n) where n is length of decoded string
    Space: O(n) for stack storage
    """
    stack = []
    current_num = 0
    current_string = ""
    
    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            # Save current state and start new
            stack.append((current_string, current_num))
            current_string = ""
            current_num = 0
        elif char == ']':
            # Pop previous state and combine
            prev_string, num = stack.pop()
            current_string = prev_string + current_string * num
        else:
            # Regular character
            current_string += char
    
    return current_string

# Test
print(decodeString("3[a]2[bc]"))      # "aaabcbc"
print(decodeString("3[a2[c]]"))       # "accaccacc"
print(decodeString("2[abc]3[cd]ef"))  # "abcabccdcdcdef"
```

---

## Problem 5: Asteroid Collision

**LeetCode 735** | **Difficulty: Medium**

### Problem Description

We are given an array asteroids of integers representing asteroids in a row. Each asteroid has a size and direction (positive = right, negative = left). When two asteroids meet, the smaller one explodes. If they are the same size, both explode.

**Example:**

```text
Input: asteroids = [5,10,-5]
Output: [5,10]
Explanation: 10 and -5 collide, 10 survives

Input: asteroids = [8,-8]
Output: []
Explanation: 8 and -8 collide and both explode
```

### Solution

```python
def asteroidCollision(asteroids):
    """
    Simulate asteroid collisions using stack.
    
    Time: O(n) - each asteroid processed once
    Space: O(n) - stack storage
    """
    stack = []
    
    for asteroid in asteroids:
        while stack and asteroid < 0 and stack[-1] > 0:
            # Collision between right-moving and left-moving
            if stack[-1] < -asteroid:
                # Right-moving asteroid destroyed
                stack.pop()
                continue
            elif stack[-1] == -asteroid:
                # Both destroyed
                stack.pop()
            
            # Left-moving asteroid destroyed or absorbed
            break
        else:
            # No collision or left-moving asteroid survives
            stack.append(asteroid)
    
    return stack

# Test
print(asteroidCollision([5,10,-5]))     # [5,10]
print(asteroidCollision([8,-8]))        # []
print(asteroidCollision([10,2,-5]))     # [10]
print(asteroidCollision([-2,-1,1,2]))   # [-2,-1,1,2]
```

---

## Problem 6: Remove K Digits

**LeetCode 402** | **Difficulty: Medium**

### Problem Description

Given string num representing a non-negative integer and an integer k, return the smallest possible integer after removing k digits from num.

**Example:**

```text
Input: num = "1432219", k = 3
Output: "1219"
Explanation: Remove digits 4, 3, and 2 to get 1219

Input: num = "10200", k = 1  
Output: "200"
Explanation: Remove the leading 1
```

### Solution

```python
def removeKdigits(num, k):
    """
    Remove k digits to get smallest number using monotonic stack.
    
    Time: O(n) - each digit processed once
    Space: O(n) - stack storage
    """
    stack = []
    to_remove = k
    
    for digit in num:
        # Remove larger digits from stack while we can
        while stack and stack[-1] > digit and to_remove > 0:
            stack.pop()
            to_remove -= 1
        
        stack.append(digit)
    
    # Remove remaining digits from end if needed
    while to_remove > 0:
        stack.pop()
        to_remove -= 1
    
    # Build result, handle leading zeros
    result = ''.join(stack).lstrip('0')
    
    return result if result else '0'

# Test
print(removeKdigits("1432219", 3))  # "1219"
print(removeKdigits("10200", 1))    # "200"
print(removeKdigits("10", 2))       # "0"
```

---

## Problem 7: Validate Stack Sequences

**LeetCode 946** | **Difficulty: Medium**

### Problem Description

Given two integer arrays pushed and popped, return true if this could have been the result of a sequence of push and pop operations on an initially empty stack.

**Example:**

```text
Input: pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
Output: true
Explanation: 
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
```

### Solution

```python
def validateStackSequences(pushed, popped):
    """
    Validate if sequences represent valid stack operations.
    
    Time: O(n) - each element pushed and popped once
    Space: O(n) - stack storage
    """
    stack = []
    pop_index = 0
    
    for num in pushed:
        stack.append(num)
        
        # Pop while top matches expected pop sequence
        while stack and pop_index < len(popped) and stack[-1] == popped[pop_index]:
            stack.pop()
            pop_index += 1
    
    # Valid if all elements were popped
    return pop_index == len(popped)

# Test
print(validateStackSequences([1,2,3,4,5], [4,5,3,2,1]))  # True
print(validateStackSequences([1,2,3,4,5], [4,3,5,1,2]))  # False
```

---

## 🎯 Advanced Stack Patterns

### 1. Monotonic Stack Template

For "next greater/smaller" type problems:

```python
def monotonic_stack_template(arr, find_greater=True):
    """
    Generic monotonic stack for next greater/smaller elements.
    
    find_greater: True for next greater, False for next smaller
    """
    n = len(arr)
    result = [-1] * n
    stack = []
    
    for i in range(n):
        # Condition depends on what we're looking for
        while stack and (
            (find_greater and arr[i] > arr[stack[-1]]) or
            (not find_greater and arr[i] < arr[stack[-1]])
        ):
            prev_index = stack.pop()
            result[prev_index] = arr[i]
        
        stack.append(i)
    
    return result
```

### 2. Expression Evaluation Pattern

For mathematical expression problems:

```python
def expression_evaluator(expression, operators):
    """
    Generic expression evaluator using stacks.
    """
    operand_stack = []
    operator_stack = []
    
    def apply_operator():
        """Apply top operator to top two operands."""
        b = operand_stack.pop()
        a = operand_stack.pop()
        op = operator_stack.pop()
        result = operators[op](a, b)
        operand_stack.append(result)
    
    for token in expression:
        if token.isdigit():
            operand_stack.append(int(token))
        elif token in operators:
            # Handle operator precedence
            while (operator_stack and 
                   has_higher_precedence(operator_stack[-1], token)):
                apply_operator()
            operator_stack.append(token)
        # Handle parentheses if needed
    
    # Apply remaining operators
    while operator_stack:
        apply_operator()
    
    return operand_stack[0]
```

### 3. Nested Structure Pattern

For problems with nested structures:

```python
def handle_nested_structure(s):
    """
    Handle nested structures using stack.
    """
    stack = []
    current_level = ""
    
    for char in s:
        if char == opening_char:
            # Save current level and start new
            stack.append(current_level)
            current_level = ""
        elif char == closing_char:
            # Process current level and restore previous
            processed = process_level(current_level)
            current_level = stack.pop()
            current_level += processed
        else:
            current_level += char
    
    return current_level
```

## 💡 Problem-Solving Tips

### 1. Recognize Monotonic Stack Problems

Look for these patterns:
- **Next greater/smaller element**
- **Maximum rectangle/area problems**
- **Temperature/stock price problems**
- **Array elements comparison**

### 2. Expression Problems

Stack is perfect for:
- **Postfix/Prefix evaluation**
- **Infix to postfix conversion**
- **Balanced parentheses with operations**
- **Nested expression parsing**

### 3. Simulation Problems

Use stack when you need to:
- **Undo recent operations**
- **Handle nested processing**
- **Maintain order of operations**
- **Process elements in reverse order**

## 🏆 Progress Checklist

- [ ] **Daily Temperatures** - Master monotonic stack pattern
- [ ] **Next Greater Element II** - Handle circular arrays
- [ ] **Evaluate RPN** - Expression evaluation technique
- [ ] **Decode String** - Nested structure processing
- [ ] **Asteroid Collision** - Collision simulation
- [ ] **Remove K Digits** - Optimization with monotonic stack
- [ ] **Validate Stack Sequences** - Stack operation validation

## 🚀 Next Steps

Ready for the ultimate challenge? Advance to **[Hard Problems](hard-problems.md)** to master:

- Complex stack-based algorithms
- Advanced optimization techniques
- Multi-stack approaches
- Real-world applications

---

*Excellent work on medium stack problems! You've learned powerful patterns that apply to many real-world scenarios.*
