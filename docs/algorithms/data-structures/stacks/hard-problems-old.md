# Stacks: Hard Problems

## 🔥 Advanced Stack Challenges

These problems require sophisticated stack techniques, multiple stacks, or complex algorithms combining stacks with other data structures.

---

## Problem 1: Largest Rectangle in Histogram

**Difficulty:** Hard  
**Pattern:** Monotonic Stack  
**Time:** O(n) | **Space:** O(n)

### Problem Statement

Given an array of integers `heights` representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

**Example:**
```
Input: heights = [2,1,5,6,2,3]
Output: 10
Explanation: The largest rectangle has area = 10 units.
```

### Solution

```python
def largest_rectangle_area(heights):
    """
    Find the largest rectangle area in histogram using monotonic stack.
    
    Key insight: For each bar, find the largest rectangle where this bar
    is the shortest bar in the rectangle.
    """
    stack = []  # Store indices
    max_area = 0
    heights.append(0)  # Add sentinel to process remaining bars
    
    for i, height in enumerate(heights):
        # Maintain increasing stack
        while stack and heights[stack[-1]] > height:
            h = heights[stack.pop()]
            # Width = current index - previous index - 1
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        
        stack.append(i)
    
    return max_area

# Test
heights = [2, 1, 5, 6, 2, 3]
print(f"Largest rectangle area: {largest_rectangle_area(heights)}")
# Output: 10
```

### Explanation

1. **Monotonic Stack**: Maintain stack with increasing heights
2. **Area Calculation**: When we pop a bar, it becomes the shortest bar in some rectangle
3. **Width Calculation**: Distance between current position and previous stack top
4. **Sentinel**: Add 0 at end to process all remaining bars

---

## Problem 2: Maximal Rectangle

**Difficulty:** Hard  
**Pattern:** Stack + Dynamic Programming  
**Time:** O(m×n) | **Space:** O(n)

### Problem Statement

Given a binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

**Example:**
```
Input: matrix = [
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
Output: 6
```

### Solution

```python
def maximal_rectangle(matrix):
    """
    Find maximal rectangle in binary matrix.
    
    Convert to histogram problem for each row.
    """
    if not matrix or not matrix[0]:
        return 0
    
    rows, cols = len(matrix), len(matrix[0])
    heights = [0] * cols
    max_area = 0
    
    for row in matrix:
        # Update heights array for current row
        for i in range(cols):
            if row[i] == '1':
                heights[i] += 1
            else:
                heights[i] = 0
        
        # Find largest rectangle in current histogram
        max_area = max(max_area, largest_rectangle_area(heights[:]))
    
    return max_area

def largest_rectangle_area(heights):
    """Helper function from previous problem."""
    stack = []
    max_area = 0
    heights.append(0)
    
    for i, height in enumerate(heights):
        while stack and heights[stack[-1]] > height:
            h = heights[stack.pop()]
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        stack.append(i)
    
    return max_area
```

---

## Problem 3: Basic Calculator II

**Difficulty:** Hard  
**Pattern:** Expression Evaluation  
**Time:** O(n) | **Space:** O(n)

### Problem Statement

Implement a basic calculator to evaluate a simple expression string containing non-negative integers, `+`, `-`, `*`, `/` and empty spaces.

**Example:**
```
Input: s = "3+2*2"
Output: 7

Input: s = " 3/2 "
Output: 1
```

### Solution

```python
def calculate(s):
    """
    Evaluate mathematical expression with +, -, *, /.
    
    Use stack to handle operator precedence.
    """
    if not s:
        return 0
    
    stack = []
    num = 0
    operator = '+'  # Start with '+' to handle first number
    
    for i, char in enumerate(s):
        if char.isdigit():
            num = num * 10 + int(char)
        
        # Process operator or end of string
        if char in '+-*/' or i == len(s) - 1:
            if operator == '+':
                stack.append(num)
            elif operator == '-':
                stack.append(-num)
            elif operator == '*':
                stack.append(stack.pop() * num)
            elif operator == '/':
                # Handle negative division in Python
                prev = stack.pop()
                stack.append(int(prev / num))
            
            operator = char
            num = 0
    
    return sum(stack)

# Test
expressions = ["3+2*2", " 3/2 ", "14-3*2"]
for expr in expressions:
    print(f"'{expr}' = {calculate(expr)}")
```

---

## Problem 4: Remove K Digits

**Difficulty:** Hard  
**Pattern:** Greedy + Monotonic Stack  
**Time:** O(n) | **Space:** O(n)

### Problem Statement

Given string `num` representing a non-negative integer and an integer `k`, return the smallest possible integer after removing `k` digits from `num`.

**Example:**
```
Input: num = "1432219", k = 3
Output: "1219"
```

### Solution

```python
def remove_k_digits(num, k):
    """
    Remove k digits to get smallest number using monotonic stack.
    
    Greedy approach: Remove larger digits that appear before smaller ones.
    """
    stack = []
    removals = k
    
    for digit in num:
        # Remove larger digits from stack
        while stack and removals > 0 and stack[-1] > digit:
            stack.pop()
            removals -= 1
        
        stack.append(digit)
    
    # If we still need to remove digits, remove from end
    while removals > 0:
        stack.pop()
        removals -= 1
    
    # Build result, handle leading zeros
    result = ''.join(stack).lstrip('0')
    return result if result else '0'

# Test
test_cases = [
    ("1432219", 3),
    ("10200", 1),
    ("10", 2)
]

for num, k in test_cases:
    result = remove_k_digits(num, k)
    print(f"Remove {k} from '{num}': '{result}'")
```

---

## Problem 5: Trapping Rain Water

**Difficulty:** Hard  
**Pattern:** Monotonic Stack  
**Time:** O(n) | **Space:** O(n)

### Problem Statement

Given `n` non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

**Example:**
```
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
```

### Solution

```python
def trap_rain_water(height):
    """
    Calculate trapped rainwater using monotonic decreasing stack.
    
    Stack stores indices of bars in decreasing height order.
    """
    if not height:
        return 0
    
    stack = []
    water = 0
    
    for i, h in enumerate(height):
        # Process bars that can form water pockets
        while stack and height[stack[-1]] < h:
            bottom = stack.pop()  # Bottom of water pocket
            
            if not stack:
                break
            
            # Calculate water level and width
            water_height = min(height[stack[-1]], h) - height[bottom]
            width = i - stack[-1] - 1
            water += water_height * width
        
        stack.append(i)
    
    return water

# Alternative: Two-pointer approach (more efficient)
def trap_rain_water_optimized(height):
    """
    Two-pointer approach with O(1) space complexity.
    """
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0
    
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    
    return water

# Test
height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
print(f"Trapped water (stack): {trap_rain_water(height)}")
print(f"Trapped water (optimized): {trap_rain_water_optimized(height)}")
```

---

## Problem 6: Shortest Subarray with Sum at Least K

**Difficulty:** Hard  
**Pattern:** Monotonic Deque  
**Time:** O(n) | **Space:** O(n)

### Problem Statement

Given an integer array `nums` and an integer `k`, return the length of the shortest non-empty subarray with a sum of at least `k`. If no such subarray exists, return `-1`.

### Solution

```python
from collections import deque

def shortest_subarray(nums, k):
    """
    Find shortest subarray with sum >= k using monotonic deque.
    
    Uses prefix sums and maintains increasing deque.
    """
    n = len(nums)
    prefix = [0] * (n + 1)
    
    # Calculate prefix sums
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]
    
    deque_indices = deque()
    min_length = float('inf')
    
    for i in range(n + 1):
        # Check if we can form a valid subarray
        while deque_indices and prefix[i] - prefix[deque_indices[0]] >= k:
            min_length = min(min_length, i - deque_indices.popleft())
        
        # Maintain increasing order in deque
        while deque_indices and prefix[i] <= prefix[deque_indices[-1]]:
            deque_indices.pop()
        
        deque_indices.append(i)
    
    return min_length if min_length != float('inf') else -1

# Test
test_cases = [
    ([1], 1),
    ([1, 2], 4),
    ([2, -1, 2], 3)
]

for nums, k in test_cases:
    result = shortest_subarray(nums, k)
    print(f"Shortest subarray in {nums} with sum >= {k}: {result}")
```

---

## 🎯 Problem-Solving Patterns

### 1. Monotonic Stack
- **Use when:** Finding next/previous greater/smaller elements
- **Pattern:** Maintain increasing/decreasing order in stack
- **Examples:** Largest rectangle, daily temperatures

### 2. Expression Evaluation
- **Use when:** Parsing mathematical expressions
- **Pattern:** Use stack for operators and operands
- **Examples:** Basic calculator, infix to postfix

### 3. Multiple Stacks
- **Use when:** Need to track multiple states
- **Pattern:** Use different stacks for different purposes
- **Examples:** Min stack, browser history

### 4. Stack with Extra Data
- **Use when:** Need additional information with each element
- **Pattern:** Store tuples or custom objects in stack
- **Examples:** Stock span, sliding window maximum

## 💡 Advanced Tips

!!! tip "Monotonic Stack Optimization"
    When you need to find the next greater/smaller element for all elements, a monotonic stack can solve it in O(n) time instead of O(n²).

!!! warning "Integer Overflow"
    In problems involving large numbers or multiplication, be careful about integer overflow. Use appropriate data types or modular arithmetic.

!!! success "Space Optimization"
    Some stack problems can be solved with O(1) space using two-pointer techniques (like trapping rain water).

## 🚀 Next Level Practice

After mastering these hard problems, try:

1. **Competitive Programming:** Stack problems on Codeforces, AtCoder
2. **System Design:** Implement expression evaluators, parsers
3. **Advanced Algorithms:** Learn about persistent stacks, stack machines

---

*🎉 Congratulations! You've mastered hard stack problems. Ready to tackle [Queue Problems](../queues/easy-problems.md)?*
