# Mathematical Algorithms - Hard Problems

## 🎯 Learning Objectives

Master the most challenging mathematical algorithms and advanced techniques:

- Advanced number theory (modular exponentiation, Chinese Remainder Theorem)
- Computational geometry algorithms
- Complex combinatorial problems
- Advanced mathematical optimization
- Algorithmic game theory and number-theoretic functions

---

## Problem 1: Super Pow

**Difficulty**: 🔴 Hard  
**Pattern**: Modular Exponentiation  
**Time**: O(n), **Space**: O(n)

### Problem Overview

Your task is to calculate `a^b mod 1337` where `a` is a positive integer and `b` is an extremely large positive integer given in the form of an array.

**Example:**
```
Input: a = 2, b = [3]
Output: 8

Input: a = 2, b = [1,0]
Output: 1024

Input: a = 1, b = [4,3,3,8,5,2]
Output: 1
```

### Solution

```python
def superPow(a, b):
    """
    Use modular exponentiation with the property:
    (a^b) mod m = ((a^(b[:-1])) mod m)^10 * (a^b[-1]) mod m
    """
    MOD = 1337
    
    def powMod(base, exp, mod):
        """Fast modular exponentiation"""
        result = 1
        base %= mod
        
        while exp > 0:
            if exp & 1:
                result = (result * base) % mod
            exp >>= 1
            base = (base * base) % mod
        
        return result
    
    if not b:
        return 1
    
    # Base case
    if len(b) == 1:
        return powMod(a, b[0], MOD)
    
    # Recursive case: a^b = (a^(b[:-1]))^10 * a^b[-1]
    last_digit = b.pop()
    
    # Calculate a^(b[:-1]) mod 1337
    sub_result = superPow(a, b)
    
    # Calculate (sub_result^10 * a^last_digit) mod 1337
    return (powMod(sub_result, 10, MOD) * powMod(a, last_digit, MOD)) % MOD

# Alternative iterative approach
def superPowIterative(a, b):
    """
    Iterative approach processing digits from left to right
    """
    MOD = 1337
    
    def powMod(base, exp):
        result = 1
        base %= MOD
        
        while exp > 0:
            if exp & 1:
                result = (result * base) % MOD
            exp >>= 1
            base = (base * base) % MOD
        
        return result
    
    result = 1
    
    for digit in b:
        # result = (result^10 * a^digit) mod 1337
        result = (powMod(result, 10) * powMod(a, digit)) % MOD
    
    return result

# Test
print(superPow(2, [3]))           # 8
print(superPow(2, [1, 0]))        # 1024
print(superPow(1, [4,3,3,8,5,2])) # 1
```

---

## Problem 2: Nth Digit

**Difficulty**: 🔴 Hard  
**Pattern**: Mathematical Analysis  
**Time**: O(log n), **Space**: O(1)

### Problem Overview

Given an integer `n`, return the `n`th digit of the infinite integer sequence `1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...`

**Example:**
```
Input: n = 3
Output: 3

Input: n = 11
Output: 0 (from number 10)
```

### Solution

```python
def findNthDigit(n):
    """
    Mathematical analysis of digit distribution:
    - 1-digit numbers: 1-9 (9 numbers, 9 digits)
    - 2-digit numbers: 10-99 (90 numbers, 180 digits)
    - 3-digit numbers: 100-999 (900 numbers, 2700 digits)
    """
    # Find which group the nth digit belongs to
    digit_length = 1
    count = 9
    start = 1
    
    while n > digit_length * count:
        n -= digit_length * count
        digit_length += 1
        count *= 10
        start *= 10
    
    # Find the actual number containing the nth digit
    number = start + (n - 1) // digit_length
    
    # Find which digit in that number
    digit_index = (n - 1) % digit_length
    
    return int(str(number)[digit_index])

# Alternative approach with explicit calculation
def findNthDigitExplicit(n):
    """
    More explicit calculation for better understanding
    """
    if n <= 9:
        return n
    
    # Calculate cumulative digit counts
    digits = 1
    numbers = 9
    start = 1
    
    while n > digits * numbers:
        n -= digits * numbers
        digits += 1
        numbers *= 10
        start *= 10
    
    # Find the exact number and position
    number = start + (n - 1) // digits
    position = (n - 1) % digits
    
    return int(str(number)[position])

# Test
print(findNthDigit(3))   # 3
print(findNthDigit(11))  # 0
print(findNthDigit(15))  # 2
```

---

## Problem 3: Largest Rectangle in Histogram

**Difficulty**: 🔴 Hard  
**Pattern**: Stack + Mathematical Optimization  
**Time**: O(n), **Space**: O(n)

### Problem Overview

Given an array of integers `heights` representing the histogram's bar height where the width of each bar is `1`, return the area of the largest rectangle in the histogram.

### Solution

```python
def largestRectangleArea(heights):
    """
    Use stack to find largest rectangle efficiently
    """
    stack = []
    max_area = 0
    
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        stack.append(i)
    
    # Process remaining bars in stack
    while stack:
        height = heights[stack.pop()]
        width = len(heights) if not stack else len(heights) - stack[-1] - 1
        max_area = max(max_area, height * width)
    
    return max_area

# Divide and conquer approach
def largestRectangleAreaDC(heights):
    """
    Divide and conquer approach
    Time: O(n log n) average, O(n^2) worst case
    """
    def solve(left, right):
        if left > right:
            return 0
        
        # Find minimum height index
        min_idx = left
        for i in range(left, right + 1):
            if heights[i] < heights[min_idx]:
                min_idx = i
        
        # Calculate area with minimum height as full width
        area = heights[min_idx] * (right - left + 1)
        
        # Recursively find max area in left and right parts
        left_area = solve(left, min_idx - 1)
        right_area = solve(min_idx + 1, right)
        
        return max(area, left_area, right_area)
    
    return solve(0, len(heights) - 1)

# Test
heights = [2, 1, 5, 6, 2, 3]
print(largestRectangleArea(heights))  # 10
```

---

## Problem 4: Basic Calculator II

**Difficulty**: 🔴 Hard  
**Pattern**: Expression Parsing  
**Time**: O(n), **Space**: O(n)

### Problem Overview

Given a string `s` which represents an expression, evaluate this expression and return its value.

The integer division should truncate toward zero.

### Solution

```python
def calculate(s):
    """
    Use stack to handle operator precedence
    """
    if not s:
        return 0
    
    stack = []
    num = 0
    op = '+'
    
    for i, char in enumerate(s):
        if char.isdigit():
            num = num * 10 + int(char)
        
        if char in '+-*/' or i == len(s) - 1:
            if op == '+':
                stack.append(num)
            elif op == '-':
                stack.append(-num)
            elif op == '*':
                stack.append(stack.pop() * num)
            elif op == '/':
                # Handle negative division (truncate toward zero)
                prev = stack.pop()
                stack.append(int(prev / num))
            
            op = char
            num = 0
    
    return sum(stack)

# Without stack (space optimized)
def calculateOptimized(s):
    """
    Space-optimized version without stack
    """
    if not s:
        return 0
    
    num = 0
    prev_num = 0
    result = 0
    op = '+'
    
    for i, char in enumerate(s):
        if char.isdigit():
            num = num * 10 + int(char)
        
        if char in '+-*/' or i == len(s) - 1:
            if op == '+':
                result += prev_num
                prev_num = num
            elif op == '-':
                result += prev_num
                prev_num = -num
            elif op == '*':
                prev_num = prev_num * num
            elif op == '/':
                prev_num = int(prev_num / num)
            
            op = char
            num = 0
    
    return result + prev_num

# Test
print(calculate("3+2*2"))     # 7
print(calculate(" 3/2 "))     # 1
print(calculate(" 3+5 / 2 ")) # 5
```

---

## Problem 5: Count of Range Sum

**Difficulty**: 🔴 Hard  
**Pattern**: Merge Sort + Prefix Sum  
**Time**: O(n log n), **Space**: O(n)

### Problem Overview

Given an integer array `nums` and two integers `lower` and `upper`, return the number of range sums that lie in `[lower, upper]` inclusive.

Range sum `S(i, j)` is defined as the sum of the elements in `nums` from indices `i` to `j` inclusive, where `i <= j`.

### Solution

```python
def countRangeSum(nums, lower, upper):
    """
    Use merge sort with prefix sums
    """
    def mergeSort(prefix_sums, start, end):
        if start >= end:
            return 0
        
        mid = (start + end) // 2
        count = mergeSort(prefix_sums, start, mid) + mergeSort(prefix_sums, mid + 1, end)
        
        # Count range sums across the split
        j = k = mid + 1
        for i in range(start, mid + 1):
            # Find range [lower + prefix[i], upper + prefix[i]]
            while j <= end and prefix_sums[j] - prefix_sums[i] < lower:
                j += 1
            while k <= end and prefix_sums[k] - prefix_sums[i] <= upper:
                k += 1
            count += k - j
        
        # Merge the sorted halves
        prefix_sums[start:end+1] = sorted(prefix_sums[start:end+1])
        
        return count
    
    # Calculate prefix sums
    prefix_sums = [0]
    for num in nums:
        prefix_sums.append(prefix_sums[-1] + num)
    
    return mergeSort(prefix_sums, 0, len(prefix_sums) - 1)

# Fenwick Tree approach
class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)
    
    def query(self, i):
        res = 0
        while i > 0:
            res += self.tree[i]
            i -= i & (-i)
        return res

def countRangeSumFenwick(nums, lower, upper):
    """
    Use Fenwick Tree (Binary Indexed Tree)
    """
    # Calculate prefix sums
    prefix_sums = [0]
    for num in nums:
        prefix_sums.append(prefix_sums[-1] + num)
    
    # Coordinate compression
    all_sums = sorted(set(prefix_sums + [s + lower for s in prefix_sums] + [s + upper + 1 for s in prefix_sums]))
    
    def binary_search(arr, val):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] < val:
                left = mid + 1
            else:
                right = mid - 1
        return left
    
    ft = FenwickTree(len(all_sums))
    result = 0
    
    for prefix_sum in prefix_sums:
        # Count existing prefix sums in range [prefix_sum - upper, prefix_sum - lower]
        left = binary_search(all_sums, prefix_sum - upper)
        right = binary_search(all_sums, prefix_sum - lower + 1)
        
        if left < right:
            result += ft.query(right) - ft.query(left)
        
        # Add current prefix sum to the tree
        idx = binary_search(all_sums, prefix_sum) + 1
        ft.update(idx, 1)
    
    return result

# Test
nums = [-2, 5, -1]
lower, upper = -2, 2
print(countRangeSum(nums, lower, upper))  # 3
```

---

## Problem 6: Robot Room Cleaner

**Difficulty**: 🔴 Hard  
**Pattern**: Backtracking + Coordinate System  
**Time**: O(4^(N-M)), **Space**: O(N-M)

### Problem Overview

Given a robot cleaner in a room represented by a binary grid, where 0 means the cell is blocked and 1 means the cell is free, clean the entire room and return to the starting point.

The robot has four APIs:
- `move()`: Returns true if next cell is open and robot moves into the cell
- `turnLeft()`: Robot turns left 90 degrees
- `turnRight()`: Robot turns right 90 degrees  
- `clean()`: Clean the current cell

### Solution

```python
def cleanRoom(robot):
    """
    Use backtracking to explore all reachable cells
    """
    # Directions: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    visited = set()
    
    def backtrack(row, col, direction):
        # Clean current cell
        robot.clean()
        visited.add((row, col))
        
        # Try all 4 directions
        for i in range(4):
            new_direction = (direction + i) % 4
            new_row = row + directions[new_direction][0]
            new_col = col + directions[new_direction][1]
            
            if (new_row, new_col) not in visited and robot.move():
                backtrack(new_row, new_col, new_direction)
                
                # Backtrack: return to current cell
                robot.turnRight()
                robot.turnRight()
                robot.move()
                robot.turnRight()
                robot.turnRight()
            
            # Turn to next direction
            robot.turnLeft()
    
    backtrack(0, 0, 0)

# Spiral cleaning approach
def cleanRoomSpiral(robot):
    """
    Alternative spiral approach
    """
    def move_back():
        robot.turnLeft()
        robot.turnLeft()
        robot.move()
        robot.turnRight()
        robot.turnRight()
    
    def backtrack(row, col, direction):
        visited.add((row, col))
        robot.clean()
        
        for i in range(4):
            new_row = row + directions[direction][0]
            new_col = col + directions[direction][1]
            
            if (new_row, new_col not in visited and robot.move():
                backtrack(new_row, new_col, direction)
                move_back()
            
            robot.turnLeft()
            direction = (direction - 1) % 4
    
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    visited = set()
    backtrack(0, 0, 0)
```

---

## Problem 7: Regular Expression Matching

**Difficulty**: 🔴 Hard  
**Pattern**: Dynamic Programming  
**Time**: O(m×n), **Space**: O(m×n)

### Problem Overview

Given an input string `s` and a pattern `p`, implement regular expression matching with support for `'.'` and `'*'` where:

- `'.'` matches any single character
- `'*'` matches zero or more of the preceding element

### Solution

```python
def isMatch(s, p):
    """
    Dynamic programming approach
    """
    m, n = len(s), len(p)
    
    # dp[i][j] = True if s[:i] matches p[:j]
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    
    # Base case: empty string matches empty pattern
    dp[0][0] = True
    
    # Handle patterns like a*, a*b*, a*b*c*
    for j in range(2, n + 1):
        dp[0][j] = p[j-1] == '*' and dp[0][j-2]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                # Two choices: use * or don't use *
                dp[i][j] = dp[i][j-2] or (dp[i-1][j] and (s[i-1] == p[j-2] or p[j-2] == '.'))
            else:
                # Character must match
                dp[i][j] = dp[i-1][j-1] and (s[i-1] == p[j-1] or p[j-1] == '.')
    
    return dp[m][n]

# Recursive approach with memoization
def isMatchRecursive(s, p):
    """
    Recursive approach with memoization
    """
    memo = {}
    
    def dp(i, j):
        if (i, j) in memo:
            return memo[(i, j)]
        
        if j == len(p):
            result = i == len(s)
        else:
            first_match = i < len(s) and (p[j] == s[i] or p[j] == '.')
            
            if j + 1 < len(p) and p[j + 1] == '*':
                result = dp(i, j + 2) or (first_match and dp(i + 1, j))
            else:
                result = first_match and dp(i + 1, j + 1)
        
        memo[(i, j)] = result
        return result
    
    return dp(0, 0)

# Test
print(isMatch("aa", "a"))      # False
print(isMatch("aa", "a*"))     # True
print(isMatch("ab", ".*"))     # True
print(isMatch("aab", "c*a*b")) # True
```

---

## 📝 Summary

### Advanced Algorithmic Techniques Mastered

1. **Modular Exponentiation** - Handle extremely large numbers efficiently
2. **Mathematical Analysis** - Break down complex problems into patterns
3. **Stack-based Optimization** - Solve geometric and parsing problems
4. **Merge Sort Extensions** - Count inversions and range queries
5. **Coordinate Compression** - Handle large ranges efficiently
6. **Backtracking with State** - Explore complex state spaces systematically
7. **Advanced Dynamic Programming** - Handle pattern matching and optimization

### Problem-Solving Strategies

| **Problem Type** | **Key Insight** | **Complexity** |
|------------------|-----------------|----------------|
| **Large Number Operations** | Use modular arithmetic properties | O(n) |
| **Digit Analysis** | Mathematical pattern recognition | O(log n) |
| **Geometric Optimization** | Stack for efficient area calculation | O(n) |
| **Expression Parsing** | Precedence handling with stack/recursion | O(n) |
| **Range Queries** | Combine sorting with counting | O(n log n) |
| **Grid Exploration** | Systematic backtracking with coordinates | O(4^n) |
| **Pattern Matching** | Dynamic programming with state transitions | O(m×n) |

### Mathematical Properties Utilized

- **Modular arithmetic**: `(a^b) mod m = ((a^c) mod m)^d * (a^e) mod m`
- **Digit distribution**: Geometric series for counting digits
- **Stack properties**: LIFO for optimal substructure problems
- **Coordinate systems**: Transform complex grids to simple coordinates
- **Pattern recognition**: Break complex patterns into simpler rules

### Advanced Optimization Techniques

1. **Space optimization**: Convert 2D DP to 1D when possible
2. **Coordinate compression**: Map large ranges to smaller indices
3. **Mathematical shortcuts**: Use number theory to avoid brute force
4. **State pruning**: Eliminate impossible states early
5. **Memory efficient backtracking**: Minimize state storage

---

## 🏆 Mastery Achievement

Congratulations! You've conquered the most challenging mathematical algorithms used in:

- **Cryptography** - Modular exponentiation for RSA encryption
- **Computational Geometry** - Optimal area and shape calculations
- **Compiler Design** - Expression parsing and evaluation
- **Database Systems** - Range queries and indexing
- **Machine Learning** - Pattern recognition and optimization
- **Competitive Programming** - Contest-level problem solving

### 📚 Next Steps

- **Apply to Real Projects** - Use these algorithms in practical applications
- **Competitive Programming** - Practice on platforms like Codeforces, TopCoder
- **Research Papers** - Study cutting-edge algorithmic research
- **System Design** - Scale these algorithms for distributed systems
- **Optimization Theory** - Explore mathematical optimization further

---

*You now possess the mathematical algorithmic skills to tackle any computational challenge!*
