# Interval Dynamic Programming

## Overview

Interval Dynamic Programming is a DP paradigm that deals with problems involving ranges or intervals of elements. The key characteristic is that the state typically depends on breaking down a larger interval into smaller subintervals.

## Problem Pattern

In interval DP problems, we typically:

1. **Define states** that represent intervals `[i, j]` of the input
2. **Break down** the problem for interval `[i, j]` by considering smaller subintervals
3. **Find the optimal way** to combine solutions from smaller intervals

## Common Recurrence Relation Pattern

The general form of the recurrence relation is often:

```
dp[i][j] = optimal(dp[i][k] + dp[k+1][j] + cost) for all k where i ≤ k < j
```

Where:
- `dp[i][j]` represents the optimal solution for the interval `[i, j]`
- `k` is the splitting point within the interval
- `cost` is some additional cost for combining the subproblems

## Classic Interval DP Problems

### 1. Matrix Chain Multiplication

**Problem**: Given a chain of matrices, find the most efficient way to multiply them together.

**State Definition**: `dp[i][j]` = minimum cost to multiply matrices from i to j

**Recurrence Relation**: `dp[i][j] = min(dp[i][k] + dp[k+1][j] + p[i-1] * p[k] * p[j])` for all k from i to j-1

```python
def matrix_chain_multiplication(p):
    """
    p[i] represents the dimension of matrix i:
    Matrix i has dimensions p[i-1] x p[i]
    """
    n = len(p) - 1  # Number of matrices
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    
    # Length is the chain length
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + p[i-1] * p[k] * p[j]
                dp[i][j] = min(dp[i][j], cost)
    
    return dp[1][n]
```

### 2. Burst Balloons

**Problem**: Given n balloons with values, burst them one by one to maximize total points. Points = value of current balloon × value of left balloon × value of right balloon.

**State Definition**: `dp[i][j]` = maximum coins from bursting all balloons in range (i, j)

**Recurrence Relation**: `dp[i][j] = max(dp[i][k-1] + dp[k+1][j] + nums[i-1] * nums[k] * nums[j+1])` for all k from i to j

```python
def max_coins(nums):
    # Add 1 at the beginning and end
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    
    # Length of interval
    for length in range(1, n - 1):
        for i in range(1, n - length):
            j = i + length - 1
            for k in range(i, j + 1):
                # Last balloon to burst in range (i,j)
                dp[i][j] = max(dp[i][j], 
                               dp[i][k-1] + dp[k+1][j] + nums[i-1] * nums[k] * nums[j+1])
    
    return dp[1][n-2]
```

### 3. Palindrome Partitioning II

**Problem**: Partition a string such that every substring is a palindrome. Return the minimum number of cuts needed.

**State Definition**: `dp[i]` = minimum cuts needed for s[0...i]

First, we compute a table `isPalindrome[i][j]` to check if s[i...j] is a palindrome:

```python
def min_cut(s):
    n = len(s)
    
    # isPalindrome[i][j] = True if s[i...j] is palindrome
    isPalindrome = [[False] * n for _ in range(n)]
    
    # All single characters are palindromes
    for i in range(n):
        isPalindrome[i][i] = True
    
    # Check for palindromes of length 2 or more
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if length == 2:
                isPalindrome[i][j] = (s[i] == s[j])
            else:
                isPalindrome[i][j] = (s[i] == s[j] and isPalindrome[i+1][j-1])
    
    # dp[i] = minimum cuts needed for s[0...i]
    dp = [float('inf')] * n
    
    for i in range(n):
        if isPalindrome[0][i]:
            dp[i] = 0  # If s[0...i] is a palindrome, no cuts needed
        else:
            for j in range(i):
                if isPalindrome[j+1][i]:
                    dp[i] = min(dp[i], dp[j] + 1)
    
    return dp[n-1]
```

### 4. Stone Game

**Problem**: Two players take turns removing stones from piles. Player with the most stones wins.

**State Definition**: `dp[i][j]` = maximum score difference for the first player for piles from i to j

**Recurrence Relation**: `dp[i][j] = max(piles[i] - dp[i+1][j], piles[j] - dp[i][j-1])`

```python
def stone_game(piles):
    n = len(piles)
    dp = [[0] * n for _ in range(n)]
    
    # Base case: Single pile
    for i in range(n):
        dp[i][i] = piles[i]
    
    # Fill the table diagonally
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = max(piles[i] - dp[i+1][j], piles[j] - dp[i][j-1])
    
    return dp[0][n-1] > 0  # First player wins if advantage > 0
```

## Advanced Interval DP Problems

### 1. Minimum Score Triangulation

**Problem**: Given a convex polygon with n vertices, find the minimum score triangulation.

**State Definition**: `dp[i][j]` = minimum score of triangulating vertices from i to j

**Recurrence Relation**: `dp[i][j] = min(dp[i][k] + dp[k][j] + values[i] * values[k] * values[j])` for all k from i+1 to j-1

```python
def min_score_triangulation(values):
    n = len(values)
    dp = [[0] * n for _ in range(n)]
    
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i + 1, j):
                dp[i][j] = min(dp[i][j], 
                               dp[i][k] + dp[k][j] + values[i] * values[k] * values[j])
    
    return dp[0][n-1]
```

### 2. Remove Boxes

**Problem**: Remove boxes of the same color to maximize points.

**State Definition**: `dp[i][j][k]` = maximum points from removing boxes in range [i, j] with k extra boxes of same color as box[j]

```python
def remove_boxes(boxes):
    n = len(boxes)
    # dp[i][j][k] = max points from [i,j] plus k same-colored boxes as boxes[j]
    dp = [[[0] * n for _ in range(n)] for _ in range(n)]
    
    def calculate(i, j, k):
        if i > j:
            return 0
        if dp[i][j][k] != 0:
            return dp[i][j][k]
        
        # Initial case: remove boxes[j] and k same-colored boxes
        dp[i][j][k] = calculate(i, j - 1, 0) + (k + 1) ** 2
        
        # Try to group boxes with same color
        for m in range(i, j):
            if boxes[m] == boxes[j]:
                dp[i][j][k] = max(dp[i][j][k],
                                 calculate(i, m, k + 1) + calculate(m + 1, j - 1, 0))
        
        return dp[i][j][k]
    
    return calculate(0, n - 1, 0)
```

### 3. Strange Printer

**Problem**: A printer can print a string by printing a contiguous substring of the same character each time.

**State Definition**: `dp[i][j]` = minimum number of operations to print s[i...j]

```python
def strange_printer(s):
    n = len(s)
    dp = [[float('inf')] * n for _ in range(n)]
    
    # Base case: single character
    for i in range(n):
        dp[i][i] = 1
    
    # Consider all substrings
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Assume printing s[i] across the range
            dp[i][j] = dp[i+1][j] + 1
            
            # Check if we can optimize
            for k in range(i + 1, j + 1):
                if s[i] == s[k]:
                    dp[i][j] = min(dp[i][j], 
                                   dp[i][k-1] + (dp[k+1][j] if k < j else 0))
    
    return dp[0][n-1]
```

## General Problem-Solving Approach

1. **Identify the interval structure**: Determine if the problem can be broken into subintervals
2. **Define the state**: What does `dp[i][j]` represent for interval `[i, j]`?
3. **Establish the recurrence relation**: How can we combine solutions from smaller intervals?
4. **Determine the base cases**: What are the solutions for the smallest intervals?
5. **Choose the traversal order**: Usually bottom-up with increasing interval length
6. **Implement memoization or tabulation**: Most interval DP problems use tabulation (bottom-up)

## Time and Space Complexity

For most interval DP problems:

- **Time Complexity**: O(n³), due to three nested loops:
  - Outer loop for interval length: O(n)
  - Middle loop for starting position: O(n)
  - Inner loop for splitting position: O(n)

- **Space Complexity**: O(n²) for the DP table

## Optimization Techniques

1. **Pruning**: Skip certain states if they're known to be non-optimal
2. **Precalculation**: Precompute certain values (e.g., palindrome checks)
3. **Monotonicity**: Sometimes the optimal split point follows a monotonic pattern

## Pattern Recognition Tips

Interval DP might be applicable when:

1. **Breaking an array/string into subarrays/substrings**
2. **Combining operations on subarrays**
3. **Optimizing costs that depend on the order of operations**
4. **Problems involving parenthesization**

## Practice Problems

1. **Matrix Chain Multiplication**: Minimize the cost of multiplying a chain of matrices
2. **Burst Balloons**: Maximize points from bursting balloons
3. **Palindrome Partitioning II**: Minimize cuts needed to partition a string into palindromes
4. **Stone Game**: Determine if the first player can win a game of taking stones
5. **Minimum Score Triangulation of Polygon**: Find the minimum score triangulation
6. **Remove Boxes**: Maximize points from removing boxes
7. **Strange Printer**: Minimize operations for a printer that prints same-character substrings
