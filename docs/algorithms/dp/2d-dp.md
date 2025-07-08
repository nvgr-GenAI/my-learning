# 2D Dynamic Programming

## Overview

2D Dynamic Programming involves problems where we need to maintain a two-dimensional state to compute optimal solutions. These problems typically involve:

- Grids or matrices
- Two sequences or strings
- Two-dimensional decision spaces

## Core Concepts

### State Representation

In 2D DP, we typically use a table `dp[i][j]` where:

- `i` often represents a position or length in the first sequence/dimension
- `j` often represents a position or length in the second sequence/dimension

### Common Problem Types

1. **Grid Traversal Problems**: Finding paths through a grid
2. **Two-Sequence Problems**: Comparing or combining two sequences
3. **Interval Problems**: Processing ranges of elements
4. **Game Theory Problems**: Optimal play on 2D structures

## Grid Traversal Problems

### Unique Paths

**Problem**: Count the number of ways to reach the bottom-right cell from the top-left cell in a grid, moving only right or down.

**State Definition**: `dp[i][j]` = number of ways to reach cell (i,j)

**Recurrence Relation**: `dp[i][j] = dp[i-1][j] + dp[i][j-1]`

```python
def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[m-1][n-1]
```

### Minimum Path Sum

**Problem**: Find the path with minimum sum from top-left to bottom-right in a grid, moving only right or down.

**State Definition**: `dp[i][j]` = minimum path sum to reach cell (i,j)

**Recurrence Relation**: `dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])`

```python
def min_path_sum(grid):
    if not grid:
        return 0
    
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    
    # Initialize first row
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    
    # Initialize first column
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    
    # Fill the rest of the table
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
    
    return dp[m-1][n-1]
```

### Maximal Square

**Problem**: Find the area of the largest square containing only 1's in a binary matrix.

**State Definition**: `dp[i][j]` = side length of largest square ending at position (i,j)

**Recurrence Relation**: If `matrix[i][j] == '1'`, then:
`dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1`

```python
def maximal_square(matrix):
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * (n+1) for _ in range(m+1)]
    max_side = 0
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if matrix[i-1][j-1] == '1':
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                max_side = max(max_side, dp[i][j])
    
    return max_side * max_side  # Area = side^2
```

## Two-Sequence Problems

### Longest Common Subsequence

**Problem**: Find the length of the longest subsequence present in both given sequences.

**State Definition**: `dp[i][j]` = length of LCS of first i characters of text1 and first j characters of text2

**Recurrence Relation**:

- If `text1[i-1] == text2[j-1]`: `dp[i][j] = dp[i-1][j-1] + 1`
- Otherwise: `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`

```python
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n+1) for _ in range(m+1)]
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

### Edit Distance

**Problem**: Find the minimum number of operations required to convert one string to another.

**State Definition**: `dp[i][j]` = minimum number of operations to convert first i characters of word1 to first j characters of word2

**Recurrence Relation**:

- If `word1[i-1] == word2[j-1]`: `dp[i][j] = dp[i-1][j-1]`
- Otherwise: `dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])` (insert, delete, replace)

```python
def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n+1) for _ in range(m+1)]
    
    # Base cases
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],     # Delete
                    dp[i][j-1],     # Insert
                    dp[i-1][j-1]    # Replace
                )
    
    return dp[m][n]
```

## Interval Problems

### Palindrome Partitioning

**Problem**: Find the minimum number of cuts needed to partition a string into palindromes.

**State Definition**: `dp[i]` = minimum cuts needed for s[0...i]
We also use a 2D table `isPalindrome[i][j]` to check if s[i...j] is a palindrome.

```python
def min_cut(s):
    n = len(s)
    # isPalindrome[i][j] = True if s[i...j] is a palindrome
    isPalindrome = [[False] * n for _ in range(n)]
    
    # All single characters are palindromes
    for i in range(n):
        isPalindrome[i][i] = True
    
    # Check for palindromes of length 2 or more
    for length in range(2, n+1):
        for i in range(n-length+1):
            j = i + length - 1
            if length == 2:
                isPalindrome[i][j] = (s[i] == s[j])
            else:
                isPalindrome[i][j] = (s[i] == s[j] and isPalindrome[i+1][j-1])
    
    # dp[i] = minimum cuts needed for s[0...i]
    dp = [float('inf')] * n
    
    for i in range(n):
        if isPalindrome[0][i]:
            dp[i] = 0  # If s[0...i] is already a palindrome, no cuts needed
        else:
            for j in range(i):
                if isPalindrome[j+1][i]:
                    dp[i] = min(dp[i], dp[j] + 1)
    
    return dp[n-1]
```

### Matrix Chain Multiplication

**Problem**: Find the most efficient way to multiply a chain of matrices.

**State Definition**: `dp[i][j]` = minimum cost to multiply matrices from i to j

**Recurrence Relation**: `dp[i][j] = min(dp[i][k] + dp[k+1][j] + p[i-1] * p[k] * p[j])` for all k from i to j-1

```python
def matrix_chain_multiplication(p):
    """
    p[i] = dimension between matrix i and i+1
    e.g., for matrices A(10x30), B(30x5), C(5x60), p = [10, 30, 5, 60]
    """
    n = len(p) - 1  # Number of matrices
    dp = [[0] * (n+1) for _ in range(n+1)]
    
    for length in range(2, n+1):
        for i in range(1, n-length+2):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + p[i-1] * p[k] * p[j]
                dp[i][j] = min(dp[i][j], cost)
    
    return dp[1][n]
```

## Advanced 2D DP Problems

### Distinct Subsequences

**Problem**: Count the number of distinct subsequences of t in s.

**State Definition**: `dp[i][j]` = number of ways to get first j characters of t from first i characters of s

**Recurrence Relation**:

- If `s[i-1] == t[j-1]`: `dp[i][j] = dp[i-1][j-1] + dp[i-1][j]` (use or don't use current character)
- Otherwise: `dp[i][j] = dp[i-1][j]` (can't use current character)

```python
def num_distinct(s, t):
    m, n = len(s), len(t)
    dp = [[0] * (n+1) for _ in range(m+1)]
    
    # Empty t is a subsequence of any s (exactly once)
    for i in range(m+1):
        dp[i][0] = 1
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Current characters match
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
            else:
                dp[i][j] = dp[i-1][j]
    
    return dp[m][n]
```

### Interleaving String

**Problem**: Determine if s3 is formed by the interleaving of s1 and s2.

**State Definition**: `dp[i][j]` = whether first i+j characters of s3 can be formed by interleaving first i characters of s1 and first j characters of s2

```python
def is_interleave(s1, s2, s3):
    m, n = len(s1), len(s2)
    if len(s3) != m + n:
        return False
    
    dp = [[False] * (n+1) for _ in range(m+1)]
    dp[0][0] = True  # Empty strings interleave to form empty string
    
    # Initialize first row
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
    
    # Initialize first column
    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
    
    # Fill the table
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or \
                       (dp[i][j-1] and s2[j-1] == s3[i+j-1])
    
    return dp[m][n]
```

### Wildcard Matching

**Problem**: Implement wildcard pattern matching with '?' (any single character) and '*' (any sequence).

**State Definition**: `dp[i][j]` = whether first i characters of s match first j characters of pattern p

```python
def is_match(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n+1) for _ in range(m+1)]
    dp[0][0] = True  # Empty pattern matches empty string
    
    # Handle patterns starting with '*'
    for j in range(1, n+1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-1]
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if p[j-1] == '*':
                dp[i][j] = dp[i][j-1] or dp[i-1][j]  # Zero or more characters
            elif p[j-1] == '?' or s[i-1] == p[j-1]:
                dp[i][j] = dp[i-1][j-1]  # Single character match
    
    return dp[m][n]
```

## Optimization Techniques for 2D DP

### Space Optimization

Many 2D DP problems can be optimized to use O(n) space instead of O(n²) by keeping only the necessary rows/columns.

#### Example: Optimized Longest Common Subsequence

```python
def lcs_optimized(text1, text2):
    m, n = len(text1), len(text2)
    
    # Use only two rows
    prev = [0] * (n+1)
    curr = [0] * (n+1)
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if text1[i-1] == text2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        
        # Swap rows
        prev, curr = curr, prev
        
        # Reset current row for next iteration
        for j in range(n+1):
            curr[j] = 0
    
    return prev[n]
```

### Diagonal Traversal

For problems where cells depend on diagonal cells, traverse the matrix diagonally.

#### Example: Optimized Palindrome Substrings

```python
def count_palindrome_substrings(s):
    n = len(s)
    count = 0
    
    # Consider odd length palindromes
    for center in range(n):
        left = right = center
        while left >= 0 and right < n and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
    
    # Consider even length palindromes
    for center in range(n-1):
        left, right = center, center + 1
        while left >= 0 and right < n and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
    
    return count
```

### Prefix Sums

For problems involving subarrays or submatrices, use prefix sums to compute ranges quickly.

#### Example: Maximum Sum Submatrix

```python
def max_submatrix_sum(matrix):
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    result = float('-inf')
    
    for left in range(n):
        temp = [0] * m
        for right in range(left, n):
            # Calculate sum of current row segment
            for i in range(m):
                temp[i] += matrix[i][right]
            
            # Find maximum subarray sum
            curr_max = kadane(temp)
            result = max(result, curr_max)
    
    return result

def kadane(arr):
    max_so_far = max_ending_here = arr[0]
    for i in range(1, len(arr)):
        max_ending_here = max(arr[i], max_ending_here + arr[i])
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far
```

## Time and Space Complexity Analysis

Most 2D DP solutions have:

- **Time Complexity**: O(m × n), where m and n are the dimensions of the state space
- **Space Complexity**: O(m × n) for the full table, often optimizable to O(min(m, n))

For interval problems like Matrix Chain Multiplication, the time complexity can be O(n³) due to the third nested loop for trying all partition points.

## Practice Problems

| Problem | Difficulty | Pattern |
|---------|-----------|---------|
| Unique Paths | Easy | Grid Traversal |
| Minimum Path Sum | Medium | Grid Traversal |
| Longest Common Subsequence | Medium | Two Sequences |
| Edit Distance | Hard | Two Sequences |
| Regular Expression Matching | Hard | Two Sequences |
| Burst Balloons | Hard | Interval |
| Maximum Submatrix Sum | Hard | Grid + Kadane |
| Distinct Subsequences | Hard | Two Sequences |
| Interleaving String | Hard | Two Sequences |
| Palindrome Partitioning II | Hard | Interval |

## Tips for Solving 2D DP Problems

1. **Identify the state**: Determine what dp[i][j] represents
2. **Define base cases**: Handle the simplest scenarios first
3. **Derive recurrence relation**: How does the current state depend on previous states?
4. **Consider traversal order**: Ensure dependencies are computed before needed
5. **Look for optimization opportunities**: Space reduction, prefix sums, etc.
6. **Test with examples**: Trace through small examples to verify logic
7. **Consider edge cases**: Empty inputs, single elements, etc.
