# Edit Distance

## Overview

Edit Distance (also known as Levenshtein Distance) is a measure of similarity between two strings. It quantifies how many operations are required to transform one string into another. The operations typically allowed are:

1. **Insertion**: Adding a character
2. **Deletion**: Removing a character
3. **Substitution**: Replacing one character with another

This algorithm has broad applications in spell checking, DNA sequence analysis, plagiarism detection, and natural language processing.

## Problem Statement

Given two strings `str1` and `str2`, find the minimum number of operations required to convert `str1` into `str2`.

**Example:**
```
str1 = "kitten"
str2 = "sitting"

Edit Distance = 3
Operations:
1. kitten → sitten (substitute 'k' with 's')
2. sitten → sittin (substitute 'e' with 'i')
3. sittin → sitting (insert 'g' at the end)
```

## Approaches

### 1. Recursive Approach

The most intuitive approach is recursive, considering all possibilities at each step:

```python
def edit_distance_recursive(str1, str2, m, n):
    # If first string is empty, insert all characters of second string
    if m == 0:
        return n
    
    # If second string is empty, remove all characters of first string
    if n == 0:
        return m
    
    # If last characters are same, ignore them and process remaining
    if str1[m-1] == str2[n-1]:
        return edit_distance_recursive(str1, str2, m-1, n-1)
    
    # If last characters are different, consider all operations
    # Insert, Remove, Replace
    return 1 + min(
        edit_distance_recursive(str1, str2, m, n-1),    # Insert
        edit_distance_recursive(str1, str2, m-1, n),    # Remove
        edit_distance_recursive(str1, str2, m-1, n-1)   # Replace
    )
```

**Time Complexity**: O(3^(m+n)) - Exponential due to many overlapping subproblems
**Space Complexity**: O(m+n) for recursion stack

### 2. Dynamic Programming Approach

We can optimize the recursive solution using dynamic programming to avoid recalculating the same subproblems:

```python
def edit_distance_dp(str1, str2):
    m, n = len(str1), len(str2)
    
    # Create a table to store results of subproblems
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
    
    # Fill dp[][] in bottom-up manner
    for i in range(m + 1):
        for j in range(n + 1):
            # If first string is empty, insert all characters of second string
            if i == 0:
                dp[i][j] = j
            
            # If second string is empty, remove all characters of first string
            elif j == 0:
                dp[i][j] = i
            
            # If last characters are same, ignore them and process remaining
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            
            # If last characters are different, consider all operations
            # Insert, Remove, Replace
            else:
                dp[i][j] = 1 + min(
                    dp[i][j-1],      # Insert
                    dp[i-1][j],      # Remove
                    dp[i-1][j-1]     # Replace
                )
    
    return dp[m][n]
```

**Time Complexity**: O(m×n) where m and n are lengths of the strings
**Space Complexity**: O(m×n) for the DP table

### 3. Space-Optimized Dynamic Programming

We can further optimize the space complexity by noting that we only need the previous row to calculate the current row:

```python
def edit_distance_dp_optimized(str1, str2):
    m, n = len(str1), len(str2)
    
    # Create two arrays to store previous and current row
    prev = [j for j in range(n + 1)]
    curr = [0 for _ in range(n + 1)]
    
    # Fill dp[][] in bottom-up manner
    for i in range(1, m + 1):
        curr[0] = i  # First column
        
        for j in range(1, n + 1):
            # If characters are same, copy diagonal value
            if str1[i-1] == str2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(
                    curr[j-1],  # Insert
                    prev[j],    # Remove
                    prev[j-1]   # Replace
                )
        
        # Copy current row to previous row for next iteration
        prev = curr.copy()
    
    return prev[n]
```

**Time Complexity**: O(m×n)
**Space Complexity**: O(n) - We only store two rows at a time

## Variations

### 1. Different Operation Costs

Sometimes, different operations might have different costs. For example, substitution might be more expensive than insertion or deletion:

```python
def weighted_edit_distance(str1, str2, insert_cost, delete_cost, substitute_cost):
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i * delete_cost
    
    for j in range(n + 1):
        dp[0][j] = j * insert_cost
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i][j-1] + insert_cost,
                    dp[i-1][j] + delete_cost,
                    dp[i-1][j-1] + substitute_cost
                )
    
    return dp[m][n]
```

### 2. Damerau-Levenshtein Distance

This variation also allows transposition of adjacent characters as an operation:

```python
def damerau_levenshtein_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if str1[i-1] == str2[j-1] else 1
            
            # Standard operations
            dp[i][j] = min(
                dp[i-1][j] + 1,          # Deletion
                dp[i][j-1] + 1,          # Insertion
                dp[i-1][j-1] + cost      # Substitution
            )
            
            # Transposition
            if (i > 1 and j > 1 and 
                str1[i-1] == str2[j-2] and 
                str1[i-2] == str2[j-1]):
                dp[i][j] = min(dp[i][j], dp[i-2][j-2] + cost)
    
    return dp[m][n]
```

### 3. Longest Common Subsequence

A related problem is finding the longest common subsequence of two strings, which can be used to visualize the differences:

```python
def longest_common_subsequence(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Length of LCS is dp[m][n]
    # We can also reconstruct the LCS itself
    return dp[m][n]
```

## Applications

1. **Spell Checking**: Suggesting corrections for misspelled words
2. **DNA Sequence Analysis**: Comparing genetic sequences
3. **Plagiarism Detection**: Identifying similarities between documents
4. **Machine Translation**: Evaluating translation quality
5. **Information Retrieval**: Fuzzy string matching for search
6. **OCR Post-Processing**: Correcting optical character recognition errors
7. **Autocorrect Systems**: Suggesting corrections for typing errors

## Implementation Strategies

### Memoization (Top-Down DP)

For a recursive approach with memoization:

```python
def edit_distance_memoized(str1, str2):
    m, n = len(str1), len(str2)
    memo = {}
    
    def dp(i, j):
        if (i, j) in memo:
            return memo[(i, j)]
        
        if i == 0:
            return j
        if j == 0:
            return i
        
        if str1[i-1] == str2[j-1]:
            result = dp(i-1, j-1)
        else:
            result = 1 + min(
                dp(i, j-1),    # Insert
                dp(i-1, j),    # Delete
                dp(i-1, j-1)   # Replace
            )
        
        memo[(i, j)] = result
        return result
    
    return dp(m, n)
```

### Backtracking to Find Operations

To identify the actual operations performed:

```python
def edit_distance_with_operations(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Fill the DP table
    for i in range(m + 1):
        dp[i][0] = i
    
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i][j-1],      # Insert
                    dp[i-1][j],      # Delete
                    dp[i-1][j-1]     # Replace
                )
    
    # Backtrack to find operations
    operations = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and str1[i-1] == str2[j-1]:
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j] == dp[i][j-1] + 1):
            operations.append(("insert", i, str2[j-1]))
            j -= 1
        elif i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] + 1):
            operations.append(("delete", i-1, str1[i-1]))
            i -= 1
        else:
            operations.append(("replace", i-1, str1[i-1], str2[j-1]))
            i -= 1
            j -= 1
    
    # Reverse to get operations in correct order
    operations.reverse()
    
    return dp[m][n], operations
```

## Performance Considerations

- **String Length**: The algorithm's complexity grows with the product of string lengths
- **Alphabet Size**: For very large alphabets, specialized algorithms might be more efficient
- **Memory Constraints**: For very long strings, consider the space-optimized approach
- **Parallelization**: Edit distance can be parallelized for multiple string comparisons

## Related Algorithms

- [Longest Common Subsequence](lcs.md)
- [String Comparison Methods](comparison.md)
- [Pattern Matching](pattern-matching.md)
- [Dynamic Programming](../dp/fundamentals.md)

## References

1. Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions, and reversals. Soviet Physics Doklady, 10(8), 707-710.
2. Wagner, R. A., & Fischer, M. J. (1974). The string-to-string correction problem. Journal of the ACM, 21(1), 168-173.
3. Navarro, G. (2001). A guided tour to approximate string matching. ACM Computing Surveys, 33(1), 31-88.
