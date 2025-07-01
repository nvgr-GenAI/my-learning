# Longest Common Subsequence (LCS)

## Problem Statement

Given two strings, find the length of their longest common subsequence. A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.

## Approaches

### Recursive Solution

```python
def lcs_recursive(text1, text2, i, j):
    # Base case
    if i == len(text1) or j == len(text2):
        return 0
    
    # If characters match
    if text1[i] == text2[j]:
        return 1 + lcs_recursive(text1, text2, i + 1, j + 1)
    
    # If characters don't match, try both possibilities
    return max(
        lcs_recursive(text1, text2, i + 1, j),
        lcs_recursive(text1, text2, i, j + 1)
    )
```

### Memoization Solution

```python
def lcs_memo(text1, text2):
    memo = {}
    
    def dp(i, j):
        if (i, j) in memo:
            return memo[(i, j)]
        
        if i == len(text1) or j == len(text2):
            return 0
        
        if text1[i] == text2[j]:
            result = 1 + dp(i + 1, j + 1)
        else:
            result = max(dp(i + 1, j), dp(i, j + 1))
        
        memo[(i, j)] = result
        return result
    
    return dp(0, 0)
```

### Tabulation Solution

```python
def lcs_dp(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

### Space Optimized Solution

```python
def lcs_optimized(text1, text2):
    m, n = len(text1), len(text2)
    
    # Use only two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                curr[j] = 1 + prev[j-1]
            else:
                curr[j] = max(prev[j], curr[j-1])
        
        prev, curr = curr, prev
    
    return prev[n]
```

## Finding the Actual LCS

```python
def lcs_string(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct the LCS
    lcs = []
    i, j = m, n
    
    while i > 0 and j > 0:
        if text1[i-1] == text2[j-1]:
            lcs.append(text1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(reversed(lcs))
```

## Variations and Applications

### Longest Common Substring

```python
def lcs_substring(text1, text2):
    """Find length of longest common substring (contiguous)"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                max_length = max(max_length, dp[i][j])
            else:
                dp[i][j] = 0
    
    return max_length
```

### Edit Distance (Levenshtein Distance)

```python
def edit_distance(word1, word2):
    """Minimum operations to convert word1 to word2"""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # No operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Delete
                    dp[i][j-1],    # Insert
                    dp[i-1][j-1]   # Replace
                )
    
    return dp[m][n]
```

### Longest Increasing Subsequence

```python
def lis_length(nums):
    """Length of longest increasing subsequence"""
    if not nums:
        return 0
    
    dp = [1] * len(nums)
    
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)
```

## Time and Space Complexity

| Approach | Time | Space |
|----------|------|-------|
| Recursive | O(2^(m+n)) | O(m+n) |
| Memoization | O(m×n) | O(m×n) |
| Tabulation | O(m×n) | O(m×n) |
| Optimized | O(m×n) | O(n) |

## Pattern Recognition

LCS pattern applies when:

1. **Comparing two sequences**
2. **Finding optimal alignment**
3. **Subsequence/substring problems**
4. **Edit distance variations**

## Practice Problems

- [ ] Longest Common Subsequence
- [ ] Longest Common Substring
- [ ] Edit Distance
- [ ] Delete Operation for Two Strings
- [ ] Minimum ASCII Delete Sum
- [ ] Longest Increasing Subsequence
- [ ] Maximum Length of Pair Chain
