# Longest Common Subsequence (LCS)

## Introduction

The Longest Common Subsequence (LCS) is a classic dynamic programming problem that involves finding the longest subsequence common to two sequences. A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.

=== "Overview"
    **Core Idea**: Find the longest subsequence (not necessarily contiguous) that appears in both sequences in the same relative order.
    
    **When to Use**:
    
    - When comparing similarities between two sequences
    - When finding elements that appear in both sequences in the same order
    - When problems involve string or array matching with gaps allowed
    - When needing to find the minimum operations to transform one sequence to another
    
    **Recurrence Relation**:
    
    - If characters match: `dp[i][j] = 1 + dp[i-1][j-1]`
    - If not: `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`
    
    **Real-World Applications**:
    
    - DNA sequence alignment in bioinformatics
    - File difference algorithms
    - Plagiarism detection systems
    - Autocorrect and text suggestion features

=== "Example Problems"
    - **Longest Common Subsequence**: Find the longest subsequence common to two strings
      - Problem: Given strings "abcde" and "ace", the LCS is "ace" with length 3
      - Insight: We build the solution incrementally, matching characters when possible
    
    - **Shortest Common Supersequence**: Find the shortest string that has both input strings as subsequences
      - Problem: For "abac" and "cab", the shortest supersequence is "cabac" with length 5
      - Solution: Find LCS first, then merge both strings by including the LCS only once
    
    - **Delete Operation for Two Strings**: Find minimum number of deletions to make two strings equal
      - Problem: To make "sea" and "eat" equal, delete 's' from first and 't' from second
      - Insight: Delete all characters not in LCS (length = total length - 2*LCS length)
    
    - **Minimum ASCII Delete Sum**: Delete characters to make strings equal, minimizing ASCII sum
      - Variation: Instead of counting deletions, we consider the ASCII values
      - Shows how the LCS pattern can be adapted to different cost metrics

=== "Visualization"
    For the LCS of "abcde" and "ace":
    
    ```text
        | "" | a | c | e |
    --------------------- 
    "" | 0  | 0 | 0 | 0 |
    --------------------- 
    a  | 0  | 1 | 1 | 1 |
    ---------------------
    b  | 0  | 1 | 1 | 1 |
    ---------------------
    c  | 0  | 1 | 2 | 2 |
    ---------------------
    d  | 0  | 1 | 2 | 2 |
    ---------------------
    e  | 0  | 1 | 2 | 3 |
    ```
    
    The table shows dp[i][j] = length of LCS for first i characters of string1 and first j characters of string2.
    
    ![LCS Pattern Visualization](https://i.imgur.com/RkoBlS5.png)

=== "Implementation"
    **Standard Implementation**:
    
    ```python
    def lcs_pattern(text1, text2):
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
    
    **With Backtracking to Reconstruct LCS**:
    
    ```python
    def lcs_with_reconstruction(text1, text2):
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the dp table
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
    
    **Space-Optimized Implementation**:
    
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

=== "Tips and Insights"
    - **Pattern Recognition**: Look for problems involving finding the longest shared elements between two sequences
    - **State Definition**: `dp[i][j]` typically represents the LCS length for first `i` characters of string1 and first `j` characters of string2
    - **Overlapping Subproblems**: Notice how many subproblems are solved repeatedly in the recursive approach
    - **Optimization**: Many LCS variants can be solved by slightly modifying the base algorithm
    - **Backtracking**: Recovering the actual LCS requires backtracking through the DP table
    - **Application Areas**: 
      - DNA sequence alignment in bioinformatics
      - Text comparison tools like diff
      - Version control systems
      - Plagiarism detection
    - **Complexity Comparison**:
      - Naive recursion: O(2^(m+n))
      - DP solution: O(m*n) time, O(m*n) space
      - Space-optimized version: O(m*n) time, O(min(m,n)) space
    - **Related Problems**: 
      - Longest Increasing Subsequence
      - Shortest Common Supersequence
      - Minimum Edit Distance
      - Longest Palindromic Subsequence (LCS of string and its reverse)
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
