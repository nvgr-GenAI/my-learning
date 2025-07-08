# Edit Distance Pattern

## Introduction

The Edit Distance pattern is a two-sequence dynamic programming pattern used to find the minimum number of operations required to transform one string into another. It's one of the most versatile and widely used patterns in string comparison and similarity measurement.

=== "Overview"
    **Core Idea**: Calculate the minimum number of operations required to transform one string into another.
    
    **When to Use**:
    
    - When comparing similarity between strings or sequences
    - When you need to find the minimum transformations between sequences
    - When implementing features like autocorrect, spell checking, or DNA sequence comparison
    - When you need to measure how different two strings are
    
    **Recurrence Relation**:
    
    - If characters match: `dp[i][j] = dp[i-1][j-1]` (no operation needed)
    - If not: `dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])` (insert, delete, replace)
    
    **Real-World Applications**:
    
    - Spell checking and autocorrect algorithms
    - DNA sequence alignment in computational biology
    - Plagiarism detection systems
    - Natural language processing for text similarity
    - Fuzzy string matching in search engines

=== "Example Problems"
    - **Edit Distance (Levenshtein Distance)**: Find minimum operations to convert one string to another
      - Problem: Convert "horse" to "ros" using minimum operations
      - Solution: Delete 'h', replace 'r' with 'o', delete 's' and 'e' = 3 operations
    
    - **One Edit Distance**: Determine if two strings are one edit away from each other
      - Problem: Check if strings differ by at most one edit operation
      - Application: Used in "Did you mean?" suggestions in search engines
    
    - **Delete Operation for Two Strings**: Find minimum number of characters to delete to make two strings equal
      - Variation: Only deletion operations are allowed
      - Solution approach: Delete all characters not in LCS (length = total length - 2*LCS length)
    
    - **Word Break**: Determine if a string can be segmented into dictionary words
      - Problem: Given a string and a dictionary of words, can the string be split into words?
      - Approach: Use DP to track which prefixes can be segmented
    
    - **Minimum ASCII Delete Sum**: Find the lowest ASCII sum of deleted characters to make two strings equal
      - Variation: Consider the ASCII value of characters when measuring cost
      - Application: Text processing with character weight considerations

=== "Visualization"
    For the Edit Distance between "horse" and "ros":
    
    ```text
        |   | r | o | s |
    --------------------- 
        | 0 | 1 | 2 | 3 |
    --------------------- 
    h   | 1 | 1 | 2 | 3 |
    ---------------------
    o   | 2 | 2 | 1 | 2 |
    ---------------------
    r   | 3 | 2 | 2 | 2 |
    ---------------------
    s   | 4 | 3 | 3 | 2 |
    ---------------------
    e   | 5 | 4 | 4 | 3 |
    ```
    
    The final edit distance is 3 (bottom-right cell).
    
    Step by step explanation:
    1. Delete 'h' from "horse" to get "orse" (1 operation)
    2. Replace 'o' in "orse" with 'r' to get "rrse" (2 operations)
    3. Delete 'r' from "rrse" to get "rse" (3 operations)
    4. Delete 'e' from "rse" to get "rs" (4 operations)
    5. Insert 'o' to get "ros" (5 operations)
    
    But there's a better way:
    1. Delete 'h' from "horse" to get "orse" (1 operation)
    2. Replace 'r' with 'o' to get "oose" (2 operations) 
    3. Delete 's' and 'e' to get "ros" (3 operations total)
    
    ![Edit Distance Visualization](https://i.imgur.com/JQSxz7j.png)

=== "Implementation"
    **Standard Implementation**:
    
    ```python
    def edit_distance(word1, word2):
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Base cases
        for i in range(m + 1):
            dp[i][0] = i  # Delete operations
        for j in range(n + 1):
            dp[0][j] = j  # Insert operations
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # Delete
                        dp[i][j-1],    # Insert
                        dp[i-1][j-1]   # Replace
                    )
        
        return dp[m][n]
    ```
    
    **Space-Optimized Implementation**:
    
    ```python
    def edit_distance_optimized(word1, word2):
        m, n = len(word1), len(word2)
        
        # Ensure word1 is shorter for space optimization
        if m > n:
            word1, word2 = word2, word1
            m, n = n, m
        
        # Previous and current row
        prev_row = list(range(n + 1))
        curr_row = [0] * (n + 1)
        
        for i in range(1, m + 1):
            curr_row[0] = i
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    curr_row[j] = prev_row[j-1]
                else:
                    curr_row[j] = 1 + min(
                        prev_row[j],    # Delete
                        curr_row[j-1],  # Insert
                        prev_row[j-1]   # Replace
                    )
            prev_row, curr_row = curr_row, [0] * (n + 1)
        
        return prev_row[n]
    ```
    
    **With Operation Tracking**:
    
    ```python
    def edit_distance_with_operations(word1, word2):
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Track operations: 0=match, 1=delete, 2=insert, 3=replace
        ops = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Base cases
        for i in range(1, m + 1):
            dp[i][0] = i
            ops[i][0] = 1  # delete
        for j in range(1, n + 1):
            dp[0][j] = j
            ops[0][j] = 2  # insert
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                    ops[i][j] = 0  # match
                else:
                    delete_cost = dp[i-1][j]
                    insert_cost = dp[i][j-1]
                    replace_cost = dp[i-1][j-1]
                    
                    dp[i][j] = 1 + min(delete_cost, insert_cost, replace_cost)
                    
                    if dp[i][j] == 1 + delete_cost:
                        ops[i][j] = 1  # delete
                    elif dp[i][j] == 1 + insert_cost:
                        ops[i][j] = 2  # insert
                    else:
                        ops[i][j] = 3  # replace
        
        # Reconstruct the sequence of operations
        operations = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ops[i][j] == 0:  # match
                operations.append(f"Match '{word1[i-1]}'")
                i -= 1
                j -= 1
            elif i > 0 and ops[i][j] == 1:  # delete
                operations.append(f"Delete '{word1[i-1]}'")
                i -= 1
            elif j > 0 and ops[i][j] == 2:  # insert
                operations.append(f"Insert '{word2[j-1]}'")
                j -= 1
            else:  # replace
                operations.append(f"Replace '{word1[i-1]}' with '{word2[j-1]}'")
                i -= 1
                j -= 1
        
        return dp[m][n], list(reversed(operations))
    ```

=== "Tips and Insights"
    - **DP State Definition**:
      - `dp[i][j]` = minimum operations to transform first `i` chars of word1 to first `j` chars of word2
    - **Initialization**:
      - `dp[i][0] = i` (delete all characters from word1)
      - `dp[0][j] = j` (insert all characters from word2)
    - **Operations Interpretation**:
      - Delete: Move up in the DP table (dp[i-1][j])
      - Insert: Move left (dp[i][j-1])
      - Replace: Move diagonally (dp[i-1][j-1])
    - **Time Complexity**: O(m×n) where m and n are the lengths of the two strings
    - **Space Complexity**: 
      - Standard: O(m×n)
      - Optimized: O(min(m,n))
    - **Variations**:
      - Different operation costs (weighted edit distance)
      - Restricted operations (only delete, only insert, etc.)
      - Allowing transposition of adjacent characters (Damerau-Levenshtein distance)
    - **Optimizations**:
      - Early termination if abs(m-n) > target_distance (impossible case)
      - Use bit manipulation for small alphabets
      - Consider hashing techniques for repeated subproblems
    - **Applications**:
      - Spell checking
      - DNA sequence alignment
      - Fuzzy string matching
      - Plagiarism detection
    - **Related Algorithms**:
      - Needleman-Wunsch algorithm for sequence alignment
      - Smith-Waterman algorithm for local sequence alignment
      - Hirschberg's algorithm for linear space complexity

## Pattern Recognition

The Edit Distance pattern appears when:

1. **Comparing similarity between sequences**
2. **Finding minimum transformations** between strings
3. **Working with approximate string matching**
4. **Need to quantify differences** between sequences
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
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

## Space-Optimized Solution

Since each cell in the DP table only depends on the current and previous row, we can optimize space to O(min(m, n)).

```python
def min_distance_optimized(word1, word2):
    # Ensure word1 is the shorter string for space optimization
    if len(word1) > len(word2):
        word1, word2 = word2, word1
    
    m, n = len(word1), len(word2)
    
    # Previous and current row
    prev_row = list(range(n + 1))
    curr_row = [0] * (n + 1)
    
    for i in range(1, m + 1):
        curr_row[0] = i
        
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                curr_row[j] = prev_row[j-1]
            else:
                curr_row[j] = 1 + min(
                    prev_row[j],     # Delete
                    curr_row[j-1],   # Insert
                    prev_row[j-1]    # Replace
                )
        
        prev_row, curr_row = curr_row, [0] * (n + 1)
    
    return prev_row[n]
```

## Reconstructing the Edit Path

To determine the specific sequence of edit operations:

```python
def edit_operations(word1, word2):
    m, n = len(word1), len(word2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],     # Delete
                    dp[i][j-1],     # Insert
                    dp[i-1][j-1]    # Replace
                )
    
    # Reconstruct the edit path
    operations = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and word1[i-1] == word2[j-1]:
            # Characters match, no operation
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            # Replace
            operations.append(f"Replace {word1[i-1]} with {word2[j-1]}")
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            # Delete
            operations.append(f"Delete {word1[i-1]}")
            i -= 1
        else:
            # Insert
            operations.append(f"Insert {word2[j-1]}")
            j -= 1
    
    return operations[::-1]  # Reverse to get operations in correct order
```

## Time and Space Complexity

- **Time Complexity**: O(m × n) where m and n are the lengths of the two strings
- **Space Complexity**: 
  - Standard DP solution: O(m × n)
  - Optimized solution: O(min(m, n))

## Variations

### One Edit Distance

Determine if two strings are exactly one edit away from each other.

```python
def is_one_edit_distance(s, t):
    m, n = len(s), len(t)
    
    # Ensure s is the shorter string
    if m > n:
        return is_one_edit_distance(t, s)
    
    # Difference in length more than 1, impossible to be one edit away
    if n - m > 1:
        return False
    
    for i in range(m):
        if s[i] != t[i]:
            # If lengths are the same, try replacing current character
            if m == n:
                return s[i+1:] == t[i+1:]
            # If lengths differ, try inserting current character of t into s
            else:
                return s[i:] == t[i+1:]
    
    # If no difference found, check if t has exactly one more character
    return n - m == 1
```

### Longest Common Subsequence

Closely related to Edit Distance, this problem asks to find the longest subsequence common to both strings.

```python
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

### Minimum ASCII Delete Sum for Two Strings

Find the minimum ASCII delete sum for two strings (sum of ASCII values of deleted characters).

```python
def minimum_delete_sum(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + ord(s1[i-1])
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] + ord(s2[j-1])
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + ord(s1[i-1]),  # Delete from s1
                    dp[i][j-1] + ord(s2[j-1])   # Delete from s2
                )
    
    return dp[m][n]
```

## Applications

1. **Spell Checking**: Suggest corrections for misspelled words
2. **DNA Sequence Alignment**: Find similarity between DNA sequences
3. **Plagiarism Detection**: Measure similarity between text documents
4. **Fuzzy String Matching**: Find approximate matches in a database
5. **Auto-correction**: Suggest corrections for typing errors

## Practice Problems

1. **Edit Distance**: Calculate minimum operations to convert one string to another
2. **One Edit Distance**: Determine if two strings are exactly one edit away
3. **Delete Operation for Two Strings**: Find minimum deletions to make two strings equal
4. **Minimum ASCII Delete Sum for Two Strings**: Minimize ASCII sum of deleted characters
5. **Uncrossed Lines**: Maximum number of uncrossed connecting lines between two arrays

## Interview Tips

1. **Understand the operations**: Make sure you know what operations are allowed (e.g., insert, delete, replace)
2. **Draw out the DP table**: Visualize how values are built up
3. **Consider optimization**: Check if space optimization is required
4. **Check for variations**: Many problems are variations of edit distance
5. **Handle edge cases**: Empty strings, identical strings, completely different strings
