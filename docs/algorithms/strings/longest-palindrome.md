# Longest Palindromic Substring

## Overview

The Longest Palindromic Substring problem involves finding the longest substring within a string that is a palindrome. A palindrome is a sequence that reads the same backward as forward, like "madam" or "racecar".

This is a classic string processing problem with applications in text analysis, bioinformatics (DNA sequence analysis), and natural language processing.

## Problem Statement

Given a string `s`, find the longest palindromic substring in `s`.

**Example:**
- Input: `s = "babad"`
- Output: `"bab"` or `"aba"` (both are valid)

- Input: `s = "cbbd"`
- Output: `"bb"`

## Approaches

### 1. Brute Force Approach

The simplest approach is to check every possible substring to see if it's a palindrome.

```python
def longest_palindrome_brute_force(s):
    if not s:
        return ""
    
    longest = s[0]
    max_length = 1
    
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            substring = s[i:j]
            if substring == substring[::-1] and len(substring) > max_length:
                longest = substring
                max_length = len(substring)
    
    return longest
```

**Time Complexity:** O(n³) - We have O(n²) substrings, and checking if each is a palindrome takes O(n) time.
**Space Complexity:** O(1) - Constant extra space.

### 2. Dynamic Programming Approach

We can use dynamic programming to avoid recomputing palindrome checks for substrings.

```python
def longest_palindrome_dp(s):
    if not s:
        return ""
    
    n = len(s)
    # dp[i][j] is True if substring s[i..j] is palindrome
    dp = [[False for _ in range(n)] for _ in range(n)]
    
    # Single characters are palindromes
    for i in range(n):
        dp[i][i] = True
    
    start = 0
    max_length = 1
    
    # Check for palindromes of length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_length = 2
    
    # Check for palindromes of length 3 or more
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1  # ending index
            
            # Check if substring from i+1 to j-1 is palindrome and chars at i and j match
            if dp[i + 1][j - 1] and s[i] == s[j]:
                dp[i][j] = True
                if length > max_length:
                    start = i
                    max_length = length
    
    return s[start:start + max_length]
```

**Time Complexity:** O(n²) - We fill an n×n table.
**Space Complexity:** O(n²) - For the DP table.

### 3. Expand Around Center Approach

This approach expands around potential palindrome centers and is usually more efficient than the DP approach.

```python
def longest_palindrome_expand(s):
    if not s:
        return ""
    
    start = end = 0
    
    for i in range(len(s)):
        # Expand around center for odd-length palindromes
        len1 = expand_around_center(s, i, i)
        
        # Expand around center for even-length palindromes
        len2 = expand_around_center(s, i, i + 1)
        
        # Get the maximum length from the two expansions
        max_len = max(len1, len2)
        
        # Update start and end if we found a longer palindrome
        if max_len > (end - start):
            start = i - (max_len - 1) // 2
            end = i + max_len // 2
    
    return s[start:end + 1]

def expand_around_center(s, left, right):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    
    # Return the length of palindrome
    return right - left - 1
```

**Time Complexity:** O(n²) - In the worst case, we might need to expand around each center.
**Space Complexity:** O(1) - Constant extra space.

### 4. Manacher's Algorithm

Manacher's algorithm is a specialized algorithm for finding the longest palindromic substring in linear time.

```python
def longest_palindrome_manacher(s):
    if not s:
        return ""
    
    # Transform the string to handle even-length palindromes
    # Add special characters to avoid bound checking
    transformed = '#' + '#'.join(s) + '#'
    n = len(transformed)
    
    # p[i] = length of palindrome centered at i
    p = [0] * n
    
    center = right = 0
    
    for i in range(n):
        # Mirror of current position
        mirror = 2 * center - i
        
        # If within right boundary, take advantage of symmetry
        if i < right:
            p[i] = min(right - i, p[mirror])
        
        # Attempt to expand palindrome centered at i
        while (i + 1 + p[i] < n and 
               i - 1 - p[i] >= 0 and 
               transformed[i + 1 + p[i]] == transformed[i - 1 - p[i]]):
            p[i] += 1
        
        # If palindrome centered at i expands past right,
        # adjust center and right boundary
        if i + p[i] > right:
            center = i
            right = i + p[i]
    
    # Find the maximum palindrome length
    max_len, center_index = max((n, i) for i, n in enumerate(p))
    
    # Calculate the start and end in original string
    start = (center_index - max_len) // 2
    end = start + max_len - 1
    
    return s[start:end + 1]
```

**Time Complexity:** O(n) - Linear time.
**Space Complexity:** O(n) - For the transformed string and p array.

## Comparison of Approaches

| Approach | Time Complexity | Space Complexity | Pros | Cons |
|----------|----------------|-----------------|------|------|
| Brute Force | O(n³) | O(1) | Simple to understand | Very inefficient for long strings |
| Dynamic Programming | O(n²) | O(n²) | Systematic approach | High space usage |
| Expand Around Center | O(n²) | O(1) | Efficient space usage | Still quadratic time |
| Manacher's Algorithm | O(n) | O(n) | Linear time | Complex implementation |

## Applications

1. **Text Analysis**: Finding palindromic patterns in text
2. **Bioinformatics**: Analyzing DNA sequences for palindromic motifs
3. **Natural Language Processing**: Word play, puzzles
4. **Computational Biology**: RNA secondary structure prediction

## Common Variations

1. **Longest Palindromic Subsequence**: Find the longest subsequence (not necessarily contiguous) that is a palindrome
2. **Count All Palindromic Substrings**: Count the total number of palindromic substrings
3. **Shortest Palindrome**: Add characters to the beginning to make the entire string a palindrome
4. **Palindrome Pairs**: Find pairs of words that form palindromes when concatenated

## Interview Tips

- Start with the expand around center approach as it's easier to understand and implement than Manacher's algorithm
- Be careful of edge cases: empty strings, single characters, all identical characters
- Clarify whether multiple answers are acceptable if there are ties for the longest palindromic substring
- Consider discussing the trade-offs between the different approaches

## Practice Problems

1. Count the number of palindromic substrings in a string
2. Find the shortest palindrome by adding characters to the beginning of a string
3. Determine if a string can be rearranged to form a palindrome
4. Find all palindromic decompositions of a string

## Related Topics

- [String Pattern Matching](pattern-matching.md)
- [Dynamic Programming](../dp/index.md)
- [Suffix Arrays and Trees](suffix-arrays.md)
