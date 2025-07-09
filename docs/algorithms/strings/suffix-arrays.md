# Suffix Arrays

## Overview

A Suffix Array is a sorted array of all suffixes of a given string. It is a powerful data structure used for many string processing problems, particularly those requiring efficient substring searches and pattern matching.

Suffix arrays provide functionalities similar to suffix trees but with lower memory requirements and simpler implementation.

## Basic Concepts

### Suffix

A suffix of a string S is any string obtained by removing zero or more characters from the beginning of S.

For example, the suffixes of "banana" are:
- "banana" (the entire string)
- "anana" (removing 'b')
- "nana" (removing 'ba')
- "ana" (removing 'ban')
- "na" (removing 'bana')
- "a" (removing 'banan')
- "" (the empty string)

### Suffix Array Definition

A suffix array SA of a string S is an array of integers representing the starting positions of all suffixes of S, sorted in lexicographical order.

For example, for S = "banana":

| Index | Suffix | Position in S |
|-------|--------|---------------|
| 0 | "a" | 5 |
| 1 | "ana" | 3 |
| 2 | "anana" | 1 |
| 3 | "banana" | 0 |
| 4 | "na" | 4 |
| 5 | "nana" | 2 |

The suffix array would be SA = [5, 3, 1, 0, 4, 2]

## Construction Algorithms

### Naive Algorithm

The naive approach sorts all suffixes directly, which can be inefficient for large strings.

```python
def naive_suffix_array(s):
    # Create a list of suffixes with their original positions
    suffixes = [(s[i:], i) for i in range(len(s))]
    
    # Sort the suffixes lexicographically
    suffixes.sort()
    
    # Extract the original positions
    return [position for _, position in suffixes]
```

**Time Complexity**: O(n² log n) where n is the length of the string
- Each suffix comparison takes O(n) time
- There are n suffixes
- Sorting takes O(n log n) comparisons

### Efficient Algorithms

Several algorithms can construct suffix arrays more efficiently:

1. **Prefix Doubling (Manber-Myers)**:
   - Iteratively sorts suffixes by their first 2^k characters
   - Time Complexity: O(n log² n) or O(n log n) with careful implementation

2. **Skew Algorithm (DC3)**:
   - Divides the problem into smaller subproblems using a divide-and-conquer approach
   - Time Complexity: O(n)

3. **SA-IS (Induced Sorting)**:
   - Uses induced sorting and performs especially well in practice
   - Time Complexity: O(n)

Here's a simplified implementation of the SA-IS algorithm (though a full implementation would be quite complex):

```python
def sa_is(s):
    # This is just a placeholder - implementing SA-IS requires significant code
    # Refer to specialized libraries for production use
    pass
```

## Applications

### 1. Pattern Matching

Suffix arrays enable efficient pattern matching using binary search.

```python
def search_pattern(text, pattern, suffix_array):
    n, m = len(text), len(pattern)
    
    # Binary search for the lower bound
    left, right = 0, n - 1
    while left <= right:
        mid = (left + right) // 2
        suffix = text[suffix_array[mid]:]
        if pattern > suffix[:m]:
            left = mid + 1
        else:
            right = mid - 1
    
    lower_bound = left
    
    # Binary search for the upper bound
    left, right = 0, n - 1
    while left <= right:
        mid = (left + right) // 2
        suffix = text[suffix_array[mid]:]
        if pattern >= suffix[:m]:
            left = mid + 1
        else:
            right = mid - 1
    
    upper_bound = right
    
    # Return all occurrences
    if lower_bound <= upper_bound:
        return [suffix_array[i] for i in range(lower_bound, upper_bound + 1)]
    return []
```

**Time Complexity**: O(m log n) for searching a pattern of length m in a text of length n

### 2. Longest Common Prefix (LCP) Array

The LCP array stores the length of the longest common prefix between adjacent suffixes in the suffix array.

```python
def build_lcp_array(text, suffix_array):
    n = len(text)
    
    # Inverse of the suffix array
    rank = [0] * n
    for i in range(n):
        rank[suffix_array[i]] = i
    
    lcp = [0] * n
    k = 0  # Length of previous LCP
    
    for i in range(n):
        if rank[i] == n - 1:
            k = 0
            continue
            
        j = suffix_array[rank[i] + 1]
        
        # Extend the previous LCP
        while i + k < n and j + k < n and text[i + k] == text[j + k]:
            k += 1
            
        lcp[rank[i]] = k
        
        # Decrease k when moving to the next suffix
        if k > 0:
            k -= 1
    
    return lcp
```

The LCP array enables efficient solutions to many string problems:
- Longest repeated substring
- Longest common substring
- Most frequent substrings
- String similarity measures

### 3. Burrows-Wheeler Transform (BWT)

Suffix arrays can be used to compute the Burrows-Wheeler Transform, a transformation used in data compression algorithms.

```python
def bwt_using_suffix_array(text):
    text += "$"  # Add sentinel character
    n = len(text)
    
    # Build suffix array
    sa = build_suffix_array(text)
    
    # Compute BWT
    bwt = ""
    for i in range(n):
        # Character before the suffix (or last character)
        bwt += text[(sa[i] - 1) % n]
    
    return bwt
```

## Enhanced Functionality with Additional Arrays

### 1. Rank Array

The rank array is the inverse of the suffix array and gives the lexicographical rank of each suffix.

```python
def build_rank_array(suffix_array, n):
    rank = [0] * n
    for i in range(n):
        rank[suffix_array[i]] = i
    return rank
```

### 2. LCP Interval Tree (LCP-IT)

LCP Interval Trees provide efficient solutions for problems involving common substrings or repeats.

## Time and Space Complexity

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Naive Construction | O(n² log n) | O(n) |
| Prefix Doubling | O(n log² n) | O(n) |
| Skew Algorithm (DC3) | O(n) | O(n) |
| SA-IS | O(n) | O(n) |
| Pattern Matching | O(m log n) | O(1) additional |
| LCP Array Construction | O(n) | O(n) |

## Advantages and Disadvantages

### Advantages

1. **Space Efficiency**: More memory-efficient than suffix trees
2. **Simplicity**: Easier to implement and understand than suffix trees
3. **Pattern Matching**: Efficient for simple pattern matching
4. **Integration**: Works well with other string algorithms and data structures

### Disadvantages

1. **Limited Navigation**: Doesn't provide direct access to internal nodes like suffix trees
2. **Prefix Matching**: Less efficient than suffix trees for some prefix-based operations
3. **Construction**: Some advanced construction algorithms are complex to implement correctly

## Practical Considerations

1. **Library Usage**: For production use, consider using established libraries for suffix array construction
2. **Character Set**: Be aware of the character encoding and range when implementing
3. **Long Strings**: For very long strings, memory usage and cache efficiency become important
4. **Integer Range**: For large strings, ensure your implementation handles sufficiently large integers

## Practice Problems

1. Find the longest repeated substring in a string
2. Find the longest common substring of two strings
3. Count the number of distinct substrings in a string
4. Implement the Burrows-Wheeler Transform and its inverse

## Related Topics

- [Suffix Trees](suffix-trees.md)
- [String Pattern Matching](pattern-matching.md)
- [String Searching Algorithms](../searching/index.md)
- [Tries](tries.md)

## References

1. Manber, U., & Myers, G. (1993). Suffix arrays: a new method for on-line string searches. SIAM Journal on Computing, 22(5), 935-948.
2. Kärkkäinen, J., & Sanders, P. (2003). Simple linear work suffix array construction. In Automata, languages and programming (pp. 943-955).
3. Nong, G., Zhang, S., & Chan, W. H. (2009). Linear suffix array construction by almost pure induced-sorting. In Data Compression Conference (pp. 193-202).
