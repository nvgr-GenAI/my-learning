# Knuth-Morris-Pratt (KMP) Algorithm

## Overview

The Knuth-Morris-Pratt (KMP) algorithm is an efficient string searching algorithm that uses the observation that when a mismatch occurs, the pattern itself contains sufficient information to determine where the next match could begin, thus bypassing re-examination of previously matched characters.

Named after its inventors—Donald Knuth, James H. Morris, and Vaughan Pratt—the KMP algorithm was one of the first linear-time string matching algorithms and was published in 1977.

## Key Concepts

### The Partial Match Table (LPS Array)

The core of the KMP algorithm is the Longest Proper Prefix which is also Suffix (LPS) array (sometimes called the "failure function" or "π table"). This table helps the algorithm determine how far to shift the pattern when a mismatch is found.

For a pattern P of length m, the LPS array lps[0...m-1] is defined as:
- lps[i] = the length of the longest proper prefix of P[0...i] which is also a suffix of P[0...i]

Example:
For pattern "ABABCABAB":
- lps[0] = 0 (no proper prefix is also a suffix for a single character)
- lps[1] = 0 (no proper prefix of "AB" is also a suffix)
- lps[2] = 1 (the prefix "A" is also a suffix of "ABA")
- lps[3] = 2 (the prefix "AB" is also a suffix of "ABAB")
- lps[4] = 0 (no proper prefix of "ABABC" is also a suffix)
- lps[5] = 1 (the prefix "A" is also a suffix of "ABABCA")
- lps[6] = 2 (the prefix "AB" is also a suffix of "ABABCAB")
- lps[7] = 3 (the prefix "ABA" is also a suffix of "ABABCABA")
- lps[8] = 4 (the prefix "ABAB" is also a suffix of "ABABCABAB")

## Algorithm

### LPS Array Construction

```python
def compute_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    
    length = 0  # Length of the previous longest prefix & suffix
    i = 1
    
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    
    return lps
```

### KMP Search Algorithm

```python
def kmp_search(text, pattern):
    n, m = len(text), len(pattern)
    
    # Edge cases
    if m == 0:
        return [i for i in range(n+1)]
    if n < m:
        return []
    
    # Compute the LPS array
    lps = compute_lps(pattern)
    
    result = []
    i = 0  # Index for text
    j = 0  # Index for pattern
    
    while i < n:
        # Current characters match, move both pointers
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        # Found a complete match
        if j == m:
            result.append(i - j)
            j = lps[j - 1]
        
        # Mismatch after j matches
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return result
```

## Time and Space Complexity

- **Time Complexity**:
  - LPS array computation: O(m) where m is the length of the pattern
  - Search algorithm: O(n) where n is the length of the text
  - Overall: O(n + m)

- **Space Complexity**: O(m) for storing the LPS array

## Advantages and Disadvantages

### Advantages

1. **Linear Time Complexity**: KMP guarantees O(n + m) time complexity regardless of the input.
2. **No Backtracking**: The algorithm never needs to move backward in the text, making it suitable for streaming data.
3. **Complete Matching**: Finds all occurrences of the pattern in the text.

### Disadvantages

1. **Complexity**: More complex to implement than the naive approach.
2. **Preprocessing Overhead**: For very short patterns or texts, the preprocessing overhead might not be worth it.
3. **Not Always Fastest in Practice**: Boyer-Moore can outperform KMP for certain types of data.

## Applications

1. **Text Editors**: For efficient find/replace operations
2. **Bioinformatics**: DNA sequence matching
3. **Network Intrusion Detection**: Pattern matching for network packets
4. **Data Compression**: Some algorithms rely on efficient string matching
5. **Plagiarism Detection**: Finding matching text segments

## Implementation in Different Languages

### Java Implementation

```java
public class KMP {
    public static int[] computeLPSArray(String pattern) {
        int m = pattern.length();
        int[] lps = new int[m];
        
        int length = 0;
        int i = 1;
        
        while (i < m) {
            if (pattern.charAt(i) == pattern.charAt(length)) {
                length++;
                lps[i] = length;
                i++;
            } else {
                if (length != 0) {
                    length = lps[length - 1];
                } else {
                    lps[i] = 0;
                    i++;
                }
            }
        }
        
        return lps;
    }
    
    public static List<Integer> search(String text, String pattern) {
        int n = text.length();
        int m = pattern.length();
        List<Integer> result = new ArrayList<>();
        
        if (m == 0) {
            for (int i = 0; i <= n; i++) {
                result.add(i);
            }
            return result;
        }
        
        if (n < m) {
            return result;
        }
        
        int[] lps = computeLPSArray(pattern);
        
        int i = 0, j = 0;
        while (i < n) {
            if (pattern.charAt(j) == text.charAt(i)) {
                i++;
                j++;
            }
            
            if (j == m) {
                result.add(i - j);
                j = lps[j - 1];
            } else if (i < n && pattern.charAt(j) != text.charAt(i)) {
                if (j != 0) {
                    j = lps[j - 1];
                } else {
                    i++;
                }
            }
        }
        
        return result;
    }
}
```

## Common Pitfalls and Tips

1. **Off-by-one Errors**: Be careful with indices when implementing KMP.
2. **Edge Cases**: Remember to handle empty patterns or texts.
3. **Understanding LPS**: The LPS array can be confusing; ensure you grasp its meaning.
4. **Optimization**: For large patterns with small alphabets, consider using a DFA-based approach.

## Practice Problems

1. **String Searching**: Find all occurrences of a pattern in a text.
2. **Periodic Strings**: Determine if a string has repetitive patterns.
3. **String Rotation**: Check if one string is a rotation of another.
4. **Longest Repeating Substring**: Find the longest substring that appears multiple times.
5. **Shortest Superstring**: Find the shortest string that contains all given strings as substrings.

## Related Algorithms

- [Rabin-Karp Algorithm](rabin-karp.md)
- [Boyer-Moore Algorithm](boyer-moore.md)
- [Z Algorithm](z-algorithm.md)
- [Aho-Corasick Algorithm](aho-corasick.md)

## References

1. Knuth, D. E., Morris, J. H., & Pratt, V. R. (1977). Fast pattern matching in strings. SIAM Journal on Computing, 6(2), 323-350.
2. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to Algorithms, Third Edition. MIT Press.
3. Sedgewick, R., & Wayne, K. (2011). Algorithms, 4th Edition. Addison-Wesley Professional.
