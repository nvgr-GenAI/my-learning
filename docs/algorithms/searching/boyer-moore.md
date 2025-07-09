# Boyer-Moore Algorithm

## Overview

The Boyer-Moore algorithm is one of the most efficient string searching algorithms, particularly for large alphabets and long patterns. It works by using two heuristics—the bad character rule and the good suffix rule—to skip comparisons and jump ahead in the text when a mismatch occurs. This allows it to skip large portions of the text, often resulting in sub-linear time complexity in practice.

## Algorithm

The Boyer-Moore algorithm scans the pattern from right to left but shifts the pattern from left to right. It uses two preprocessing steps:

1. **Bad Character Heuristic**: When a mismatch occurs, shift the pattern so that the mismatched character in the text aligns with its rightmost occurrence in the pattern.

2. **Good Suffix Heuristic**: When a mismatch occurs after matching some characters, shift the pattern based on the next occurrence of the already matched suffix.

The algorithm then applies the maximum shift suggested by either heuristic.

```python
def boyer_moore(text, pattern):
    """
    Searches for all occurrences of pattern in text using the Boyer-Moore algorithm.
    
    Parameters:
    - text: the text to search in
    - pattern: the pattern to search for
    
    Returns:
    - List of starting indices of matches
    """
    m = len(pattern)
    n = len(text)
    
    if m == 0:
        return [i for i in range(n+1)]
    if n == 0:
        return []
    
    # Preprocessing for bad character heuristic
    # For each character, store its rightmost position in the pattern
    bad_char = {}
    for i in range(m):
        bad_char[pattern[i]] = i
    
    # Search
    results = []
    s = 0  # s is the shift of the pattern with respect to the text
    
    while s <= n - m:
        j = m - 1  # Start matching from the last character
        
        # Keep matching characters from right to left
        while j >= 0 and pattern[j] == text[s+j]:
            j -= 1
        
        # If the pattern is found
        if j < 0:
            results.append(s)
            # Shift to align with the next possible occurrence
            s += 1
        else:
            # Shift based on bad character heuristic
            # If the character is not in the pattern, shift the whole pattern
            # Otherwise, shift to align the bad character with its rightmost occurrence
            char_shift = j - bad_char.get(text[s+j], -1)
            s += max(1, char_shift)
    
    return results
```

## Time and Space Complexity

- **Time Complexity**: 
  - Worst case: O(m*n) where n is the length of the text and m is the length of the pattern
  - Best case: O(n/m) when few characters match
  - Average case: O(n) but often performs better in practice
- **Space Complexity**: O(k) where k is the size of the alphabet

## Advantages and Disadvantages

### Advantages

- Very fast in practice, especially for long patterns and large alphabets
- Can skip large portions of the text, resulting in sub-linear performance
- Particularly effective when the pattern ends with characters that rarely appear in the text
- Widely used in real-world applications like text editors and search utilities

### Disadvantages

- Complex to implement correctly with both heuristics
- Preprocessing step can be expensive for small searches
- Worst-case performance is still O(m*n)
- Not as effective for small alphabets or very short patterns

## Use Cases

- Text editors (find/replace functionality)
- DNA sequence matching
- Pattern matching in large documents
- Search engines
- File comparison tools
- Virus signature detection

## Implementation Details

### Full Boyer-Moore Implementation (with Good Suffix Rule)

```python
def boyer_moore_full(text, pattern):
    """
    Full Boyer-Moore implementation with both bad character and good suffix rules.
    """
    m = len(pattern)
    n = len(text)
    
    if m == 0:
        return [i for i in range(n+1)]
    if n == 0 or m > n:
        return []
        
    # Preprocessing for bad character rule
    def preprocess_bad_char(pattern):
        # For each character, store its rightmost position
        bad_char = {}
        for i in range(m):
            bad_char[pattern[i]] = i
        return bad_char
    
    # Preprocessing for good suffix rule
    def preprocess_good_suffix(pattern):
        m = len(pattern)
        
        # suffix[i] = longest suffix of pattern[0:i+1] that is also a suffix of pattern
        suffix = [0] * m
        suffix[m-1] = m
        
        # Case 1: matching suffixes
        f = 0  # The first position from right where suffix match fails
        g = m - 1  # The position where the suffix starts
        
        for i in range(m-2, -1, -1):
            if i > g and suffix[i + m - 1 - f] < i - g:
                suffix[i] = suffix[i + m - 1 - f]
            else:
                if i < g:
                    g = i
                f = i
                while g >= 0 and pattern[g] == pattern[g + m - 1 - f]:
                    g -= 1
                suffix[i] = f - g
                
        # shift[i] = amount to shift if mismatch at pattern[i]
        shift = [0] * m
        
        # Case 2: no matching suffix exists
        j = 0
        for i in range(m):
            shift[i] = m
            
        # Case 1: matching suffix exists
        for i in range(m-1, -1, -1):
            if suffix[i] == i + 1:
                while j < m - 1 - i:
                    if shift[j] == m:
                        shift[j] = m - 1 - i
                    j += 1
                    
        # Case 3: partial matching suffix exists
        for i in range(m-1):
            shift[m - 1 - suffix[i]] = m - 1 - i
            
        return shift
    
    bad_char = preprocess_bad_char(pattern)
    good_suffix = preprocess_good_suffix(pattern)
    
    results = []
    s = 0  # s is the shift of the pattern with respect to the text
    
    while s <= n - m:
        j = m - 1
        
        # Keep matching characters from right to left
        while j >= 0 and pattern[j] == text[s+j]:
            j -= 1
            
        # If the pattern is found
        if j < 0:
            results.append(s)
            # Use good suffix rule for shift
            s += good_suffix[0]
        else:
            # Use the maximum of bad character and good suffix rules
            char_shift = j - bad_char.get(text[s+j], -1)
            suffix_shift = good_suffix[j]
            s += max(char_shift, suffix_shift, 1)
    
    return results
```

### Java Implementation (Simplified Version)

```java
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BoyerMoore {
    public static List<Integer> boyerMoore(String text, String pattern) {
        List<Integer> results = new ArrayList<>();
        
        int m = pattern.length();
        int n = text.length();
        
        if (m == 0) {
            for (int i = 0; i <= n; i++) {
                results.add(i);
            }
            return results;
        }
        
        if (n == 0 || m > n) {
            return results;
        }
        
        // Preprocessing for bad character rule
        Map<Character, Integer> badChar = new HashMap<>();
        for (int i = 0; i < m; i++) {
            badChar.put(pattern.charAt(i), i);
        }
        
        int s = 0;  // s is the shift of the pattern with respect to the text
        
        while (s <= n - m) {
            int j = m - 1;
            
            // Keep matching characters from right to left
            while (j >= 0 && pattern.charAt(j) == text.charAt(s+j)) {
                j--;
            }
            
            // If the pattern is found
            if (j < 0) {
                results.add(s);
                s++;  // Move to the next character
            } else {
                // Shift based on bad character rule
                int badCharShift = j - badChar.getOrDefault(text.charAt(s+j), -1);
                s += Math.max(1, badCharShift);
            }
        }
        
        return results;
    }
    
    public static void main(String[] args) {
        String text = "ABAAABCD";
        String pattern = "ABC";
        List<Integer> matches = boyerMoore(text, pattern);
        System.out.println("Pattern found at indices: " + matches);  // Output: [4]
    }
}
```

## Understanding the Heuristics

### Bad Character Rule

When a mismatch occurs at position j in the pattern:
1. Look at the character in the text that caused the mismatch.
2. Find the rightmost occurrence of this character in the pattern (to the left of the current position).
3. Align this occurrence with the mismatched character in the text.

If the character doesn't exist in the pattern, shift the pattern past the mismatched character.

### Good Suffix Rule

When a mismatch occurs at position j in the pattern:
1. The characters pattern[j+1...m-1] have already matched with the text.
2. Look for the next occurrence of this suffix in the pattern.
3. Shift the pattern to align this occurrence with the matched suffix in the text.

If no such occurrence exists, look for the longest prefix of the pattern that matches a suffix of the matched portion, and shift accordingly.

## Variations

### Turbo Boyer-Moore

```python
def turbo_boyer_moore(text, pattern):
    """
    Turbo Boyer-Moore algorithm, which uses a factor that allows larger shifts.
    """
    m = len(pattern)
    n = len(text)
    
    # Preprocessing (same as regular Boyer-Moore)
    bad_char = {}
    for i in range(m):
        bad_char[pattern[i]] = i
    
    # Additional factor for turbo shifts
    factor = 0
    
    s = 0  # Pattern shift
    while s <= n - m:
        j = m - 1
        
        # Apply the current factor (skip comparisons)
        if factor > 0:
            j -= factor
            # If the factor suggests a guaranteed match, just verify the rest
            if pattern[j] == text[s+j]:
                j -= 1  # Start checking from the position before
            else:
                # If the guaranteed match fails, we need to reset
                s += 1
                factor = 0
                continue
        
        # Continue matching from right to left
        while j >= 0 and pattern[j] == text[s+j]:
            j -= 1
            
        if j < 0:
            # Found a match
            results.append(s)
            s += 1
            factor = 0
        else:
            # Calculate shift based on bad character rule
            shift = j - bad_char.get(text[s+j], -1)
            
            # Update factor for next iteration (matched positions)
            factor = m - 1 - j
            
            s += max(1, shift)
    
    return results
```

### Boyer-Moore-Horspool

A simplified version focusing only on the bad character rule:

```python
def boyer_moore_horspool(text, pattern):
    """
    Boyer-Moore-Horspool algorithm, a simplified version of Boyer-Moore.
    """
    m = len(pattern)
    n = len(text)
    
    if m == 0:
        return [i for i in range(n+1)]
    if n == 0:
        return []
    
    # Preprocessing: For each character, calculate how far to shift
    # If the character is not in the pattern, shift by m
    # Otherwise, shift to align the rightmost occurrence
    skip = {}
    for i in range(m - 1):
        skip[pattern[i]] = m - 1 - i
    
    results = []
    s = 0  # Pattern shift
    
    while s <= n - m:
        j = m - 1
        
        # Match from right to left
        while j >= 0 and pattern[j] == text[s+j]:
            j -= 1
            
        if j < 0:
            # Found a match
            results.append(s)
            s += 1
        else:
            # Shift based on the character that caused the mismatch
            # If not in the last position of pattern and not in skip table, shift by m
            char = text[s+m-1]
            s += skip.get(char, m)
    
    return results
```

## Interview Tips

- Explain how Boyer-Moore can achieve sub-linear time complexity in practice (by skipping characters)
- Contrast Boyer-Moore with other string searching algorithms like KMP or Rabin-Karp
- Describe the two main heuristics (bad character and good suffix rules) and how they work together
- Highlight when Boyer-Moore is most effective (long patterns, large alphabets)
- Discuss the trade-offs between implementing the full algorithm versus simplified versions
- Be prepared to implement at least the bad character rule portion of the algorithm

## Practice Problems

1. Implement the full Boyer-Moore algorithm with both heuristics
2. Find all occurrences of multiple patterns in a text using Boyer-Moore
3. Compare the performance of Boyer-Moore with KMP for different types of texts and patterns
4. Modify Boyer-Moore to handle case-insensitive searching
5. Implement a text editor's search functionality using Boyer-Moore
