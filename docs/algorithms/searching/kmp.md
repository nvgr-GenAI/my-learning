# KMP (Knuth-Morris-Pratt) Algorithm

## Overview

The KMP (Knuth-Morris-Pratt) algorithm is an efficient string searching algorithm that uses the observation that when a mismatch occurs, the pattern itself contains sufficient information to determine where the next match could begin, avoiding the need to reexamine previously matched characters. This makes it particularly efficient for patterns with repeated substrings.

## Algorithm

The KMP algorithm has two main parts:
1. Preprocessing: Build a "Longest Prefix Suffix" (LPS) array
2. Searching: Use the LPS array to skip characters when a mismatch occurs

```python
def kmp_search(text, pattern):
    """
    Searches for all occurrences of pattern in text using KMP algorithm.
    Returns a list of starting indices of matches.
    """
    if not pattern:
        return [i for i in range(len(text) + 1)]  # Empty pattern matches everywhere
    
    # Preprocessing: build the LPS array
    lps = compute_lps_array(pattern)
    
    results = []
    i = 0  # Index for text
    j = 0  # Index for pattern
    
    while i < len(text):
        # Characters match, move both pointers
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        # Found a complete match
        if j == len(pattern):
            results.append(i - j)  # Record the starting position
            # Look for the next match, using LPS to avoid rechecking
            j = lps[j - 1]
        
        # Mismatch after some matches
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]  # Use LPS to skip comparisons
            else:
                i += 1  # No match at the beginning, move to next character in text
    
    return results

def compute_lps_array(pattern):
    """
    Computes the Longest Prefix Suffix (LPS) array for the pattern.
    LPS[i] = length of the longest proper prefix of pattern[0...i] 
             that is also a suffix of pattern[0...i]
    """
    length = 0  # Length of previous longest prefix suffix
    lps = [0] * len(pattern)
    i = 1
    
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                # Try matching with the longest prefix suffix
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
                
    return lps
```

## Time and Space Complexity

- **Time Complexity**: O(m + n) where m is the length of the text and n is the length of the pattern
- **Space Complexity**: O(n) for the LPS array

## Advantages and Disadvantages

### Advantages

- Linear time complexity O(m + n)
- No backtracking in the main text
- Particularly efficient for patterns with repeated substrings
- Handles large texts and patterns efficiently

### Disadvantages

- More complex to implement compared to naive string matching
- Preprocessing step adds overhead for short patterns or one-time searches
- Not always the fastest for very short patterns or simple cases
- The LPS computation can be tricky to get right

## Use Cases

- Efficient text search in large documents
- Pattern matching in genetic sequences
- Intrusion detection in network security
- Plagiarism detection
- File searching and text processing
- Real-time string matching applications

## Implementation Details

### Python Implementation

```python
def kmp_search(text, pattern):
    """
    Searches for all occurrences of pattern in text using KMP algorithm.
    Returns a list of starting indices of matches.
    """
    if not pattern:
        return [i for i in range(len(text) + 1)]  # Empty pattern matches everywhere
    
    if not text:
        return []
    
    # Preprocessing: build the LPS array
    lps = compute_lps_array(pattern)
    
    results = []
    i = 0  # Index for text
    j = 0  # Index for pattern
    
    while i < len(text):
        # Characters match, move both pointers
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        # Found a complete match
        if j == len(pattern):
            results.append(i - j)  # Record the starting position
            # Look for the next match, using LPS to avoid rechecking
            j = lps[j - 1]
        
        # Mismatch after some matches
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]  # Use LPS to skip comparisons
            else:
                i += 1  # No match at the beginning, move to next character in text
    
    return results

def compute_lps_array(pattern):
    """
    Computes the Longest Prefix Suffix (LPS) array for the pattern.
    LPS[i] = length of the longest proper prefix of pattern[0...i] 
             that is also a suffix of pattern[0...i]
    """
    length = 0  # Length of previous longest prefix suffix
    lps = [0] * len(pattern)
    i = 1
    
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                # Try matching with the longest prefix suffix
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
                
    return lps

# Example usage
text = "ABABDABACDABABCABAB"
pattern = "ABABCABAB"
matches = kmp_search(text, pattern)
print(f"Pattern found at indices: {matches}")  # Output: [10]
```

### Java Implementation

```java
import java.util.ArrayList;
import java.util.List;

public class KMP {
    public static List<Integer> kmpSearch(String text, String pattern) {
        List<Integer> results = new ArrayList<>();
        
        if (pattern.isEmpty()) {
            for (int i = 0; i <= text.length(); i++) {
                results.add(i);
            }
            return results;
        }
        
        if (text.isEmpty()) {
            return results;
        }
        
        // Preprocessing: build the LPS array
        int[] lps = computeLPSArray(pattern);
        
        int i = 0; // Index for text
        int j = 0; // Index for pattern
        
        while (i < text.length()) {
            // Characters match, move both pointers
            if (pattern.charAt(j) == text.charAt(i)) {
                i++;
                j++;
            }
            
            // Found a complete match
            if (j == pattern.length()) {
                results.add(i - j);
                // Look for the next match, using LPS to avoid rechecking
                j = lps[j - 1];
            }
            // Mismatch after some matches
            else if (i < text.length() && pattern.charAt(j) != text.charAt(i)) {
                if (j != 0) {
                    j = lps[j - 1];
                } else {
                    i++;
                }
            }
        }
        
        return results;
    }
    
    private static int[] computeLPSArray(String pattern) {
        int length = 0; // Length of previous longest prefix suffix
        int[] lps = new int[pattern.length()];
        int i = 1;
        
        while (i < pattern.length()) {
            if (pattern.charAt(i) == pattern.charAt(length)) {
                length++;
                lps[i] = length;
                i++;
            } else {
                if (length != 0) {
                    // Try matching with the longest prefix suffix
                    length = lps[length - 1];
                } else {
                    lps[i] = 0;
                    i++;
                }
            }
        }
        
        return lps;
    }
    
    public static void main(String[] args) {
        String text = "ABABDABACDABABCABAB";
        String pattern = "ABABCABAB";
        List<Integer> matches = kmpSearch(text, pattern);
        System.out.println("Pattern found at indices: " + matches);  // Output: [10]
    }
}
```

## Understanding the LPS Array

The Longest Prefix Suffix (LPS) array is key to the KMP algorithm's efficiency. For each position i in the pattern, LPS[i] contains the length of the longest proper prefix that is also a suffix of the pattern[0...i].

Example for pattern "ABABCABAB":

```
Pattern: A B A B C A B A B
LPS:     0 0 1 2 0 1 2 3 4
```

Interpretation:
- LPS[0] = 0: No proper prefix for a single character
- LPS[1] = 0: "AB" has no common prefix and suffix
- LPS[2] = 1: In "ABA", "A" is both a prefix and suffix
- LPS[3] = 2: In "ABAB", "AB" is both a prefix and suffix
- LPS[4] = 0: In "ABABC", no common prefix and suffix
- And so on...

This array helps us skip comparisons when a mismatch occurs. If we have matched j characters and then find a mismatch, we know that the first LPS[j-1] characters of the pattern are already matched with the current position in the text.

## Variations

### KMP with Count Only

If we only need to count the occurrences without storing the positions:

```python
def kmp_count(text, pattern):
    """
    Counts occurrences of pattern in text using KMP algorithm.
    """
    if not pattern:
        return len(text) + 1
    
    if not text:
        return 0
        
    lps = compute_lps_array(pattern)
    
    count = 0
    i = 0  # Index for text
    j = 0  # Index for pattern
    
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == len(pattern):
            count += 1
            j = lps[j - 1]
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return count
```

### KMP for Multiple Patterns (Aho-Corasick)

For searching multiple patterns, the Aho-Corasick algorithm builds upon KMP's principles:

```python
from collections import deque

class AhoCorasick:
    def __init__(self, patterns):
        self.patterns = patterns
        self.trie = {}
        self.fail = {}
        self.outputs = {}
        
        self.build_trie()
        self.build_failure_links()
        
    def build_trie(self):
        # Building the trie (prefix tree)
        for index, pattern in enumerate(self.patterns):
            node = self.trie
            for char in pattern:
                if char not in node:
                    node[char] = {}
                node = node[char]
            self.outputs[id(node)] = [pattern]
        
    def build_failure_links(self):
        # Building failure links with BFS
        queue = deque()
        for char, node in self.trie.items():
            self.fail[id(node)] = self.trie
            queue.append(node)
            
        while queue:
            current = queue.popleft()
            for char, node in current.items():
                queue.append(node)
                state = self.fail[id(current)]
                
                while state != self.trie and char not in state:
                    state = self.fail[id(state)]
                    
                self.fail[id(node)] = state.get(char, self.trie)
                if id(self.fail[id(node)]) in self.outputs:
                    if id(node) not in self.outputs:
                        self.outputs[id(node)] = []
                    self.outputs[id(node)].extend(self.outputs[id(self.fail[id(node)])])
    
    def search(self, text):
        results = {}
        state = self.trie
        
        for i, char in enumerate(text):
            while state != self.trie and char not in state:
                state = self.fail[id(state)]
                
            if char in state:
                state = state[char]
            else:
                state = self.trie
                
            if id(state) in self.outputs:
                for pattern in self.outputs[id(state)]:
                    if pattern not in results:
                        results[pattern] = []
                    results[pattern].append(i - len(pattern) + 1)
                    
        return results
```

## Interview Tips

- Explain the difference between naive string matching (O(m*n)) and KMP (O(m+n))
- Describe how the LPS array helps avoid unnecessary comparisons
- Highlight that KMP is particularly efficient for patterns with repeated substrings
- Be prepared to explain how the LPS array is constructed and what each value represents
- Discuss when KMP might be overkill (short patterns, single searches) vs. when it's necessary (large texts, repeated searches)

## Practice Problems

1. Find all occurrences of a pattern in a text using KMP
2. Implement a function to check if a string is a rotation of another string using KMP
3. Count the number of times a pattern appears as a substring in a text
4. Implement KMP for circular string matching
5. Use KMP to find the longest repeated substring in a text
