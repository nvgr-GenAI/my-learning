# Rabin-Karp Algorithm

## Overview

The Rabin-Karp algorithm is a string searching algorithm that uses hashing to find patterns in strings. It calculates a hash value for the pattern and for each possible substring of the text, and then compares only the hash values. When hash values match, it does a full comparison to confirm. This approach allows for efficient batch processing of multiple patterns.

## Algorithm

1. Compute the hash value of the pattern
2. Compute the hash value of the first m characters of the text (where m is the pattern length)
3. For each position in the text:
   - Compare the hash value of the current substring with the pattern's hash
   - If hash values match, verify character by character
   - Update the rolling hash to get the next substring's hash in O(1) time

```python
def rabin_karp(text, pattern, d=256, q=101):
    """
    Searches for pattern in text using the Rabin-Karp algorithm.
    
    Parameters:
    - text: the text to search in
    - pattern: the pattern to search for
    - d: number of characters in the alphabet
    - q: a prime number for hash calculation
    
    Returns:
    - List of starting indices of matches
    """
    results = []
    
    # Edge cases
    if not pattern:
        return [i for i in range(len(text) + 1)]
    if not text or len(pattern) > len(text):
        return []
    
    m = len(pattern)
    n = len(text)
    
    # Calculate hash for pattern and first window of text
    pattern_hash = 0
    text_hash = 0
    h = 1  # d^(m-1) % q
    
    # Calculate the value of h
    for i in range(m-1):
        h = (h * d) % q
    
    # Calculate initial hash values
    for i in range(m):
        pattern_hash = (d * pattern_hash + ord(pattern[i])) % q
        text_hash = (d * text_hash + ord(text[i])) % q
    
    # Slide the pattern over text one by one
    for i in range(n - m + 1):
        # Check if hash values match
        if pattern_hash == text_hash:
            # Verify character by character
            match = True
            for j in range(m):
                if text[i+j] != pattern[j]:
                    match = False
                    break
            
            if match:
                results.append(i)
        
        # Calculate hash for next window
        if i < n - m:
            text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i+m])) % q
            
            # Make sure we have a positive hash
            if text_hash < 0:
                text_hash += q
    
    return results
```

## Time and Space Complexity

- **Time Complexity**: 
  - Average case: O(n + m) where n is the length of the text and m is the length of the pattern
  - Worst case: O(n*m) if there are many hash collisions
- **Space Complexity**: O(1) - Only a constant amount of extra space is used

## Advantages and Disadvantages

### Advantages

- Efficient for multiple pattern searching (can hash all patterns once)
- Average case time complexity is linear
- Simple to implement once the rolling hash concept is understood
- Can be extended to 2D pattern matching and other applications

### Disadvantages

- Worst case time complexity is still O(n*m) due to potential hash collisions
- Performance depends heavily on the choice of hash function and parameters
- Generally slower than specialized algorithms like KMP and Boyer-Moore for single pattern searching
- Requires careful implementation to avoid overflow in hash calculations

## Use Cases

- Multiple pattern matching (searching for many patterns simultaneously)
- Plagiarism detection
- DNA sequence matching
- File integrity verification
- Finding duplicated substrings or files

## Implementation Details

### Python Implementation

```python
def rabin_karp(text, pattern, d=256, q=101):
    """
    Searches for pattern in text using the Rabin-Karp algorithm.
    
    Parameters:
    - text: the text to search in
    - pattern: the pattern to search for
    - d: number of characters in the alphabet
    - q: a prime number for hash calculation
    
    Returns:
    - List of starting indices of matches
    """
    results = []
    
    # Edge cases
    if not pattern:
        return [i for i in range(len(text) + 1)]
    if not text or len(pattern) > len(text):
        return []
    
    m = len(pattern)
    n = len(text)
    
    # Calculate hash for pattern and first window of text
    pattern_hash = 0
    text_hash = 0
    h = 1  # d^(m-1) % q
    
    # Calculate the value of h
    for i in range(m-1):
        h = (h * d) % q
    
    # Calculate initial hash values
    for i in range(m):
        pattern_hash = (d * pattern_hash + ord(pattern[i])) % q
        text_hash = (d * text_hash + ord(text[i])) % q
    
    # Slide the pattern over text one by one
    for i in range(n - m + 1):
        # Check if hash values match
        if pattern_hash == text_hash:
            # Verify character by character
            match = True
            for j in range(m):
                if text[i+j] != pattern[j]:
                    match = False
                    break
            
            if match:
                results.append(i)
        
        # Calculate hash for next window
        if i < n - m:
            text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i+m])) % q
            
            # Make sure we have a positive hash
            if text_hash < 0:
                text_hash += q
    
    return results

# Example usage
text = "ABCCDDAEFG"
pattern = "CDD"
matches = rabin_karp(text, pattern)
print(f"Pattern found at indices: {matches}")  # Output: [2]
```

### Java Implementation

```java
import java.util.ArrayList;
import java.util.List;

public class RabinKarp {
    public static List<Integer> rabinKarp(String text, String pattern, int d, int q) {
        List<Integer> results = new ArrayList<>();
        
        // Edge cases
        if (pattern.isEmpty()) {
            for (int i = 0; i <= text.length(); i++) {
                results.add(i);
            }
            return results;
        }
        if (text.isEmpty() || pattern.length() > text.length()) {
            return results;
        }
        
        int m = pattern.length();
        int n = text.length();
        
        // Calculate hash for pattern and first window of text
        int patternHash = 0;
        int textHash = 0;
        int h = 1;  // d^(m-1) % q
        
        // Calculate the value of h
        for (int i = 0; i < m-1; i++) {
            h = (h * d) % q;
        }
        
        // Calculate initial hash values
        for (int i = 0; i < m; i++) {
            patternHash = (d * patternHash + pattern.charAt(i)) % q;
            textHash = (d * textHash + text.charAt(i)) % q;
        }
        
        // Slide the pattern over text one by one
        for (int i = 0; i <= n - m; i++) {
            // Check if hash values match
            if (patternHash == textHash) {
                // Verify character by character
                boolean match = true;
                for (int j = 0; j < m; j++) {
                    if (text.charAt(i+j) != pattern.charAt(j)) {
                        match = false;
                        break;
                    }
                }
                
                if (match) {
                    results.add(i);
                }
            }
            
            // Calculate hash for next window
            if (i < n - m) {
                textHash = (d * (textHash - text.charAt(i) * h) + text.charAt(i+m)) % q;
                
                // Make sure we have a positive hash
                if (textHash < 0) {
                    textHash += q;
                }
            }
        }
        
        return results;
    }
    
    public static void main(String[] args) {
        String text = "ABCCDDAEFG";
        String pattern = "CDD";
        List<Integer> matches = rabinKarp(text, pattern, 256, 101);
        System.out.println("Pattern found at indices: " + matches);  // Output: [2]
    }
}
```

## Understanding the Rolling Hash

The key to Rabin-Karp's efficiency is the rolling hash function, which allows us to compute the hash of the next substring in O(1) time based on the hash of the current substring.

For a window of characters [c₁, c₂, ..., cₘ], the hash is calculated as:

Hash = (c₁ × d^(m-1) + c₂ × d^(m-2) + ... + cₘ × d^0) mod q

Where:
- d is the size of the character set (e.g., 256 for ASCII)
- q is a prime number to reduce collisions
- m is the pattern length

When sliding the window from [c₁, c₂, ..., cₘ] to [c₂, c₃, ..., cₘ₊₁], the new hash is computed as:

New_Hash = (d × (Old_Hash - c₁ × d^(m-1)) + cₘ₊₁) mod q

This formula efficiently removes the contribution of the first character and adds the contribution of the new character.

## Variations

### Multi-Pattern Rabin-Karp

```python
def multi_pattern_rabin_karp(text, patterns, d=256, q=101):
    """Search for multiple patterns in the text."""
    results = {pattern: [] for pattern in patterns}
    
    # Compute hash for each pattern
    pattern_hashes = {}
    for pattern in patterns:
        m = len(pattern)
        pattern_hash = 0
        
        for i in range(m):
            pattern_hash = (d * pattern_hash + ord(pattern[i])) % q
            
        pattern_hashes[pattern] = (pattern_hash, m)
    
    # Group patterns by length
    patterns_by_length = {}
    for pattern in patterns:
        length = len(pattern)
        if length not in patterns_by_length:
            patterns_by_length[length] = []
        patterns_by_length[length].append(pattern)
    
    # Search for each group of patterns with the same length
    for length, group in patterns_by_length.items():
        n = len(text)
        if length > n:
            continue
            
        # Initialize hash for first window
        text_hash = 0
        h = 1
        for i in range(length-1):
            h = (h * d) % q
            
        for i in range(length):
            text_hash = (d * text_hash + ord(text[i])) % q
            
        # Check each position in text
        for i in range(n - length + 1):
            # Check if hash matches any pattern of this length
            for pattern in group:
                pattern_hash = pattern_hashes[pattern][0]
                if text_hash == pattern_hash:
                    # Verify character by character
                    match = True
                    for j in range(length):
                        if text[i+j] != pattern[j]:
                            match = False
                            break
                    
                    if match:
                        results[pattern].append(i)
            
            # Calculate hash for next window
            if i < n - length:
                text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i+length])) % q
                if text_hash < 0:
                    text_hash += q
    
    return results
```

### Fingerprinting Documents

Rabin-Karp's rolling hash is also used for document fingerprinting:

```python
def document_fingerprint(text, k=5, d=256, q=101):
    """
    Generate k-gram fingerprints of a document for plagiarism detection.
    Returns a set of hash values for k-character sequences.
    """
    n = len(text)
    if n < k:
        return set()
        
    fingerprints = set()
    
    # Compute hash for first k-gram
    window_hash = 0
    h = 1
    for i in range(k-1):
        h = (h * d) % q
        
    for i in range(k):
        window_hash = (d * window_hash + ord(text[i])) % q
    
    fingerprints.add(window_hash)
    
    # Compute hash for remaining k-grams using rolling hash
    for i in range(n - k):
        window_hash = (d * (window_hash - ord(text[i]) * h) + ord(text[i+k])) % q
        if window_hash < 0:
            window_hash += q
        fingerprints.add(window_hash)
    
    return fingerprints
```

## Interview Tips

- Explain the concept of rolling hash and how it allows for efficient string searching
- Discuss the choice of parameters (d and q) and their impact on hash collisions
- Compare Rabin-Karp with other string matching algorithms (KMP, Boyer-Moore)
- Highlight Rabin-Karp's strength in multi-pattern searching
- Mention applications beyond simple string matching (document similarity, plagiarism detection)
- Be prepared to analyze worst-case scenarios and hash collision probability

## Practice Problems

1. Implement Rabin-Karp to find all occurrences of multiple patterns in a text
2. Use Rabin-Karp to find the longest repeated substring in a text
3. Implement a plagiarism detector using document fingerprinting
4. Find all palindromes of length > k in a text using Rabin-Karp
5. Implement a 2D pattern matching algorithm using Rabin-Karp for image processing
