# Z Algorithm

## Overview
The Z algorithm is an efficient string matching algorithm that finds all occurrences of a pattern in a text in linear time. It uses a Z array that stores the length of the longest substring starting at each position that is also a prefix of the string.

## Algorithm

1. Concatenate pattern and text with a special character in between, e.g., pattern$text
2. Build the Z array for the concatenated string
3. Search for pattern by looking for Z values equal to the pattern length

## Implementation

### Python Implementation

```python
def z_function(s):
    n = len(s)
    z = [0] * n
    left, right = 0, 0
    
    for i in range(1, n):
        # If i is inside the z-box, use previously computed values
        if i <= right:
            z[i] = min(right - i + 1, z[i - left])
            
        # Try to extend the z-box
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
            
        # Update z-box if needed
        if i + z[i] - 1 > right:
            left, right = i, i + z[i] - 1
            
    return z

def z_algorithm(text, pattern):
    concat = pattern + "$" + text
    z = z_function(concat)
    results = []
    
    # Search for pattern occurrences
    for i in range(len(pattern) + 1, len(concat)):
        if z[i] == len(pattern):
            results.append(i - len(pattern) - 1)  # Adjust index to match original text
            
    return results

# Example usage
text = "ABABDABACDABABCABAB"
pattern = "ABABCABAB"
matches = z_algorithm(text, pattern)
print(f"Pattern found at positions: {matches}")
```

### Java Implementation

```java
public class ZAlgorithm {
    public static int[] zFunction(String s) {
        int n = s.length();
        int[] z = new int[n];
        int left = 0, right = 0;
        
        for (int i = 1; i < n; i++) {
            // If i is inside the z-box
            if (i <= right) {
                z[i] = Math.min(right - i + 1, z[i - left]);
            }
            
            // Try to extend the z-box
            while (i + z[i] < n && s.charAt(z[i]) == s.charAt(i + z[i])) {
                z[i]++;
            }
            
            // Update z-box if needed
            if (i + z[i] - 1 > right) {
                left = i;
                right = i + z[i] - 1;
            }
        }
        
        return z;
    }
    
    public static List<Integer> search(String text, String pattern) {
        String concat = pattern + "$" + text;
        int[] z = zFunction(concat);
        List<Integer> results = new ArrayList<>();
        
        // Search for pattern occurrences
        for (int i = pattern.length() + 1; i < concat.length(); i++) {
            if (z[i] == pattern.length()) {
                results.add(i - pattern.length() - 1);
            }
        }
        
        return results;
    }
    
    public static void main(String[] args) {
        String text = "ABABDABACDABABCABAB";
        String pattern = "ABABCABAB";
        List<Integer> matches = search(text, pattern);
        System.out.println("Pattern found at positions: " + matches);
    }
}
```

## Complexity Analysis

- **Time Complexity**: O(n + m), where n is the length of the text and m is the length of the pattern
- **Space Complexity**: O(n + m) for the Z array of the concatenated string

## Advantages and Disadvantages

### Advantages
- Linear time complexity O(n + m)
- Simple to implement compared to KMP
- Efficient for multiple pattern searching with preprocessing

### Disadvantages
- Requires extra space for the Z array
- Not as widely used as KMP or Rabin-Karp

## Use Cases
- String matching and searching
- Detecting repeating patterns in a string
- Finding all occurrences of a pattern in a text
- Used in bioinformatics for DNA sequence matching

## Variations
- Z algorithm with suffix arrays for multiple pattern matching
- Z algorithm with rolling hash for approximate string matching

## Interview Tips
- Know how to construct the Z array efficiently
- Understand the concept of Z-box (or Z-window)
- Be able to explain the difference between Z algorithm and KMP
- Practice implementing it without the concatenation trick (separate Z array calculation for pattern and text)

## Practice Problems

1. [Implement the Z Algorithm](https://leetcode.com/problems/implement-strstr/) - Implement the Z algorithm for string searching
2. [Repeated String Match](https://leetcode.com/problems/repeated-string-match/) - Find how many times A has to be repeated so that B is a substring of the repeated A
3. [Longest Prefix that is also a Suffix](https://leetcode.com/problems/longest-happy-prefix/) - Find the longest prefix that is also a suffix in a string
4. [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/) - Find the minimum window in a string that contains all characters of another string
5. [String Matching in an Array](https://leetcode.com/problems/string-matching-in-an-array/) - Find all strings that are a substring of another string in an array

## References
1. Dan Gusfield. "Algorithms on Strings, Trees, and Sequences: Computer Science and Computational Biology". Cambridge University Press, 1997.
2. Cormen, Thomas H., et al. "Introduction to Algorithms". MIT Press, 2009.
