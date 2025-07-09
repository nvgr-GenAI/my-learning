# String Comparison

## Overview

String comparison is a fundamental operation in computer science that determines the relationship between two strings. While simple equality checks are straightforward, string comparison encompasses a range of techniques from lexicographical ordering to advanced similarity measures.

Effective string comparison is essential for sorting, searching, data validation, and text analysis applications.

## Basic String Comparison

### 1. Lexicographical Comparison

The most common form of string comparison is lexicographical (dictionary) ordering, where strings are compared character by character from left to right.

```python
def lexicographical_compare(str1, str2):
    # Returns: -1 if str1 < str2, 0 if str1 == str2, 1 if str1 > str2
    
    for i in range(min(len(str1), len(str2))):
        if str1[i] < str2[i]:
            return -1
        elif str1[i] > str2[i]:
            return 1
    
    # If we get here, one string might be a prefix of the other
    if len(str1) < len(str2):
        return -1
    elif len(str1) > len(str2):
        return 1
    else:
        return 0  # Strings are equal
```

In most programming languages, this is built-in:

```python
# Python
comparison = "apple" < "banana"  # True

# Case-insensitive comparison
comparison = "Apple".lower() < "banana".lower()  # True
```

```java
// Java
int comparison = "apple".compareTo("banana");  // Negative value
boolean lessThan = comparison < 0;  // True

// Case-insensitive comparison
int caseInsensitiveComparison = "Apple".compareToIgnoreCase("banana");  // Negative value
```

**Time Complexity**: O(min(m,n)) where m and n are the lengths of the strings
**Space Complexity**: O(1)

### 2. Numeric String Comparison

When comparing strings that represent numbers, lexicographical comparison may not give the expected result:

```
"10" < "2" lexicographically, but 10 > 2 numerically
```

To handle this:

```python
def numeric_string_compare(str1, str2):
    try:
        num1 = int(str1)
        num2 = int(str2)
        return -1 if num1 < num2 else (1 if num1 > num2 else 0)
    except ValueError:
        # Fall back to lexicographical comparison if not valid numbers
        return lexicographical_compare(str1, str2)
```

### 3. Natural Sort Order

Natural sort order mimics how humans compare strings containing numbers by treating numeric parts as numbers:

```python
import re

def natural_sort_key(s):
    """Key function for natural sorting"""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', s)]
    
# Usage
sorted_list = sorted(["file1.txt", "file10.txt", "file2.txt"], key=natural_sort_key)
# Result: ["file1.txt", "file2.txt", "file10.txt"]
```

## String Similarity Measures

### 1. Edit Distance (Levenshtein Distance)

Measures the minimum number of single-character edits required to change one string into another.

```python
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]
```

[Read more about Edit Distance](edit-distance.md)

### 2. Hamming Distance

Counts the number of positions at which corresponding characters differ:

```python
def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Strings must have equal length")
    
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))
```

**Time Complexity**: O(n) where n is the length of the strings
**Space Complexity**: O(1)

### 3. Longest Common Subsequence (LCS)

Finds the longest subsequence present in both strings:

```python
def longest_common_subsequence(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

**Time Complexity**: O(m×n)
**Space Complexity**: O(m×n)

### 4. Jaro-Winkler Distance

Particularly effective for short strings like names:

```python
def jaro_distance(s1, s2):
    # If strings are identical
    if s1 == s2:
        return 1.0
    
    # If either string is empty
    if len(s1) == 0 or len(s2) == 0:
        return 0.0
    
    # Maximum distance for matching characters
    match_distance = max(len(s1), len(s2)) // 2 - 1
    
    # Arrays to track matched characters
    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)
    
    # Count of matching characters
    matches = 0
    
    # Count of transpositions
    transpositions = 0
    
    # Find matching characters within match_distance
    for i in range(len(s1)):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(s2))
        
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
    
    # If no matches, return 0
    if matches == 0:
        return 0.0
    
    # Count transpositions
    k = 0
    for i in range(len(s1)):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
    
    # Calculate Jaro distance
    return (matches / len(s1) + matches / len(s2) + 
            (matches - transpositions / 2) / matches) / 3.0
```

The Jaro-Winkler distance adds a prefix scale to the Jaro distance, giving more favorable ratings to strings that match from the beginning:

```python
def jaro_winkler_distance(s1, s2, p=0.1):
    jaro = jaro_distance(s1, s2)
    
    # Common prefix length up to 4 characters
    prefix_len = 0
    for i in range(min(min(len(s1), len(s2)), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
    
    return jaro + prefix_len * p * (1 - jaro)
```

### 5. Cosine Similarity

Treats strings as vectors in a high-dimensional space:

```python
from collections import Counter
import math

def cosine_similarity(s1, s2):
    # Count character frequencies
    counter1 = Counter(s1)
    counter2 = Counter(s2)
    
    # Find common characters
    common_chars = set(counter1.keys()) & set(counter2.keys())
    
    # Calculate dot product
    dot_product = sum(counter1[char] * counter2[char] for char in common_chars)
    
    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(count * count for count in counter1.values()))
    magnitude2 = math.sqrt(sum(count * count for count in counter2.values()))
    
    if magnitude1 * magnitude2 == 0:
        return 0
    
    return dot_product / (magnitude1 * magnitude2)
```

## Specialized String Comparison Techniques

### 1. Phonetic Matching

Phonetic algorithms compare strings based on their pronunciation:

#### Soundex

```python
def soundex(s):
    # Maps letters to soundex digits
    soundex_dict = {
        'b': 1, 'f': 1, 'p': 1, 'v': 1,
        'c': 2, 'g': 2, 'j': 2, 'k': 2, 'q': 2, 's': 2, 'x': 2, 'z': 2,
        'd': 3, 't': 3,
        'l': 4,
        'm': 5, 'n': 5,
        'r': 6
    }
    
    # Convert to uppercase and keep only letters
    s = ''.join(c for c in s.upper() if c.isalpha())
    
    if not s:
        return "0000"
    
    # Keep first letter
    result = s[0]
    
    # Replace consonants with digits
    for i in range(1, len(s)):
        if s[i].lower() in soundex_dict and s[i] != s[i-1]:
            digit = str(soundex_dict[s[i].lower()])
            # Don't add if same as previous digit
            if digit != result[-1]:
                result += digit
    
    # Remove vowels and 'h', 'w', 'y'
    result = result[0] + ''.join(c for c in result[1:] if c != '0')
    
    # Pad with zeros and limit to length 4
    result = result + '0' * 4
    return result[:4]
```

#### Double Metaphone

This is more complex and typically implemented using specialized libraries.

### 2. Fuzzy Matching

Fuzzy matching combines multiple similarity measures for flexible string comparison:

```python
def fuzzy_match(s1, s2, threshold=0.7):
    # Normalize strings
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    
    # Exact match
    if s1 == s2:
        return 1.0
    
    # Combine different similarity measures
    ed_sim = 1.0 - levenshtein_distance(s1, s2) / max(len(s1), len(s2))
    jw_sim = jaro_winkler_distance(s1, s2)
    
    # You could add more measures like Soundex comparison or cosine similarity
    
    # Weighted combination
    similarity = 0.4 * ed_sim + 0.6 * jw_sim
    
    return similarity >= threshold, similarity
```

## String Comparison in Different Languages

### Python

```python
# Case-sensitive comparison
"apple" == "Apple"  # False

# Case-insensitive comparison
"apple".lower() == "Apple".lower()  # True

# Lexicographical comparison
"apple" < "banana"  # True

# startswith and endswith
"hello world".startswith("hello")  # True
"hello world".endswith("world")  # True
```

### Java

```java
String str1 = "apple";
String str2 = "Apple";

// Case-sensitive comparison
boolean equals = str1.equals(str2);  // False

// Case-insensitive comparison
boolean equalsIgnoreCase = str1.equalsIgnoreCase(str2);  // True

// Lexicographical comparison
int comparison = str1.compareTo(str2);  // Positive value (lowercase > uppercase)

// Case-insensitive lexicographical comparison
int caseInsensitiveComparison = str1.compareToIgnoreCase(str2);  // 0 (equal)

// startsWith and endsWith
boolean starts = "hello world".startsWith("hello");  // True
boolean ends = "hello world".endsWith("world");  // True
```

## Practical Applications

1. **Spell Checking**: Finding close matches to misspelled words
2. **Search Systems**: Supporting approximate matching for user queries
3. **Data Deduplication**: Identifying similar or duplicate records
4. **Plagiarism Detection**: Comparing text documents for similarity
5. **Auto-Correction**: Suggesting corrections for user input
6. **Name Matching**: Finding matching names despite spelling variations

## Comparison of Approaches

| Method | Best For | Time Complexity | Key Advantage | Key Limitation |
|--------|----------|----------------|---------------|----------------|
| Lexicographical | Sorting | O(min(m,n)) | Simple, standard | Not robust to errors |
| Edit Distance | Error tolerance | O(m×n) | Accounts for edits | Computationally expensive |
| Hamming Distance | Fixed-length | O(n) | Fast | Only equal length strings |
| LCS | Finding common parts | O(m×n) | Identifies shared content | Computationally expensive |
| Jaro-Winkler | Short strings, names | O(m×n) | Good for typos in names | Not ideal for long strings |
| Phonetic | Name sounds | O(n) | Matches pronunciation | Language dependent |
| Fuzzy Match | Flexible comparison | Varies | Combines multiple methods | Requires calibration |

## Best Practices

1. **Normalize strings** before comparison (e.g., lowercase, trim, remove special characters)
2. **Choose the appropriate algorithm** for your specific use case
3. **Consider performance implications** for large-scale comparisons
4. **Set appropriate thresholds** for approximate matching
5. **Use established libraries** for complex algorithms

## Related Topics

- [Edit Distance](edit-distance.md)
- [String Pattern Matching](pattern-matching.md)
- [String Searching Algorithms](../searching/index.md)
- [Regular Expressions](../advanced/regex.md)
