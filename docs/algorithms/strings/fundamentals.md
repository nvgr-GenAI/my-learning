# String Fundamentals

## Overview

Strings are sequences of characters and are one of the most common data types in programming. Understanding string manipulation, algorithms, and optimizations is crucial for solving a wide range of problems efficiently.

## Basic Concepts

### String Representation

In most programming languages, strings are represented as arrays of characters. However, the implementation details vary:

- **Immutable strings**: Languages like Java, Python, and JavaScript use immutable strings, meaning once a string is created, it cannot be modified.
- **Mutable strings**: Some languages (like C++) allow direct modification of string contents.

### String Operations

Common string operations include:

- **Concatenation**: Combining two or more strings
- **Substring**: Extracting a portion of a string
- **Search**: Finding occurrences of a pattern within a string
- **Comparison**: Comparing two strings lexicographically
- **Modification**: Replacing, inserting, or deleting characters

## Time Complexity Analysis

Understanding the time complexity of string operations is essential:

| Operation | Average Case | Worst Case | Notes |
|-----------|--------------|------------|-------|
| Access | O(1) | O(1) | Direct access to a character at a specific index |
| Search | O(n+m) | O(n*m) | Searching for a pattern of length m in a string of length n |
| Insertion | O(n) | O(n) | For immutable strings, requires creating a new string |
| Deletion | O(n) | O(n) | For immutable strings, requires creating a new string |
| Concatenation | O(n+m) | O(n+m) | Combining strings of lengths n and m |

## Common String Challenges

1. **Pattern Matching**: Finding occurrences of a pattern within a larger text
2. **String Manipulation**: Transforming strings according to specific rules
3. **Lexicographical Problems**: Sorting, comparing, and analyzing strings based on character order
4. **Dynamic Programming on Strings**: Problems like edit distance, longest common subsequence, etc.

## String Algorithms

Some of the most important string algorithms include:

1. **Naive String Matching**: O(n*m) time complexity
2. **Knuth-Morris-Pratt (KMP)**: O(n+m) time complexity
3. **Rabin-Karp**: O(n+m) average case, O(n*m) worst case
4. **Boyer-Moore**: O(n/m) best case, O(n*m) worst case
5. **Z Algorithm**: O(n+m) time complexity
6. **Suffix Trees and Arrays**: Advanced data structures for string problems

## String in Different Programming Languages

### Python
```python
# String declaration and basic operations
s = "Hello, World!"
length = len(s)
char = s[1]  # 'e'
substring = s[0:5]  # "Hello"
```

### Java
```java
// String declaration and basic operations
String s = "Hello, World!";
int length = s.length();
char c = s.charAt(1);  // 'e'
String substring = s.substring(0, 5);  // "Hello"
```

### JavaScript
```javascript
// String declaration and basic operations
const s = "Hello, World!";
const length = s.length;
const char = s[1];  // 'e'
const substring = s.substring(0, 5);  // "Hello"
```

## Best Practices

1. **Choose the Right Algorithm**: Different string algorithms perform better in different scenarios
2. **Be Aware of Immutability**: In languages with immutable strings, excessive string modifications can lead to performance issues
3. **Use Built-in Functions**: Most languages provide optimized implementations of common string operations
4. **Consider Space Complexity**: Some string algorithms require additional space proportional to the input size

## Further Reading

- [Knuth-Morris-Pratt Algorithm](kmp.md)
- [Boyer-Moore Algorithm](boyer-moore.md)
- [Rabin-Karp Algorithm](rabin-karp.md)
- [Tries Data Structure](tries.md)
- [Suffix Arrays and Trees](suffix-arrays.md)
