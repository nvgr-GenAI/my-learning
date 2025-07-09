# String Basic Operations

## Overview

Basic string operations form the foundation of string manipulation in programming. Understanding these operations and their time complexities is essential for efficient string processing.

## Common Basic String Operations

### 1. String Creation and Access

Creating strings and accessing individual characters are fundamental operations.

```python
# Python
string = "Hello World"
first_char = string[0]  # 'H'
last_char = string[-1]  # 'd'
```

```java
// Java
String string = "Hello World";
char firstChar = string.charAt(0);  // 'H'
char lastChar = string.charAt(string.length() - 1);  // 'd'
```

**Time Complexity**: O(1) for access operations

### 2. String Length

Finding the length or size of a string.

```python
# Python
length = len("Hello World")  # 11
```

```java
// Java
int length = "Hello World".length();  // 11
```

**Time Complexity**: O(1) in most modern languages

### 3. String Concatenation

Joining two or more strings together.

```python
# Python
string1 = "Hello"
string2 = "World"
result = string1 + " " + string2  # "Hello World"

# Using join for multiple strings
result = " ".join(["Hello", "World"])  # "Hello World"
```

```java
// Java
String string1 = "Hello";
String string2 = "World";
String result = string1 + " " + string2;  // "Hello World"

// Using StringBuilder for efficient concatenation
StringBuilder sb = new StringBuilder();
sb.append(string1);
sb.append(" ");
sb.append(string2);
String result2 = sb.toString();  // "Hello World"
```

**Time Complexity**: 
- O(n) for simple concatenation
- O(n) for join operations
- Using StringBuilder in Java: O(n) amortized for multiple operations

### 4. Substring Extraction

Extracting a portion of a string.

```python
# Python
string = "Hello World"
sub = string[0:5]  # "Hello"
end_sub = string[6:]  # "World"
```

```java
// Java
String string = "Hello World";
String sub = string.substring(0, 5);  // "Hello"
String endSub = string.substring(6);  // "World"
```

**Time Complexity**: O(k) where k is the length of the substring

### 5. String Comparison

Comparing strings for equality or lexicographical ordering.

```python
# Python
string1 = "apple"
string2 = "apple"
string3 = "banana"

equal = string1 == string2  # True
less_than = string1 < string3  # True (lexicographical comparison)
```

```java
// Java
String string1 = "apple";
String string2 = "apple";
String string3 = "banana";

boolean equal = string1.equals(string2);  // true
boolean lessThan = string1.compareTo(string3) < 0;  // true
```

**Time Complexity**: O(min(n, m)) where n and m are the lengths of the strings being compared

### 6. Case Conversion

Converting string case (uppercase, lowercase, title case).

```python
# Python
upper = "hello".upper()  # "HELLO"
lower = "WORLD".lower()  # "world"
title = "hello world".title()  # "Hello World"
```

```java
// Java
String upper = "hello".toUpperCase();  // "HELLO"
String lower = "WORLD".toLowerCase();  // "world"
```

**Time Complexity**: O(n) where n is the string length

### 7. Trimming and Padding

Removing whitespace or adding padding characters.

```python
# Python
trimmed = "  hello  ".strip()  # "hello"
left_trimmed = "  hello  ".lstrip()  # "hello  "
right_trimmed = "  hello  ".rstrip()  # "  hello"
padded = "hello".ljust(10, '*')  # "hello*****"
```

```java
// Java
String trimmed = "  hello  ".trim();  // "hello"
String padded = String.format("%-10s", "hello").replace(' ', '*');  // "hello*****"
```

**Time Complexity**: O(n) where n is the string length

### 8. Searching and Contains

Checking if a string contains a substring or character.

```python
# Python
string = "Hello World"
contains = "World" in string  # True
index = string.find("World")  # 6
not_found = string.find("Python")  # -1
```

```java
// Java
String string = "Hello World";
boolean contains = string.contains("World");  // true
int index = string.indexOf("World");  # 6
int notFound = string.indexOf("Python");  # -1
```

**Time Complexity**: O(n*m) for naive search where n is the string length and m is the pattern length

### 9. Replacing

Replacing occurrences of a substring.

```python
# Python
string = "Hello World"
replaced = string.replace("World", "Python")  # "Hello Python"
```

```java
// Java
String string = "Hello World";
String replaced = string.replace("World", "Python");  # "Hello Python"
```

**Time Complexity**: O(n) where n is the string length

### 10. Splitting and Joining

Breaking a string into parts or joining parts into a string.

```python
# Python
string = "apple,banana,orange"
parts = string.split(",")  # ["apple", "banana", "orange"]
joined = "-".join(parts)  # "apple-banana-orange"
```

```java
// Java
String string = "apple,banana,orange";
String[] parts = string.split(",");  // ["apple", "banana", "orange"]
String joined = String.join("-", parts);  // "apple-banana-orange"
```

**Time Complexity**: O(n) for both operations where n is the string length

## Optimization Tips

1. **Use StringBuilder/StringBuffer in Java**: For multiple string concatenations
2. **Use join() instead of + for multiple concatenations**: More efficient and cleaner
3. **Prefer in-place operations when available**: To avoid creating unnecessary copies
4. **Use specialized string search algorithms**: For large-scale text processing
5. **Consider memory usage**: String operations can create many temporary objects

## Common Pitfalls

1. **String Immutability**: In languages with immutable strings (Java, Python), operations create new strings
2. **Off-by-one errors**: When working with indices and substrings
3. **Encoding issues**: Be aware of character encoding (ASCII, UTF-8, etc.)
4. **Performance concerns**: Naive string operations can be inefficient for large strings

## Practice Examples

1. Reverse a string
2. Check if a string is a palindrome
3. Convert between cases (camelCase, snake_case, kebab-case)
4. Implement a simple string compression algorithm
5. Create a function that counts occurrences of each character in a string

## Related String Algorithms

- [String Pattern Matching](pattern-matching.md)
- [String Parsing Techniques](parsing.md)
- [String Comparison Algorithms](comparison.md)
