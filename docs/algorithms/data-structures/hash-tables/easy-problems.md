# Hash Tables - Easy Problems

## 🎯 Learning Objectives

Master basic hash table operations and common patterns:

- Key-value lookups and storage
- Frequency counting
- Set operations
- Simple mapping problems

---

## Problem 1: Two Sum

**Difficulty**: 🟢 Easy  
**Pattern**: Index Mapping  
**Time**: O(n), **Space**: O(n)

### Problem Statement

Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

```
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: nums[0] + nums[1] = 2 + 7 = 9
```

### Solution

```python
def two_sum(nums, target):
    """
    Use hash map to store number -> index mapping
    For each number, check if complement exists
    """
    seen = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        
        if complement in seen:
            return [seen[complement], i]
        
        seen[num] = i
    
    return []  # No solution found

# Test
nums = [2, 7, 11, 15]
target = 9
print(two_sum(nums, target))  # [0, 1]
```

### Key Insights

- Hash map allows O(1) lookup of complements
- Store index as value for result construction
- Single pass through array

---

## Problem 2: Valid Anagram

**Difficulty**: 🟢 Easy  
**Pattern**: Frequency Counting  
**Time**: O(n), **Space**: O(1) - limited alphabet

### Problem Statement

Given two strings `s` and `t`, return true if `t` is an anagram of `s`.

```
Input: s = "anagram", t = "nagaram"
Output: true

Input: s = "rat", t = "car"
Output: false
```

### Solution

```python
def is_anagram(s, t):
    """
    Method 1: Frequency counting with hash map
    """
    if len(s) != len(t):
        return False
    
    freq = {}
    
    # Count characters in s
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    
    # Subtract characters in t
    for char in t:
        if char not in freq:
            return False
        freq[char] -= 1
        if freq[char] == 0:
            del freq[char]
    
    return len(freq) == 0

def is_anagram_v2(s, t):
    """
    Method 2: Using Counter from collections
    """
    from collections import Counter
    return Counter(s) == Counter(t)

def is_anagram_v3(s, t):
    """
    Method 3: Sorting approach
    """
    return sorted(s) == sorted(t)

# Test
s = "anagram"
t = "nagaram"
print(is_anagram(s, t))  # True
```

### Key Insights

- Anagrams have same character frequencies
- Hash map efficiently tracks character counts
- Multiple approaches: counting, sorting, built-in Counter

---

## Problem 3: First Unique Character

**Difficulty**: 🟢 Easy  
**Pattern**: Frequency Counting  
**Time**: O(n), **Space**: O(1)

### Problem Statement

Given a string `s`, find the first non-repeating character and return its index. If it doesn't exist, return -1.

```
Input: s = "leetcode"
Output: 0 (first 'l' appears once)

Input: s = "loveleetcode"
Output: 2 (first 'v' appears once)
```

### Solution

```python
def first_unique_char(s):
    """
    Two-pass approach: count frequencies, then find first unique
    """
    # Count frequencies
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    
    # Find first character with frequency 1
    for i, char in enumerate(s):
        if freq[char] == 1:
            return i
    
    return -1

def first_unique_char_v2(s):
    """
    Using Counter from collections
    """
    from collections import Counter
    freq = Counter(s)
    
    for i, char in enumerate(s):
        if freq[char] == 1:
            return i
    
    return -1

# Test
s = "leetcode"
print(first_unique_char(s))  # 0
```

### Key Insights

- Two-pass solution: count then find
- Hash map provides O(1) frequency lookup
- Return index of first unique character

---

## Problem 4: Contains Duplicate

**Difficulty**: 🟢 Easy  
**Pattern**: Set Operations  
**Time**: O(n), **Space**: O(n)

### Problem Statement

Given an integer array `nums`, return true if any value appears at least twice in the array.

```
Input: nums = [1,2,3,1]
Output: true

Input: nums = [1,2,3,4]
Output: false
```

### Solution

```python
def contains_duplicate(nums):
    """
    Method 1: Using set for O(1) lookups
    """
    seen = set()
    
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    
    return False

def contains_duplicate_v2(nums):
    """
    Method 2: Compare lengths
    """
    return len(nums) != len(set(nums))

def contains_duplicate_v3(nums):
    """
    Method 3: Using hash map (if need to track positions)
    """
    positions = {}
    
    for i, num in enumerate(nums):
        if num in positions:
            return True
        positions[num] = i
    
    return False

# Test
nums = [1, 2, 3, 1]
print(contains_duplicate(nums))  # True
```

### Key Insights

- Set provides O(1) membership testing
- Early termination when duplicate found
- Multiple approaches based on requirements

---

## Problem 5: Valid Parentheses (Hash Map Approach)

**Difficulty**: 🟢 Easy  
**Pattern**: Mapping + Stack  
**Time**: O(n), **Space**: O(n)

### Problem Statement

Given a string containing just characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

```
Input: s = "()[]{}"
Output: true

Input: s = "([)]"
Output: false
```

### Solution

```python
def is_valid_parentheses(s):
    """
    Use hash map for bracket mapping and stack for matching
    """
    # Hash map for bracket pairs
    bracket_map = {
        ')': '(',
        '}': '{',
        ']': '['
    }
    
    stack = []
    
    for char in s:
        if char in bracket_map:  # Closing bracket
            if not stack or stack.pop() != bracket_map[char]:
                return False
        else:  # Opening bracket
            stack.append(char)
    
    return len(stack) == 0

# Test
s = "()[]{}"
print(is_valid_parentheses(s))  # True
```

### Key Insights

- Hash map provides clean bracket pair mapping
- Stack tracks opening brackets
- Combined data structures solve the problem elegantly

---

## Problem 6: Roman to Integer

**Difficulty**: 🟢 Easy  
**Pattern**: Character Mapping  
**Time**: O(n), **Space**: O(1)

### Problem Statement

Convert a roman numeral to an integer.

```
Input: s = "III"
Output: 3

Input: s = "MCMXC"
Output: 1990
```

### Solution

```python
def roman_to_int(s):
    """
    Use hash map for roman numeral values
    Handle subtraction cases (IV, IX, etc.)
    """
    # Hash map for roman values
    roman_values = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    total = 0
    
    for i in range(len(s)):
        # If current value < next value, subtract (IV, IX, etc.)
        if (i < len(s) - 1 and 
            roman_values[s[i]] < roman_values[s[i + 1]]):
            total -= roman_values[s[i]]
        else:
            total += roman_values[s[i]]
    
    return total

def roman_to_int_v2(s):
    """
    Alternative: process from right to left
    """
    roman_values = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    total = 0
    prev_value = 0
    
    for char in reversed(s):
        value = roman_values[char]
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value
    
    return total

# Test
s = "MCMXC"
print(roman_to_int(s))  # 1990
```

### Key Insights

- Hash map stores character-to-value mapping
- Handle subtraction cases with lookahead or reverse processing
- Consider edge cases and special combinations

---

## 🎯 Practice Summary

### Patterns Covered

1. **Index Mapping**: Store value-to-index for lookups
2. **Frequency Counting**: Track occurrences of elements  
3. **Set Operations**: Membership testing and duplicates
4. **Character Mapping**: Map characters to values/properties

### Key Techniques

- Use `dict.get(key, default)` for safe access
- Combine hash maps with other data structures
- Consider space-time tradeoffs
- Handle edge cases (empty inputs, single elements)

### Next Steps

Ready for more challenges? Try **[Medium Hash Table Problems](medium-problems.md)** to tackle:

- Group Anagrams
- Top K Frequent Elements  
- Subarray Sum problems
- Hash Map design challenges

---

*Hash tables are fundamental - master these patterns and you'll solve many problems efficiently!*
