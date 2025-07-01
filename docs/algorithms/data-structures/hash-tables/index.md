# Hash Tables & Hash Maps

## 📋 Overview

Hash tables (also known as hash maps) are one of the most important and widely used data structures in computer science. They provide average O(1) time complexity for insertion, deletion, and lookup operations by using a hash function to map keys to array indices.

## 🔍 What You'll Learn

- **Fundamentals**: Hash functions, collision handling, load factor
- **Implementation**: Building hash tables from scratch
- **Problem Solving**: Common patterns and techniques
- **Advanced Topics**: Perfect hashing, bloom filters, consistent hashing

## 📚 Section Contents

### 🎯 Fundamentals
- **[Hash Tables Fundamentals](fundamentals.md)** - Core concepts, hash functions, collision resolution
- **[Implementation Details](implementation.md)** - Building hash tables from scratch

### 💪 Practice Problems

#### 🟢 Easy Problems
- **[Easy Hash Table Problems](easy-problems.md)**
  - Two Sum, Valid Anagram, First Unique Character
  - Hash Set operations, Simple frequency counting

#### 🟡 Medium Problems  
- **[Medium Hash Table Problems](medium-problems.md)**
  - Group Anagrams, Top K Frequent Elements
  - Subarray problems, Hash map design

#### 🔴 Hard Problems
- **[Hard Hash Table Problems](hard-problems.md)**
  - LRU Cache, Design Twitter, Alien Dictionary
  - Advanced hash table applications

## 🎨 Key Patterns

### 1. **Frequency Counting**
```python
# Count occurrences of elements
freq = {}
for item in array:
    freq[item] = freq.get(item, 0) + 1
```

### 2. **Two Sum Pattern**
```python
# Find pairs that sum to target
seen = {}
for i, num in enumerate(nums):
    complement = target - num
    if complement in seen:
        return [seen[complement], i]
    seen[num] = i
```

### 3. **Sliding Window with Hash Map**
```python
# Track elements in current window
window = {}
for right in range(len(s)):
    window[s[right]] = window.get(s[right], 0) + 1
    # Shrink window if needed
    while condition:
        window[s[left]] -= 1
        if window[s[left]] == 0:
            del window[s[left]]
        left += 1
```

## 📊 Complexity Summary

| Operation | Average | Worst Case |
|-----------|---------|------------|
| **Search** | O(1) | O(n) |
| **Insert** | O(1) | O(n) |
| **Delete** | O(1) | O(n) |
| **Space** | O(n) | O(n) |

## 🚀 Getting Started

Start with **[Hash Tables Fundamentals](fundamentals.md)** to understand the core concepts, then practice with problems based on your skill level:

- New to hash tables? → **[Easy Problems](easy-problems.md)**
- Comfortable with basics? → **[Medium Problems](medium-problems.md)**  
- Ready for challenges? → **[Hard Problems](hard-problems.md)**

---

*Hash tables are fundamental to many algorithms and system designs. Master them to unlock powerful problem-solving techniques!*
