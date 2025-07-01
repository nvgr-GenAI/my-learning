# Hash Tables

## Overview

Hash Tables (also called Hash Maps) are data structures that implement an associative array abstract data type, mapping keys to values.

## Key Concepts

### Hash Function

A function that converts a key into an array index.

### Collision Handling

- **Chaining**: Store multiple elements in the same bucket using linked lists
- **Open Addressing**: Find another empty slot using probing

## Implementation

### Basic Hash Table

```python
class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(self.size)]
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def put(self, key, value):
        index = self._hash(key)
        bucket = self.table[index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        bucket.append((key, value))
    
    def get(self, key):
        index = self._hash(key)
        bucket = self.table[index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        raise KeyError(key)
    
    def remove(self, key):
        index = self._hash(key)
        bucket = self.table[index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                return
        
        raise KeyError(key)
```

## Common Hash Table Problems

1. **Two Sum**
2. **Group Anagrams**
3. **Valid Anagram**
4. **Longest Substring Without Repeating Characters**
5. **Top K Frequent Elements**

## Techniques

### Frequency Counting

```python
def count_frequency(arr):
    freq = {}
    for item in arr:
        freq[item] = freq.get(item, 0) + 1
    return freq
```

### Using Sets for Uniqueness

```python
def has_duplicates(arr):
    seen = set()
    for item in arr:
        if item in seen:
            return True
        seen.add(item)
    return False
```

## Time Complexities

| Operation | Average | Worst Case |
|-----------|---------|------------|
| Search    | O(1)    | O(n)       |
| Insert    | O(1)    | O(n)       |
| Delete    | O(1)    | O(n)       |

## Practice Problems

- [ ] Two Sum
- [ ] Valid Anagram
- [ ] Group Anagrams
- [ ] Intersection of Two Arrays
- [ ] Happy Number
- [ ] Contains Duplicate
