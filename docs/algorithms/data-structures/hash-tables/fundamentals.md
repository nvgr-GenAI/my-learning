# Hash Tables Fundamentals

## 📋 What are Hash Tables?

A **hash table** (or hash map) is a data structure that maps keys to values using a hash function. It provides average O(1) time complexity for basic operations, making it one of the most efficient data structures for key-value storage and retrieval.

## 🔧 Core Components

### 1. Hash Function
A function that converts keys into array indices:

```python
def simple_hash(key, table_size):
    """Simple hash function using modulo"""
    return hash(key) % table_size

def djb2_hash(key):
    """DJB2 hash algorithm - better distribution"""
    hash_value = 5381
    for char in str(key):
        hash_value = ((hash_value << 5) + hash_value) + ord(char)
    return hash_value & 0xFFFFFFFF  # 32-bit
```

### 2. Collision Resolution

#### Separate Chaining
Store colliding elements in linked lists:

```python
class HashTableChaining:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def put(self, key, value):
        index = self._hash(key)
        bucket = self.table[index]
        
        # Update existing key
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        # Add new key-value pair
        bucket.append((key, value))
    
    def get(self, key):
        index = self._hash(key)
        bucket = self.table[index]
        
        for k, v in bucket:
            if k == key:
                return v
        raise KeyError(key)
    
    def delete(self, key):
        index = self._hash(key)
        bucket = self.table[index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                return
        raise KeyError(key)
```

#### Open Addressing (Linear Probing)
Find next available slot when collision occurs:

```python
class HashTableLinearProbing:
    def __init__(self, size=10):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size
        self.count = 0
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def _resize(self):
        """Resize when load factor exceeds 0.7"""
        if self.count >= self.size * 0.7:
            old_keys, old_values = self.keys, self.values
            self.size *= 2
            self.keys = [None] * self.size
            self.values = [None] * self.size
            self.count = 0
            
            for i in range(len(old_keys)):
                if old_keys[i] is not None:
                    self.put(old_keys[i], old_values[i])
    
    def put(self, key, value):
        self._resize()
        index = self._hash(key)
        
        while self.keys[index] is not None:
            if self.keys[index] == key:
                self.values[index] = value  # Update
                return
            index = (index + 1) % self.size  # Linear probing
        
        # Insert new
        self.keys[index] = key
        self.values[index] = value
        self.count += 1
    
    def get(self, key):
        index = self._hash(key)
        
        while self.keys[index] is not None:
            if self.keys[index] == key:
                return self.values[index]
            index = (index + 1) % self.size
        
        raise KeyError(key)
```

## 📊 Load Factor and Performance

The **load factor** (α) = number of elements / table size

- **α < 0.7**: Good performance, low collision rate
- **α > 0.7**: Performance degrades, consider resizing
- **α = 1.0**: Table full (open addressing), many collisions

```python
def calculate_load_factor(num_elements, table_size):
    return num_elements / table_size

# Automatic resizing strategy
def should_resize(num_elements, table_size):
    return calculate_load_factor(num_elements, table_size) > 0.7
```

## 🎯 Hash Function Properties

### Good Hash Function Characteristics:
1. **Deterministic**: Same input → same output
2. **Uniform Distribution**: Minimizes clustering
3. **Fast Computation**: O(1) time complexity
4. **Avalanche Effect**: Small input change → large output change

### Common Hash Functions:

```python
def multiplicative_hash(key, table_size):
    """Multiplicative hashing using golden ratio"""
    A = 0.6180339887  # (√5 - 1) / 2
    return int(table_size * ((key * A) % 1))

def polynomial_hash(string, base=31):
    """Polynomial rolling hash for strings"""
    hash_value = 0
    for char in string:
        hash_value = (hash_value * base + ord(char)) % (10**9 + 7)
    return hash_value
```

## 🚀 Advanced Techniques

### 1. Robin Hood Hashing
Minimize variance in probe distances:

```python
class RobinHoodHashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [(None, None, -1)] * size  # (key, value, distance)
    
    def put(self, key, value):
        index = hash(key) % self.size
        distance = 0
        
        while True:
            current_key, current_value, current_distance = self.table[index]
            
            if current_key is None:
                self.table[index] = (key, value, distance)
                break
            
            if current_key == key:
                self.table[index] = (key, value, distance)
                break
            
            # Robin Hood: steal from the rich
            if distance > current_distance:
                self.table[index] = (key, value, distance)
                key, value, distance = current_key, current_value, current_distance
            
            index = (index + 1) % self.size
            distance += 1
```

### 2. Cuckoo Hashing
Guarantees O(1) worst-case lookup:

```python
class CuckooHashTable:
    def __init__(self, size=10):
        self.size = size
        self.table1 = [None] * size
        self.table2 = [None] * size
        self.hash1 = lambda x: hash(x) % size
        self.hash2 = lambda x: (hash(x) // size) % size
    
    def put(self, key, value):
        if self._insert(key, value, 0):
            return
        
        # Rehash if insertion fails
        self._rehash()
        self.put(key, value)
    
    def _insert(self, key, value, count):
        if count >= self.size:
            return False  # Cycle detected
        
        pos1 = self.hash1(key)
        if self.table1[pos1] is None:
            self.table1[pos1] = (key, value)
            return True
        
        # Evict and move to second table
        old_key, old_value = self.table1[pos1]
        self.table1[pos1] = (key, value)
        
        pos2 = self.hash2(old_key)
        if self.table2[pos2] is None:
            self.table2[pos2] = (old_key, old_value)
            return True
        
        # Continue eviction chain
        evicted_key, evicted_value = self.table2[pos2]
        self.table2[pos2] = (old_key, old_value)
        
        return self._insert(evicted_key, evicted_value, count + 1)
```

## 🎨 Common Patterns

### 1. Frequency Counter
```python
def count_frequencies(items):
    freq = {}
    for item in items:
        freq[item] = freq.get(item, 0) + 1
    return freq

# Using defaultdict
from collections import defaultdict

def count_frequencies_default(items):
    freq = defaultdict(int)
    for item in items:
        freq[item] += 1
    return dict(freq)
```

### 2. Index Mapping
```python
def two_sum(nums, target):
    """Find two numbers that add up to target"""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

### 3. Grouping
```python
def group_anagrams(strs):
    """Group strings that are anagrams"""
    groups = defaultdict(list)
    for s in strs:
        # Use sorted string as key
        key = ''.join(sorted(s))
        groups[key].append(s)
    return list(groups.values())
```

## 📈 Time & Space Complexity

| Operation | Average | Worst Case | Space |
|-----------|---------|------------|-------|
| **Search** | O(1) | O(n) | O(n) |
| **Insert** | O(1) | O(n) | O(n) |
| **Delete** | O(1) | O(n) | O(n) |

**Note**: Worst case occurs when all keys hash to same index (poor hash function or adversarial input).

## 🔍 When to Use Hash Tables

**✅ Good for:**
- Fast lookups by key
- Counting/frequency analysis  
- Caching/memoization
- Set operations
- Database indexing

**❌ Avoid when:**
- Need sorted order
- Range queries required
- Memory is very limited
- Need predictable worst-case performance

---

**Next**: Practice with [Easy Hash Table Problems](easy-problems.md) to apply these concepts!
