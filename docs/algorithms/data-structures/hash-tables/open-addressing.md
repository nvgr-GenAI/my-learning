# Hash Tables: Open Addressing

## 🔍 Overview

Open addressing is a collision resolution technique where all elements are stored directly in the hash table array. When a collision occurs, the algorithm probes for the next available slot using a predetermined sequence.

---

## 📊 Characteristics

### Key Properties

- **No Extra Memory**: All elements stored in the main array
- **Cache Friendly**: Better memory locality than chaining
- **Load Factor Limit**: Must maintain load factor below 1.0
- **Probing Required**: Need strategy to find next available slot
- **Clustering**: Can lead to primary and secondary clustering

### Memory Layout

```text
Open Addressing Example (Linear Probing):
Index:  0   1   2   3   4   5   6   7
Array: [A] [B] [C] [ ] [D] [E] [ ] [F]
        ↑       ↑       ↑       ↑
    hash(A)=0  hash(C)=2  hash(D)=4  hash(F)=7
              (B collided with A, placed at 1)
```

---

## ⏱️ Time Complexities

| Operation | Average Case | Worst Case | Notes |
|-----------|--------------|------------|-------|
| **Insert** | O(1) | O(n) | Depends on clustering |
| **Search** | O(1) | O(n) | May need to probe multiple slots |
| **Delete** | O(1) | O(n) | May require tombstone markers |
| **Space** | O(n) | O(n) | No extra memory for pointers |

---

## 🔍 Probing Strategies

### 1. Linear Probing

```python
class LinearProbingHashTable:
    """Hash table with linear probing."""
    
    def __init__(self, initial_capacity=16):
        """Initialize hash table."""
        self.capacity = initial_capacity
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.deleted = [False] * self.capacity  # Tombstone markers
        self.load_factor_threshold = 0.75
    
    def _hash(self, key):
        """Hash function."""
        return hash(key) % self.capacity
    
    def _resize(self):
        """Resize when load factor exceeds threshold."""
        old_keys = self.keys
        old_values = self.values
        old_capacity = self.capacity
        
        # Double capacity
        self.capacity *= 2
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.deleted = [False] * self.capacity
        
        # Rehash all elements
        for i in range(old_capacity):
            if old_keys[i] is not None:
                self.put(old_keys[i], old_values[i])
    
    def _probe(self, key):
        """Find slot for key using linear probing."""
        index = self._hash(key)
        
        while True:
            # Found empty slot or matching key
            if self.keys[index] is None or self.keys[index] == key:
                return index
            
            # Move to next slot (linear probing)
            index = (index + 1) % self.capacity
    
    def put(self, key, value):
        """Insert or update key-value pair."""
        # Resize if needed
        if self.size >= self.capacity * self.load_factor_threshold:
            self._resize()
        
        index = self._probe(key)
        
        # New key
        if self.keys[index] is None:
            self.size += 1
        
        self.keys[index] = key
        self.values[index] = value
        self.deleted[index] = False
    
    def get(self, key):
        """Retrieve value for key."""
        index = self._hash(key)
        
        while self.keys[index] is not None:
            if self.keys[index] == key and not self.deleted[index]:
                return self.values[index]
            index = (index + 1) % self.capacity
        
        raise KeyError(f"Key '{key}' not found")
    
    def delete(self, key):
        """Delete key-value pair using tombstone."""
        index = self._hash(key)
        
        while self.keys[index] is not None:
            if self.keys[index] == key and not self.deleted[index]:
                self.deleted[index] = True
                self.size -= 1
                return self.values[index]
            index = (index + 1) % self.capacity
        
        raise KeyError(f"Key '{key}' not found")
    
    def contains(self, key):
        """Check if key exists."""
        try:
            self.get(key)
            return True
        except KeyError:
            return False
    
    def load_factor(self):
        """Calculate load factor."""
        return self.size / self.capacity
    
    def __len__(self):
        return self.size
    
    def __str__(self):
        items = []
        for i in range(self.capacity):
            if self.keys[i] is not None and not self.deleted[i]:
                items.append((self.keys[i], self.values[i]))
        return f"HashTable({dict(items)})"
```

### 2. Quadratic Probing

```python
class QuadraticProbingHashTable:
    """Hash table with quadratic probing."""
    
    def __init__(self, initial_capacity=16):
        self.capacity = initial_capacity
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.deleted = [False] * self.capacity
        self.load_factor_threshold = 0.5  # Lower threshold for quadratic
    
    def _hash(self, key):
        return hash(key) % self.capacity
    
    def _probe(self, key):
        """Find slot using quadratic probing: h(k) + i²."""
        index = self._hash(key)
        i = 0
        
        while i < self.capacity:
            probe_index = (index + i * i) % self.capacity
            
            if (self.keys[probe_index] is None or 
                self.keys[probe_index] == key):
                return probe_index
            
            i += 1
        
        raise Exception("Hash table is full")
    
    def put(self, key, value):
        """Insert using quadratic probing."""
        if self.size >= self.capacity * self.load_factor_threshold:
            self._resize()
        
        index = self._probe(key)
        
        if self.keys[index] is None:
            self.size += 1
        
        self.keys[index] = key
        self.values[index] = value
        self.deleted[index] = False
    
    def _resize(self):
        """Resize hash table."""
        old_keys = self.keys
        old_values = self.values
        old_capacity = self.capacity
        
        self.capacity *= 2
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.deleted = [False] * self.capacity
        
        for i in range(old_capacity):
            if old_keys[i] is not None:
                self.put(old_keys[i], old_values[i])
```

### 3. Double Hashing

```python
class DoubleHashingHashTable:
    """Hash table with double hashing."""
    
    def __init__(self, initial_capacity=16):
        self.capacity = initial_capacity
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.deleted = [False] * self.capacity
        self.load_factor_threshold = 0.75
    
    def _hash1(self, key):
        """Primary hash function."""
        return hash(key) % self.capacity
    
    def _hash2(self, key):
        """Secondary hash function."""
        # Use a prime number less than capacity
        return 7 - (hash(key) % 7)
    
    def _probe(self, key):
        """Find slot using double hashing."""
        index = self._hash1(key)
        step = self._hash2(key)
        
        while True:
            if (self.keys[index] is None or 
                self.keys[index] == key):
                return index
            
            index = (index + step) % self.capacity
    
    def put(self, key, value):
        """Insert using double hashing."""
        if self.size >= self.capacity * self.load_factor_threshold:
            self._resize()
        
        index = self._probe(key)
        
        if self.keys[index] is None:
            self.size += 1
        
        self.keys[index] = key
        self.values[index] = value
        self.deleted[index] = False
    
    def _resize(self):
        """Resize hash table."""
        old_keys = self.keys
        old_values = self.values
        old_capacity = self.capacity
        
        self.capacity *= 2
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.deleted = [False] * self.capacity
        
        for i in range(old_capacity):
            if old_keys[i] is not None:
                self.put(old_keys[i], old_values[i])
```

---

## 📊 Probing Comparison

| **Method** | **Clustering** | **Performance** | **Complexity** |
|------------|----------------|-----------------|----------------|
| **Linear** | High (Primary) | Good average | Simple |
| **Quadratic** | Medium (Secondary) | Better distribution | Moderate |
| **Double** | Low | Best distribution | Complex |

---

## 🔧 Advanced Techniques

### Robin Hood Hashing

```python
class RobinHoodHashTable:
    """Hash table with Robin Hood hashing for better distribution."""
    
    def __init__(self, initial_capacity=16):
        self.capacity = initial_capacity
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.distances = [0] * self.capacity  # Distance from ideal position
    
    def _hash(self, key):
        return hash(key) % self.capacity
    
    def _distance(self, key, index):
        """Calculate distance from ideal position."""
        ideal = self._hash(key)
        return (index - ideal) % self.capacity
    
    def put(self, key, value):
        """Insert using Robin Hood hashing."""
        index = self._hash(key)
        distance = 0
        
        while True:
            # Empty slot
            if self.keys[index] is None:
                self.keys[index] = key
                self.values[index] = value
                self.distances[index] = distance
                self.size += 1
                return
            
            # Update existing key
            if self.keys[index] == key:
                self.values[index] = value
                return
            
            # Robin Hood: steal from the rich
            if distance > self.distances[index]:
                # Swap with current occupant
                self.keys[index], key = key, self.keys[index]
                self.values[index], value = value, self.values[index]
                self.distances[index], distance = distance, self.distances[index]
            
            index = (index + 1) % self.capacity
            distance += 1
```

---

## ✅ Advantages

- **Memory Efficient**: No extra memory for pointers
- **Cache Friendly**: Better memory locality
- **Simple Memory Model**: All data in contiguous array
- **Predictable Memory Usage**: Fixed memory footprint

## ❌ Disadvantages

- **Load Factor Limit**: Must keep load factor below 1.0
- **Clustering Issues**: Can lead to performance degradation
- **Complex Deletion**: Requires tombstone markers
- **Resize Overhead**: Must rehash all elements when resizing

---

## 🎯 When to Use

### ✅ Choose Open Addressing When

- **Memory Efficiency**: Want to minimize memory usage
- **Cache Performance**: Need good memory locality
- **Predictable Size**: Know approximate number of elements
- **Simple Data Types**: Working with primitive types

### ❌ Avoid Open Addressing When

- **High Load Factors**: Need to store many elements
- **Frequent Deletions**: Many delete operations
- **Unknown Size**: Unpredictable number of elements
- **Complex Keys**: Large or complex key types

---

## 🚀 Next Steps

After mastering open addressing, explore:

- **[Separate Chaining](chaining.md)**: Alternative collision resolution
- **[Cuckoo Hashing](cuckoo-hashing.md)**: Guaranteed O(1) worst-case lookup
- **[Consistent Hashing](consistent-hashing.md)**: For distributed systems

---

Open addressing provides excellent performance and memory efficiency when used appropriately! 🎯
