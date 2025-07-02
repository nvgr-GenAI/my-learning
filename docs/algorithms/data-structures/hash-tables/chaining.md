# Hash Tables: Separate Chaining

## 🔍 Overview

Separate chaining is a collision resolution technique where each slot in the hash table contains a linked list (or chain) of elements that hash to the same index. This approach is simple to implement and handles collisions gracefully.

---

## 📊 Characteristics

### Key Properties

- **Collision Handling**: Multiple elements can exist at the same index
- **Dynamic Size**: Each bucket can grow as needed
- **Memory Overhead**: Requires additional memory for pointers
- **Simple Implementation**: Straightforward to understand and code
- **No Clustering**: Collisions don't affect other slots

### Memory Layout

```text
Hash Table with Chaining:
Index:  0    1    2    3
       [•]  [•]  [ ]  [•]
        ↓    ↓         ↓
     [A,1]→[D,4] [B,2] [C,3]
       ↓
     [NULL]
```

---

## ⏱️ Time Complexities

| Operation | Average Case | Worst Case | Notes |
|-----------|--------------|------------|-------|
| **Insert** | O(1) | O(n) | O(n) when all elements hash to same slot |
| **Search** | O(1) | O(n) | Depends on chain length |
| **Delete** | O(1) | O(n) | Must search through chain |
| **Space** | O(n) | O(n) | Extra space for pointers |

---

## 💻 Implementation

### Basic Chaining Implementation

```python
class HashTableChaining:
    """Hash table with separate chaining collision resolution."""
    
    def __init__(self, initial_capacity=16):
        """Initialize hash table with given capacity."""
        self.capacity = initial_capacity
        self.size = 0
        self.table = [[] for _ in range(self.capacity)]
        self.load_factor_threshold = 0.75
    
    def _hash(self, key):
        """Hash function to compute index."""
        return hash(key) % self.capacity
    
    def _resize(self):
        """Resize table when load factor exceeds threshold."""
        old_table = self.table
        old_capacity = self.capacity
        
        # Double the capacity
        self.capacity *= 2
        self.size = 0
        self.table = [[] for _ in range(self.capacity)]
        
        # Rehash all existing elements
        for bucket in old_table:
            for key, value in bucket:
                self.put(key, value)
    
    def put(self, key, value):
        """Insert or update key-value pair."""
        # Check if resize is needed
        if self.size >= self.capacity * self.load_factor_threshold:
            self._resize()
        
        index = self._hash(key)
        bucket = self.table[index]
        
        # Check if key already exists
        for i, (existing_key, existing_value) in enumerate(bucket):
            if existing_key == key:
                bucket[i] = (key, value)  # Update existing
                return
        
        # Add new key-value pair
        bucket.append((key, value))
        self.size += 1
    
    def get(self, key):
        """Retrieve value for given key."""
        index = self._hash(key)
        bucket = self.table[index]
        
        for existing_key, value in bucket:
            if existing_key == key:
                return value
        
        raise KeyError(f"Key '{key}' not found")
    
    def delete(self, key):
        """Remove key-value pair."""
        index = self._hash(key)
        bucket = self.table[index]
        
        for i, (existing_key, value) in enumerate(bucket):
            if existing_key == key:
                del bucket[i]
                self.size -= 1
                return value
        
        raise KeyError(f"Key '{key}' not found")
    
    def contains(self, key):
        """Check if key exists in hash table."""
        try:
            self.get(key)
            return True
        except KeyError:
            return False
    
    def keys(self):
        """Return all keys."""
        result = []
        for bucket in self.table:
            for key, _ in bucket:
                result.append(key)
        return result
    
    def values(self):
        """Return all values."""
        result = []
        for bucket in self.table:
            for _, value in bucket:
                result.append(value)
        return result
    
    def items(self):
        """Return all key-value pairs."""
        result = []
        for bucket in self.table:
            for key, value in bucket:
                result.append((key, value))
        return result
    
    def load_factor(self):
        """Calculate current load factor."""
        return self.size / self.capacity
    
    def __len__(self):
        """Return number of elements."""
        return self.size
    
    def __str__(self):
        """String representation."""
        items = self.items()
        return f"HashTable({dict(items)})"
    
    def __repr__(self):
        return self.__str__()

# Usage Example
if __name__ == "__main__":
    # Create hash table
    ht = HashTableChaining()
    
    # Insert elements
    ht.put("apple", 5)
    ht.put("banana", 3)
    ht.put("orange", 8)
    ht.put("grape", 12)
    
    print(f"Hash table: {ht}")
    print(f"Size: {len(ht)}")
    print(f"Load factor: {ht.load_factor():.2f}")
    
    # Access elements
    print(f"apple: {ht.get('apple')}")
    print(f"Contains 'banana': {ht.contains('banana')}")
    
    # Update element
    ht.put("apple", 10)
    print(f"Updated apple: {ht.get('apple')}")
    
    # Delete element
    ht.delete("banana")
    print(f"After deleting banana: {ht}")
    
    # List all items
    print(f"Keys: {ht.keys()}")
    print(f"Values: {ht.values()}")
```

---

## 🔧 Advanced Features

### Custom Hash Functions

```python
class CustomHashTable(HashTableChaining):
    """Hash table with custom hash functions."""
    
    def __init__(self, hash_function=None, initial_capacity=16):
        super().__init__(initial_capacity)
        self.hash_function = hash_function or self._default_hash
    
    def _default_hash(self, key):
        """Default hash function."""
        return hash(key)
    
    def _hash(self, key):
        """Use custom hash function."""
        return self.hash_function(key) % self.capacity

# Example with custom hash function
def djb2_hash(key):
    """DJB2 hash algorithm for better distribution."""
    hash_value = 5381
    for char in str(key):
        hash_value = ((hash_value << 5) + hash_value) + ord(char)
    return hash_value & 0xFFFFFFFF

# Usage
custom_ht = CustomHashTable(hash_function=djb2_hash)
```

### Statistics and Analysis

```python
def analyze_hash_table(ht):
    """Analyze hash table performance."""
    chain_lengths = [len(bucket) for bucket in ht.table]
    
    stats = {
        'capacity': ht.capacity,
        'size': ht.size,
        'load_factor': ht.load_factor(),
        'max_chain_length': max(chain_lengths),
        'min_chain_length': min(chain_lengths),
        'avg_chain_length': sum(chain_lengths) / len(chain_lengths),
        'empty_buckets': chain_lengths.count(0),
        'chain_distribution': chain_lengths
    }
    
    return stats

# Usage
stats = analyze_hash_table(ht)
print(f"Analysis: {stats}")
```

---

## ✅ Advantages

- **Simple Implementation**: Easy to understand and code
- **Handles Collisions Well**: No limit on number of collisions
- **Dynamic Growth**: Chains can grow as needed
- **Good Average Performance**: O(1) average time complexity
- **No Clustering**: Collisions are isolated to specific buckets

## ❌ Disadvantages

- **Memory Overhead**: Additional memory for pointers/links
- **Cache Performance**: Poor cache locality due to linked structures
- **Worst-Case Scenarios**: Can degrade to O(n) with poor hash function
- **Complex Memory Management**: Need to handle dynamic allocation

---

## 🎯 When to Use

### ✅ Choose Chaining When

- **Simple Implementation**: Want straightforward collision handling
- **Unknown Data Size**: Don't know maximum number of elements
- **High Load Factors**: Need to handle dense hash tables
- **Frequent Insertions**: Adding elements is common operation

### ❌ Avoid Chaining When

- **Memory Constrained**: Limited memory available
- **Cache Performance Critical**: Need optimal memory locality
- **Predictable Size**: Know exact number of elements in advance
- **Simple Data Types**: Working with integers or simple keys

---

## 🚀 Next Steps

After mastering separate chaining, explore:

- **[Open Addressing](open-addressing.md)**: Alternative collision resolution
- **[Hash Functions](hash-functions.md)**: Learn about different hash algorithms
- **[Performance Optimization](../fundamentals.md#optimization)**: Advanced optimization techniques

---

Chaining provides a robust and flexible approach to hash table implementation, making it an excellent choice for most general-purpose applications! 🎯
