# Hash Functions

## 🔍 Overview

Hash functions are the foundation of hash tables, converting keys into array indices. A good hash function distributes keys uniformly across the available slots, minimizing collisions and ensuring optimal performance.

---

## 🎯 Properties of Good Hash Functions

### Essential Characteristics

- **Deterministic**: Same key always produces same hash value
- **Uniform Distribution**: Keys spread evenly across hash space
- **Fast Computation**: Quick to calculate hash values
- **Avalanche Effect**: Small key changes cause large hash changes
- **Low Collision Rate**: Minimize keys mapping to same index

### Mathematical Properties

```text
Good Hash Function Requirements:
1. h(k) = h(k') implies k = k' (ideally)
2. P(h(k₁) = h(k₂)) = 1/m for k₁ ≠ k₂
3. Computation time = O(1)
4. Uniform distribution across [0, m-1]
```

---

## 🔢 Common Hash Functions

### 1. Division Method

```python
def division_hash(key, table_size):
    """Simple modulo hash function."""
    return hash(key) % table_size

# Choose table_size as prime number for better distribution
def division_hash_prime(key, table_size=101):
    """Division method with prime table size."""
    return hash(key) % table_size

# Example
keys = ["apple", "banana", "cherry", "date"]
table_size = 7
for key in keys:
    print(f"{key}: {division_hash(key, table_size)}")
```

### 2. Multiplication Method

```python
import math

def multiplication_hash(key, table_size):
    """Multiplication method using golden ratio."""
    A = (math.sqrt(5) - 1) / 2  # Golden ratio constant
    key_hash = hash(key)
    return int(table_size * ((key_hash * A) % 1))

def multiplication_hash_custom(key, table_size, A=0.6180339887):
    """Multiplication method with custom constant."""
    key_hash = hash(key)
    return int(table_size * ((key_hash * A) % 1))

# Example
keys = ["apple", "banana", "cherry", "date"]
table_size = 8
for key in keys:
    print(f"{key}: {multiplication_hash(key, table_size)}")
```

### 3. Universal Hashing

```python
import random

class UniversalHashFunction:
    """Universal hash function family."""
    
    def __init__(self, table_size):
        self.table_size = table_size
        self.p = self._find_prime(table_size * 2)  # Prime > table_size
        self.a = random.randint(1, self.p - 1)
        self.b = random.randint(0, self.p - 1)
    
    def _find_prime(self, n):
        """Find prime number greater than n."""
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num ** 0.5) + 1):
                if num % i == 0:
                    return False
            return True
        
        while not is_prime(n):
            n += 1
        return n
    
    def hash(self, key):
        """Universal hash function: ((ak + b) mod p) mod m."""
        key_int = hash(key) if isinstance(key, str) else key
        return ((self.a * key_int + self.b) % self.p) % self.table_size

# Example
uhf = UniversalHashFunction(10)
keys = ["apple", "banana", "cherry", "date"]
for key in keys:
    print(f"{key}: {uhf.hash(key)}")
```

---

## 🚀 Advanced Hash Functions

### 1. DJB2 Hash

```python
def djb2_hash(key):
    """DJB2 hash algorithm - excellent distribution."""
    hash_value = 5381
    for char in str(key):
        hash_value = ((hash_value << 5) + hash_value) + ord(char)
    return hash_value & 0xFFFFFFFF  # Keep 32-bit

def djb2_hash_table(key, table_size):
    """DJB2 hash for hash table."""
    return djb2_hash(key) % table_size

# Example
keys = ["apple", "banana", "cherry", "date"]
for key in keys:
    print(f"{key}: {djb2_hash(key):08x}")
```

### 2. FNV Hash (Fowler-Noll-Vo)

```python
def fnv1_hash(key):
    """FNV-1 hash algorithm."""
    FNV_PRIME = 16777619
    FNV_OFFSET_BASIS = 2166136261
    
    hash_value = FNV_OFFSET_BASIS
    for byte in str(key).encode('utf-8'):
        hash_value *= FNV_PRIME
        hash_value ^= byte
        hash_value &= 0xFFFFFFFF  # Keep 32-bit
    
    return hash_value

def fnv1a_hash(key):
    """FNV-1a hash algorithm (better avalanche)."""
    FNV_PRIME = 16777619
    FNV_OFFSET_BASIS = 2166136261
    
    hash_value = FNV_OFFSET_BASIS
    for byte in str(key).encode('utf-8'):
        hash_value ^= byte
        hash_value *= FNV_PRIME
        hash_value &= 0xFFFFFFFF
    
    return hash_value

# Example
key = "hello"
print(f"FNV-1:  {fnv1_hash(key):08x}")
print(f"FNV-1a: {fnv1a_hash(key):08x}")
```

### 3. MurmurHash

```python
def murmur_hash3_32(key, seed=0):
    """Simplified MurmurHash3 32-bit implementation."""
    def rotl32(x, r):
        return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF
    
    def fmix32(h):
        h ^= h >> 16
        h = (h * 0x85ebca6b) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 0xc2b2ae35) & 0xFFFFFFFF
        h ^= h >> 16
        return h
    
    data = str(key).encode('utf-8')
    length = len(data)
    c1 = 0xcc9e2d51
    c2 = 0x1b873593
    r1 = 15
    r2 = 13
    m = 5
    n = 0xe6546b64
    
    hash_value = seed
    
    # Process 4-byte chunks
    for i in range(0, length - 3, 4):
        k = int.from_bytes(data[i:i+4], 'little')
        k = (k * c1) & 0xFFFFFFFF
        k = rotl32(k, r1)
        k = (k * c2) & 0xFFFFFFFF
        
        hash_value ^= k
        hash_value = rotl32(hash_value, r2)
        hash_value = ((hash_value * m) + n) & 0xFFFFFFFF
    
    # Handle remaining bytes
    remaining = length % 4
    if remaining:
        k = 0
        for i in range(remaining):
            k |= data[length - remaining + i] << (8 * i)
        
        k = (k * c1) & 0xFFFFFFFF
        k = rotl32(k, r1)
        k = (k * c2) & 0xFFFFFFFF
        hash_value ^= k
    
    # Finalization
    hash_value ^= length
    return fmix32(hash_value)

# Example
keys = ["apple", "banana", "cherry"]
for key in keys:
    print(f"{key}: {murmur_hash3_32(key):08x}")
```

---

## 🔍 Hash Function Analysis

### Distribution Testing

```python
def test_hash_distribution(hash_func, keys, table_size):
    """Test hash function distribution quality."""
    hash_counts = [0] * table_size
    
    # Count hash values
    for key in keys:
        hash_value = hash_func(key, table_size)
        hash_counts[hash_value] += 1
    
    # Calculate statistics
    total_keys = len(keys)
    expected_per_bucket = total_keys / table_size
    
    # Chi-square test for uniformity
    chi_square = sum((count - expected_per_bucket) ** 2 / expected_per_bucket 
                     for count in hash_counts)
    
    # Load factor variance
    variance = sum((count - expected_per_bucket) ** 2 for count in hash_counts) / table_size
    
    return {
        'distribution': hash_counts,
        'chi_square': chi_square,
        'variance': variance,
        'max_bucket': max(hash_counts),
        'min_bucket': min(hash_counts),
        'empty_buckets': hash_counts.count(0)
    }

# Test different hash functions
def compare_hash_functions():
    """Compare different hash functions."""
    keys = [f"key_{i}" for i in range(1000)]
    table_size = 100
    
    functions = {
        'Division': division_hash,
        'Multiplication': multiplication_hash,
        'DJB2': djb2_hash_table
    }
    
    results = {}
    for name, func in functions.items():
        results[name] = test_hash_distribution(func, keys, table_size)
    
    return results

# Example usage
results = compare_hash_functions()
for name, stats in results.items():
    print(f"{name}: Chi-square={stats['chi_square']:.2f}, "
          f"Variance={stats['variance']:.2f}")
```

### Collision Analysis

```python
def collision_analysis(hash_func, keys, table_size):
    """Analyze collision patterns."""
    hash_table = {}
    collisions = []
    
    for key in keys:
        hash_value = hash_func(key, table_size)
        
        if hash_value in hash_table:
            collisions.append((key, hash_table[hash_value], hash_value))
        else:
            hash_table[hash_value] = key
    
    collision_rate = len(collisions) / len(keys)
    unique_positions = len(hash_table)
    
    return {
        'total_collisions': len(collisions),
        'collision_rate': collision_rate,
        'unique_positions': unique_positions,
        'load_factor': len(keys) / table_size,
        'collision_details': collisions[:10]  # First 10 collisions
    }

# Example
keys = [f"user_{i}" for i in range(200)]
analysis = collision_analysis(djb2_hash_table, keys, 100)
print(f"Collision rate: {analysis['collision_rate']:.2%}")
```

---

## 🎯 Choosing Hash Functions

### For Different Data Types

```python
class CustomHashTable:
    """Hash table with type-specific hash functions."""
    
    def __init__(self, data_type='string'):
        self.data_type = data_type
        self.hash_function = self._select_hash_function(data_type)
    
    def _select_hash_function(self, data_type):
        """Select appropriate hash function for data type."""
        if data_type == 'string':
            return self._string_hash
        elif data_type == 'integer':
            return self._integer_hash
        elif data_type == 'float':
            return self._float_hash
        else:
            return self._generic_hash
    
    def _string_hash(self, key, table_size):
        """Optimized hash for strings."""
        return djb2_hash_table(key, table_size)
    
    def _integer_hash(self, key, table_size):
        """Hash for integers."""
        # Use multiplication method for integers
        A = 0.6180339887
        return int(table_size * ((key * A) % 1))
    
    def _float_hash(self, key, table_size):
        """Hash for floating point numbers."""
        # Convert to integer representation
        import struct
        int_key = struct.unpack('I', struct.pack('f', key))[0]
        return self._integer_hash(int_key, table_size)
    
    def _generic_hash(self, key, table_size):
        """Generic hash using built-in hash."""
        return hash(key) % table_size
```

---

## ✅ Best Practices

### Hash Function Selection

- **Strings**: Use DJB2, FNV, or MurmurHash
- **Integers**: Use multiplication method or universal hashing
- **General Purpose**: Use language built-in hash with modulo
- **Cryptographic**: Use SHA-256 or other cryptographic functions

### Performance Optimization

```python
class OptimizedHashFunction:
    """Optimized hash function with caching."""
    
    def __init__(self, table_size):
        self.table_size = table_size
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def hash(self, key):
        """Hash with caching for repeated keys."""
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        
        hash_value = djb2_hash_table(key, self.table_size)
        self.cache[key] = hash_value
        self.cache_misses += 1
        
        # Limit cache size
        if len(self.cache) > 1000:
            self.cache.clear()
        
        return hash_value
    
    def get_stats(self):
        """Get cache performance statistics."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return {'hit_rate': 0, 'total_calls': 0}
        
        return {
            'hit_rate': self.cache_hits / total,
            'total_calls': total,
            'cache_size': len(self.cache)
        }
```

---

## 🚀 Next Steps

After mastering hash functions, explore:

- **[Collision Resolution](chaining.md)**: Handle hash collisions effectively
- **[Performance Tuning](../fundamentals.md#performance)**: Optimize hash table performance
- **[Cryptographic Hashing](cryptographic-hashing.md)**: Secure hash functions

---

Choosing the right hash function is crucial for optimal hash table performance! 🎯
