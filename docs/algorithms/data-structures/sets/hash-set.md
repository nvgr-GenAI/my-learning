# Hash Set Implementation

## 🔍 Overview

Hash Sets are unordered collections that store unique elements using a hash table as the underlying data structure. They provide excellent average-case performance for membership testing, insertion, and deletion operations.

---

## 📊 Characteristics

### Key Properties

- **Unique Elements**: Automatically prevents duplicates
- **Unordered**: No guaranteed iteration order
- **Dynamic Size**: Grows and shrinks as needed
- **Hash-Based**: Uses hash function for fast access
- **Average O(1)**: Constant time operations on average

### Memory Layout

```text
Hash Set Structure:
Buckets: [0] [1] [2] [3] [4] [5] [6] [7]
          •   •       •   •       •
          ↓   ↓       ↓   ↓       ↓
         {5} {9}     {2} {12}    {6}
                      ↓
                     {10} (collision chain)
```

---

## ⏱️ Time Complexities

| Operation | Average Case | Worst Case | Notes |
|-----------|--------------|------------|-------|
| **Add** | O(1) | O(n) | O(n) when all elements hash to same bucket |
| **Remove** | O(1) | O(n) | Must search through collision chain |
| **Contains** | O(1) | O(n) | Depends on hash function quality |
| **Iteration** | O(n) | O(n) | Must visit all buckets |

---

## 💻 Implementation

### Basic Hash Set

```python
class HashSet:
    """Hash set implementation with separate chaining."""
    
    def __init__(self, initial_capacity=16):
        """Initialize hash set with given capacity."""
        self.capacity = initial_capacity
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        self.load_factor_threshold = 0.75
    
    def _hash(self, element):
        """Hash function to compute bucket index."""
        return hash(element) % self.capacity
    
    def _resize(self):
        """Resize when load factor exceeds threshold."""
        old_buckets = self.buckets
        old_capacity = self.capacity
        
        # Double the capacity
        self.capacity *= 2
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        
        # Rehash all existing elements
        for bucket in old_buckets:
            for element in bucket:
                self.add(element)
    
    def add(self, element):
        """Add element to the set."""
        # Check if resize is needed
        if self.size >= self.capacity * self.load_factor_threshold:
            self._resize()
        
        bucket_index = self._hash(element)
        bucket = self.buckets[bucket_index]
        
        # Check if element already exists
        if element not in bucket:
            bucket.append(element)
            self.size += 1
            return True
        return False
    
    def remove(self, element):
        """Remove element from the set."""
        bucket_index = self._hash(element)
        bucket = self.buckets[bucket_index]
        
        if element in bucket:
            bucket.remove(element)
            self.size -= 1
            return True
        
        raise KeyError(f"Element '{element}' not found in set")
    
    def discard(self, element):
        """Remove element if present, no error if not found."""
        try:
            self.remove(element)
            return True
        except KeyError:
            return False
    
    def contains(self, element):
        """Check if element exists in the set."""
        bucket_index = self._hash(element)
        return element in self.buckets[bucket_index]
    
    def __contains__(self, element):
        """Support 'in' operator."""
        return self.contains(element)
    
    def clear(self):
        """Remove all elements from the set."""
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
    
    def copy(self):
        """Create a shallow copy of the set."""
        new_set = HashSet(self.capacity)
        for element in self:
            new_set.add(element)
        return new_set
    
    def load_factor(self):
        """Calculate current load factor."""
        return self.size / self.capacity
    
    def __len__(self):
        """Return number of elements."""
        return self.size
    
    def __bool__(self):
        """Return True if set is not empty."""
        return self.size > 0
    
    def __iter__(self):
        """Iterate over all elements."""
        for bucket in self.buckets:
            for element in bucket:
                yield element
    
    def __str__(self):
        """String representation."""
        elements = list(self)
        return f"HashSet({{{', '.join(map(str, elements))}}})"
    
    def __repr__(self):
        return self.__str__()

# Usage Example
if __name__ == "__main__":
    # Create hash set
    my_set = HashSet()
    
    # Add elements
    my_set.add(1)
    my_set.add(2)
    my_set.add(3)
    my_set.add(2)  # Duplicate, won't be added
    
    print(f"Set: {my_set}")
    print(f"Size: {len(my_set)}")
    print(f"Contains 2: {2 in my_set}")
    
    # Remove element
    my_set.remove(2)
    print(f"After removing 2: {my_set}")
    
    # Iterate
    print("Elements:")
    for element in my_set:
        print(f"  {element}")
```

---

## 🔧 Set Operations

### Union, Intersection, Difference

```python
class HashSetWithOperations(HashSet):
    """Hash set with set operations."""
    
    def union(self, other):
        """Return union of two sets."""
        result = self.copy()
        for element in other:
            result.add(element)
        return result
    
    def intersection(self, other):
        """Return intersection of two sets."""
        result = HashSet()
        for element in self:
            if element in other:
                result.add(element)
        return result
    
    def difference(self, other):
        """Return difference of two sets (elements in self but not in other)."""
        result = HashSet()
        for element in self:
            if element not in other:
                result.add(element)
        return result
    
    def symmetric_difference(self, other):
        """Return symmetric difference (elements in either set but not both)."""
        result = HashSet()
        
        # Add elements from self that are not in other
        for element in self:
            if element not in other:
                result.add(element)
        
        # Add elements from other that are not in self
        for element in other:
            if element not in self:
                result.add(element)
        
        return result
    
    def is_subset(self, other):
        """Check if this set is a subset of other."""
        for element in self:
            if element not in other:
                return False
        return True
    
    def is_superset(self, other):
        """Check if this set is a superset of other."""
        for element in other:
            if element not in self:
                return False
        return True
    
    def is_disjoint(self, other):
        """Check if sets have no common elements."""
        for element in self:
            if element in other:
                return False
        return True
    
    # Operator overloading
    def __or__(self, other):
        """Union operator |"""
        return self.union(other)
    
    def __and__(self, other):
        """Intersection operator &"""
        return self.intersection(other)
    
    def __sub__(self, other):
        """Difference operator -"""
        return self.difference(other)
    
    def __xor__(self, other):
        """Symmetric difference operator ^"""
        return self.symmetric_difference(other)
    
    def __le__(self, other):
        """Subset operator <="""
        return self.is_subset(other)
    
    def __ge__(self, other):
        """Superset operator >="""
        return self.is_superset(other)

# Usage
set1 = HashSetWithOperations()
set1.add(1)
set1.add(2)
set1.add(3)

set2 = HashSetWithOperations()
set2.add(2)
set2.add(3)
set2.add(4)

print(f"Set1: {set1}")
print(f"Set2: {set2}")
print(f"Union: {set1 | set2}")
print(f"Intersection: {set1 & set2}")
print(f"Difference: {set1 - set2}")
print(f"Symmetric Difference: {set1 ^ set2}")
```

---

## 🔍 Advanced Features

### Custom Hash Functions

```python
class CustomHashSet(HashSet):
    """Hash set with custom hash function."""
    
    def __init__(self, hash_function=None, initial_capacity=16):
        super().__init__(initial_capacity)
        self.custom_hash = hash_function
    
    def _hash(self, element):
        """Use custom hash function if provided."""
        if self.custom_hash:
            return self.custom_hash(element) % self.capacity
        return super()._hash(element)

# Example with custom hash for strings
def string_hash(s):
    """Simple string hash function."""
    hash_value = 0
    for char in s:
        hash_value = (hash_value * 31 + ord(char)) % (2**32)
    return hash_value

string_set = CustomHashSet(hash_function=string_hash)
string_set.add("hello")
string_set.add("world")
```

### Statistics and Analysis

```python
def analyze_hash_set(hash_set):
    """Analyze hash set performance."""
    bucket_lengths = [len(bucket) for bucket in hash_set.buckets]
    
    return {
        'capacity': hash_set.capacity,
        'size': hash_set.size,
        'load_factor': hash_set.load_factor(),
        'max_bucket_length': max(bucket_lengths) if bucket_lengths else 0,
        'min_bucket_length': min(bucket_lengths) if bucket_lengths else 0,
        'avg_bucket_length': sum(bucket_lengths) / len(bucket_lengths),
        'empty_buckets': bucket_lengths.count(0),
        'collision_rate': (hash_set.size - (hash_set.capacity - bucket_lengths.count(0))) / hash_set.size if hash_set.size > 0 else 0
    }

# Usage
my_set = HashSet()
for i in range(100):
    my_set.add(i)

stats = analyze_hash_set(my_set)
print(f"Hash Set Analysis: {stats}")
```

---

## ✅ Advantages

- **Fast Average Performance**: O(1) operations on average
- **Automatic Uniqueness**: No duplicate elements
- **Dynamic Sizing**: Grows as needed
- **Memory Efficient**: No wasted space for unused elements
- **Simple Interface**: Easy to use and understand

## ❌ Disadvantages

- **No Ordering**: Elements not stored in any particular order
- **Hash Function Dependency**: Performance depends on hash quality
- **Worst-Case Performance**: Can degrade to O(n) with poor hashing
- **Memory Overhead**: Hash table structure requires extra memory

---

## 🎯 When to Use

### ✅ Choose Hash Set When

- **Need fast membership testing**: O(1) contains operations
- **Duplicate removal**: Automatically ensures uniqueness
- **Set operations**: Union, intersection, difference operations
- **Caching unique items**: Store items that have been seen
- **Order doesn't matter**: Don't need sorted or insertion order

### ❌ Avoid Hash Set When

- **Need ordering**: Use TreeSet or maintain separate sorted list
- **Memory constrained**: Consider BitSet for integer ranges
- **Predictable performance**: Use TreeSet for guaranteed O(log n)
- **Range queries**: TreeSet provides efficient range operations

---

## 🚀 Next Steps

After mastering hash sets, explore:

- **[Tree Set](tree-set.md)**: Ordered set implementation
- **[Bit Set](bit-set.md)**: Memory-efficient integer sets
- **[Set Applications](../fundamentals.md#applications)**: Real-world use cases

---

Hash sets provide the foundation for many set-based algorithms and are essential for efficient duplicate detection and membership testing! 🎯
