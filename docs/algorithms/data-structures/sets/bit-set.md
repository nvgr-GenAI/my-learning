# Bit Set Implementation

## 🔍 Overview

Bit Sets are memory-efficient data structures for storing sets of integers within a fixed range. They use a single bit per possible element, making them extremely space-efficient for dense sets or when the universe of possible elements is known and bounded.

---

## 📊 Characteristics

### Key Properties

- **Memory Efficient**: One bit per element in universe
- **Fixed Universe**: Pre-defined range of possible values
- **Bitwise Operations**: Extremely fast set operations
- **Dense Storage**: Efficient for large dense sets
- **Integer Only**: Limited to non-negative integers

### Memory Layout

```text
Bit Set for universe [0, 15]:
Elements: {1, 3, 5, 7, 10, 14}

Bit Array:
Index:  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
Bits:   0 1 0 1 0 1 0 1 0 0  1  0  0  0  1  0
        └─────────────────────────────────────┘
        16 bits = 2 bytes (vs 6 integers = 24 bytes)
```

---

## ⏱️ Time Complexities

| Operation | Time | Notes |
|-----------|------|-------|
| **Add** | O(1) | Set bit at index |
| **Remove** | O(1) | Clear bit at index |
| **Contains** | O(1) | Check bit at index |
| **Union** | O(n/w) | Bitwise OR on words |
| **Intersection** | O(n/w) | Bitwise AND on words |
| **Size** | O(n/w) | Count set bits |

Where n = universe size, w = word size (32 or 64 bits)

---

## 💻 Implementation

### Basic Bit Set

```python
class BitSet:
    """Bit set implementation for integers in range [0, capacity)."""
    
    def __init__(self, capacity):
        """Initialize bit set with given capacity."""
        self.capacity = capacity
        self.word_size = 64  # Use 64-bit integers
        self.num_words = (capacity + self.word_size - 1) // self.word_size
        self.bits = [0] * self.num_words
    
    def _validate_index(self, index):
        """Validate that index is within bounds."""
        if not (0 <= index < self.capacity):
            raise IndexError(f"Index {index} out of range [0, {self.capacity})")
    
    def _word_index(self, index):
        """Get word index for bit position."""
        return index // self.word_size
    
    def _bit_index(self, index):
        """Get bit position within word."""
        return index % self.word_size
    
    def add(self, index):
        """Add element to the set."""
        self._validate_index(index)
        word_idx = self._word_index(index)
        bit_idx = self._bit_index(index)
        self.bits[word_idx] |= (1 << bit_idx)
    
    def remove(self, index):
        """Remove element from the set."""
        self._validate_index(index)
        if not self.contains(index):
            raise KeyError(f"Element {index} not found in set")
        word_idx = self._word_index(index)
        bit_idx = self._bit_index(index)
        self.bits[word_idx] &= ~(1 << bit_idx)
    
    def discard(self, index):
        """Remove element if present, no error if not found."""
        try:
            self.remove(index)
            return True
        except (KeyError, IndexError):
            return False
    
    def contains(self, index):
        """Check if element exists in the set."""
        if not (0 <= index < self.capacity):
            return False
        word_idx = self._word_index(index)
        bit_idx = self._bit_index(index)
        return bool(self.bits[word_idx] & (1 << bit_idx))
    
    def __contains__(self, index):
        """Support 'in' operator."""
        return self.contains(index)
    
    def clear(self):
        """Remove all elements from the set."""
        self.bits = [0] * self.num_words
    
    def size(self):
        """Count number of elements in the set."""
        count = 0
        for word in self.bits:
            count += bin(word).count('1')
        return count
    
    def __len__(self):
        """Return number of elements."""
        return self.size()
    
    def __bool__(self):
        """Return True if set is not empty."""
        return any(word != 0 for word in self.bits)
    
    def __iter__(self):
        """Iterate over all elements in the set."""
        for i in range(self.capacity):
            if self.contains(i):
                yield i
    
    def __str__(self):
        """String representation."""
        elements = list(self)
        return f"BitSet({{{', '.join(map(str, elements))}}})"
    
    def __repr__(self):
        return self.__str__()

# Usage Example
if __name__ == "__main__":
    # Create bit set for integers 0-99
    bit_set = BitSet(100)
    
    # Add elements
    bit_set.add(5)
    bit_set.add(10)
    bit_set.add(15)
    bit_set.add(25)
    
    print(f"Bit Set: {bit_set}")
    print(f"Size: {len(bit_set)}")
    print(f"Contains 10: {10 in bit_set}")
    print(f"Contains 20: {20 in bit_set}")
    
    # Remove element
    bit_set.remove(10)
    print(f"After removing 10: {bit_set}")
    
    # Iterate
    print("Elements:")
    for element in bit_set:
        print(f"  {element}")
```

---

## 🔧 Set Operations

### Bitwise Set Operations

```python
class BitSetWithOperations(BitSet):
    """Bit set with set operations."""
    
    def _ensure_compatible(self, other):
        """Ensure other bit set has same capacity."""
        if not isinstance(other, BitSet):
            raise TypeError("Operation requires another BitSet")
        if self.capacity != other.capacity:
            raise ValueError("BitSets must have same capacity")
    
    def union(self, other):
        """Return union of two bit sets."""
        self._ensure_compatible(other)
        result = BitSet(self.capacity)
        for i in range(self.num_words):
            result.bits[i] = self.bits[i] | other.bits[i]
        return result
    
    def intersection(self, other):
        """Return intersection of two bit sets."""
        self._ensure_compatible(other)
        result = BitSet(self.capacity)
        for i in range(self.num_words):
            result.bits[i] = self.bits[i] & other.bits[i]
        return result
    
    def difference(self, other):
        """Return difference of two bit sets."""
        self._ensure_compatible(other)
        result = BitSet(self.capacity)
        for i in range(self.num_words):
            result.bits[i] = self.bits[i] & ~other.bits[i]
        return result
    
    def symmetric_difference(self, other):
        """Return symmetric difference of two bit sets."""
        self._ensure_compatible(other)
        result = BitSet(self.capacity)
        for i in range(self.num_words):
            result.bits[i] = self.bits[i] ^ other.bits[i]
        return result
    
    def complement(self):
        """Return complement of the bit set."""
        result = BitSet(self.capacity)
        for i in range(self.num_words):
            result.bits[i] = ~self.bits[i]
        
        # Clear bits beyond capacity
        if self.capacity % self.word_size != 0:
            last_word_bits = self.capacity % self.word_size
            mask = (1 << last_word_bits) - 1
            result.bits[-1] &= mask
        
        return result
    
    def is_subset(self, other):
        """Check if this set is a subset of other."""
        self._ensure_compatible(other)
        for i in range(self.num_words):
            if (self.bits[i] & other.bits[i]) != self.bits[i]:
                return False
        return True
    
    def is_superset(self, other):
        """Check if this set is a superset of other."""
        return other.is_subset(self)
    
    def is_disjoint(self, other):
        """Check if sets have no common elements."""
        self._ensure_compatible(other)
        for i in range(self.num_words):
            if self.bits[i] & other.bits[i] != 0:
                return False
        return True
    
    # In-place operations
    def update(self, other):
        """Update this set with union of self and other."""
        self._ensure_compatible(other)
        for i in range(self.num_words):
            self.bits[i] |= other.bits[i]
    
    def intersection_update(self, other):
        """Update this set with intersection of self and other."""
        self._ensure_compatible(other)
        for i in range(self.num_words):
            self.bits[i] &= other.bits[i]
    
    def difference_update(self, other):
        """Update this set with difference of self and other."""
        self._ensure_compatible(other)
        for i in range(self.num_words):
            self.bits[i] &= ~other.bits[i]
    
    def symmetric_difference_update(self, other):
        """Update this set with symmetric difference of self and other."""
        self._ensure_compatible(other)
        for i in range(self.num_words):
            self.bits[i] ^= other.bits[i]
    
    # Operator overloading
    def __or__(self, other):
        return self.union(other)
    
    def __and__(self, other):
        return self.intersection(other)
    
    def __sub__(self, other):
        return self.difference(other)
    
    def __xor__(self, other):
        return self.symmetric_difference(other)
    
    def __invert__(self):
        return self.complement()
    
    def __le__(self, other):
        return self.is_subset(other)
    
    def __ge__(self, other):
        return self.is_superset(other)
    
    def __ior__(self, other):
        self.update(other)
        return self
    
    def __iand__(self, other):
        self.intersection_update(other)
        return self
    
    def __isub__(self, other):
        self.difference_update(other)
        return self
    
    def __ixor__(self, other):
        self.symmetric_difference_update(other)
        return self

# Usage
set1 = BitSetWithOperations(50)
set1.add(1)
set1.add(3)
set1.add(5)

set2 = BitSetWithOperations(50)
set2.add(3)
set2.add(5)
set2.add(7)

print(f"Set1: {set1}")
print(f"Set2: {set2}")
print(f"Union: {set1 | set2}")
print(f"Intersection: {set1 & set2}")
print(f"Difference: {set1 - set2}")
print(f"Symmetric Difference: {set1 ^ set2}")
print(f"Complement of Set1: {~set1}")
```

---

## 🔍 Advanced Features

### Range Operations

```python
class AdvancedBitSet(BitSetWithOperations):
    """Bit set with range operations."""
    
    def add_range(self, start, end):
        """Add all elements in range [start, end)."""
        for i in range(start, min(end, self.capacity)):
            self.add(i)
    
    def remove_range(self, start, end):
        """Remove all elements in range [start, end)."""
        for i in range(start, min(end, self.capacity)):
            self.discard(i)
    
    def count_range(self, start, end):
        """Count elements in range [start, end)."""
        count = 0
        for i in range(start, min(end, self.capacity)):
            if self.contains(i):
                count += 1
        return count
    
    def next_set_bit(self, start=0):
        """Find next set bit starting from index."""
        for i in range(start, self.capacity):
            if self.contains(i):
                return i
        return -1
    
    def next_clear_bit(self, start=0):
        """Find next clear bit starting from index."""
        for i in range(start, self.capacity):
            if not self.contains(i):
                return i
        return -1
    
    def flip(self, index):
        """Flip bit at index."""
        self._validate_index(index)
        if self.contains(index):
            self.remove(index)
        else:
            self.add(index)
    
    def flip_range(self, start, end):
        """Flip all bits in range [start, end)."""
        for i in range(start, min(end, self.capacity)):
            self.flip(i)

# Usage
advanced_set = AdvancedBitSet(20)
advanced_set.add_range(5, 10)
print(f"After adding range [5, 10): {advanced_set}")

advanced_set.flip_range(7, 12)
print(f"After flipping range [7, 12): {advanced_set}")

print(f"Next set bit from 0: {advanced_set.next_set_bit(0)}")
print(f"Next clear bit from 0: {advanced_set.next_clear_bit(0)}")
```

### Performance Analysis

```python
def analyze_bit_set_performance():
    """Compare bit set with regular set performance."""
    import time
    import random
    
    capacity = 100000
    test_size = 10000
    
    # Generate random integers
    test_data = [random.randint(0, capacity-1) for _ in range(test_size)]
    
    # Test BitSet
    bit_set = BitSet(capacity)
    start_time = time.time()
    for num in test_data:
        bit_set.add(num)
    bit_set_time = time.time() - start_time
    
    # Test regular set
    regular_set = set()
    start_time = time.time()
    for num in test_data:
        regular_set.add(num)
    regular_set_time = time.time() - start_time
    
    # Memory usage (approximate)
    import sys
    bit_set_memory = sys.getsizeof(bit_set.bits) + len(bit_set.bits) * 8
    regular_set_memory = sys.getsizeof(regular_set) + sum(sys.getsizeof(x) for x in regular_set)
    
    print(f"Performance Comparison:")
    print(f"BitSet time: {bit_set_time:.4f}s")
    print(f"Regular set time: {regular_set_time:.4f}s")
    print(f"BitSet memory: {bit_set_memory} bytes")
    print(f"Regular set memory: {regular_set_memory} bytes")
    print(f"Memory savings: {((regular_set_memory - bit_set_memory) / regular_set_memory * 100):.1f}%")

# Uncomment to run performance analysis
# analyze_bit_set_performance()
```

---

## ✅ Advantages

- **Memory Efficient**: Uses only 1 bit per possible element
- **Fast Operations**: Bitwise operations are extremely fast
- **Cache Friendly**: Compact representation improves cache performance
- **Parallel Operations**: Set operations work on multiple bits simultaneously
- **Predictable Performance**: All operations have constant time complexity

## ❌ Disadvantages

- **Fixed Universe**: Must know range of possible values in advance
- **Integer Only**: Limited to non-negative integers
- **Sparse Sets**: Inefficient for sparse sets with large universe
- **No Dynamic Resizing**: Cannot change capacity after creation
- **Limited Flexibility**: Less flexible than general-purpose sets

---

## 🎯 When to Use

### ✅ Choose Bit Set When

- **Known integer range**: Working with integers in fixed range
- **Memory constrained**: Need memory-efficient storage
- **Dense sets**: Large percentage of universe is in the set
- **Fast set operations**: Need efficient union/intersection operations
- **Performance critical**: Bitwise operations provide maximum speed

### ❌ Avoid Bit Set When

- **Unknown range**: Don't know bounds of possible values
- **Sparse sets**: Very few elements relative to universe size
- **Non-integer elements**: Need to store strings, objects, etc.
- **Dynamic universe**: Range of values changes over time
- **General-purpose use**: Need flexibility of general sets

---

## 🚀 Next Steps

After mastering bit sets, explore:

- **[Hash Set](hash-set.md)**: General-purpose unordered sets
- **[Tree Set](tree-set.md)**: Ordered sets with range queries
- **[Bloom Filters](../advanced/bloom-filters.md)**: Probabilistic set membership

---

Bit sets are the most memory-efficient way to represent sets of integers and provide blazing-fast set operations! 🎯
