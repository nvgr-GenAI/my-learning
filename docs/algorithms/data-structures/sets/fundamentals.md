# Set Fundamentals 🧮

## 🎯 Overview

Understanding different set types, their implementations, and time complexities is crucial for choosing the right data structure for your algorithmic problems.

## 📋 Types of Sets

### Hash Sets

- **Unordered**: Fast O(1) average operations
- **Memory efficient**: Hash table based  
- **No duplicates**: Automatic uniqueness guarantee
- **Use case**: Fast lookups and membership testing

### Tree Sets (Ordered Sets)

- **Sorted order**: Elements maintained in sorted order
- **Logarithmic operations**: O(log n) for most operations
- **Range queries**: Efficient range operations
- **Use case**: When you need sorted iteration

### Bit Sets

- **Memory efficient**: One bit per element
- **Fast operations**: Bitwise operations
- **Fixed size**: Pre-defined universe size
- **Use case**: Integer sets with known bounds

## 🔧 Set Implementations

### Basic Hash Set

```python
class HashSet:
    def __init__(self, initial_capacity=16):
        self.capacity = initial_capacity
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        self.load_factor = 0.75
    
    def _hash(self, key):
        """Hash function for the key"""
        return hash(key) % self.capacity
    
    def _resize(self):
        """Resize when load factor exceeds threshold"""
        old_buckets = self.buckets
        self.capacity *= 2
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        
        for bucket in old_buckets:
            for key in bucket:
                self.add(key)
    
    def add(self, key):
        """Add element to set"""
        if self.size >= self.capacity * self.load_factor:
            self._resize()
        
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        if key not in bucket:
            bucket.append(key)
            self.size += 1
            return True
        return False
    
    def remove(self, key):
        """Remove element from set"""
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        if key in bucket:
            bucket.remove(key)
            self.size -= 1
            return True
        return False
    
    def contains(self, key):
        """Check if element exists in set"""
        bucket_index = self._hash(key)
        return key in self.buckets[bucket_index]
    
    def __len__(self):
        return self.size
    
    def __iter__(self):
        for bucket in self.buckets:
            for key in bucket:
                yield key

# Usage
my_set = HashSet()
my_set.add(1)
my_set.add(2)
my_set.add(3)
print(my_set.contains(2))  # True
print(len(my_set))         # 3
```

### Tree Set (Using BST)

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

class TreeSet:
    def __init__(self):
        self.root = None
        self.size = 0
    
    def add(self, val):
        """Add element to tree set"""
        if not self.root:
            self.root = TreeNode(val)
            self.size += 1
            return True
        
        return self._add_helper(self.root, val)
    
    def _add_helper(self, node, val):
        if val == node.val:
            return False  # Already exists
        elif val < node.val:
            if not node.left:
                node.left = TreeNode(val)
                self.size += 1
                return True
            return self._add_helper(node.left, val)
        else:
            if not node.right:
                node.right = TreeNode(val)
                self.size += 1
                return True
            return self._add_helper(node.right, val)
    
    def contains(self, val):
        """Check if element exists"""
        return self._contains_helper(self.root, val)
    
    def _contains_helper(self, node, val):
        if not node:
            return False
        if val == node.val:
            return True
        elif val < node.val:
            return self._contains_helper(node.left, val)
        else:
            return self._contains_helper(node.right, val)
    
    def remove(self, val):
        """Remove element from tree set"""
        self.root, removed = self._remove_helper(self.root, val)
        if removed:
            self.size -= 1
        return removed
    
    def _remove_helper(self, node, val):
        if not node:
            return None, False
        
        if val < node.val:
            node.left, removed = self._remove_helper(node.left, val)
            return node, removed
        elif val > node.val:
            node.right, removed = self._remove_helper(node.right, val)
            return node, removed
        else:
            # Node to remove found
            if not node.left:
                return node.right, True
            elif not node.right:
                return node.left, True
            else:
                # Node has two children
                successor = self._find_min(node.right)
                node.val = successor.val
                node.right, _ = self._remove_helper(node.right, successor.val)
                return node, True
    
    def _find_min(self, node):
        while node.left:
            node = node.left
        return node
    
    def inorder(self):
        """Return sorted list of elements"""
        result = []
        self._inorder_helper(self.root, result)
        return result
    
    def _inorder_helper(self, node, result):
        if node:
            self._inorder_helper(node.left, result)
            result.append(node.val)
            self._inorder_helper(node.right, result)
    
    def __len__(self):
        return self.size

# Usage
tree_set = TreeSet()
tree_set.add(5)
tree_set.add(3)
tree_set.add(7)
tree_set.add(1)
print(tree_set.inorder())  # [1, 3, 5, 7] - sorted order
```

### Bit Set Implementation

```python
class BitSet:
    def __init__(self, size):
        self.size = size
        self.bits = [0] * ((size + 31) // 32)  # 32-bit integers
    
    def set_bit(self, index):
        """Set bit at index to 1"""
        if 0 <= index < self.size:
            word_index = index // 32
            bit_index = index % 32
            self.bits[word_index] |= (1 << bit_index)
    
    def clear_bit(self, index):
        """Set bit at index to 0"""
        if 0 <= index < self.size:
            word_index = index // 32
            bit_index = index % 32
            self.bits[word_index] &= ~(1 << bit_index)
    
    def get_bit(self, index):
        """Get bit value at index"""
        if 0 <= index < self.size:
            word_index = index // 32
            bit_index = index % 32
            return (self.bits[word_index] >> bit_index) & 1
        return 0
    
    def flip_bit(self, index):
        """Flip bit at index"""
        if 0 <= index < self.size:
            word_index = index // 32
            bit_index = index % 32
            self.bits[word_index] ^= (1 << bit_index)
    
    def union(self, other):
        """Union with another BitSet"""
        result = BitSet(max(self.size, other.size))
        min_words = min(len(self.bits), len(other.bits))
        
        for i in range(min_words):
            result.bits[i] = self.bits[i] | other.bits[i]
        
        # Copy remaining bits
        if len(self.bits) > min_words:
            result.bits[min_words:len(self.bits)] = self.bits[min_words:]
        elif len(other.bits) > min_words:
            result.bits[min_words:len(other.bits)] = other.bits[min_words:]
        
        return result
    
    def intersection(self, other):
        """Intersection with another BitSet"""
        result = BitSet(min(self.size, other.size))
        min_words = min(len(self.bits), len(other.bits))
        
        for i in range(min_words):
            result.bits[i] = self.bits[i] & other.bits[i]
        
        return result
    
    def difference(self, other):
        """Difference with another BitSet"""
        result = BitSet(self.size)
        min_words = min(len(self.bits), len(other.bits))
        
        for i in range(min_words):
            result.bits[i] = self.bits[i] & ~other.bits[i]
        
        # Copy remaining bits from self
        if len(self.bits) > min_words:
            result.bits[min_words:] = self.bits[min_words:]
        
        return result
    
    def count(self):
        """Count number of set bits"""
        count = 0
        for word in self.bits:
            count += bin(word).count('1')
        return count
    
    def __str__(self):
        return ''.join(str(self.get_bit(i)) for i in range(self.size))

# Usage
bs1 = BitSet(8)
bs1.set_bit(1)
bs1.set_bit(3)
bs1.set_bit(5)

bs2 = BitSet(8)
bs2.set_bit(1)
bs2.set_bit(2)
bs2.set_bit(5)

union_set = bs1.union(bs2)
print(f"Union: {union_set}")  # Union: 01110100
```

## 🚀 Set Operations

### Basic Python Set Operations

```python
# Creating sets
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Union (all elements)
union_result = set1 | set2  # {1, 2, 3, 4, 5, 6}
union_result = set1.union(set2)

# Intersection (common elements)
intersection_result = set1 & set2  # {3, 4}
intersection_result = set1.intersection(set2)

# Difference (elements in first but not second)
difference_result = set1 - set2  # {1, 2}
difference_result = set1.difference(set2)

# Symmetric difference (elements in either but not both)
sym_diff_result = set1 ^ set2  # {1, 2, 5, 6}
sym_diff_result = set1.symmetric_difference(set2)

# Subset and superset checks
is_subset = {1, 2}.issubset(set1)  # True
is_superset = set1.issuperset({1, 2})  # True
is_disjoint = set1.isdisjoint({7, 8})  # True
```

### Union-Find (Disjoint Set)

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x):
        """Find root with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
            
            self.components -= 1
            return True
        return False
    
    def connected(self, x, y):
        """Check if two elements are connected"""
        return self.find(x) == self.find(y)
    
    def get_components(self):
        """Get number of connected components"""
        return self.components

# Usage
uf = UnionFind(5)
uf.union(0, 1)
uf.union(2, 3)
print(uf.connected(0, 1))  # True
print(uf.connected(0, 2))  # False
print(uf.get_components())  # 3
```

## 📊 Complexity Analysis

| **Operation** | **Hash Set** | **Tree Set** | **Bit Set** | **Union-Find** |
|---------------|-------------|-------------|-------------|----------------|
| **Insert** | O(1) avg, O(n) worst | O(log n) | O(1) | O(α(n)) |
| **Delete** | O(1) avg, O(n) worst | O(log n) | O(1) | N/A |
| **Search** | O(1) avg, O(n) worst | O(log n) | O(1) | O(α(n)) |
| **Union** | O(n + m) | O(n + m) | O(k/32) | O(α(n)) |
| **Intersection** | O(min(n,m)) | O(n + m) | O(k/32) | N/A |
| **Space** | O(n) | O(n) | O(k/8) bytes | O(n) |

Where:
- n, m = number of elements in sets
- k = universe size for bit sets  
- α(n) = inverse Ackermann function (practically constant)

## 🎯 When to Use Each Set Type

### Hash Set
✅ **Use when:**
- Need fastest average-case performance
- Order doesn't matter
- Working with general data types

❌ **Avoid when:**
- Need guaranteed worst-case performance
- Need sorted iteration
- Memory is extremely constrained

### Tree Set
✅ **Use when:**
- Need sorted order
- Need range queries
- Want predictable O(log n) performance

❌ **Avoid when:**
- Performance is critical and order isn't needed
- Working with simple integer ranges

### Bit Set
✅ **Use when:**
- Working with integers in known range
- Memory is critical
- Need very fast set operations

❌ **Avoid when:**
- Working with arbitrary data types
- Universe size is unknown or very large

### Union-Find
✅ **Use when:**
- Need to track connected components
- Solving connectivity problems
- Dynamic equivalence relationships

❌ **Avoid when:**
- Need to iterate through set elements
- Need traditional set operations like intersection

## 🚀 Next Steps

Now that you understand set fundamentals, practice with:

1. **[Easy Problems](easy-problems.md)** - Start with basic set operations
2. **[Medium Problems](medium-problems.md)** - Apply sets to solve algorithms  
3. **[Hard Problems](hard-problems.md)** - Master complex set applications
