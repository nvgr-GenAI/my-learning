# Tree Set Implementation

## 🔍 Overview

Tree Sets are ordered collections that maintain elements in sorted order using a balanced binary search tree as the underlying data structure. They provide guaranteed O(log n) performance for all operations while maintaining the natural ordering of elements.

---

## 📊 Characteristics

### Key Properties

- **Sorted Order**: Elements maintained in natural or custom order
- **Unique Elements**: No duplicates allowed
- **Balanced Tree**: Self-balancing for optimal performance
- **Logarithmic Operations**: O(log n) for all operations
- **Range Queries**: Efficient range and predecessor/successor operations

### Memory Layout

```text
Tree Set Structure (BST):
        [5]
       /   \
    [3]     [8]
   /  \    /  \
 [1]  [4][6]  [9]
              /
            [7]
```

---

## ⏱️ Time Complexities

| Operation | Average Case | Worst Case | Notes |
|-----------|--------------|------------|-------|
| **Add** | O(log n) | O(log n) | Balanced tree maintains height |
| **Remove** | O(log n) | O(log n) | May need rebalancing |
| **Contains** | O(log n) | O(log n) | Binary search in tree |
| **Min/Max** | O(log n) | O(log n) | Leftmost/rightmost nodes |
| **Range Query** | O(log n + k) | O(log n + k) | k = number of results |

---

## 💻 Implementation

### Basic Tree Set with AVL Tree

```python
class AVLNode:
    """Node for AVL tree implementation."""
    
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1

class TreeSet:
    """Tree set implementation using AVL tree."""
    
    def __init__(self, compare_func=None):
        """Initialize tree set with optional comparison function."""
        self.root = None
        self.size = 0
        self.compare = compare_func or (lambda a, b: (a > b) - (a < b))
    
    def _height(self, node):
        """Get height of node."""
        return node.height if node else 0
    
    def _balance_factor(self, node):
        """Calculate balance factor of node."""
        return self._height(node.left) - self._height(node.right)
    
    def _update_height(self, node):
        """Update height of node."""
        if node:
            node.height = 1 + max(self._height(node.left), self._height(node.right))
    
    def _rotate_right(self, y):
        """Right rotation for AVL balancing."""
        x = y.left
        T2 = x.right
        
        # Perform rotation
        x.right = y
        y.left = T2
        
        # Update heights
        self._update_height(y)
        self._update_height(x)
        
        return x
    
    def _rotate_left(self, x):
        """Left rotation for AVL balancing."""
        y = x.right
        T2 = y.left
        
        # Perform rotation
        y.left = x
        x.right = T2
        
        # Update heights
        self._update_height(x)
        self._update_height(y)
        
        return y
    
    def _insert(self, node, value):
        """Insert value into AVL tree."""
        # Standard BST insertion
        if not node:
            self.size += 1
            return AVLNode(value)
        
        cmp = self.compare(value, node.value)
        if cmp < 0:
            node.left = self._insert(node.left, value)
        elif cmp > 0:
            node.right = self._insert(node.right, value)
        else:
            # Duplicate value, don't insert
            return node
        
        # Update height
        self._update_height(node)
        
        # Get balance factor
        balance = self._balance_factor(node)
        
        # Left Left Case
        if balance > 1 and self.compare(value, node.left.value) < 0:
            return self._rotate_right(node)
        
        # Right Right Case
        if balance < -1 and self.compare(value, node.right.value) > 0:
            return self._rotate_left(node)
        
        # Left Right Case
        if balance > 1 and self.compare(value, node.left.value) > 0:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        
        # Right Left Case
        if balance < -1 and self.compare(value, node.right.value) < 0:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        
        return node
    
    def _find_min(self, node):
        """Find minimum value node."""
        while node.left:
            node = node.left
        return node
    
    def _delete(self, node, value):
        """Delete value from AVL tree."""
        if not node:
            return node
        
        cmp = self.compare(value, node.value)
        if cmp < 0:
            node.left = self._delete(node.left, value)
        elif cmp > 0:
            node.right = self._delete(node.right, value)
        else:
            # Node to be deleted found
            self.size -= 1
            
            # Node with only one child or no child
            if not node.left or not node.right:
                temp = node.left if node.left else node.right
                
                # No child case
                if not temp:
                    temp = node
                    node = None
                else:
                    # One child case
                    node = temp
            else:
                # Node with two children
                temp = self._find_min(node.right)
                node.value = temp.value
                node.right = self._delete(node.right, temp.value)
                self.size += 1  # Compensate for decrement in recursive call
        
        # If tree has only one node
        if not node:
            return node
        
        # Update height
        self._update_height(node)
        
        # Get balance factor
        balance = self._balance_factor(node)
        
        # Left Left Case
        if balance > 1 and self._balance_factor(node.left) >= 0:
            return self._rotate_right(node)
        
        # Left Right Case
        if balance > 1 and self._balance_factor(node.left) < 0:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        
        # Right Right Case
        if balance < -1 and self._balance_factor(node.right) <= 0:
            return self._rotate_left(node)
        
        # Right Left Case
        if balance < -1 and self._balance_factor(node.right) > 0:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        
        return node
    
    def _search(self, node, value):
        """Search for value in tree."""
        if not node:
            return False
        
        cmp = self.compare(value, node.value)
        if cmp == 0:
            return True
        elif cmp < 0:
            return self._search(node.left, value)
        else:
            return self._search(node.right, value)
    
    def add(self, value):
        """Add value to the tree set."""
        old_size = self.size
        self.root = self._insert(self.root, value)
        return self.size > old_size
    
    def remove(self, value):
        """Remove value from the tree set."""
        if not self.contains(value):
            raise KeyError(f"Value '{value}' not found in set")
        self.root = self._delete(self.root, value)
    
    def discard(self, value):
        """Remove value if present, no error if not found."""
        try:
            self.remove(value)
            return True
        except KeyError:
            return False
    
    def contains(self, value):
        """Check if value exists in the tree set."""
        return self._search(self.root, value)
    
    def __contains__(self, value):
        """Support 'in' operator."""
        return self.contains(value)
    
    def __len__(self):
        """Return number of elements."""
        return self.size
    
    def __bool__(self):
        """Return True if set is not empty."""
        return self.size > 0
    
    def _inorder(self, node, result):
        """Inorder traversal for iteration."""
        if node:
            self._inorder(node.left, result)
            result.append(node.value)
            self._inorder(node.right, result)
    
    def __iter__(self):
        """Iterate over elements in sorted order."""
        result = []
        self._inorder(self.root, result)
        return iter(result)
    
    def __str__(self):
        """String representation."""
        elements = list(self)
        return f"TreeSet({{{', '.join(map(str, elements))}}})"
    
    def __repr__(self):
        return self.__str__()

# Usage Example
if __name__ == "__main__":
    # Create tree set
    tree_set = TreeSet()
    
    # Add elements
    tree_set.add(5)
    tree_set.add(3)
    tree_set.add(8)
    tree_set.add(1)
    tree_set.add(4)
    tree_set.add(6)
    tree_set.add(9)
    
    print(f"Tree Set: {tree_set}")  # Elements in sorted order
    print(f"Size: {len(tree_set)}")
    print(f"Contains 4: {4 in tree_set}")
    
    # Remove element
    tree_set.remove(3)
    print(f"After removing 3: {tree_set}")
    
    # Iterate in sorted order
    print("Elements in order:")
    for element in tree_set:
        print(f"  {element}")
```

---

## 🔧 Advanced Operations

### Range Queries and Navigation

```python
class AdvancedTreeSet(TreeSet):
    """Tree set with advanced navigation operations."""
    
    def _find_min_node(self, node):
        """Find minimum node in subtree."""
        while node and node.left:
            node = node.left
        return node
    
    def _find_max_node(self, node):
        """Find maximum node in subtree."""
        while node and node.right:
            node = node.right
        return node
    
    def min(self):
        """Get minimum element."""
        if not self.root:
            raise ValueError("Set is empty")
        return self._find_min_node(self.root).value
    
    def max(self):
        """Get maximum element."""
        if not self.root:
            raise ValueError("Set is empty")
        return self._find_max_node(self.root).value
    
    def _floor(self, node, value):
        """Find largest element <= value."""
        if not node:
            return None
        
        cmp = self.compare(value, node.value)
        if cmp == 0:
            return node
        elif cmp < 0:
            return self._floor(node.left, value)
        else:
            # value > node.value, so node is a candidate
            right_result = self._floor(node.right, value)
            return right_result if right_result else node
    
    def floor(self, value):
        """Get largest element <= value."""
        result = self._floor(self.root, value)
        if result:
            return result.value
        raise ValueError(f"No element <= {value}")
    
    def _ceiling(self, node, value):
        """Find smallest element >= value."""
        if not node:
            return None
        
        cmp = self.compare(value, node.value)
        if cmp == 0:
            return node
        elif cmp > 0:
            return self._ceiling(node.right, value)
        else:
            # value < node.value, so node is a candidate
            left_result = self._ceiling(node.left, value)
            return left_result if left_result else node
    
    def ceiling(self, value):
        """Get smallest element >= value."""
        result = self._ceiling(self.root, value)
        if result:
            return result.value
        raise ValueError(f"No element >= {value}")
    
    def _range_query(self, node, low, high, result):
        """Find all elements in range [low, high]."""
        if not node:
            return
        
        if self.compare(low, node.value) <= 0 and self.compare(node.value, high) <= 0:
            result.append(node.value)
        
        if self.compare(low, node.value) < 0:
            self._range_query(node.left, low, high, result)
        
        if self.compare(high, node.value) > 0:
            self._range_query(node.right, low, high, result)
    
    def range(self, low, high):
        """Get all elements in range [low, high]."""
        result = []
        self._range_query(self.root, low, high, result)
        return sorted(result, key=lambda x: x)
    
    def subset(self, from_element, to_element):
        """Get subset of elements in range."""
        return self.range(from_element, to_element)

# Usage
advanced_set = AdvancedTreeSet()
for i in [5, 3, 8, 1, 4, 6, 9, 7]:
    advanced_set.add(i)

print(f"Min: {advanced_set.min()}")
print(f"Max: {advanced_set.max()}")
print(f"Floor of 5.5: {advanced_set.floor(5.5)}")
print(f"Ceiling of 5.5: {advanced_set.ceiling(5.5)}")
print(f"Range [3, 7]: {advanced_set.range(3, 7)}")
```

---

## 🎯 Custom Comparators

```python
# Custom comparator for strings (case-insensitive)
def case_insensitive_compare(a, b):
    a_lower = a.lower()
    b_lower = b.lower()
    return (a_lower > b_lower) - (a_lower < b_lower)

string_set = TreeSet(case_insensitive_compare)
string_set.add("apple")
string_set.add("Banana")
string_set.add("cherry")
string_set.add("APPLE")  # Won't be added (case-insensitive)

print(f"String set: {string_set}")

# Custom comparator for objects
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __str__(self):
        return f"{self.name}({self.age})"
    
    def __repr__(self):
        return self.__str__()

def person_age_compare(p1, p2):
    return (p1.age > p2.age) - (p1.age < p2.age)

person_set = TreeSet(person_age_compare)
person_set.add(Person("Alice", 30))
person_set.add(Person("Bob", 25))
person_set.add(Person("Charlie", 35))

print(f"Person set (by age): {person_set}")
```

---

## ✅ Advantages

- **Sorted Order**: Elements always maintained in sorted order
- **Guaranteed Performance**: O(log n) for all operations
- **Range Operations**: Efficient range queries and navigation
- **Balanced Structure**: Self-balancing prevents worst-case scenarios
- **No Hash Collisions**: No dependency on hash function quality

## ❌ Disadvantages

- **Slower than Hash Set**: O(log n) vs O(1) average for hash sets
- **Memory Overhead**: Tree structure requires parent/child pointers
- **Complex Implementation**: More complex than hash-based structures
- **No Random Access**: Can't access elements by index efficiently

---

## 🎯 When to Use

### ✅ Choose Tree Set When

- **Need sorted order**: Want elements in natural or custom order
- **Range queries**: Need to find elements in specific ranges
- **Navigation operations**: Need min, max, floor, ceiling operations
- **Guaranteed performance**: Need predictable O(log n) operations
- **Ordered iteration**: Want to iterate in sorted order

### ❌ Avoid Tree Set When

- **Performance critical**: Hash set provides better average performance
- **Simple membership**: Only need contains/add/remove operations
- **Memory constrained**: Tree structure has higher memory overhead
- **Frequent updates**: High insertion/deletion rate favors hash sets

---

## 🚀 Next Steps

After mastering tree sets, explore:

- **[Hash Set](hash-set.md)**: Unordered set for better average performance
- **[Bit Set](bit-set.md)**: Memory-efficient integer sets
- **[Advanced Trees](../trees/index.md)**: Other tree data structures

---

Tree sets provide powerful ordered collection capabilities and are essential when you need both uniqueness and ordering! 🎯
