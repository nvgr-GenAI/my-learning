# Binary Search Trees (BST) 🔍

## Introduction

Binary Search Trees are binary trees with the ordering property: for every node, all values in the left subtree are less than the node's value, and all values in the right subtree are greater.

=== "Overview"
    **Core Concept**:
    
    - Binary tree with an ordering property that enables efficient search, insertion, and deletion
    - For every node: all values in left subtree < node.value < all values in right subtree
    
    **When to Use**:
    
    - When you need ordered data with fast search, insert, and delete operations
    - When you need to perform in-order traversal to get sorted data
    - For implementing sets and maps with ordered keys
    
    **Time Complexity**:
    
    - Average case: O(log n) for search, insert, delete
    - Worst case: O(n) if tree becomes unbalanced
    
    **Real-World Applications**:
    
    - Database indexing
    - Symbol tables in compilers
    - Priority queues
    - Implementing sets and maps

=== "BST Properties"
    **Key Invariant**:
    
    - **Left subtree**: All values < node.value
    - **Right subtree**: All values > node.value
    - **No duplicates**: Typically, duplicates are not allowed (or handled specially)
- **Right subtree**: All values > node.val
- **In-order traversal**: Yields sorted sequence
- **No duplicates**: Traditional BSTs don't allow duplicate values

### Performance Characteristics
- **Average case**: O(log n) for search, insert, delete
- **Worst case**: O(n) when tree becomes skewed
- **Best case**: O(1) for root operations

## 🔧 Implementation

### Basic BST Class

```python
class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        """Insert value into BST"""
        self.root = self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, node, val):
        if not node:
            return TreeNode(val)
        
        if val < node.val:
            node.left = self._insert_recursive(node.left, val)
        elif val > node.val:
            node.right = self._insert_recursive(node.right, val)
        
        return node
    
    def search(self, val):
        """Search for value in BST"""
        return self._search_recursive(self.root, val)
    
    def _search_recursive(self, node, val):
        if not node or node.val == val:
            return node
        
        if val < node.val:
            return self._search_recursive(node.left, val)
        return self._search_recursive(node.right, val)
    
    def delete(self, val):
        """Delete value from BST"""
        self.root = self._delete_recursive(self.root, val)
    
    def _delete_recursive(self, node, val):
        if not node:
            return node
        
        if val < node.val:
            node.left = self._delete_recursive(node.left, val)
        elif val > node.val:
            node.right = self._delete_recursive(node.right, val)
        else:
            # Node to be deleted found
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            
            # Node with two children
            successor = self._find_min(node.right)
            node.val = successor.val
            node.right = self._delete_recursive(node.right, successor.val)
        
        return node
    
    def _find_min(self, node):
        """Find minimum value in subtree"""
        while node.left:
            node = node.left
        return node
    
    def _find_max(self, node):
        """Find maximum value in subtree"""
        while node.right:
            node = node.right
        return node
```

### Iterative Implementations

```python
def search_iterative(root, val):
    """Iterative search"""
    while root and root.val != val:
        if val < root.val:
            root = root.left
        else:
            root = root.right
    return root

def insert_iterative(root, val):
    """Iterative insert"""
    if not root:
        return TreeNode(val)
    
    current = root
    while True:
        if val < current.val:
            if not current.left:
                current.left = TreeNode(val)
                break
            current = current.left
        elif val > current.val:
            if not current.right:
                current.right = TreeNode(val)
                break
            current = current.right
        else:
            break  # Duplicate value
    
    return root
```

## 🎨 Common Operations

### Range Queries
```python
def range_sum_bst(root, low, high):
    """Sum of values in range [low, high]"""
    if not root:
        return 0
    
    total = 0
    if low <= root.val <= high:
        total += root.val
    
    if root.val > low:
        total += range_sum_bst(root.left, low, high)
    if root.val < high:
        total += range_sum_bst(root.right, low, high)
    
    return total

def range_search(root, low, high):
    """Find all values in range [low, high]"""
    result = []
    
    def inorder(node):
        if not node:
            return
        
        if node.val > low:
            inorder(node.left)
        
        if low <= node.val <= high:
            result.append(node.val)
        
        if node.val < high:
            inorder(node.right)
    
    inorder(root)
    return result
```

### Validation
```python
def is_valid_bst(root):
    """Check if tree is valid BST"""
    def validate(node, min_val, max_val):
        if not node:
            return True
        
        if node.val <= min_val or node.val >= max_val:
            return False
        
        return (validate(node.left, min_val, node.val) and 
                validate(node.right, node.val, max_val))
    
    return validate(root, float('-inf'), float('inf'))

def is_valid_bst_inorder(root):
    """Validate using inorder traversal"""
    def inorder(node):
        if not node:
            return []
        return inorder(node.left) + [node.val] + inorder(node.right)
    
    values = inorder(root)
    return all(values[i] < values[i+1] for i in range(len(values)-1))
```

### Successor/Predecessor
```python
def inorder_successor(root, p):
    """Find inorder successor of node p"""
    successor = None
    
    while root:
        if p.val < root.val:
            successor = root
            root = root.left
        else:
            root = root.right
    
    return successor

def inorder_predecessor(root, p):
    """Find inorder predecessor of node p"""
    predecessor = None
    
    while root:
        if p.val > root.val:
            predecessor = root
            root = root.right
        else:
            root = root.left
    
    return predecessor
```

## 🚀 Advanced Techniques

### BST from Sorted Array
```python
def sorted_array_to_bst(nums):
    """Convert sorted array to balanced BST"""
    if not nums:
        return None
    
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    
    root.left = sorted_array_to_bst(nums[:mid])
    root.right = sorted_array_to_bst(nums[mid+1:])
    
    return root
```

### BST Iterator
```python
class BSTIterator:
    """Iterator for BST inorder traversal"""
    
    def __init__(self, root):
        self.stack = []
        self._push_left(root)
    
    def _push_left(self, node):
        while node:
            self.stack.append(node)
            node = node.left
    
    def next(self):
        node = self.stack.pop()
        self._push_left(node.right)
        return node.val
    
    def hasNext(self):
        return len(self.stack) > 0
```

### Kth Smallest/Largest
```python
def kth_smallest(root, k):
    """Find kth smallest element"""
    def inorder(node):
        if not node:
            return []
        return inorder(node.left) + [node.val] + inorder(node.right)
    
    return inorder(root)[k-1]

def kth_smallest_optimized(root, k):
    """Find kth smallest with early termination"""
    def inorder(node):
        nonlocal k
        if not node:
            return None
        
        result = inorder(node.left)
        if result is not None:
            return result
        
        k -= 1
        if k == 0:
            return node.val
        
        return inorder(node.right)
    
    return inorder(root)
```

## 📊 Complexity Analysis

| **Operation** | **Average** | **Worst** | **Best** | **Space** |
|---------------|-------------|-----------|----------|-----------|
| **Search** | O(log n) | O(n) | O(1) | O(1) |
| **Insert** | O(log n) | O(n) | O(1) | O(1) |
| **Delete** | O(log n) | O(n) | O(1) | O(1) |
| **Min/Max** | O(log n) | O(n) | O(1) | O(1) |
| **Successor** | O(log n) | O(n) | O(1) | O(1) |
| **Range Query** | O(log n + k) | O(n) | O(k) | O(1) |

## 🎯 Common Patterns

### 1. **Inorder Traversal**
- Gives sorted order
- Useful for validation and range queries

### 2. **Bounds Checking**
- Maintain min/max bounds while traversing
- Essential for validation

### 3. **Predecessor/Successor**
- Navigate tree structure efficiently
- Key for iterator implementations

### 4. **Reconstruction**
- Build BST from different traversals
- Convert between representations

## 🏆 Practice Problems

### Easy
- [Search in BST](https://leetcode.com/problems/search-in-a-binary-search-tree/)
- [Insert into BST](https://leetcode.com/problems/insert-into-a-binary-search-tree/)
- [Range Sum of BST](https://leetcode.com/problems/range-sum-of-bst/)

### Medium
- [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
- [Kth Smallest Element in BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)
- [Delete Node in BST](https://leetcode.com/problems/delete-node-in-a-bst/)

### Hard
- [Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)
- [Closest Binary Search Tree Value II](https://leetcode.com/problems/closest-binary-search-tree-value-ii/)

## 📚 Key Takeaways

1. **Master the invariant** - BST property is fundamental
2. **Practice both recursive and iterative** - Different use cases
3. **Understand tree traversal** - Inorder gives sorted sequence
4. **Handle edge cases** - Empty trees, single nodes, duplicates
5. **Optimize for specific operations** - Choose right approach for the problem
