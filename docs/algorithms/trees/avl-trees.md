# AVL Trees 🌳⚖️

## Introduction

AVL Trees are self-balancing binary search trees where the height difference between left and right subtrees of any node is at most 1. Named after Adelson-Velsky and Landis.

=== "Overview"
    **Core Concept**:
    
    - Self-balancing binary search tree that maintains O(log n) height
    - Named after inventors Adelson-Velsky and Landis (1962)
    - For any node, the height difference between left and right subtrees is at most 1
    
    **When to Use**:
    
    - When you need guaranteed O(log n) operations for search, insert, and delete
    - When the tree will undergo frequent modifications
    - When balanced performance is critical for all operations
    
    **Time Complexity**:
    
    - Search, Insert, Delete: O(log n) guaranteed
    - Space complexity: O(n)
    
    **Real-World Applications**:
    
    - Database indexing requiring strict performance guarantees
    - Memory management systems
    - In-memory caches with predictable performance

=== "Balance Properties"
    **Balance Factor**:
    
    - **Balance Factor (BF)** = height(left) - height(right)
    - Valid values: {-1, 0, 1}
    - BF > 1: Left subtree is too tall (left-heavy)
    - BF < -1: Right subtree is too tall (right-heavy)
- **Valid range**: -1, 0, +1
- **Violation**: |BF| > 1 requires rebalancing

### Rotations
- **Left Rotation**: Fix right-heavy imbalance
- **Right Rotation**: Fix left-heavy imbalance
- **Left-Right Rotation**: Fix left-right imbalance
- **Right-Left Rotation**: Fix right-left imbalance

## 🔧 Implementation

### AVL Node Structure

```python
class AVLNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.height = 1  # Height of node

class AVLTree:
    def __init__(self):
        self.root = None
    
    def get_height(self, node):
        """Get height of node"""
        if not node:
            return 0
        return node.height
    
    def get_balance(self, node):
        """Get balance factor of node"""
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)
    
    def update_height(self, node):
        """Update height of node"""
        if node:
            node.height = 1 + max(self.get_height(node.left), 
                                  self.get_height(node.right))
    
    def right_rotate(self, y):
        """Right rotation around y"""
        x = y.left
        T2 = x.right
        
        # Perform rotation
        x.right = y
        y.left = T2
        
        # Update heights
        self.update_height(y)
        self.update_height(x)
        
        return x
    
    def left_rotate(self, x):
        """Left rotation around x"""
        y = x.right
        T2 = y.left
        
        # Perform rotation
        y.left = x
        x.right = T2
        
        # Update heights
        self.update_height(x)
        self.update_height(y)
        
        return y
    
    def insert(self, val):
        """Insert value and maintain AVL property"""
        self.root = self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, node, val):
        # Step 1: Perform normal BST insertion
        if not node:
            return AVLNode(val)
        
        if val < node.val:
            node.left = self._insert_recursive(node.left, val)
        elif val > node.val:
            node.right = self._insert_recursive(node.right, val)
        else:
            return node  # Duplicate values not allowed
        
        # Step 2: Update height
        self.update_height(node)
        
        # Step 3: Get balance factor
        balance = self.get_balance(node)
        
        # Step 4: Perform rotations if needed
        # Left Left Case
        if balance > 1 and val < node.left.val:
            return self.right_rotate(node)
        
        # Right Right Case
        if balance < -1 and val > node.right.val:
            return self.left_rotate(node)
        
        # Left Right Case
        if balance > 1 and val > node.left.val:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)
        
        # Right Left Case
        if balance < -1 and val < node.right.val:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)
        
        return node
    
    def delete(self, val):
        """Delete value and maintain AVL property"""
        self.root = self._delete_recursive(self.root, val)
    
    def _delete_recursive(self, node, val):
        # Step 1: Perform normal BST deletion
        if not node:
            return node
        
        if val < node.val:
            node.left = self._delete_recursive(node.left, val)
        elif val > node.val:
            node.right = self._delete_recursive(node.right, val)
        else:
            # Node to be deleted found
            if not node.left or not node.right:
                temp = node.left if node.left else node.right
                if not temp:
                    temp = node
                    node = None
                else:
                    node = temp
            else:
                # Node with two children
                temp = self._find_min(node.right)
                node.val = temp.val
                node.right = self._delete_recursive(node.right, temp.val)
        
        if not node:
            return node
        
        # Step 2: Update height
        self.update_height(node)
        
        # Step 3: Get balance factor
        balance = self.get_balance(node)
        
        # Step 4: Perform rotations if needed
        # Left Left Case
        if balance > 1 and self.get_balance(node.left) >= 0:
            return self.right_rotate(node)
        
        # Left Right Case
        if balance > 1 and self.get_balance(node.left) < 0:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)
        
        # Right Right Case
        if balance < -1 and self.get_balance(node.right) <= 0:
            return self.left_rotate(node)
        
        # Right Left Case
        if balance < -1 and self.get_balance(node.right) > 0:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)
        
        return node
    
    def _find_min(self, node):
        """Find minimum value in subtree"""
        while node.left:
            node = node.left
        return node
```

## ⚙️ Rotation Examples

### Single Rotations

```python
def visualize_right_rotation():
    """
    Right Rotation (LL Case):
    
    Before:         After:
        y              x
       / \\            / \\
      x   C    -->    A   y
     / \\                / \\
    A   B              B   C
    """
    pass

def visualize_left_rotation():
    """
    Left Rotation (RR Case):
    
    Before:         After:
      x                y
     / \\             / \\
    A   y    -->     x   C
       / \\         / \\
      B   C        A   B
    """
    pass
```

### Double Rotations

```python
def visualize_left_right_rotation():
    """
    Left-Right Rotation (LR Case):
    
    Step 1: Left rotate on x        Step 2: Right rotate on z
        z                              z                  y
       / \\                           / \\                / \\
      x   D     Left rotate x        y   D   Right rot z  x   z
     / \\       ------------->      / \\   ------------> / \\ / \\
    A   y                          x   C                A  B C  D
       / \\                       / \\
      B   C                     A   B
    """
    pass

def visualize_right_left_rotation():
    """
    Right-Left Rotation (RL Case):
    
    Step 1: Right rotate on z       Step 2: Left rotate on x
      x                               x                    y
     / \\                            / \\                  / \\
    A   z     Right rotate z        A   y    Left rot x   x   z
       / \\   --------------->         / \\  -----------> / \\ / \\
      y   D                          B   z              A  B C  D
     / \\                               / \\
    B   C                             C   D
    """
    pass
```

## 🎨 Common Operations

### Search
```python
def search(self, val):
    """Search for value in AVL tree"""
    return self._search_recursive(self.root, val)

def _search_recursive(self, node, val):
    if not node or node.val == val:
        return node
    
    if val < node.val:
        return self._search_recursive(node.left, val)
    return self._search_recursive(node.right, val)
```

### Tree Validation
```python
def is_balanced(self, node):
    """Check if tree maintains AVL property"""
    if not node:
        return True
    
    balance = self.get_balance(node)
    if abs(balance) > 1:
        return False
    
    return (self.is_balanced(node.left) and 
            self.is_balanced(node.right))

def is_avl_tree(self, node, min_val=float('-inf'), max_val=float('inf')):
    """Check if tree is valid AVL tree"""
    if not node:
        return True
    
    # Check BST property
    if node.val <= min_val or node.val >= max_val:
        return False
    
    # Check balance property
    if abs(self.get_balance(node)) > 1:
        return False
    
    # Check height property
    expected_height = 1 + max(self.get_height(node.left), 
                             self.get_height(node.right))
    if node.height != expected_height:
        return False
    
    return (self.is_avl_tree(node.left, min_val, node.val) and 
            self.is_avl_tree(node.right, node.val, max_val))
```

## 📊 Complexity Analysis

| **Operation** | **Time Complexity** | **Space Complexity** |
|---------------|-------------------|---------------------|
| **Search** | O(log n) | O(log n) |
| **Insert** | O(log n) | O(log n) |
| **Delete** | O(log n) | O(log n) |
| **Min/Max** | O(log n) | O(log n) |
| **Predecessor/Successor** | O(log n) | O(log n) |

### Height Bounds
- **Maximum height**: 1.44 × log₂(n + 2) - 0.328
- **Minimum height**: log₂(n + 1)

## 🚀 Advanced Techniques

### Bulk Operations
```python
def build_from_sorted_array(self, arr):
    """Build AVL tree from sorted array"""
    def build_balanced(start, end):
        if start > end:
            return None
        
        mid = (start + end) // 2
        node = AVLNode(arr[mid])
        
        node.left = build_balanced(start, mid - 1)
        node.right = build_balanced(mid + 1, end)
        
        self.update_height(node)
        return node
    
    self.root = build_balanced(0, len(arr) - 1)
```

### Range Operations
```python
def range_query(self, low, high):
    """Find all values in range [low, high]"""
    result = []
    
    def inorder_range(node):
        if not node:
            return
        
        if low < node.val:
            inorder_range(node.left)
        
        if low <= node.val <= high:
            result.append(node.val)
        
        if node.val < high:
            inorder_range(node.right)
    
    inorder_range(self.root)
    return result
```

## 🆚 AVL vs Other Trees

| **Feature** | **AVL** | **Red-Black** | **Splay** |
|-------------|---------|---------------|-----------|
| **Balance** | Strict | Relaxed | None |
| **Rotations per Insert** | ≤ 2 | ≤ 3 | O(log n) |
| **Rotations per Delete** | O(log n) | ≤ 3 | O(log n) |
| **Search Time** | O(log n) | O(log n) | Amortized O(log n) |
| **Use Case** | Frequent searches | General purpose | Temporal locality |

## 🎯 Practice Problems

### Easy
- [Height of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
- [Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)

### Medium
- [Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)
- [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)

### Hard
- [Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)

## 📚 Applications

### Database Indexing
- **B+ Trees**: Variant used in databases
- **Guaranteed performance**: O(log n) operations
- **Range queries**: Efficient for database operations

### Memory Management
- **Virtual memory**: Page table implementations
- **Allocation algorithms**: Balanced free block trees

### Computational Geometry
- **Range trees**: Multi-dimensional range queries
- **Interval trees**: Overlapping intervals

## 🏆 Key Takeaways

1. **Height guarantee** - Never degrades to O(n)
2. **Rotation mastery** - Four types of rotations
3. **Balance factor** - Core invariant to maintain
4. **More rotations than Red-Black** - Stricter balancing
5. **Best for read-heavy** - Optimal for frequent searches
