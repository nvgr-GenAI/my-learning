# Binary Trees 🌱

## 🎯 Overview

Binary trees are the foundation of tree data structures, where each node has at most two children, typically called left and right child.

## 📋 Key Concepts

### Basic Properties
- **Node**: Contains data and references to children
- **Root**: Top node of the tree
- **Leaf**: Node with no children
- **Height**: Longest path from root to leaf
- **Depth**: Distance from root to a node

### Tree Types
- **Full Binary Tree**: Every node has 0 or 2 children
- **Complete Binary Tree**: All levels filled except possibly the last
- **Perfect Binary Tree**: All internal nodes have 2 children, all leaves at same level
- **Balanced Binary Tree**: Height difference between subtrees ≤ 1

## 🔧 Implementation

### Node Structure

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

```java
public class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    
    TreeNode() {}
    TreeNode(int val) { this.val = val; }
    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}
```

### Basic Operations

```python
class BinaryTree:
    def __init__(self):
        self.root = None
    
    def insert_level_order(self, arr):
        """Build tree from level-order array"""
        if not arr:
            return None
        
        self.root = TreeNode(arr[0])
        queue = [self.root]
        i = 1
        
        while queue and i < len(arr):
            node = queue.pop(0)
            
            # Left child
            if i < len(arr) and arr[i] is not None:
                node.left = TreeNode(arr[i])
                queue.append(node.left)
            i += 1
            
            # Right child
            if i < len(arr) and arr[i] is not None:
                node.right = TreeNode(arr[i])
                queue.append(node.right)
            i += 1
    
    def height(self, node):
        """Calculate height of tree"""
        if not node:
            return -1
        return 1 + max(self.height(node.left), self.height(node.right))
    
    def size(self, node):
        """Count total nodes"""
        if not node:
            return 0
        return 1 + self.size(node.left) + self.size(node.right)
    
    def is_balanced(self, node):
        """Check if tree is balanced"""
        def check_balance(node):
            if not node:
                return 0, True
            
            left_height, left_balanced = check_balance(node.left)
            right_height, right_balanced = check_balance(node.right)
            
            balanced = (left_balanced and right_balanced and 
                       abs(left_height - right_height) <= 1)
            height = 1 + max(left_height, right_height)
            
            return height, balanced
        
        _, balanced = check_balance(node)
        return balanced
```

## 🎨 Common Patterns

### 1. Tree Construction
```python
def build_tree_from_arrays(preorder, inorder):
    """Build tree from preorder and inorder traversals"""
    if not preorder or not inorder:
        return None
    
    root = TreeNode(preorder[0])
    mid = inorder.index(preorder[0])
    
    root.left = build_tree_from_arrays(preorder[1:mid+1], inorder[:mid])
    root.right = build_tree_from_arrays(preorder[mid+1:], inorder[mid+1:])
    
    return root
```

### 2. Path Problems
```python
def has_path_sum(root, target):
    """Check if tree has root-to-leaf path with given sum"""
    if not root:
        return False
    
    if not root.left and not root.right:
        return root.val == target
    
    target -= root.val
    return (has_path_sum(root.left, target) or 
            has_path_sum(root.right, target))

def all_paths(root):
    """Find all root-to-leaf paths"""
    if not root:
        return []
    
    if not root.left and not root.right:
        return [[root.val]]
    
    paths = []
    for path in all_paths(root.left) + all_paths(root.right):
        paths.append([root.val] + path)
    
    return paths
```

### 3. Tree Comparison
```python
def is_same_tree(p, q):
    """Check if two trees are identical"""
    if not p and not q:
        return True
    if not p or not q:
        return False
    return (p.val == q.val and 
            is_same_tree(p.left, q.left) and 
            is_same_tree(p.right, q.right))

def is_subtree(root, subRoot):
    """Check if subRoot is subtree of root"""
    if not subRoot:
        return True
    if not root:
        return False
    
    return (is_same_tree(root, subRoot) or 
            is_subtree(root.left, subRoot) or 
            is_subtree(root.right, subRoot))
```

## 📊 Complexity Analysis

| **Operation** | **Time** | **Space** | **Notes** |
|---------------|----------|-----------|-----------|
| **Search** | O(n) | O(h) | h = height, worst case O(n) |
| **Insert** | O(n) | O(h) | Level-order insertion |
| **Delete** | O(n) | O(h) | Need to find node first |
| **Traversal** | O(n) | O(h) | Visit all nodes |
| **Height** | O(n) | O(h) | Visit all nodes |

## 🚀 Advanced Techniques

### Serialization/Deserialization
```python
def serialize(root):
    """Serialize tree to string"""
    def dfs(node):
        if not node:
            return "null"
        return f"{node.val},{dfs(node.left)},{dfs(node.right)}"
    
    return dfs(root)

def deserialize(data):
    """Deserialize string to tree"""
    def dfs():
        val = next(vals)
        if val == "null":
            return None
        return TreeNode(int(val), dfs(), dfs())
    
    vals = iter(data.split(','))
    return dfs()
```

### Morris Traversal (O(1) Space)
```python
def morris_inorder(root):
    """Inorder traversal without recursion or stack"""
    result = []
    current = root
    
    while current:
        if not current.left:
            result.append(current.val)
            current = current.right
        else:
            # Find predecessor
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right
            
            if not predecessor.right:
                predecessor.right = current
                current = current.left
            else:
                predecessor.right = None
                result.append(current.val)
                current = current.right
    
    return result
```

## 🎯 Practice Problems

### Easy
- [Same Tree](https://leetcode.com/problems/same-tree/)
- [Maximum Depth](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
- [Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)

### Medium
- [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- [Construct Binary Tree from Preorder and Inorder](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
- [Path Sum II](https://leetcode.com/problems/path-sum-ii/)

### Hard
- [Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)
- [Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)
- [Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/) (O(1) space)

## 📚 Key Takeaways

1. **Master traversals** - Understanding DFS and BFS is crucial
2. **Think recursively** - Many tree problems have elegant recursive solutions
3. **Consider base cases** - Always handle null nodes properly
4. **Use helper functions** - Pass additional parameters for state
5. **Practice serialization** - Important for system design interviews
