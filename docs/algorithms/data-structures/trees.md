# Trees

## Overview

Trees are hierarchical data structures with a root node and child nodes forming a parent-child relationship.

## Types of Trees

### Binary Tree

A tree where each node has at most two children (left and right).

### Binary Search Tree (BST)

A binary tree where left child < parent < right child.

### AVL Tree

A self-balancing binary search tree.

### Red-Black Tree

A balanced binary search tree with color properties.

## Implementation

### Binary Tree Node

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

### Binary Search Tree Operations

```python
class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        self.root = self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, node, val):
        if not node:
            return TreeNode(val)
        
        if val < node.val:
            node.left = self._insert_recursive(node.left, val)
        else:
            node.right = self._insert_recursive(node.right, val)
        
        return node
    
    def search(self, val):
        return self._search_recursive(self.root, val)
    
    def _search_recursive(self, node, val):
        if not node or node.val == val:
            return node
        
        if val < node.val:
            return self._search_recursive(node.left, val)
        else:
            return self._search_recursive(node.right, val)
```

## Tree Traversals

### Depth-First Search (DFS)

```python
def inorder_traversal(root):
    result = []
    if root:
        result.extend(inorder_traversal(root.left))
        result.append(root.val)
        result.extend(inorder_traversal(root.right))
    return result

def preorder_traversal(root):
    result = []
    if root:
        result.append(root.val)
        result.extend(preorder_traversal(root.left))
        result.extend(preorder_traversal(root.right))
    return result

def postorder_traversal(root):
    result = []
    if root:
        result.extend(postorder_traversal(root.left))
        result.extend(postorder_traversal(root.right))
        result.append(root.val)
    return result
```

### Breadth-First Search (BFS)

```python
from collections import deque

def level_order_traversal(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

## Common Tree Problems

1. **Maximum Depth of Binary Tree**
2. **Validate Binary Search Tree**
3. **Lowest Common Ancestor**
4. **Path Sum**
5. **Serialize and Deserialize Binary Tree**

## Time Complexities

| Operation | BST (Balanced) | BST (Worst) | Binary Tree |
|-----------|----------------|-------------|-------------|
| Search    | O(log n)       | O(n)        | O(n)        |
| Insert    | O(log n)       | O(n)        | O(n)        |
| Delete    | O(log n)       | O(n)        | O(n)        |

## Practice Problems

- [ ] Binary Tree Inorder Traversal
- [ ] Maximum Depth of Binary Tree
- [ ] Same Tree
- [ ] Invert Binary Tree
- [ ] Binary Tree Level Order Traversal
