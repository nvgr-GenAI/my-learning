# Tree Traversals 🌲🚶

## 🎯 Overview

Tree traversal is the process of visiting all nodes in a tree data structure. There are multiple ways to traverse a tree, each serving different purposes and applications.

## 📋 Types of Traversals

### Depth-First Search (DFS)
- **Preorder**: Root → Left → Right
- **Inorder**: Left → Root → Right  
- **Postorder**: Left → Right → Root

### Breadth-First Search (BFS)
- **Level Order**: Visit nodes level by level

## 🔧 Implementation

### DFS Traversals (Recursive)

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def preorder_recursive(root):
    """Preorder: Root → Left → Right"""
    if not root:
        return []
    
    result = [root.val]
    result.extend(preorder_recursive(root.left))
    result.extend(preorder_recursive(root.right))
    return result

def inorder_recursive(root):
    """Inorder: Left → Root → Right"""
    if not root:
        return []
    
    result = []
    result.extend(inorder_recursive(root.left))
    result.append(root.val)
    result.extend(inorder_recursive(root.right))
    return result

def postorder_recursive(root):
    """Postorder: Left → Right → Root"""
    if not root:
        return []
    
    result = []
    result.extend(postorder_recursive(root.left))
    result.extend(postorder_recursive(root.right))
    result.append(root.val)
    return result
```

### DFS Traversals (Iterative)

```python
def preorder_iterative(root):
    """Iterative preorder using stack"""
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        # Push right first, then left (stack is LIFO)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result

def inorder_iterative(root):
    """Iterative inorder using stack"""
    result = []
    stack = []
    current = root
    
    while stack or current:
        # Go to leftmost node
        while current:
            stack.append(current)
            current = current.left
        
        # Process current node
        current = stack.pop()
        result.append(current.val)
        
        # Move to right subtree
        current = current.right
    
    return result

def postorder_iterative(root):
    """Iterative postorder using two stacks"""
    if not root:
        return []
    
    result = []
    stack1 = [root]
    stack2 = []
    
    # Fill stack2 with reverse postorder
    while stack1:
        node = stack1.pop()
        stack2.append(node)
        
        if node.left:
            stack1.append(node.left)
        if node.right:
            stack1.append(node.right)
    
    # Pop from stack2 to get postorder
    while stack2:
        result.append(stack2.pop().val)
    
    return result

def postorder_iterative_one_stack(root):
    """Iterative postorder using one stack"""
    if not root:
        return []
    
    result = []
    stack = []
    last_visited = None
    current = root
    
    while stack or current:
        if current:
            stack.append(current)
            current = current.left
        else:
            peek = stack[-1]
            # If right child exists and hasn't been visited
            if peek.right and last_visited != peek.right:
                current = peek.right
            else:
                result.append(peek.val)
                last_visited = stack.pop()
    
    return result
```

### BFS Traversal (Level Order)

```python
from collections import deque

def level_order(root):
    """Level order traversal using queue"""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        result.append(node.val)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return result

def level_order_by_levels(root):
    """Level order with each level as separate list"""
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

## 🎨 Advanced Traversals

### Morris Traversals (O(1) Space)

```python
def morris_inorder(root):
    """Inorder traversal without recursion or stack"""
    result = []
    current = root
    
    while current:
        if not current.left:
            # No left child, process current and go right
            result.append(current.val)
            current = current.right
        else:
            # Find inorder predecessor
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right
            
            if not predecessor.right:
                # First time visiting, create thread
                predecessor.right = current
                current = current.left
            else:
                # Second time visiting, remove thread and process
                predecessor.right = None
                result.append(current.val)
                current = current.right
    
    return result

def morris_preorder(root):
    """Preorder traversal without recursion or stack"""
    result = []
    current = root
    
    while current:
        if not current.left:
            result.append(current.val)
            current = current.right
        else:
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right
            
            if not predecessor.right:
                predecessor.right = current
                result.append(current.val)  # Process before going left
                current = current.left
            else:
                predecessor.right = None
                current = current.right
    
    return result
```

### Vertical Order Traversal

```python
def vertical_order(root):
    """Traverse tree in vertical order"""
    if not root:
        return []
    
    from collections import defaultdict, deque
    
    # Map from column to list of (row, value) pairs
    column_map = defaultdict(list)
    
    # Queue stores (node, row, column)
    queue = deque([(root, 0, 0)])
    
    while queue:
        node, row, col = queue.popleft()
        column_map[col].append((row, node.val))
        
        if node.left:
            queue.append((node.left, row + 1, col - 1))
        if node.right:
            queue.append((node.right, row + 1, col + 1))
    
    # Sort columns and within each column sort by row, then by value
    result = []
    for col in sorted(column_map.keys()):
        column_map[col].sort()  # Sort by row, then by value
        result.append([val for row, val in column_map[col]])
    
    return result
```

### Boundary Traversal

```python
def boundary_traversal(root):
    """Traverse tree boundary: left boundary + leaves + right boundary"""
    if not root:
        return []
    
    def is_leaf(node):
        return not node.left and not node.right
    
    def left_boundary(node):
        """Left boundary excluding leaves"""
        path = []
        while node:
            if not is_leaf(node):
                path.append(node.val)
            if node.left:
                node = node.left
            else:
                node = node.right
        return path
    
    def right_boundary(node):
        """Right boundary excluding leaves"""
        path = []
        while node:
            if not is_leaf(node):
                path.append(node.val)
            if node.right:
                node = node.right
            else:
                node = node.left
        return path[::-1]  # Reverse for correct order
    
    def leaves(node):
        """All leaf nodes"""
        if not node:
            return []
        if is_leaf(node):
            return [node.val]
        return leaves(node.left) + leaves(node.right)
    
    if is_leaf(root):
        return [root.val]
    
    result = [root.val]
    result.extend(left_boundary(root.left))
    result.extend(leaves(root))
    result.extend(right_boundary(root.right))
    
    return result
```

## 📊 Complexity Analysis

| **Traversal** | **Time** | **Space (Recursive)** | **Space (Iterative)** | **Space (Morris)** |
|---------------|----------|----------------------|----------------------|-------------------|
| **Preorder** | O(n) | O(h) | O(h) | O(1) |
| **Inorder** | O(n) | O(h) | O(h) | O(1) |
| **Postorder** | O(n) | O(h) | O(h) | - |
| **Level Order** | O(n) | O(w) | O(w) | - |

Where:
- n = number of nodes
- h = height of tree  
- w = maximum width of tree

## 🎯 Applications

### Inorder Traversal
- **Binary Search Trees**: Gets sorted sequence
- **Expression trees**: Infix notation
- **Validation**: Check BST property

### Preorder Traversal
- **Tree construction**: Serialize/deserialize
- **File systems**: Directory traversal
- **Expression trees**: Prefix notation

### Postorder Traversal
- **Tree deletion**: Delete children before parent
- **Expression evaluation**: Postfix notation
- **Directory size**: Calculate folder sizes

### Level Order Traversal
- **Tree printing**: Print tree level by level
- **Shortest path**: In unweighted trees
- **Tree width**: Find maximum width

## 🚀 Advanced Patterns

### Tree Iterator

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

### Tree Path Tracking

```python
def find_path(root, target):
    """Find path from root to target node"""
    def dfs(node, path):
        if not node:
            return False
        
        path.append(node.val)
        
        if node.val == target:
            return True
        
        if (dfs(node.left, path) or dfs(node.right, path)):
            return True
        
        path.pop()  # Backtrack
        return False
    
    path = []
    dfs(root, path)
    return path
```

## 🏆 Practice Problems

### Easy
- [Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)
- [Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)
- [Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)

### Medium
- [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- [Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)
- [Vertical Order Traversal](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/)

### Hard
- [Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/) (O(1) space)
- [Recover Binary Search Tree](https://leetcode.com/problems/recover-binary-search-tree/)

## 📚 Key Takeaways

1. **Choose the right traversal** for your use case
2. **Understand recursion vs iteration** trade-offs
3. **Master Morris traversals** for O(1) space solutions
4. **Practice iterative implementations** for interviews
5. **Use level order** for level-based problems
