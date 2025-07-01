# Trees

Hierarchical data structures that model parent-child relationships, forming the backbone of many algorithms and applications.

## Overview

A **Tree** is a connected acyclic graph consisting of nodes connected by edges. It has a hierarchical structure with a single root node and child nodes forming parent-child relationships.

### Key Terminology

- **Root**: Top node with no parent
- **Parent**: Node with child nodes
- **Child**: Node connected to parent above
- **Leaf**: Node with no children
- **Siblings**: Nodes with the same parent  
- **Depth**: Distance from root to node
- **Height**: Maximum depth of any node
- **Level**: All nodes at same depth
- **Subtree**: Tree formed by node and descendants

### Tree Properties

- **N nodes** → **N-1 edges** (in any tree)
- **Exactly one path** between any two nodes
- **Connected and acyclic** by definition  
- **Hierarchical structure** enables divide-and-conquer

### Real-World Applications

- **File Systems**: Directory/folder structures
- **Database Indexing**: B-trees, B+ trees for fast lookups
- **Compilers**: Abstract Syntax Trees (AST) 
- **Decision Making**: Decision trees in ML
- **Network Routing**: Spanning trees in networks
- **Game AI**: Minimax trees for game playing
- **Web Development**: DOM (Document Object Model)

---

## Types of Trees

### 1. Binary Tree

A tree where each node has **at most two children** (left and right).

**Properties:**
- Maximum 2 children per node
- Can be empty, have left child only, right child only, or both
- Used as base for many specialized trees

**Special Binary Trees:**

**Complete Binary Tree:**
- All levels filled except possibly the last
- Last level filled left to right
- Used in heaps

**Full Binary Tree:**
- Every node has either 0 or 2 children
- No node has exactly 1 child

**Perfect Binary Tree:**
- All internal nodes have 2 children
- All leaves at same level
- Exactly 2^h - 1 nodes for height h

### 2. Binary Search Tree (BST)

A binary tree with **ordering property**: left child < parent < right child.

**Properties:**
- **Left subtree** values < parent value
- **Right subtree** values > parent value  
- **Inorder traversal** gives sorted sequence
- **Average O(log n)** operations, **worst O(n)**

**Applications:**
- Database indexing
- Expression parsing  
- Priority queues
- Set/Map implementations

### 3. AVL Tree

A **self-balancing** binary search tree maintaining height balance.

**Properties:**
- **Balance Factor**: |height(left) - height(right)| ≤ 1
- **Automatic rebalancing** through rotations
- **Guaranteed O(log n)** operations
- **Height ≤ 1.44 * log n**

**Rotations:**
- Single Right, Single Left
- Double Left-Right, Double Right-Left

### 4. Red-Black Tree

A balanced binary search tree with **color properties**.

**Properties:**
- Each node colored **red** or **black**
- Root is black
- Red nodes have black children
- All paths from root to leaves have same number of black nodes
- **Guaranteed O(log n)** operations

### 5. Other Important Trees

**B-Tree:**
- Multi-way search tree
- Used in databases and file systems
- Nodes can have many children

**Trie (Prefix Tree):**
- Specialized for string operations
- Each path represents a string
- Used in autocomplete, spell checkers

**Segment Tree:**
- Binary tree for range queries
- Each node stores aggregate information
- Used for range sum/min/max queries

**Fenwick Tree (Binary Indexed Tree):**
- Efficient for prefix sum operations
- More space-efficient than segment trees

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
