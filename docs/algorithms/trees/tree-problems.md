# Tree Problems 🌳🧩

## 🎯 Overview

This section covers common tree problems patterns and their solutions. These problems frequently appear in coding interviews and help develop a deep understanding of tree structures.

## 📋 Problem Categories

### Tree Construction
- Building trees from traversals
- Converting between representations
- Creating balanced trees

### Tree Properties
- Height, depth, diameter calculations
- Path sum problems
- Tree validation

### Tree Modifications
- Tree transformations
- Node insertion/deletion
- Tree pruning

### Tree Traversal Applications
- Finding specific nodes
- Path tracking
- Level-based operations

## 🔧 Common Patterns

### Pattern 1: Tree Construction

#### Build Tree from Preorder and Inorder

```python
def build_tree(preorder, inorder):
    """
    Construct binary tree from preorder and inorder traversals
    Time: O(n), Space: O(n)
    """
    if not preorder or not inorder:
        return None
    
    # First element in preorder is root
    root = TreeNode(preorder[0])
    
    # Find root position in inorder
    mid = inorder.index(preorder[0])
    
    # Recursively build left and right subtrees
    root.left = build_tree(preorder[1:mid+1], inorder[:mid])
    root.right = build_tree(preorder[mid+1:], inorder[mid+1:])
    
    return root

def build_tree_optimized(preorder, inorder):
    """
    Optimized version using hashmap and indices
    Time: O(n), Space: O(n)
    """
    inorder_map = {val: i for i, val in enumerate(inorder)}
    self.preorder_idx = 0
    
    def helper(left, right):
        if left > right:
            return None
        
        root_val = preorder[self.preorder_idx]
        root = TreeNode(root_val)
        self.preorder_idx += 1
        
        # Build left subtree first (preorder)
        root.left = helper(left, inorder_map[root_val] - 1)
        root.right = helper(inorder_map[root_val] + 1, right)
        
        return root
    
    return helper(0, len(inorder) - 1)
```

#### Convert Sorted Array to BST

```python
def sorted_array_to_bst(nums):
    """
    Convert sorted array to height-balanced BST
    Time: O(n), Space: O(log n)
    """
    def helper(left, right):
        if left > right:
            return None
        
        mid = (left + right) // 2
        root = TreeNode(nums[mid])
        
        root.left = helper(left, mid - 1)
        root.right = helper(mid + 1, right)
        
        return root
    
    return helper(0, len(nums) - 1)
```

### Pattern 2: Tree Properties

#### Maximum Depth

```python
def max_depth(root):
    """
    Find maximum depth of binary tree
    Time: O(n), Space: O(h)
    """
    if not root:
        return 0
    
    return 1 + max(max_depth(root.left), max_depth(root.right))

def max_depth_iterative(root):
    """
    Iterative approach using level order traversal
    Time: O(n), Space: O(w) where w is max width
    """
    if not root:
        return 0
    
    from collections import deque
    queue = deque([root])
    depth = 0
    
    while queue:
        depth += 1
        level_size = len(queue)
        
        for _ in range(level_size):
            node = queue.popleft()
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return depth
```

#### Diameter of Binary Tree

```python
def diameter_of_binary_tree(root):
    """
    Find diameter (longest path between any two nodes)
    Time: O(n), Space: O(h)
    """
    self.max_diameter = 0
    
    def height(node):
        if not node:
            return 0
        
        left_height = height(node.left)
        right_height = height(node.right)
        
        # Update diameter at this node
        self.max_diameter = max(self.max_diameter, 
                               left_height + right_height)
        
        return 1 + max(left_height, right_height)
    
    height(root)
    return self.max_diameter
```

#### Path Sum Problems

```python
def has_path_sum(root, target_sum):
    """
    Check if tree has root-to-leaf path with given sum
    Time: O(n), Space: O(h)
    """
    if not root:
        return False
    
    if not root.left and not root.right:
        return root.val == target_sum
    
    target_sum -= root.val
    return (has_path_sum(root.left, target_sum) or 
            has_path_sum(root.right, target_sum))

def path_sum_ii(root, target_sum):
    """
    Find all root-to-leaf paths with given sum
    Time: O(n²), Space: O(h)
    """
    def dfs(node, remaining, path, result):
        if not node:
            return
        
        path.append(node.val)
        
        if not node.left and not node.right and node.val == remaining:
            result.append(path[:])  # Copy current path
        else:
            dfs(node.left, remaining - node.val, path, result)
            dfs(node.right, remaining - node.val, path, result)
        
        path.pop()  # Backtrack
    
    result = []
    dfs(root, target_sum, [], result)
    return result

def max_path_sum(root):
    """
    Find maximum path sum between any two nodes
    Time: O(n), Space: O(h)
    """
    self.max_sum = float('-inf')
    
    def helper(node):
        if not node:
            return 0
        
        # Get max sum from children (ignore negative sums)
        left_sum = max(0, helper(node.left))
        right_sum = max(0, helper(node.right))
        
        # Path through current node
        current_max = node.val + left_sum + right_sum
        self.max_sum = max(self.max_sum, current_max)
        
        # Return max sum ending at current node
        return node.val + max(left_sum, right_sum)
    
    helper(root)
    return self.max_sum
```

### Pattern 3: Tree Validation

#### Validate Binary Search Tree

```python
def is_valid_bst(root):
    """
    Check if tree is valid BST
    Time: O(n), Space: O(h)
    """
    def validate(node, min_val, max_val):
        if not node:
            return True
        
        if node.val <= min_val or node.val >= max_val:
            return False
        
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))
    
    return validate(root, float('-inf'), float('inf'))

def is_balanced(root):
    """
    Check if tree is height-balanced
    Time: O(n), Space: O(h)
    """
    def check_balance(node):
        if not node:
            return 0, True  # height, is_balanced
        
        left_height, left_balanced = check_balance(node.left)
        right_height, right_balanced = check_balance(node.right)
        
        height = 1 + max(left_height, right_height)
        balanced = (left_balanced and right_balanced and 
                   abs(left_height - right_height) <= 1)
        
        return height, balanced
    
    _, balanced = check_balance(root)
    return balanced
```

### Pattern 4: Tree Transformations

#### Invert Binary Tree

```python
def invert_tree(root):
    """
    Invert/mirror a binary tree
    Time: O(n), Space: O(h)
    """
    if not root:
        return None
    
    # Swap children
    root.left, root.right = root.right, root.left
    
    # Recursively invert subtrees
    invert_tree(root.left)
    invert_tree(root.right)
    
    return root

def invert_tree_iterative(root):
    """
    Iterative approach using queue
    Time: O(n), Space: O(w)
    """
    if not root:
        return None
    
    from collections import deque
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        
        # Swap children
        node.left, node.right = node.right, node.left
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return root
```

#### Flatten Binary Tree to Linked List

```python
def flatten(root):
    """
    Flatten binary tree to linked list in-place
    Time: O(n), Space: O(h)
    """
    def helper(node):
        if not node:
            return None
        
        # Store original right child
        right_child = node.right
        
        # If left child exists, process it
        if node.left:
            node.right = node.left
            node.left = None
            
            # Find tail of flattened left subtree
            tail = node.right
            while tail.right:
                tail = tail.right
            
            # Connect to original right child
            tail.right = right_child
        
        # Recursively flatten right subtree
        helper(node.right)
    
    helper(root)
```

### Pattern 5: Lowest Common Ancestor

```python
def lowest_common_ancestor(root, p, q):
    """
    Find LCA of two nodes in binary tree
    Time: O(n), Space: O(h)
    """
    if not root or root == p or root == q:
        return root
    
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    
    if left and right:
        return root  # Both nodes found in different subtrees
    
    return left or right  # Return non-None result

def lowest_common_ancestor_bst(root, p, q):
    """
    Find LCA in BST (optimized)
    Time: O(h), Space: O(1)
    """
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root  # Found LCA
    
    return None
```

### Pattern 6: Serialization

```python
def serialize(root):
    """
    Serialize tree to string
    Time: O(n), Space: O(n)
    """
    def dfs(node):
        if not node:
            return "null"
        return f"{node.val},{dfs(node.left)},{dfs(node.right)}"
    
    return dfs(root)

def deserialize(data):
    """
    Deserialize string to tree
    Time: O(n), Space: O(n)
    """
    def dfs():
        val = next(vals)
        if val == "null":
            return None
        return TreeNode(int(val), dfs(), dfs())
    
    vals = iter(data.split(','))
    return dfs()
```

## 🏆 LeetCode Problems by Pattern

### Tree Construction
- [105. Construct Binary Tree from Preorder and Inorder](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
- [108. Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)
- [1008. Construct BST from Preorder](https://leetcode.com/problems/construct-binary-search-tree-from-preorder-traversal/)

### Tree Properties
- [104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
- [543. Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)
- [112. Path Sum](https://leetcode.com/problems/path-sum/)
- [113. Path Sum II](https://leetcode.com/problems/path-sum-ii/)
- [124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

### Tree Validation
- [98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
- [110. Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)
- [100. Same Tree](https://leetcode.com/problems/same-tree/)

### Tree Transformations
- [226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)
- [114. Flatten Binary Tree to Linked List](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)
- [617. Merge Two Binary Trees](https://leetcode.com/problems/merge-two-binary-trees/)

### Lowest Common Ancestor
- [236. Lowest Common Ancestor of Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)
- [235. Lowest Common Ancestor of BST](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

### Serialization
- [297. Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)
- [449. Serialize and Deserialize BST](https://leetcode.com/problems/serialize-and-deserialize-bst/)

## 📚 Key Problem-Solving Strategies

### 1. **Recursive Thinking**
- Most tree problems have elegant recursive solutions
- Define base cases clearly
- Think about what each recursive call should return

### 2. **State Tracking**
- Use class variables or helper functions to track global state
- Pass additional parameters for local state

### 3. **Two-Pass vs One-Pass**
- Some problems can be solved in one traversal
- Others require separate passes for different computations

### 4. **Space Optimization**
- Consider iterative solutions for constant space
- Morris traversal for O(1) space tree traversal

### 5. **Pattern Recognition**
- Identify if it's a construction, validation, or transformation problem  
- Choose appropriate traversal method based on requirements

## 🎯 Practice Strategy

1. **Master basic traversals** first
2. **Solve construction problems** to understand tree building
3. **Practice property calculations** for recursive thinking
4. **Work on transformations** to understand tree manipulation
5. **Tackle serialization** for advanced tree operations
