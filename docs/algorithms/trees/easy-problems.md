# Trees - Easy Problems

## 🎯 Learning Objectives

Master fundamental tree operations and basic tree patterns. These 15 problems cover the essential tree algorithms asked in technical interviews.

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Maximum Depth of Binary Tree | Tree DFS | Easy | O(n) | O(h) |
    | 2 | Same Tree | Tree Comparison | Easy | O(n) | O(h) |
    | 3 | Invert Binary Tree | Tree Transformation | Easy | O(n) | O(h) |
    | 4 | Symmetric Tree | Tree Comparison | Easy | O(n) | O(h) |
    | 5 | Binary Tree Paths | Tree DFS + Backtracking | Easy | O(n) | O(h) |
    | 6 | Path Sum | Tree DFS | Easy | O(n) | O(h) |
    | 7 | Minimum Depth of Binary Tree | Tree BFS/DFS | Easy | O(n) | O(h) |
    | 8 | Balanced Binary Tree | Tree DFS | Easy | O(n) | O(h) |
    | 9 | Diameter of Binary Tree | Tree DFS | Easy | O(n) | O(h) |
    | 10 | Binary Tree Level Order Traversal | Tree BFS | Easy | O(n) | O(w) |
    | 11 | Convert Sorted Array to BST | Tree Construction | Easy | O(n) | O(h) |
    | 12 | Validate Binary Search Tree | Tree DFS | Easy | O(n) | O(h) |
    | 13 | Range Sum of BST | Tree DFS | Easy | O(n) | O(h) |
    | 14 | Merge Two Binary Trees | Tree DFS | Easy | O(n) | O(h) |
    | 15 | Binary Tree Preorder Traversal | Tree Traversal | Easy | O(n) | O(h) |

=== "🎯 Core Tree Patterns"

    **🌲 Tree Traversal:**
    - DFS: Preorder, Inorder, Postorder
    - BFS: Level-order traversal
    
    **🔍 Tree Properties:**
    - Height, depth, balance checks
    - Path calculations and validations
    
    **🔄 Tree Transformations:**
    - Inversion, merging, construction
    - Structure modifications
    
    **🎯 Tree Search:**
    - BST properties and validations
    - Path finding and sum calculations

=== "⚡ Interview Strategy"

    **💡 Problem Recognition:**
    
    - **Recursive structure**: Most tree problems use recursion naturally
    - **Base cases**: Empty tree (null) handling is crucial
    - **Traversal patterns**: Choose DFS vs BFS based on requirements
    
    **🔄 Solution Approach:**
    
    1. **Define base case**: What happens with null/empty tree?
    2. **Recursive case**: How to combine left and right subtree results?
    3. **Choose traversal**: Pre/in/post-order or level-order?
    4. **Space consideration**: Recursion depth vs iterative space

---

## Problem 1: Maximum Depth of Binary Tree

**Difficulty**: 🟢 Easy  
**Pattern**: Tree DFS (Postorder)  
**Time**: O(n), **Space**: O(h) where h = height

=== "Problem Statement"

    Given the root of a binary tree, return its maximum depth. The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

    **Example:**
    ```text
    Input: root = [3,9,20,null,null,15,7]
    Output: 3
    Explanation: The maximum depth is 3 (3 -> 20 -> 7 or 3 -> 20 -> 15)
    ```

=== "Optimal Solution"

    ```python
    def max_depth(root):
        """
        DFS approach - O(n) time, O(h) space for recursion.
        """
        if not root:
            return 0
        
        left_depth = max_depth(root.left)
        right_depth = max_depth(root.right)
        
        return max(left_depth, right_depth) + 1

    # Iterative BFS approach
    def max_depth_bfs(root):
        """
        BFS level-order traversal approach.
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

=== "Pattern Recognition"

    ```python
    # Tree depth/height template:
    def tree_height(root):
        if not root:
            return 0  # Base case
        
        # Postorder: process children first, then current
        left_height = tree_height(root.left)
        right_height = tree_height(root.right)
        
        # Combine results and add current level
        return max(left_height, right_height) + 1
    ```

---

## Problem 2: Same Tree

**Difficulty**: 🟢 Easy  
**Pattern**: Tree Comparison DFS  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Given the roots of two binary trees p and q, check if they are the same (structurally identical with same node values).

=== "Optimal Solution"

    ```python
    def is_same_tree(p, q):
        """
        Recursive comparison of trees.
        """
        # Base cases
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False
        
        # Recursively check left and right subtrees
        return (is_same_tree(p.left, q.left) and 
                is_same_tree(p.right, q.right))
    ```

---

## Problem 3: Invert Binary Tree

**Difficulty**: 🟢 Easy  
**Pattern**: Tree Transformation DFS  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Given the root of a binary tree, invert it (swap left and right children for every node).

=== "Optimal Solution"

    ```python
    def invert_tree(root):
        """
        Recursive inversion - swap children at each node.
        """
        if not root:
            return None
        
        # Swap children
        root.left, root.right = root.right, root.left
        
        # Recursively invert subtrees
        invert_tree(root.left)
        invert_tree(root.right)
        
        return root
    ```

---

## Problem 4: Symmetric Tree

**Difficulty**: 🟢 Easy  
**Pattern**: Tree Comparison DFS  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Check if a binary tree is symmetric (mirror of itself around its center).

=== "Optimal Solution"

    ```python
    def is_symmetric(root):
        """
        Check if tree is symmetric by comparing left and right subtrees.
        """
        def is_mirror(left, right):
            if not left and not right:
                return True
            if not left or not right:
                return False
            
            return (left.val == right.val and
                    is_mirror(left.left, right.right) and
                    is_mirror(left.right, right.left))
        
        return is_mirror(root.left, root.right) if root else True
    ```

---

## Problem 5: Binary Tree Paths

**Difficulty**: 🟢 Easy  
**Pattern**: Tree DFS + Backtracking  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Return all root-to-leaf paths in a binary tree.

=== "Optimal Solution"

    ```python
    def binary_tree_paths(root):
        """
        DFS with path tracking to find all root-to-leaf paths.
        """
        def dfs(node, path, result):
            if not node:
                return
            
            path.append(str(node.val))
            
            # If leaf node, add path to result
            if not node.left and not node.right:
                result.append("->".join(path))
            else:
                # Continue DFS
                dfs(node.left, path, result)
                dfs(node.right, path, result)
            
            # Backtrack
            path.pop()
        
        result = []
        dfs(root, [], result)
        return result
    ```

---

## Problem 6: Path Sum

**Difficulty**: 🟢 Easy  
**Pattern**: Tree DFS with Target  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Check if there exists a root-to-leaf path with a given target sum.

=== "Optimal Solution"

    ```python
    def has_path_sum(root, target_sum):
        """
        DFS to check if any root-to-leaf path equals target sum.
        """
        if not root:
            return False
        
        # If leaf node, check if remaining sum equals node value
        if not root.left and not root.right:
            return root.val == target_sum
        
        # Recursively check subtrees with reduced target
        remaining = target_sum - root.val
        return (has_path_sum(root.left, remaining) or 
                has_path_sum(root.right, remaining))
    ```

---

## Problem 7: Minimum Depth of Binary Tree

**Difficulty**: 🟢 Easy  
**Pattern**: Tree BFS/DFS  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Find the minimum depth (shortest path from root to any leaf).

=== "Optimal Solution"

    ```python
    def min_depth(root):
        """
        DFS approach for minimum depth.
        """
        if not root:
            return 0
        
        # If one subtree is empty, return depth of other + 1
        if not root.left:
            return min_depth(root.right) + 1
        if not root.right:
            return min_depth(root.left) + 1
        
        # Both subtrees exist
        return min(min_depth(root.left), min_depth(root.right)) + 1

    def min_depth_bfs(root):
        """
        BFS approach - stops at first leaf found.
        """
        if not root:
            return 0
        
        from collections import deque
        queue = deque([(root, 1)])
        
        while queue:
            node, depth = queue.popleft()
            
            # If leaf found, return depth
            if not node.left and not node.right:
                return depth
            
            if node.left:
                queue.append((node.left, depth + 1))
            if node.right:
                queue.append((node.right, depth + 1))
        
        return 0
    ```

---

## Problem 8: Balanced Binary Tree

**Difficulty**: 🟢 Easy  
**Pattern**: Tree DFS with Height Check  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Check if a binary tree is height-balanced (height difference between left and right subtrees ≤ 1 for every node).

=== "Optimal Solution"

    ```python
    def is_balanced(root):
        """
        Check if tree is balanced using height calculation.
        """
        def check_height(node):
            if not node:
                return 0
            
            left_height = check_height(node.left)
            if left_height == -1:  # Left subtree not balanced
                return -1
            
            right_height = check_height(node.right)
            if right_height == -1:  # Right subtree not balanced
                return -1
            
            # Check if current node is balanced
            if abs(left_height - right_height) > 1:
                return -1
            
            return max(left_height, right_height) + 1
        
        return check_height(root) != -1
    ```

---

## Problem 9: Diameter of Binary Tree

**Difficulty**: 🟢 Easy  
**Pattern**: Tree DFS with Path Length  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Find the diameter (longest path between any two nodes) of a binary tree.

=== "Optimal Solution"

    ```python
    def diameter_of_binary_tree(root):
        """
        Calculate diameter while computing heights.
        """
        def dfs(node):
            if not node:
                return 0
            
            left_height = dfs(node.left)
            right_height = dfs(node.right)
            
            # Update diameter (path through current node)
            self.diameter = max(self.diameter, left_height + right_height)
            
            # Return height of current subtree
            return max(left_height, right_height) + 1
        
        self.diameter = 0
        dfs(root)
        return self.diameter
    ```

---

## Problem 10: Binary Tree Level Order Traversal

**Difficulty**: 🟢 Easy  
**Pattern**: Tree BFS  
**Time**: O(n), **Space**: O(w) where w = max width

=== "Problem Statement"

    Return the level order traversal of nodes' values (left to right, level by level).

=== "Optimal Solution"

    ```python
    def level_order(root):
        """
        BFS level-order traversal.
        """
        if not root:
            return []
        
        from collections import deque
        queue = deque([root])
        result = []
        
        while queue:
            level_size = len(queue)
            level_values = []
            
            for _ in range(level_size):
                node = queue.popleft()
                level_values.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(level_values)
        
        return result
    ```

---

## Problem 11: Convert Sorted Array to BST

**Difficulty**: 🟢 Easy  
**Pattern**: Tree Construction with Divide & Conquer  
**Time**: O(n), **Space**: O(log n)

=== "Problem Statement"

    Convert a sorted array to a height-balanced BST.

=== "Optimal Solution"

    ```python
    def sorted_array_to_bst(nums):
        """
        Build BST by choosing middle element as root recursively.
        """
        def build_tree(left, right):
            if left > right:
                return None
            
            mid = (left + right) // 2
            root = TreeNode(nums[mid])
            
            root.left = build_tree(left, mid - 1)
            root.right = build_tree(mid + 1, right)
            
            return root
        
        return build_tree(0, len(nums) - 1)
    ```

---

## Problem 12: Validate Binary Search Tree

**Difficulty**: 🟢 Easy  
**Pattern**: Tree DFS with Bounds  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Validate if a binary tree is a valid BST.

=== "Optimal Solution"

    ```python
    def is_valid_bst(root):
        """
        Validate BST using min/max bounds.
        """
        def validate(node, min_val, max_val):
            if not node:
                return True
            
            if node.val <= min_val or node.val >= max_val:
                return False
            
            return (validate(node.left, min_val, node.val) and
                    validate(node.right, node.val, max_val))
        
        return validate(root, float('-inf'), float('inf'))
    ```

---

## Problem 13: Range Sum of BST

**Difficulty**: 🟢 Easy  
**Pattern**: Tree DFS with Pruning  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Find sum of all node values in BST within a given range [low, high].

=== "Optimal Solution"

    ```python
    def range_sum_bst(root, low, high):
        """
        DFS with pruning based on BST properties.
        """
        if not root:
            return 0
        
        # If current value is in range, include it
        current_sum = 0
        if low <= root.val <= high:
            current_sum = root.val
        
        # Recursively sum from left subtree (if worth exploring)
        if root.val > low:
            current_sum += range_sum_bst(root.left, low, high)
        
        # Recursively sum from right subtree (if worth exploring)
        if root.val < high:
            current_sum += range_sum_bst(root.right, low, high)
        
        return current_sum
    ```

---

## Problem 14: Merge Two Binary Trees

**Difficulty**: 🟢 Easy  
**Pattern**: Tree DFS with Parallel Traversal  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Merge two binary trees by summing overlapping nodes.

=== "Optimal Solution"

    ```python
    def merge_trees(root1, root2):
        """
        Merge trees by creating new nodes with summed values.
        """
        if not root1:
            return root2
        if not root2:
            return root1
        
        # Create new node with sum of both values
        merged = TreeNode(root1.val + root2.val)
        
        # Recursively merge left and right subtrees
        merged.left = merge_trees(root1.left, root2.left)
        merged.right = merge_trees(root1.right, root2.right)
        
        return merged
    ```

---

## Problem 15: Binary Tree Preorder Traversal

**Difficulty**: 🟢 Easy  
**Pattern**: Tree Traversal  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Return the preorder traversal of a binary tree's node values.

=== "Optimal Solution"

    ```python
    def preorder_traversal(root):
        """
        Recursive preorder traversal.
        """
        if not root:
            return []
        
        result = [root.val]
        result.extend(preorder_traversal(root.left))
        result.extend(preorder_traversal(root.right))
        return result

    def preorder_iterative(root):
        """
        Iterative preorder using stack.
        """
        if not root:
            return []
        
        stack = [root]
        result = []
        
        while stack:
            node = stack.pop()
            result.append(node.val)
            
            # Push right first, then left (stack is LIFO)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        
        return result
    ```

---

## 🎯 Practice Summary

### Key Tree Patterns Mastered

1. **Tree Traversal**: DFS (pre/in/post-order) and BFS (level-order)
2. **Tree Properties**: Height, depth, balance, symmetry
3. **Tree Comparison**: Structure and value equality
4. **Tree Construction**: From arrays, balanced BST creation
5. **Tree Validation**: BST properties, balance checks
6. **Path Problems**: Root-to-leaf paths, path sums
7. **Tree Transformation**: Inversion, merging, modifications

### Common Tree Patterns

- **Base Case**: Always handle `null/None` nodes
- **Recursive Structure**: Most tree problems naturally use recursion
- **Postorder Pattern**: Process children first, then current (height, balance)
- **Preorder Pattern**: Process current first, then children (copying, printing)
- **Level Traversal**: Use BFS with queue for level-by-level processing

### Space Complexity Notes

- **Recursive Space**: O(h) where h is tree height
- **Balanced Tree**: h = O(log n)
- **Skewed Tree**: h = O(n) worst case
- **BFS Space**: O(w) where w is maximum width of tree

### Interview Success Tips

1. **Clarify tree structure**: BST vs regular binary tree
2. **Handle edge cases**: Empty tree, single node, unbalanced
3. **Choose traversal**: DFS for path problems, BFS for level problems
4. **Space optimization**: Consider iterative solutions for deep trees
5. **Test with examples**: Draw small trees to verify logic

### Next Steps

Ready for more challenges? Try **[Medium Tree Problems](medium-problems.md)** to explore:

- Advanced tree constructions (from multiple traversals)
- Complex path problems (path sum variants, lowest common ancestor)
- Tree modifications (deletion, insertion in BST)
- Serialization and deserialization of trees

---

*These foundational tree problems are essential for building intuition about recursive tree algorithms. Master these patterns before advancing to more complex tree challenges!*
