# Trees - Medium Problems

## 🎯 Learning Objectives

Master intermediate tree algorithms and complex tree manipulation patterns. These 15 problems build upon basic tree operations and introduce advanced concepts.

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Construct Binary Tree from Traversals | Tree Construction | Medium | O(n) | O(n) |
    | 2 | Binary Tree Level Order Traversal II | Tree BFS | Medium | O(n) | O(w) |
    | 3 | Binary Tree Zigzag Level Order | Tree BFS | Medium | O(n) | O(w) |
    | 4 | Validate Binary Search Tree | Tree DFS + Bounds | Medium | O(n) | O(h) |
    | 5 | Recover Binary Search Tree | Tree DFS + Morris | Medium | O(n) | O(h) |
    | 6 | Binary Tree Right Side View | Tree BFS/DFS | Medium | O(n) | O(h) |
    | 7 | Populating Next Right Pointers | Tree BFS | Medium | O(n) | O(1) |
    | 8 | Sum Root to Leaf Numbers | Tree DFS | Medium | O(n) | O(h) |
    | 9 | Binary Tree Maximum Path Sum | Tree DFS | Medium | O(n) | O(h) |
    | 10 | Lowest Common Ancestor of BST | Tree DFS | Medium | O(h) | O(h) |
    | 11 | Kth Smallest Element in BST | Tree DFS/Morris | Medium | O(h+k) | O(h) |
    | 12 | House Robber III | Tree DP | Medium | O(n) | O(h) |
    | 13 | Path Sum II | Tree DFS + Backtracking | Medium | O(n) | O(h) |
    | 14 | Flatten Binary Tree to Linked List | Tree DFS | Medium | O(n) | O(h) |
    | 15 | Serialize and Deserialize Binary Tree | Tree Traversal | Medium | O(n) | O(n) |

=== "🎯 Advanced Tree Patterns"

    **🏗️ Tree Construction:**
    - Building from traversal sequences
    - Reconstruction with constraints
    
    **🔄 Tree Traversal Variants:**
    - Level-order modifications (reverse, zigzag)
    - Morris traversal (O(1) space)
    
    **🎯 Tree Properties & Validation:**
    - Complex BST operations and repairs
    - Path calculations and optimizations
    
    **🔗 Tree Modifications:**
    - Structural transformations
    - Connecting nodes across levels

=== "⚡ Advanced Interview Strategy"

    **💡 Pattern Recognition:**
    
    - **Multiple constraints**: Problems with 2+ conditions to satisfy
    - **Optimization required**: Space or time complexity improvements
    - **State tracking**: Problems requiring additional information during traversal
    
    **🔄 Advanced Techniques:**
    
    1. **Morris Traversal**: O(1) space tree traversal
    2. **Path State Management**: Tracking cumulative values during DFS
    3. **Level-based Processing**: Complex BFS patterns
    4. **Tree DP**: Combining tree traversal with dynamic programming

---

## Problem 1: Construct Binary Tree from Preorder and Inorder

**Difficulty**: 🟡 Medium  
**Pattern**: Tree Construction with Divide & Conquer  
**Time**: O(n), **Space**: O(n)

=== "Problem Statement"

    Given two arrays representing preorder and inorder traversal of a binary tree, construct and return the binary tree.

    **Example:**
    ```text
    Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
    Output: [3,9,20,null,null,15,7]
    ```

=== "Optimal Solution"

    ```python
    def build_tree(preorder, inorder):
        """
        Build tree using preorder for root and inorder for subtree splits.
        """
        if not preorder or not inorder:
            return None
        
        # First element in preorder is always root
        root = TreeNode(preorder[0])
        
        # Find root position in inorder to split left/right subtrees
        mid = inorder.index(preorder[0])
        
        # Recursively build left and right subtrees
        root.left = build_tree(preorder[1:mid+1], inorder[:mid])
        root.right = build_tree(preorder[mid+1:], inorder[mid+1:])
        
        return root

    def build_tree_optimized(preorder, inorder):
        """
        Optimized version using hashmap for O(1) lookups.
        """
        inorder_map = {val: i for i, val in enumerate(inorder)}
        self.preorder_idx = 0
        
        def build(left, right):
            if left > right:
                return None
            
            root_val = preorder[self.preorder_idx]
            self.preorder_idx += 1
            root = TreeNode(root_val)
            
            # Split point in inorder
            inorder_idx = inorder_map[root_val]
            
            # Build left subtree first (matches preorder)
            root.left = build(left, inorder_idx - 1)
            root.right = build(inorder_idx + 1, right)
            
            return root
        
        return build(0, len(inorder) - 1)
    ```

---

## Problem 2: Binary Tree Level Order Traversal II

**Difficulty**: 🟡 Medium  
**Pattern**: Tree BFS with Result Manipulation  
**Time**: O(n), **Space**: O(w)

=== "Problem Statement"

    Return the bottom-up level order traversal (last level first).

=== "Optimal Solution"

    ```python
    def level_order_bottom(root):
        """
        BFS with result reversal.
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
        
        return result[::-1]  # Reverse for bottom-up
    ```

---

## Problem 3: Binary Tree Zigzag Level Order Traversal

**Difficulty**: 🟡 Medium  
**Pattern**: Tree BFS with Direction Toggle  
**Time**: O(n), **Space**: O(w)

=== "Problem Statement"

    Return zigzag level order traversal (alternating left-to-right and right-to-left).

=== "Optimal Solution"

    ```python
    def zigzag_level_order(root):
        """
        BFS with alternating direction using deque.
        """
        if not root:
            return []
        
        from collections import deque
        queue = deque([root])
        result = []
        left_to_right = True
        
        while queue:
            level_size = len(queue)
            level_values = deque()
            
            for _ in range(level_size):
                node = queue.popleft()
                
                # Add to front or back based on direction
                if left_to_right:
                    level_values.append(node.val)
                else:
                    level_values.appendleft(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(list(level_values))
            left_to_right = not left_to_right
        
        return result
    ```

---

## Problem 4: Binary Tree Right Side View

**Difficulty**: 🟡 Medium  
**Pattern**: Tree BFS/DFS with Level Tracking  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Return the values of nodes you can see from the right side of the tree.

=== "Optimal Solution"

    ```python
    def right_side_view(root):
        """
        DFS approach - visit right subtree first.
        """
        def dfs(node, level, result):
            if not node:
                return
            
            # If first time seeing this level, add to result
            if level == len(result):
                result.append(node.val)
            
            # Visit right first, then left
            dfs(node.right, level + 1, result)
            dfs(node.left, level + 1, result)
        
        result = []
        dfs(root, 0, result)
        return result

    def right_side_view_bfs(root):
        """
        BFS approach - take last node at each level.
        """
        if not root:
            return []
        
        from collections import deque
        queue = deque([root])
        result = []
        
        while queue:
            level_size = len(queue)
            
            for i in range(level_size):
                node = queue.popleft()
                
                # Last node at this level
                if i == level_size - 1:
                    result.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        return result
    ```

---

## Problem 5: Sum Root to Leaf Numbers

**Difficulty**: 🟡 Medium  
**Pattern**: Tree DFS with Path State  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Each root-to-leaf path represents a number. Return the sum of all such numbers.

    **Example:**
    ```text
    Input: root = [1,2,3]
    Output: 25
    Explanation: 12 (1->2) + 13 (1->3) = 25
    ```

=== "Optimal Solution"

    ```python
    def sum_numbers(root):
        """
        DFS with cumulative number building.
        """
        def dfs(node, current_num):
            if not node:
                return 0
            
            current_num = current_num * 10 + node.val
            
            # If leaf, return the complete number
            if not node.left and not node.right:
                return current_num
            
            # Sum from both subtrees
            return dfs(node.left, current_num) + dfs(node.right, current_num)
        
        return dfs(root, 0)
    ```

---

## Problem 6: Binary Tree Maximum Path Sum

**Difficulty**: 🟡 Medium  
**Pattern**: Tree DFS with Global State  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Find the maximum path sum in a binary tree (path can start and end at any nodes).

=== "Optimal Solution"

    ```python
    def max_path_sum(root):
        """
        DFS with global maximum tracking.
        """
        def dfs(node):
            if not node:
                return 0
            
            # Get max path sum from left and right (ignore negative)
            left_max = max(0, dfs(node.left))
            right_max = max(0, dfs(node.right))
            
            # Path through current node (for global maximum)
            path_through_node = node.val + left_max + right_max
            self.max_sum = max(self.max_sum, path_through_node)
            
            # Return max path ending at this node (for parent)
            return node.val + max(left_max, right_max)
        
        self.max_sum = float('-inf')
        dfs(root)
        return self.max_sum
    ```

---

## Problem 7: Lowest Common Ancestor of BST

**Difficulty**: 🟡 Medium  
**Pattern**: Tree DFS with BST Properties  
**Time**: O(h), **Space**: O(h)

=== "Problem Statement"

    Find the lowest common ancestor of two nodes in a BST.

=== "Optimal Solution"

    ```python
    def lowest_common_ancestor(root, p, q):
        """
        Use BST property to navigate efficiently.
        """
        if not root:
            return None
        
        # Both nodes in left subtree
        if p.val < root.val and q.val < root.val:
            return lowest_common_ancestor(root.left, p, q)
        
        # Both nodes in right subtree
        if p.val > root.val and q.val > root.val:
            return lowest_common_ancestor(root.right, p, q)
        
        # Nodes on different sides or one is root
        return root

    def lowest_common_ancestor_iterative(root, p, q):
        """
        Iterative approach for O(1) space.
        """
        while root:
            if p.val < root.val and q.val < root.val:
                root = root.left
            elif p.val > root.val and q.val > root.val:
                root = root.right
            else:
                return root
        return None
    ```

---

## Problem 8: Kth Smallest Element in BST

**Difficulty**: 🟡 Medium  
**Pattern**: Tree DFS (Inorder) with Early Termination  
**Time**: O(h + k), **Space**: O(h)

=== "Problem Statement"

    Find the kth smallest element in a BST.

=== "Optimal Solution"

    ```python
    def kth_smallest(root, k):
        """
        Inorder traversal with count tracking.
        """
        def inorder(node):
            if not node or self.result is not None:
                return
            
            inorder(node.left)
            
            self.count += 1
            if self.count == k:
                self.result = node.val
                return
            
            inorder(node.right)
        
        self.count = 0
        self.result = None
        inorder(root)
        return self.result

    def kth_smallest_iterative(root, k):
        """
        Iterative inorder with early termination.
        """
        stack = []
        current = root
        count = 0
        
        while stack or current:
            # Go to leftmost node
            while current:
                stack.append(current)
                current = current.left
            
            # Process current node
            current = stack.pop()
            count += 1
            
            if count == k:
                return current.val
            
            # Move to right subtree
            current = current.right
        
        return -1
    ```

---

## Problem 9: House Robber III

**Difficulty**: 🟡 Medium  
**Pattern**: Tree DP with State Tracking  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Houses are arranged in a binary tree. Cannot rob two directly connected houses. Find maximum money that can be robbed.

=== "Optimal Solution"

    ```python
    def rob(root):
        """
        Tree DP: return [max_without_root, max_with_root].
        """
        def dfs(node):
            if not node:
                return [0, 0]  # [without_node, with_node]
            
            left = dfs(node.left)
            right = dfs(node.right)
            
            # Without current node: can take max from children
            without_node = max(left) + max(right)
            
            # With current node: take node + without children
            with_node = node.val + left[0] + right[0]
            
            return [without_node, with_node]
        
        return max(dfs(root))
    ```

---

## Problem 10: Path Sum II

**Difficulty**: 🟡 Medium  
**Pattern**: Tree DFS with Backtracking  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Find all root-to-leaf paths where the path sum equals the target sum.

=== "Optimal Solution"

    ```python
    def path_sum(root, target_sum):
        """
        DFS with path tracking and backtracking.
        """
        def dfs(node, remaining, path, result):
            if not node:
                return
            
            path.append(node.val)
            remaining -= node.val
            
            # If leaf and sum matches, add to result
            if not node.left and not node.right and remaining == 0:
                result.append(path[:])  # Create copy
            
            # Continue DFS
            dfs(node.left, remaining, path, result)
            dfs(node.right, remaining, path, result)
            
            # Backtrack
            path.pop()
        
        result = []
        dfs(root, target_sum, [], result)
        return result
    ```

---

## Problem 11: Flatten Binary Tree to Linked List

**Difficulty**: 🟡 Medium  
**Pattern**: Tree DFS with Pointer Manipulation  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Flatten a binary tree to a linked list in preorder traversal order.

=== "Optimal Solution"

    ```python
    def flatten(root):
        """
        Postorder approach: flatten children first, then connect.
        """
        def dfs(node):
            if not node:
                return None
            
            # If leaf, return itself
            if not node.left and not node.right:
                return node
            
            # Flatten left and right subtrees
            left_tail = dfs(node.left)
            right_tail = dfs(node.right)
            
            # If left subtree exists, connect it
            if left_tail:
                left_tail.right = node.right
                node.right = node.left
                node.left = None
            
            # Return the tail of flattened tree
            return right_tail if right_tail else left_tail
        
        dfs(root)
    ```

---

## Problem 12: Serialize and Deserialize Binary Tree

**Difficulty**: 🟡 Medium  
**Pattern**: Tree Traversal with Encoding  
**Time**: O(n), **Space**: O(n)

=== "Problem Statement"

    Design algorithms to serialize a binary tree to a string and deserialize string back to tree.

=== "Optimal Solution"

    ```python
    def serialize(root):
        """
        Preorder serialization with null markers.
        """
        def preorder(node):
            if not node:
                vals.append("null")
            else:
                vals.append(str(node.val))
                preorder(node.left)
                preorder(node.right)
        
        vals = []
        preorder(root)
        return ",".join(vals)

    def deserialize(data):
        """
        Preorder deserialization using iterator.
        """
        def build():
            val = next(vals)
            if val == "null":
                return None
            
            node = TreeNode(int(val))
            node.left = build()
            node.right = build()
            return node
        
        vals = iter(data.split(","))
        return build()
    ```

---

## Problem 13: Populating Next Right Pointers

**Difficulty**: 🟡 Medium  
**Pattern**: Tree BFS with O(1) Space  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Populate next right pointers in a perfect binary tree.

=== "Optimal Solution"

    ```python
    def connect(root):
        """
        Level-by-level connection using existing pointers.
        """
        if not root:
            return root
        
        # Start with root level
        leftmost = root
        
        while leftmost.left:  # While not leaf level
            head = leftmost
            
            # Connect all nodes at current level
            while head:
                # Connect children
                head.left.next = head.right
                
                # Connect across parents
                if head.next:
                    head.right.next = head.next.left
                
                head = head.next
            
            # Move to next level
            leftmost = leftmost.left
        
        return root
    ```

---

## Problem 14: Recover Binary Search Tree

**Difficulty**: 🟡 Medium  
**Pattern**: Tree DFS with Morris Traversal  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Two nodes in a BST have been swapped. Recover the tree without changing structure.

=== "Optimal Solution"

    ```python
    def recover_tree(root):
        """
        Morris inorder traversal to find swapped nodes.
        """
        def morris_inorder():
            current = root
            first = second = prev = None
            
            while current:
                if not current.left:
                    # Process current
                    if prev and prev.val > current.val:
                        if not first:
                            first = prev
                        second = current
                    prev = current
                    current = current.right
                else:
                    # Find inorder predecessor
                    predecessor = current.left
                    while predecessor.right and predecessor.right != current:
                        predecessor = predecessor.right
                    
                    if not predecessor.right:
                        # Create thread
                        predecessor.right = current
                        current = current.left
                    else:
                        # Remove thread and process
                        predecessor.right = None
                        if prev and prev.val > current.val:
                            if not first:
                                first = prev
                            second = current
                        prev = current
                        current = current.right
            
            return first, second
        
        # Find and swap the two nodes
        first, second = morris_inorder()
        if first and second:
            first.val, second.val = second.val, first.val
    ```

---

## Problem 15: Validate Binary Search Tree (Advanced)

**Difficulty**: 🟡 Medium  
**Pattern**: Tree DFS with Multiple Validation Methods  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Validate if a binary tree is a valid BST using multiple approaches.

=== "Optimal Solution"

    ```python
    def is_valid_bst_inorder(root):
        """
        Inorder traversal approach - should be strictly increasing.
        """
        def inorder(node):
            if not node:
                return True
            
            if not inorder(node.left):
                return False
            
            if self.prev is not None and self.prev >= node.val:
                return False
            self.prev = node.val
            
            return inorder(node.right)
        
        self.prev = None
        return inorder(root)

    def is_valid_bst_bounds(root):
        """
        Bounds checking approach.
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

## 🎯 Practice Summary

### Advanced Tree Patterns Mastered

1. **Tree Construction**: From traversal sequences, with optimizations
2. **Complex Traversals**: Zigzag, bottom-up, right side view
3. **Tree State Management**: Path tracking, cumulative calculations
4. **Tree DP**: Combining traversal with dynamic programming decisions
5. **BST Advanced Operations**: Validation, recovery, k-th element
6. **Tree Transformations**: Flattening, serialization, pointer connections
7. **Space Optimization**: Morris traversal, O(1) space solutions

### Key Algorithmic Techniques

- **Morris Traversal**: O(1) space tree traversal using threading
- **State Propagation**: Passing cumulative information during DFS
- **Global Variables**: Tracking maximum/minimum across entire tree
- **Path Reconstruction**: Building and maintaining paths during traversal
- **Tree DP**: Making optimal decisions at each node based on subtrees

### Time Complexity Patterns

- **Standard Traversal**: O(n) to visit all nodes
- **BST Operations**: O(h) leveraging tree structure
- **Path Problems**: O(n) but early termination possible
- **Construction**: O(n) for building from linear structures

### Space Complexity Optimization

- **Recursive**: O(h) for call stack
- **Iterative**: O(h) for explicit stack/queue
- **Morris**: O(1) using tree threading
- **In-place**: Modifying existing structure when possible

### Interview Success Strategy

1. **Identify tree type**: BST vs binary tree affects approach
2. **Choose traversal**: DFS for paths, BFS for levels
3. **State management**: Track necessary information during traversal
4. **Edge cases**: Empty tree, single node, unbalanced structure
5. **Optimization**: Consider Morris traversal for space constraints

### Next Steps

Ready for the hardest challenges? Try **[Hard Tree Problems](hard-problems.md)** to explore:

- Complex tree constructions and transformations
- Advanced tree DP with multiple states
- Tree algorithms with geometric interpretations
- Multi-tree problems and forest algorithms

---

*These medium tree problems introduce advanced concepts essential for mastering tree algorithms. Focus on understanding state management and optimization techniques!*
