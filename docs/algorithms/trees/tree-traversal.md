# Tree Traversals 🌲🚶

## Introduction

Tree traversal is the systematic process of visiting each node in a tree data structure exactly once. The order in which nodes are visited defines the traversal type and determines how the tree's data is processed. Different traversal methods serve different purposes and are crucial for various tree operations and algorithms.

=== "Overview"
    **Core Concept**:

    - Systematic processes for visiting every node in a tree exactly once
    - Different traversal strategies yield different node visit orders
    - Can be implemented recursively or iteratively
    
    **Major Categories**:
    
    - **Depth-First Search (DFS)**: Explores as far as possible along a branch before backtracking
    - **Breadth-First Search (BFS)**: Visits all nodes at the current depth before moving to nodes at the next depth
    
    **Common Applications**:
    
    - Expression evaluation
    - Directory/file system traversal
    - Tree serialization and deserialization
    - Search operations

=== "Depth-First Search"
    **Preorder Traversal** (Root → Left → Right):

    - Visit the root node first
    - Then recursively traverse the left subtree
    - Finally, recursively traverse the right subtree
    - Applications: Creating a copy of the tree, prefix expression evaluation
    
    **Inorder Traversal** (Left → Root → Right):
    
    - Visit the left subtree first
    - Then visit the root node
    - Finally, visit the right subtree
    - Applications: Get sorted elements from a BST, infix expression evaluation
    
    **Postorder Traversal** (Left → Right → Root):
    
    - Visit the left subtree first
    - Then visit the right subtree
    - Finally, visit the root node
    - Applications: Delete tree (free memory), postfix expression evaluation

=== "Breadth-First Search"
    **Level Order Traversal**:

    - Visit nodes level by level, from left to right
    - Start at the root (level 0)
    - Process all nodes at current level before moving to next level
    - Uses a queue to track nodes to be processed
    
    **Applications**:
    
    - Level-aware processing
    - Finding the minimum depth of a tree
    - Connecting nodes at same level
    - Tree serialization
    
    **Variations**:
    
    - **Standard Level Order**: Process level by level
    - **Zigzag Level Order**: Alternate between left-to-right and right-to-left
    - **Reverse Level Order**: Process from bottom level to top

=== "Advanced Traversals"
    **Morris Traversal** (O(1) Space):

    - Uses threading technique to eliminate need for stack or recursion
    - Creates temporary links from predecessor nodes
    - Restores original tree structure before completion
    
    **Vertical Order Traversal**:
    
    - Assigns horizontal distance values to nodes
    - Nodes at same distance form a vertical line
    - Can incorporate level information for tie-breaking
    
    **Boundary Traversal**:
    
    - Traces the boundary of the tree
    - Usually combines: left boundary, leaves, right boundary (reverse)
    
    **Diagonal Traversal**:
    
    - Groups nodes by diagonal lines
    - All nodes where path from root involves the same number of right turns

## 🔧 Implementation

=== "Recursive Implementations"
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

=== "Iterative Implementations"
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

=== "BFS Implementation"
    ### Level Order Traversal

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

=== "Morris Traversal"
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

=== "Vertical Order Traversal"
    ### Vertical Traversal Implementation

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

=== "Boundary Traversal"
    ### Boundary Traversal Implementation

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

=== "Time Complexity"
    | **Traversal** | **Best Case** | **Average Case** | **Worst Case** | **Notes** |
    |---------------|---------------|-----------------|----------------|-----------|
    | **Preorder** | O(n) | O(n) | O(n) | Always visits all nodes |
    | **Inorder** | O(n) | O(n) | O(n) | Always visits all nodes |
    | **Postorder** | O(n) | O(n) | O(n) | Always visits all nodes |
    | **Level Order** | O(n) | O(n) | O(n) | Always visits all nodes |
    | **Morris** | O(n) | O(n) | O(n) | Has additional operations but still linear |

=== "Space Complexity"
    | **Traversal** | **Recursive** | **Iterative** | **Morris** | **Notes** |
    |---------------|--------------|--------------|------------|-----------|
    | **Preorder** | O(h) | O(h) | O(1) | Stack depth equals tree height |
    | **Inorder** | O(h) | O(h) | O(1) | Stack depth equals tree height |
    | **Postorder** | O(h) | O(h) | - | Stack depth equals tree height |
    | **Level Order** | O(w) | O(w) | - | Queue size equals level width |
    | **Vertical Order** | O(n) | O(n) | - | Needs to store all nodes with coordinates |

    Where:
    
    - n = number of nodes
    - h = height of tree (best: log n, worst: n)
    - w = maximum width of tree (best: 1, worst: n/2)

## 🎯 Applications

=== "Inorder Applications"
    - **Binary Search Trees**: Gets sorted sequence
    - **Expression trees**: Infix notation
    - **Validation**: Check BST property
    - **Symbol tables**: Process in-order for ordered output
    - **Thread safety**: Safe tree deletion order

=== "Preorder Applications"
    - **Tree construction**: Serialize/deserialize
    - **File systems**: Directory traversal
    - **Expression trees**: Prefix notation
    - **Copy operations**: Create duplicate trees
    - **DOM parsing**: Web document processing

=== "Postorder Applications"
    - **Tree deletion**: Delete children before parent
    - **Expression evaluation**: Postfix notation
    - **Directory size**: Calculate folder sizes
    - **Resource cleanup**: Free resources bottom-up
    - **Dependency resolution**: Process dependencies first

=== "Level Order Applications"
    - **Tree printing**: Print tree level by level
    - **Shortest path**: In unweighted trees
    - **Tree width**: Find maximum width
    - **Nearest neighbors**: Find closest nodes
    - **Tree visualization**: UI rendering of trees

## 🚀 Advanced Patterns

=== "Tree Iterator"
    ### Tree Iterator Implementation

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

=== "Path Tracking"
    ### Path Finding Implementation

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

=== "Easy Problems"
    - [Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)
    - [Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)
    - [Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)
    - [Same Tree](https://leetcode.com/problems/same-tree/)
    - [Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)

=== "Medium Problems"
    - [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
    - [Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)
    - [Vertical Order Traversal](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/)
    - [Construct Binary Tree from Preorder and Inorder](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
    - [Boundary of Binary Tree](https://leetcode.com/problems/boundary-of-binary-tree/)

=== "Hard Problems"
    - [Binary Tree Postorder Traversal (O(1) space)](https://leetcode.com/problems/binary-tree-postorder-traversal/)
    - [Recover Binary Search Tree](https://leetcode.com/problems/recover-binary-search-tree/)
    - [Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)
    - [Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)

## 📚 Key Takeaways

1. **Choose the right traversal** for your use case
2. **Understand recursion vs iteration** trade-offs
3. **Master Morris traversals** for O(1) space solutions
4. **Practice iterative implementations** for interviews
5. **Use level order** for level-based problems
