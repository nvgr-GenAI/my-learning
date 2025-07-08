# Tree Implementation Guide 🛠️

This guide provides practical implementations for creating and manipulating tree data structures, including operations for analyzing tree properties, traversing trees, and modifying tree structures.

## 🌱 Basic Tree Creation

=== "Binary Tree"
    ```python
    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right
    
    # Creating a simple binary tree
    #       1
    #      / \
    #     2   3
    #    / \   \
    #   4   5   6
    
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    root.right.right = TreeNode(6)
    ```

=== "N-ary Tree"
    ```python
    class TreeNode:
        def __init__(self, val=0, children=None):
            self.val = val
            self.children = children if children else []
    
    # Creating an N-ary tree
    #       1
    #    / / \ \
    #   2 3  4  5
    #     / \    \
    #    6   7    8
    
    root = TreeNode(1)
    root.children = [TreeNode(2), TreeNode(3), TreeNode(4), TreeNode(5)]
    root.children[1].children = [TreeNode(6), TreeNode(7)]  # Node 3's children
    root.children[3].children = [TreeNode(8)]  # Node 5's children
    ```

=== "BST"
    ```python
    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right
    
    def insert_into_bst(root, val):
        if not root:
            return TreeNode(val)
            
        if val < root.val:
            root.left = insert_into_bst(root.left, val)
        else:
            root.right = insert_into_bst(root.right, val)
            
        return root
    
    # Create a BST
    values = [5, 3, 7, 2, 4, 6, 8]
    root = None
    for val in values:
        root = insert_into_bst(root, val)
    
    # Result:
    #       5
    #      / \
    #     3   7
    #    / \ / \
    #   2  4 6  8
    ```

=== "AVL Tree"
    ```python
    class AVLNode:
        def __init__(self, val=0):
            self.val = val
            self.left = None
            self.right = None
            self.height = 1  # New node is initially at height 1
    
    def get_height(node):
        if not node:
            return 0
        return node.height
    
    def get_balance(node):
        if not node:
            return 0
        return get_height(node.left) - get_height(node.right)
    
    def right_rotate(y):
        x = y.left
        T3 = x.right
        
        # Rotation
        x.right = y
        y.left = T3
        
        # Update heights
        y.height = max(get_height(y.left), get_height(y.right)) + 1
        x.height = max(get_height(x.left), get_height(x.right)) + 1
        
        return x
    
    def left_rotate(x):
        y = x.right
        T2 = y.left
        
        # Rotation
        y.left = x
        x.right = T2
        
        # Update heights
        x.height = max(get_height(x.left), get_height(x.right)) + 1
        y.height = max(get_height(y.left), get_height(y.right)) + 1
        
        return y
    
    def insert_avl(root, val):
        # Standard BST insert
        if not root:
            return AVLNode(val)
            
        if val < root.val:
            root.left = insert_avl(root.left, val)
        else:
            root.right = insert_avl(root.right, val)
            
        # Update height of current node
        root.height = max(get_height(root.left), get_height(root.right)) + 1
        
        # Get balance factor
        balance = get_balance(root)
        
        # Left Left Case
        if balance > 1 and val < root.left.val:
            return right_rotate(root)
        
        # Right Right Case
        if balance < -1 and val > root.right.val:
            return left_rotate(root)
        
        # Left Right Case
        if balance > 1 and val > root.left.val:
            root.left = left_rotate(root.left)
            return right_rotate(root)
        
        # Right Left Case
        if balance < -1 and val < root.right.val:
            root.right = right_rotate(root.right)
            return left_rotate(root)
            
        return root
    
    # Create an AVL tree
    values = [9, 5, 10, 0, 6, 11, -1, 1, 2]
    root = None
    for val in values:
        root = insert_avl(root, val)
    ```

## 🔍 Tree Property Analysis

=== "Height & Depth"
    ```python
    def height(node):
        """
        Calculate height of a tree (maximum distance from node to leaf)
        Height of empty tree is -1, height of leaf node is 0
        """
        if not node:
            return -1
        return 1 + max(height(node.left), height(node.right))
    
    def depth(root, node, current_depth=0):
        """
        Calculate depth of a node (distance from root to node)
        Depth of root is 0
        """
        if not root:
            return -1  # Node not found
            
        if root == node:
            return current_depth
            
        left_depth = depth(root.left, node, current_depth + 1)
        if left_depth != -1:
            return left_depth
            
        return depth(root.right, node, current_depth + 1)
    
    # For N-ary trees
    def height_nary(node):
        if not node:
            return -1
        if not node.children:
            return 0
        return 1 + max(height_nary(child) for child in node.children)
    ```

=== "Node Identification"
    ```python
    def is_leaf(node):
        """Check if a node is a leaf node (has no children)"""
        return node and not node.left and not node.right
    
    def is_internal(node):
        """Check if node is an internal node (has at least one child)"""
        return node and (node.left or node.right)
    
    def count_leaves(root):
        """Count number of leaf nodes in a tree"""
        if not root:
            return 0
        if is_leaf(root):
            return 1
        return count_leaves(root.left) + count_leaves(root.right)
    
    def count_internal_nodes(root):
        """Count number of internal nodes in a tree"""
        if not root or is_leaf(root):
            return 0
        return 1 + count_internal_nodes(root.left) + count_internal_nodes(root.right)
    
    def count_total_nodes(root):
        """Count total number of nodes in a tree"""
        if not root:
            return 0
        return 1 + count_total_nodes(root.left) + count_total_nodes(root.right)
    
    # For N-ary trees
    def is_leaf_nary(node):
        return node and not node.children
    
    def count_leaves_nary(root):
        if not root:
            return 0
        if is_leaf_nary(root):
            return 1
        return sum(count_leaves_nary(child) for child in root.children)
    ```

=== "Level & Width"
    ```python
    from collections import deque
    
    def get_level_order(root):
        """Return all nodes by level"""
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
    
    def get_node_level(root, target_node, level=1):
        """Find the level of a given node"""
        if not root:
            return 0
            
        if root == target_node:
            return level
            
        downlevel = get_node_level(root.left, target_node, level + 1)
        if downlevel != 0:
            return downlevel
            
        return get_node_level(root.right, target_node, level + 1)
    
    def tree_width(root):
        """Calculate the maximum width of the tree"""
        if not root:
            return 0
            
        levels = get_level_order(root)
        return max(len(level) for level in levels)
    ```

=== "Relationships"
    ```python
    def find_siblings(root, node):
        """Find siblings of a node (nodes with same parent)"""
        if not root or not node:
            return []
            
        # BFS to find parent of the target node
        queue = deque([root])
        
        while queue:
            current = queue.popleft()
            
            # Check if either left or right child is the target
            children = []
            if current.left:
                children.append(current.left)
            if current.right:
                children.append(current.right)
                
            if node in children:
                # Return all siblings (excluding the node itself)
                return [child for child in children if child != node]
                
            # Continue BFS
            for child in children:
                queue.append(child)
                
        return []  # No siblings found
    
    def find_parent(root, node):
        """Find parent of a given node"""
        if not root or root == node:
            return None
            
        # Check if either child is the target node
        if (root.left and root.left == node) or (root.right and root.right == node):
            return root
            
        # Recursively search left and right subtrees
        left_result = find_parent(root.left, node)
        if left_result:
            return left_result
            
        return find_parent(root.right, node)
    
    def are_cousins(root, node1, node2):
        """Check if two nodes are cousins (same level, different parents)"""
        if not root:
            return False
            
        level1 = get_node_level(root, node1)
        level2 = get_node_level(root, node2)
        
        parent1 = find_parent(root, node1)
        parent2 = find_parent(root, node2)
        
        # Cousins must be at same level but have different parents
        return level1 == level2 and parent1 != parent2
    ```

## 🛠️ Tree Modification

=== "Adding Nodes"
    ```python
    def insert_in_binary_tree(root, val):
        """Insert a value in the first available position level by level"""
        if not root:
            return TreeNode(val)
            
        queue = deque([root])
        
        while queue:
            node = queue.popleft()
            
            if not node.left:
                node.left = TreeNode(val)
                return root
            if not node.right:
                node.right = TreeNode(val)
                return root
                
            queue.append(node.left)
            queue.append(node.right)
        
        return root  # Should never reach here
    
    def insert_in_bst(root, val):
        """Insert a value in a BST, maintaining BST property"""
        if not root:
            return TreeNode(val)
            
        if val < root.val:
            root.left = insert_in_bst(root.left, val)
        elif val > root.val:
            root.right = insert_in_bst(root.right, val)
        # If val == root.val, do nothing (no duplicates)
        
        return root
    
    def add_child_to_nary_node(node, val):
        """Add a child to an N-ary tree node"""
        if not node:
            return
        
        new_node = TreeNode(val)
        node.children.append(new_node)
        return new_node
    ```

=== "Deleting Nodes"
    ```python
    def delete_in_bst(root, val):
        """Delete a node with given value from BST"""
        if not root:
            return None
            
        # Find the node
        if val < root.val:
            root.left = delete_in_bst(root.left, val)
        elif val > root.val:
            root.right = delete_in_bst(root.right, val)
        else:
            # Node with only one child or no child
            if not root.left:
                return root.right
            elif not root.right:
                return root.left
                
            # Node with two children
            # Get inorder successor (smallest in right subtree)
            temp = root.right
            while temp.left:
                temp = temp.left
                
            # Replace with inorder successor
            root.val = temp.val
            
            # Delete the inorder successor
            root.right = delete_in_bst(root.right, temp.val)
        
        return root
    
    def delete_node_in_binary_tree(root, key):
        """
        Delete a node with the given value from a binary tree
        Replace with deepest rightmost node
        """
        if not root:
            return None
            
        # If tree has only one node
        if not root.left and not root.right:
            if root.val == key:
                return None
            return root
            
        # Find the node to delete and the deepest node
        key_node = None
        last_node = None
        queue = deque([root])
        
        while queue:
            node = queue.popleft()
            
            if node.val == key:
                key_node = node
                
            if node.left:
                queue.append(node.left)
                last_node = node.left
                
            if node.right:
                queue.append(node.right)
                last_node = node.right
        
        # If key node found, replace with deepest node and delete deepest
        if key_node:
            # Save the value of last node
            key_node.val = last_node.val
            
            # Now delete the last node
            queue = deque([root])
            while queue:
                node = queue.popleft()
                
                if node.left:
                    if node.left == last_node:
                        node.left = None
                        return root
                    queue.append(node.left)
                    
                if node.right:
                    if node.right == last_node:
                        node.right = None
                        return root
                    queue.append(node.right)
        
        return root
    ```

=== "Tree Balancing"
    ```python
    def is_balanced(root):
        """Check if binary tree is balanced (height difference ≤ 1)"""
        def check_height(node):
            if not node:
                return 0
            
            left = check_height(node.left)
            if left == -1:
                return -1
                
            right = check_height(node.right)
            if right == -1:
                return -1
                
            if abs(left - right) > 1:
                return -1
                
            return max(left, right) + 1
            
        return check_height(root) != -1
    
    def balance_bst(root):
        """Convert a BST to a balanced BST"""
        # Step 1: Store nodes in sorted order (inorder traversal)
        nodes = []
        
        def inorder(node):
            if not node:
                return
            inorder(node.left)
            nodes.append(node.val)
            inorder(node.right)
            
        inorder(root)
        
        # Step 2: Construct a balanced BST from sorted array
        def construct_balanced_bst(arr, start, end):
            if start > end:
                return None
                
            # Get the middle element and make it root
            mid = (start + end) // 2
            node = TreeNode(arr[mid])
            
            # Recursively construct left and right subtrees
            node.left = construct_balanced_bst(arr, start, mid - 1)
            node.right = construct_balanced_bst(arr, mid + 1, end)
            
            return node
            
        return construct_balanced_bst(nodes, 0, len(nodes) - 1)
    ```

## 🔄 Tree Transformations

=== "Inversion/Mirroring"
    ```python
    def invert_binary_tree(root):
        """Invert a binary tree (swap left and right children)"""
        if not root:
            return None
            
        # Swap children
        root.left, root.right = root.right, root.left
        
        # Recursively invert subtrees
        invert_binary_tree(root.left)
        invert_binary_tree(root.right)
        
        return root
    ```

=== "Conversion"
    ```python
    def binary_tree_to_linked_list(root):
        """Convert binary tree to DLL in-order"""
        from collections import deque
        
        if not root:
            return None
            
        # Get inorder traversal
        nodes = []
        
        def inorder(node):
            if not node:
                return
            inorder(node.left)
            nodes.append(node)
            inorder(node.right)
            
        inorder(root)
        
        # Create DLL
        for i in range(len(nodes) - 1):
            nodes[i].right = nodes[i + 1]  # Next pointer
            nodes[i + 1].left = nodes[i]   # Prev pointer
            
        # Clear original tree connections
        for node in nodes:
            if node.left and node.left not in (nodes[nodes.index(node) - 1] if nodes.index(node) > 0 else None, None):
                node.left = None
            if node.right and node.right not in (nodes[nodes.index(node) + 1] if nodes.index(node) < len(nodes) - 1 else None, None):
                node.right = None
        
        return nodes[0]  # Head of the linked list
    
    def sorted_array_to_bst(nums):
        """Convert sorted array to balanced BST"""
        if not nums:
            return None
            
        mid = len(nums) // 2
        
        root = TreeNode(nums[mid])
        root.left = sorted_array_to_bst(nums[:mid])
        root.right = sorted_array_to_bst(nums[mid + 1:])
        
        return root
    ```

=== "Serialization"
    ```python
    def serialize_binary_tree(root):
        """Serialize binary tree to string"""
        if not root:
            return "X,"  # Null marker
            
        # Preorder traversal
        return str(root.val) + "," + serialize_binary_tree(root.left) + serialize_binary_tree(root.right)
    
    def deserialize_binary_tree(data):
        """Deserialize string to binary tree"""
        def helper(nodes):
            val = next(nodes)
            
            if val == "X":
                return None
                
            node = TreeNode(int(val))
            node.left = helper(nodes)
            node.right = helper(nodes)
            
            return node
            
        nodes_iter = iter(data.split(","))
        return helper(nodes_iter)
    ```

## 🔍 Tree Validation

=== "Structure Validation"
    ```python
    def is_valid_bst(root):
        """Check if a binary tree is a valid BST"""
        def validate(node, low=float('-inf'), high=float('inf')):
            if not node:
                return True
                
            # Current node's value must be between low and high
            if node.val <= low or node.val >= high:
                return False
                
            # Left subtree must be < node.val and right subtree must be > node.val
            return (validate(node.left, low, node.val) and
                    validate(node.right, node.val, high))
                    
        return validate(root)
    
    def is_complete(root):
        """Check if a binary tree is complete"""
        if not root:
            return True
            
        queue = deque([root])
        flag = False  # Flag to mark if we've seen a non-full node
        
        while queue:
            node = queue.popleft()
            
            # If we see a node with missing left child
            if node.left:
                if flag:  # If we've already seen a non-full node
                    return False
                queue.append(node.left)
            else:
                flag = True
                
            # If we see a node with missing right child
            if node.right:
                if flag:  # If we've already seen a non-full node
                    return False
                queue.append(node.right)
            else:
                flag = True
                
        return True
    
    def is_perfect(root):
        """Check if a binary tree is perfect"""
        # Perfect tree has 2^h - 1 nodes, where h is height
        def count_nodes(node):
            if not node:
                return 0
            return 1 + count_nodes(node.left) + count_nodes(node.right)
            
        def height(node):
            if not node:
                return -1
            return 1 + max(height(node.left), height(node.right))
            
        h = height(root)
        n = count_nodes(root)
        
        return n == (2 ** (h + 1)) - 1
    
    def is_full(root):
        """Check if a binary tree is full (every node has 0 or 2 children)"""
        if not root:
            return True
            
        # Leaf node
        if not root.left and not root.right:
            return True
            
        # Both children exist
        if root.left and root.right:
            return is_full(root.left) and is_full(root.right)
            
        # One child exists and other doesn't
        return False
    ```

=== "Edge & Path Validation"
    ```python
    def count_edges(root):
        """Count edges in a binary tree"""
        # In a tree with n nodes, there are n-1 edges
        def count_nodes(node):
            if not node:
                return 0
            return 1 + count_nodes(node.left) + count_nodes(node.right)
            
        nodes = count_nodes(root)
        return max(0, nodes - 1)  # Ensure we don't return negative
    
    def has_path_sum(root, target_sum):
        """Check if tree has a root-to-leaf path with given sum"""
        if not root:
            return False
            
        # If leaf node, check if value equals remaining sum
        if not root.left and not root.right:
            return root.val == target_sum
            
        # Check both subtrees with reduced target
        return (has_path_sum(root.left, target_sum - root.val) or
                has_path_sum(root.right, target_sum - root.val))
                
    def find_paths(root, target_sum):
        """Find all root-to-leaf paths with given sum"""
        result = []
        
        def dfs(node, remaining, path):
            if not node:
                return
                
            # Add current node to path
            path.append(node.val)
            
            # If leaf node and sum matches
            if not node.left and not node.right and node.val == remaining:
                result.append(path.copy())
                
            # Continue DFS with updated remaining sum
            dfs(node.left, remaining - node.val, path)
            dfs(node.right, remaining - node.val, path)
            
            # Backtrack
            path.pop()
            
        dfs(root, target_sum, [])
        return result
    ```

## 📊 Visualization & Debugging

=== "Print Tree"
    ```python
    def print_tree(root):
        """Print binary tree level by level with indentation"""
        def print_level(node, level):
            if not node:
                return
                
            print_level(node.right, level + 1)
            print("    " * level + str(node.val))
            print_level(node.left, level + 1)
            
        print_level(root, 0)
    
    def pretty_print_tree(root):
        """Print binary tree in a more visual format"""
        lines, _, _, _ = _pretty_print_helper(root)
        for line in lines:
            print(line)
            
    def _pretty_print_helper(node):
        """Helper function for pretty printing a binary tree"""
        if not node:
            return [], 0, 0, 0
            
        # Convert value to string
        node_repr = str(node.val)
        
        # Base case: no child
        if not node.left and not node.right:
            line = node_repr
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle
            
        # Recursive case: compute width of left and right children
        left_lines, left_width, left_height, left_middle = _pretty_print_helper(node.left)
        right_lines, right_width, right_height, right_middle = _pretty_print_helper(node.right)
        
        # Construct node line
        node_width = max(len(node_repr), left_width + right_width + 1)
        left_indent = (node_width - left_width) // 2
        right_indent = node_width - right_width - left_indent
        
        # Construct branches
        top_line = " " * left_indent + "/" + " " * (left_width - left_indent - 1)
        bottom_line = " " * right_indent + "\\" + " " * (right_width - right_indent - 1)
        
        # Construct all lines
        lines = [" " * ((node_width - len(node_repr)) // 2) + node_repr]
        
        # Add branches if children exist
        if node.left or node.right:
            for i in range(max(left_height, right_height)):
                left_part = left_lines[i] if i < len(left_lines) else " " * left_width
                right_part = right_lines[i] if i < len(right_lines) else " " * right_width
                lines.append(left_part + " " * (node_width - left_width - right_width) + right_part)
        
        return lines, node_width, len(lines), node_width // 2
    ```

=== "Debugging Helpers"
    ```python
    def validate_tree_structure(root):
        """Validate that tree structure is consistent"""
        if not root:
            return True
            
        # Check that left child points back to parent
        if root.left and hasattr(root.left, 'parent'):
            if root.left.parent != root:
                return False
                
        # Check that right child points back to parent
        if root.right and hasattr(root.right, 'parent'):
            if root.right.parent != root:
                return False
                
        # Recursively check children
        return (validate_tree_structure(root.left) and
                validate_tree_structure(root.right))
    
    def get_tree_state(root):
        """Get a dictionary representation of tree state for debugging"""
        if not root:
            return {"type": "null"}
            
        result = {
            "value": root.val,
            "height": height(root),
            "leaf": is_leaf(root),
            "left": get_tree_state(root.left),
            "right": get_tree_state(root.right)
        }
        
        # Add BST property validation if needed
        if hasattr(root, 'is_bst'):
            result["is_valid_bst"] = is_valid_bst(root)
            
        return result
    ```

## 🌟 Advanced Tree Algorithms

=== "Lowest Common Ancestor"
    ```python
    def lowest_common_ancestor(root, p, q):
        """Find the lowest common ancestor of two nodes in binary tree"""
        if not root or root == p or root == q:
            return root
            
        left = lowest_common_ancestor(root.left, p, q)
        right = lowest_common_ancestor(root.right, p, q)
        
        if left and right:  # p and q are in different subtrees
            return root
            
        return left or right  # Either one is in one subtree or none exists
    
    def lca_with_parent_pointers(p, q):
        """LCA when nodes have parent pointers"""
        # Track ancestors of p
        ancestors = set()
        while p:
            ancestors.add(p)
            p = p.parent
            
        # Check if q or its ancestors are in p's ancestors
        while q:
            if q in ancestors:
                return q
            q = q.parent
            
        return None
    ```

=== "Tree Diameter"
    ```python
    def diameter_of_binary_tree(root):
        """Find the diameter (longest path) of a binary tree"""
        diameter = 0
        
        def height(node):
            nonlocal diameter
            if not node:
                return 0
                
            left_height = height(node.left)
            right_height = height(node.right)
            
            # Update diameter (path through current node)
            diameter = max(diameter, left_height + right_height)
            
            # Return height of tree rooted at current node
            return 1 + max(left_height, right_height)
            
        height(root)
        return diameter
    ```

=== "Tree Views"
    ```python
    def left_view(root):
        """Get left view of binary tree (first node at each level)"""
        if not root:
            return []
            
        result = []
        queue = deque([root])
        
        while queue:
            level_size = len(queue)
            for i in range(level_size):
                node = queue.popleft()
                
                # First node at this level
                if i == 0:
                    result.append(node.val)
                    
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                    
        return result
    
    def right_view(root):
        """Get right view of binary tree (last node at each level)"""
        if not root:
            return []
            
        result = []
        queue = deque([root])
        
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
    
    def top_view(root):
        """Get top view of binary tree (first node at each vertical line)"""
        if not root:
            return []
            
        # Map to store nodes at each horizontal distance
        hd_map = {}
        
        # Queue for level order traversal (node, horizontal distance)
        queue = deque([(root, 0)])
        
        while queue:
            node, hd = queue.popleft()
            
            # If this horizontal distance is not in map yet
            if hd not in hd_map:
                hd_map[hd] = node.val
                
            if node.left:
                queue.append((node.left, hd - 1))
            if node.right:
                queue.append((node.right, hd + 1))
                
        # Return values sorted by horizontal distance
        return [hd_map[hd] for hd in sorted(hd_map.keys())]
    
    def bottom_view(root):
        """Get bottom view of binary tree (last node at each vertical line)"""
        if not root:
            return []
            
        # Map to store nodes at each horizontal distance
        hd_map = {}
        
        # Queue for level order traversal (node, horizontal distance)
        queue = deque([(root, 0)])
        
        while queue:
            node, hd = queue.popleft()
            
            # Always update map (we want the last node at each hd)
            hd_map[hd] = node.val
                
            if node.left:
                queue.append((node.left, hd - 1))
            if node.right:
                queue.append((node.right, hd + 1))
                
        # Return values sorted by horizontal distance
        return [hd_map[hd] for hd in sorted(hd_map.keys())]
    ```

=== "Vertical Traversal"
    ```python
    def vertical_order_traversal(root):
        """Get vertical order traversal of binary tree"""
        if not root:
            return []
            
        # Dictionary to store nodes at each horizontal distance
        node_map = defaultdict(list)
        
        # Queue for level order traversal (node, hd, level)
        queue = deque([(root, 0, 0)])
        
        while queue:
            node, hd, level = queue.popleft()
            
            # Store node info (value and level for tie-breaking)
            node_map[hd].append((level, node.val))
            
            if node.left:
                queue.append((node.left, hd - 1, level + 1))
            if node.right:
                queue.append((node.right, hd + 1, level + 1))
                
        # Construct result with proper ordering
        result = []
        for hd in sorted(node_map.keys()):
            # Sort nodes at same horizontal distance by level then value
            column = [val for _, val in sorted(node_map[hd])]
            result.append(column)
            
        return result
    ```

## 📝 Best Practices

1. **Always Handle Edge Cases**: Check for `None`/empty trees at the beginning of your functions
2. **Use Helper Functions**: Keep public API clean with private helpers for complex operations
3. **Be Mindful of Recursion Depth**: Very deep trees can cause stack overflow; use iterative approach when needed
4. **Leverage Tree Properties**: Different tree types have special properties (BST ordering, etc.) that can optimize algorithms
5. **Test with Various Tree Shapes**: Ensure algorithms work on skewed, balanced, and degenerate trees
6. **Consider Time/Space Tradeoffs**: Sometimes you can cache computations for faster lookup at the expense of memory
7. **Document Assumptions**: Make clear what tree properties your functions assume
8. **Use Visualization**: For complex operations, visualize the tree before and after to verify correctness

---

This implementation guide covers the core operations you'll need to work with tree data structures effectively. For additional implementations or specific tree types, refer to the dedicated pages for each tree type.
