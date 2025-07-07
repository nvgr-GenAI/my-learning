# Tree Fundamentals 🌳

## 🎯 Overview

Trees are hierarchical data structures that consist of nodes connected by edges. Understanding tree fundamentals is essential for solving complex algorithmic problems and designing efficient systems.

=== "📋 Core Concepts"

    ## **Tree Terminology**
    
    | Term | Definition | Example |
    |------|------------|---------|
    | **Node** | Basic unit containing data | Each circle in a tree diagram |
    | **Edge** | Connection between nodes | Lines connecting nodes |
    | **Root** | Top node with no parent | Starting point of the tree |
    | **Leaf** | Node with no children | End nodes of branches |
    | **Parent** | Node with children | Node one level up |
    | **Child** | Node with a parent | Node one level down |
    | **Sibling** | Nodes with same parent | Nodes at same level |
    | **Ancestor** | All nodes on path to root | All nodes above current node |
    | **Descendant** | All nodes in subtree | All nodes below current node |
    | **Height** | Max distance from node to leaf | Longest path downward |
    | **Depth** | Distance from root to node | Path length from root |
    | **Level** | All nodes at same depth | Nodes at same distance from root |

=== "🌲 Tree Properties"

    ## **Binary Tree Properties**
    
    ```
    Maximum nodes at level i: 2^i
    Maximum nodes in tree of height h: 2^(h+1) - 1
    Minimum height with n nodes: ⌊log₂n⌋
    Maximum height with n nodes: n - 1
    ```
    
    ## **Tree Types**
    
    | Type | Properties | Use Cases |
    |------|------------|-----------|
    | **Binary Tree** | At most 2 children per node | General hierarchy |
    | **Complete** | All levels filled except possibly last | Heaps, efficient storage |
    | **Perfect** | All internal nodes have 2 children | Mathematical analysis |
    | **Balanced** | Height difference ≤ 1 | Guaranteed performance |
    | **BST** | Left < Root < Right | Searching, sorting |
    | **AVL** | Self-balancing BST | Guaranteed O(log n) |
    | **Red-Black** | Colored nodes for balance | Standard library implementations |

=== "🔍 Tree Traversals"

    ## **Depth-First Search (DFS)**
    
    ```python
    # Inorder (Left, Root, Right)
    def inorder(root):
        if root:
            inorder(root.left)
            print(root.val)
            inorder(root.right)
    
    # Preorder (Root, Left, Right)
    def preorder(root):
        if root:
            print(root.val)
            preorder(root.left)
            preorder(root.right)
    
    # Postorder (Left, Right, Root)
    def postorder(root):
        if root:
            postorder(root.left)
            postorder(root.right)
            print(root.val)
    ```
    
    ## **Breadth-First Search (BFS)**
    
    ```python
    from collections import deque
    
    def level_order(root):
        if not root:
            return
        
        queue = deque([root])
        while queue:
            node = queue.popleft()
            print(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    ```

=== "⚡ Implementation Patterns"

    ## **Tree Node Structure**
    
    ```python
    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right
    ```
    
    ## **Common Tree Operations**
    
    ```python
    # Calculate height
    def height(root):
        if not root:
            return -1
        return 1 + max(height(root.left), height(root.right))
    
    # Count nodes
    def count_nodes(root):
        if not root:
            return 0
        return 1 + count_nodes(root.left) + count_nodes(root.right)
    
    # Check if balanced
    def is_balanced(root):
        def check_balance(node):
            if not node:
                return 0, True
            
            left_height, left_balanced = check_balance(node.left)
            right_height, right_balanced = check_balance(node.right)
            
            balanced = (left_balanced and right_balanced and 
                       abs(left_height - right_height) <= 1)
            height = 1 + max(left_height, right_height)
            
            return height, balanced
        
        return check_balance(root)[1]
    ```

=== "🎯 Problem Solving Strategies"

    ## **Tree Problem Patterns**
    
    | Pattern | When to Use | Example Problems |
    |---------|-------------|------------------|
    | **DFS Recursion** | Path problems, tree properties | Path sum, tree height |
    | **BFS Level Order** | Level-by-level processing | Level order traversal, right side view |
    | **Divide & Conquer** | Combine subtree results | Diameter, lowest common ancestor |
    | **Tree DP** | Optimization problems | House robber III, maximum path sum |
    | **Tree Construction** | Build from traversals | Construct from preorder/inorder |
    | **Morris Traversal** | O(1) space traversal | Inorder without recursion/stack |
    
    ## **Common Tree Algorithms**
    
    ```python
    # Lowest Common Ancestor
    def lca(root, p, q):
        if not root or root == p or root == q:
            return root
        
        left = lca(root.left, p, q)
        right = lca(root.right, p, q)
        
        if left and right:
            return root
        return left or right
    
    # Path Sum
    def has_path_sum(root, target):
        if not root:
            return False
        
        if not root.left and not root.right:
            return root.val == target
        
        return (has_path_sum(root.left, target - root.val) or
                has_path_sum(root.right, target - root.val))
    
    # Maximum Path Sum
    def max_path_sum(root):
        max_sum = float('-inf')
        
        def max_gain(node):
            nonlocal max_sum
            if not node:
                return 0
            
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)
            
            price_newpath = node.val + left_gain + right_gain
            max_sum = max(max_sum, price_newpath)
            
            return node.val + max(left_gain, right_gain)
        
        max_gain(root)
        return max_sum
    ```

=== "🏆 Advanced Concepts"

    ## **Tree Balancing**
    
    **AVL Tree Rotations:**
    ```
    Left Rotation:     Right Rotation:
         A                 A
        / \               / \
       B   C             B   C
      / \       -->     / \
     D   E             D   E
    ```
    
    **Red-Black Tree Properties:**
    1. Every node is either red or black
    2. Root is black
    3. All leaves (NIL) are black
    4. Red nodes have black children
    5. All paths from root to leaves contain same number of black nodes
    
    ## **Segment Trees**
    
    ```python
    class SegmentTree:
        def __init__(self, arr):
            self.n = len(arr)
            self.tree = [0] * (4 * self.n)
            self.build(arr, 0, 0, self.n - 1)
        
        def build(self, arr, node, start, end):
            if start == end:
                self.tree[node] = arr[start]
            else:
                mid = (start + end) // 2
                self.build(arr, 2 * node + 1, start, mid)
                self.build(arr, 2 * node + 2, mid + 1, end)
                self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    ```

=== "📊 Complexity Analysis"

    ## **Time Complexities**
    
    | Operation | Binary Tree | BST (Average) | BST (Worst) | AVL/Red-Black |
    |-----------|-------------|---------------|-------------|---------------|
    | **Search** | O(n) | O(log n) | O(n) | O(log n) |
    | **Insert** | O(n) | O(log n) | O(n) | O(log n) |
    | **Delete** | O(n) | O(log n) | O(n) | O(log n) |
    | **Traversal** | O(n) | O(n) | O(n) | O(n) |
    | **Height** | O(n) | O(n) | O(n) | O(log n) |
    
    ## **Space Complexities**
    
    | Structure | Space | Notes |
    |-----------|-------|-------|
    | **Binary Tree** | O(n) | One node per element |
    | **Balanced Tree** | O(n) | Additional balance info |
    | **Segment Tree** | O(4n) | Internal nodes for ranges |
    | **Trie** | O(ALPHABET_SIZE * N * M) | Where M is average string length |

---

*Master these tree fundamentals to build a strong foundation for advanced tree algorithms and data structures!*
