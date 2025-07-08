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
    Number of leaf nodes in perfect binary tree: 2^h
    Number of internal nodes in perfect binary tree: 2^h - 1
    ```
    
    ## **Tree Type Definitions**
    
    - **Binary Tree**: Each node has at most 2 children (left and right)
    - **Complete Binary Tree**: All levels filled except possibly the last, which is filled from left to right
    - **Full Binary Tree**: Every node has either 0 or 2 children (no node has exactly 1 child)
    - **Perfect Binary Tree**: All internal nodes have exactly 2 children and all leaf nodes are at the same level
    - **Balanced Binary Tree**: Height difference between left and right subtrees is at most 1 for all nodes
    - **Binary Search Tree (BST)**: For each node, all values in left subtree ≤ node value ≤ all values in right subtree

    ## **Self-Balancing Trees**
    
    - **AVL Tree**: First self-balancing BST, maintains height balance factor (-1, 0, or 1) for each node
    - **Red-Black Tree**: Each node colored red or black, maintains color properties to ensure balance
    - **Splay Tree**: Recently accessed elements moved to root, good for caching
    - **B-Tree**: Generalizes BST by allowing nodes to have more than two children, optimized for disk operations
    
    ## **Advanced Tree Structures**
    
    - **Trie (Prefix Tree)**: Character-based tree for efficient string operations
    - **Segment Tree**: Binary tree for range queries and updates in arrays
    - **Fenwick Tree (BIT)**: Efficiently updates and calculates prefix sums
    - **Heap**: Complete binary tree with heap property (parent ≥ or ≤ children)
    - **Suffix Tree**: Compressed trie containing all suffixes of a string
    
    ## **Tree Types**
    
    | Type | Properties | Use Cases |
    |------|------------|-----------|
    | **Binary Tree** | At most 2 children per node | General hierarchy |
    | **Complete** | All levels filled except possibly last | Heaps, efficient storage |
    | **Perfect** | All internal nodes have 2 children | Mathematical analysis |
    | **Balanced** | Height difference ≤ 1 | Guaranteed performance |
    | **Full** | Nodes have either 0 or 2 children | Expression trees |
    | **Skewed** | Each node has only 1 child | Degenerated cases |
    | **BST** | Left < Root < Right | Searching, sorting |
    | **AVL** | Self-balancing BST | Guaranteed O(log n) |
    | **Red-Black** | Colored nodes for balance | Standard library implementations |
    | **B-Tree** | Multi-way search tree | Databases, file systems |
    | **Trie** | Character-based tree | Dictionary, autocomplete |
    | **Segment Tree** | Range queries | Query optimizations |
    | **Heap** | Complete binary tree | Priority queues |

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
    | **Serialization** | Convert tree to string | Tree serialization/deserialization |
    | **Path Recording** | Track node-to-node paths | Path sum II, all paths |
    | **View Problems** | External perspectives | Left/right/top/bottom view |
    | **Parent Pointers** | Upward navigation | Find all ancestors |
    | **Recursive State** | Additional state tracking | Count good nodes, path with given sum |
    
    ## **Interview Problem Categories**
    
    - **Tree Structure Validation**: Is it BST, balanced, symmetric, etc.
    - **Tree Modification**: Insert, delete, rebalance operations
    - **Tree Comparison**: Same tree, subtree of another, flip equivalent
    - **Tree Transformation**: Convert to linked list, mirror/invert
    - **Tree Metrics**: Height, diameter, count nodes of specific type
    - **Path Problems**: Existence, enumeration, optimization
    - **Ancestor/Descendant**: Lowest common ancestor, all ancestors
    - **Tree Construction**: From traversals, sorted array
    
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
    
    ## **Additional Tree Algorithms**
    
    ```python
    # Tree diameter (longest path between any two nodes)
    def diameter_of_binary_tree(root):
        diameter = 0
        
        def depth(node):
            nonlocal diameter
            if not node:
                return 0
                
            left = depth(node.left)
            right = depth(node.right)
            
            # Update diameter at every node
            diameter = max(diameter, left + right)
            
            # Return the longest path going through the node
            return max(left, right) + 1
            
        depth(root)
        return diameter
    
    # Serialize and deserialize a binary tree
    def serialize(root):
        """Encodes a tree to a single string."""
        if not root:
            return "X,"  # Null marker
        
        return str(root.val) + "," + serialize(root.left) + serialize(root.right)
    
    def deserialize(data):
        """Decodes your encoded data to tree."""
        def dfs(nodes):
            val = next(nodes)
            if val == "X":
                return None
                
            node = TreeNode(int(val))
            node.left = dfs(nodes)
            node.right = dfs(nodes)
            return node
            
        nodes_iter = iter(data.split(','))
        return dfs(nodes_iter)
    
    # Check if tree is symmetric
    def is_symmetric(root):
        def mirror(left, right):
            if not left and not right:
                return True
            if not left or not right:
                return False
            
            return (left.val == right.val and 
                    mirror(left.left, right.right) and 
                    mirror(left.right, right.left))
        
        return not root or mirror(root.left, root.right)
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
    | **B-Tree (order m)** | O(n) | Higher fan-out reduces height |
    | **Heap** | O(n) | Complete binary tree structure |
    | **Huffman Tree** | O(k) | k is number of unique characters |
    | **Red-Black Tree** | O(n) | One bit per node for color |
    
    ## **Auxiliary Space in Tree Algorithms**
    
    | Algorithm | Space | Notes |
    |-----------|-------|-------|
    | **Recursive DFS** | O(h) | Stack frames, h = tree height |
    | **Iterative DFS** | O(h) | Explicit stack size |
    | **BFS** | O(w) | Queue size, w = max width |
    | **Morris Traversal** | O(1) | Constant extra space |
    | **Serialization** | O(n) | String representation size |

=== "🚶 Tree Traversal Methods"

    ## **Traversal Approaches**
    
    > For detailed implementations and examples, see [Tree Traversal](tree-traversal.md)
    
    | Traversal | Order | Applications |
    |-----------|-------|-------------|
    | **Preorder** | Root → Left → Right | Create copy of tree, prefix expressions |
    | **Inorder** | Left → Root → Right | BST gives sorted order, infix expressions |
    | **Postorder** | Left → Right → Root | Delete tree, postfix expressions |
    | **Level Order** | Level by level from top | Level-aware processing, breadth-first |
    
    ## **Implementation Approaches**
    
    - **Recursive**: Simple to implement using function call stack
    - **Iterative**: Using explicit stack/queue, better memory control
    - **Morris Traversal**: O(1) space, uses temporary threading links
    
    ## **Advanced Traversal Techniques**
    
    - **Zigzag/Spiral Traversal**: Alternating direction at each level
    - **Vertical Order**: Nodes grouped by horizontal distance
    - **Boundary Traversal**: Tracing the outline of the tree
    - **Diagonal Traversal**: Nodes along same diagonal lines

=== "🌎 Real-World Applications"

    ## **Trees in Computing**
    
    | Application | Tree Type | Why Trees? |
    |-------------|-----------|------------|
    | **File Systems** | General Trees | Natural hierarchy for directories |
    | **Databases** | B-Trees, B+ Trees | Optimized for disk access, range queries |
    | **Compilers** | AST, Parse Trees | Representing code structure |
    | **Network Routing** | Tries | Fast prefix matching for IP addresses |
    | **Compression** | Huffman Trees | Optimal prefix codes |
    | **Graphics** | Quadtrees, Octrees | Spatial partitioning for rendering |
    | **AI/Game Theory** | Minimax Trees | Decision making, game states |
    | **Machine Learning** | Decision Trees | Classification and regression |
    
    ## **Advantages of Tree Structures**
    
    - **Hierarchical Representation**: Natural for nested or hierarchical data
    - **Efficient Search**: Logarithmic search time in balanced trees
    - **Ordered Data**: Maintains relationships between elements
    - **Flexible Growth**: Can grow and shrink dynamically
    - **Rebalancing**: Self-adjusting structures maintain performance
    
    ## **Tree Data Structure Selection Guide**
    
    - **Need sorted data with frequent lookups?** → BST/AVL/Red-Black Tree
    - **Need to represent hierarchical data?** → General Tree
    - **Need prefix matching for strings?** → Trie
    - **Need range queries/updates?** → Segment Tree
    - **Need to efficiently extract minimums/maximums?** → Heap
    - **Need disk-friendly search structure?** → B-Tree
    - **Need to store huge dictionaries with prefix searching?** → Compressed Trie

=== "💡 Best Practices & Pitfalls"

    ## **Common Implementation Pitfalls**
    
    - **Null/Empty Tree Handling**: Always check if the tree is null/empty
    - **Base Case Definition**: Clear recursion termination conditions
    - **Leaf Node Checks**: Properly identify and handle leaf nodes
    - **Tree Modification**: Preserve structure during modifications
    - **Balance Maintenance**: Rebalance after insertions/deletions
    - **Reference Updates**: Update parent references when restructuring
    - **Memory Management**: Clean up resources to avoid memory leaks
    
    ## **Best Practices**
    
    | Practice | Benefit |
    |----------|---------|
    | **Recursive Helper Functions** | Keep public API clean, manage additional state |
    | **Iterative When Possible** | Avoid stack overflow for deep trees |
    | **Nullability Documentation** | Clear communication about null handling |
    | **Immutable Operations** | Return new trees instead of modifying |
    | **Tree Validation** | Verify structure constraints (BST property, etc.) |
    | **Caching Computed Values** | Avoid redundant calculations |
    | **Bottom-up Processing** | Often more efficient than top-down |
    
    ## **Debugging Techniques**
    
    - **Visual Tree Representation**: Draw the tree for complex cases
    - **Level-by-level Printing**: See the tree structure clearly
    - **Trace Small Examples**: Work through algorithms on small trees
    - **State Tracking**: Monitor value changes during recursion
    - **Invariant Checking**: Verify expected properties at key points

---

## 🔑 Key Takeaways

- **Trees are versatile** data structures that efficiently represent hierarchical relationships
- **Understanding tree properties** helps select the right tree type for specific problems
- **Tree traversals** (see [Tree Traversal](tree-traversal.md)) provide different ways to process tree nodes
- **Self-balancing trees** maintain logarithmic operations regardless of insertion order
- **Special-purpose trees** (B-Trees, Tries, Segment Trees) are optimized for specific use cases
- **Tree problems** follow common patterns that can be recognized and applied
- **Real-world applications** of trees span across multiple domains in computer science

*Master these tree fundamentals to build a strong foundation for advanced tree algorithms and data structures!*
