# Trees - Hard Problems

## 🎯 Learning Objectives

Master the most challenging tree algorithms and advanced tree manipulation techniques. These 10 problems represent the pinnacle of tree-based algorithmic thinking.

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Binary Tree Maximum Path Sum II | Tree DP + Global State | Hard | O(n) | O(h) |
    | 2 | Vertical Order Traversal | Tree DFS + Sorting | Hard | O(n log n) | O(n) |
    | 3 | Binary Tree Cameras | Tree DP + Greedy | Hard | O(n) | O(h) |
    | 4 | Distribute Coins in Binary Tree | Tree DFS + Flow | Hard | O(n) | O(h) |
    | 5 | Longest Univalue Path | Tree DFS + Path Tracking | Hard | O(n) | O(h) |
    | 6 | Binary Tree Postorder Traversal | Tree Morris + Advanced | Hard | O(n) | O(1) |
    | 7 | Count Complete Tree Nodes | Tree Binary Search | Hard | O(log²n) | O(log n) |
    | 8 | Serialize Deserialize N-ary Tree | Tree Encoding | Hard | O(n) | O(n) |
    | 9 | Tree Isomorphism | Tree Canonical Form | Hard | O(n) | O(n) |
    | 10 | Minimum Height Trees | Tree Topological Sort | Hard | O(n) | O(n) |

=== "🎯 Expert Tree Patterns"

    **🧠 Advanced Tree DP:**
    - Multi-state decisions at each node
    - Global optimization with local choices
    
    **🎯 Tree Geometry & Structure:**
    - Spatial relationships and ordering
    - Complete tree properties and optimizations
    
    **🔄 Tree Transformations:**
    - Advanced serialization techniques
    - Canonical representations
    
    **🌐 Multi-Tree Problems:**
    - Forest algorithms and tree collections
    - Graph-tree hybrid approaches

=== "⚡ Expert Interview Strategy"

    **💡 Advanced Problem Recognition:**
    
    - **Multi-constraint optimization**: Complex state spaces
    - **Geometric tree problems**: Coordinate-based analysis
    - **Tree structure exploitation**: Complete/perfect tree properties
    
    **🔄 Expert Techniques:**
    
    1. **Multi-State Tree DP**: Tracking multiple possibilities per node
    2. **Tree Canonical Forms**: Unique representations for comparison
    3. **Advanced Morris Traversal**: O(1) space complex traversals
    4. **Tree-Graph Hybrid**: Using graph algorithms on trees

---

## Problem 1: Binary Tree Cameras

**Difficulty**: 🔴 Hard  
**Pattern**: Tree DP + Greedy with Multi-State  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Install the minimum number of cameras to monitor all nodes. Each camera monitors its parent, itself, and immediate children.

    **States:**
    - 0: Node needs to be monitored
    - 1: Node has a camera  
    - 2: Node is monitored but has no camera

=== "Optimal Solution"

    ```python
    def min_camera_cover(root):
        """
        Tree DP with three states: need_monitor, has_camera, monitored.
        """
        def dfs(node):
            if not node:
                return 2  # null nodes are considered monitored
            
            left = dfs(node.left)
            right = dfs(node.right)
            
            # If any child needs monitoring, place camera here
            if left == 0 or right == 0:
                self.cameras += 1
                return 1
            
            # If any child has camera, this node is monitored
            if left == 1 or right == 1:
                return 2
            
            # Both children are monitored, this node needs monitoring
            return 0
        
        self.cameras = 0
        
        # If root needs monitoring, place camera there
        if dfs(root) == 0:
            self.cameras += 1
        
        return self.cameras
    ```

=== "State Analysis"

    ```python
    # State transitions:
    # 0 (need monitor) -> parent must place camera
    # 1 (has camera) -> monitors self, parent, children
    # 2 (monitored) -> covered by camera elsewhere
    
    def analyze_states(left_state, right_state):
        """
        Decision matrix for camera placement.
        """
        # Child needs monitoring -> place camera here
        if left_state == 0 or right_state == 0:
            return 1  # has_camera
        
        # Child has camera -> we're monitored
        if left_state == 1 or right_state == 1:
            return 2  # monitored
        
        # Both children monitored -> we need monitoring
        return 0  # need_monitor
    ```

---

## Problem 2: Vertical Order Traversal of Binary Tree

**Difficulty**: 🔴 Hard  
**Pattern**: Tree DFS + Custom Sorting  
**Time**: O(n log n), **Space**: O(n)

=== "Problem Statement"

    Return vertical order traversal where nodes in same column are ordered by row, then by value.

=== "Optimal Solution"

    ```python
    def vertical_traversal(root):
        """
        DFS with coordinate tracking and custom sorting.
        """
        def dfs(node, row, col):
            if not node:
                return
            
            nodes.append((col, row, node.val))
            dfs(node.left, row + 1, col - 1)
            dfs(node.right, row + 1, col + 1)
        
        nodes = []
        dfs(root, 0, 0)
        
        # Sort by column, then row, then value
        nodes.sort(key=lambda x: (x[0], x[1], x[2]))
        
        # Group by column
        from collections import defaultdict
        columns = defaultdict(list)
        for col, row, val in nodes:
            columns[col].append(val)
        
        # Return in column order
        return [columns[col] for col in sorted(columns.keys())]

    def vertical_traversal_optimized(root):
        """
        More efficient grouping during traversal.
        """
        from collections import defaultdict
        
        def dfs(node, row, col):
            if not node:
                return
            
            column_map[col].append((row, node.val))
            dfs(node.left, row + 1, col - 1)
            dfs(node.right, row + 1, col + 1)
        
        column_map = defaultdict(list)
        dfs(root, 0, 0)
        
        result = []
        for col in sorted(column_map.keys()):
            # Sort by row, then by value
            column_nodes = sorted(column_map[col])
            result.append([val for row, val in column_nodes])
        
        return result
    ```

---

## Problem 3: Distribute Coins in Binary Tree

**Difficulty**: 🔴 Hard  
**Pattern**: Tree DFS + Flow Calculation  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Each node has some coins. Move coins so each node has exactly 1 coin. Find minimum moves (moving 1 coin between adjacent nodes = 1 move).

=== "Optimal Solution"

    ```python
    def distribute_coins(root):
        """
        Calculate coin flow between parent and child.
        """
        def dfs(node):
            if not node:
                return 0
            
            left_flow = dfs(node.left)
            right_flow = dfs(node.right)
            
            # Total moves = absolute flow from children
            self.moves += abs(left_flow) + abs(right_flow)
            
            # Net flow from this subtree to parent
            # = coins in subtree - nodes in subtree
            return node.val + left_flow + right_flow - 1
        
        self.moves = 0
        dfs(root)
        return self.moves

    def distribute_coins_with_explanation(root):
        """
        Detailed version with flow explanation.
        """
        def dfs(node):
            if not node:
                return 0, 0  # coins, nodes
            
            left_coins, left_nodes = dfs(node.left)
            right_coins, right_nodes = dfs(node.right)
            
            # Flow = excess/deficit coins in subtree
            left_flow = left_coins - left_nodes
            right_flow = right_coins - right_nodes
            
            # Moves = coins that must flow through this node
            self.moves += abs(left_flow) + abs(right_flow)
            
            # Return total coins and nodes in this subtree
            total_coins = node.val + left_coins + right_coins
            total_nodes = 1 + left_nodes + right_nodes
            
            return total_coins, total_nodes
        
        self.moves = 0
        dfs(root)
        return self.moves
    ```

---

## Problem 4: Longest Univalue Path

**Difficulty**: 🔴 Hard  
**Pattern**: Tree DFS + Path Extension  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Find the length of the longest path where all nodes have the same value.

=== "Optimal Solution"

    ```python
    def longest_univalue_path(root):
        """
        For each node, calculate longest univalue path through it.
        """
        def dfs(node):
            if not node:
                return 0
            
            left_length = dfs(node.left)
            right_length = dfs(node.right)
            
            # Reset lengths if values don't match
            if node.left and node.left.val != node.val:
                left_length = 0
            if node.right and node.right.val != node.val:
                right_length = 0
            
            # Path through this node
            path_through_node = left_length + right_length
            self.max_path = max(self.max_path, path_through_node)
            
            # Return max path ending at this node
            return max(left_length, right_length) + 1
        
        self.max_path = 0
        dfs(root)
        return self.max_path
    ```

---

## Problem 5: Count Complete Tree Nodes

**Difficulty**: 🔴 Hard  
**Pattern**: Tree Binary Search + Complete Tree Properties  
**Time**: O(log²n), **Space**: O(log n)

=== "Problem Statement"

    Count nodes in a complete binary tree more efficiently than O(n).

=== "Optimal Solution"

    ```python
    def count_nodes(root):
        """
        Use complete tree properties for O(log²n) solution.
        """
        if not root:
            return 0
        
        def get_height(node, go_left=True):
            height = 0
            while node:
                height += 1
                node = node.left if go_left else node.right
            return height
        
        left_height = get_height(root, True)
        right_height = get_height(root, False)
        
        # If heights equal, tree is perfect
        if left_height == right_height:
            return (1 << left_height) - 1  # 2^h - 1
        
        # Otherwise, recurse on subtrees
        return 1 + count_nodes(root.left) + count_nodes(root.right)

    def count_nodes_binary_search(root):
        """
        Binary search approach on last level.
        """
        if not root:
            return 0
        
        def get_depth(node):
            depth = 0
            while node:
                depth += 1
                node = node.left
            return depth
        
        def exists(idx, depth, node):
            """Check if node at index exists in last level."""
            left, right = 0, (1 << depth) - 1
            
            for _ in range(depth):
                mid = (left + right) // 2
                if idx <= mid:
                    node = node.left
                    right = mid
                else:
                    node = node.right
                    left = mid + 1
            
            return node is not None
        
        depth = get_depth(root)
        if depth == 1:
            return 1
        
        # Binary search on last level
        left, right = 0, (1 << (depth - 1)) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if exists(mid, depth - 1, root):
                left = mid + 1
            else:
                right = mid - 1
        
        # Full levels + nodes in last level
        return (1 << (depth - 1)) - 1 + left
    ```

---

## Problem 6: Binary Tree Postorder Traversal (Morris)

**Difficulty**: 🔴 Hard  
**Pattern**: Morris Traversal + Advanced Pointer Manipulation  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Implement postorder traversal with O(1) space using Morris traversal.

=== "Optimal Solution"

    ```python
    def postorder_traversal_morris(root):
        """
        Morris postorder traversal with O(1) space.
        """
        def reverse_path(start, end):
            """Reverse the path from start to end."""
            if start == end:
                return
            
            prev, curr = None, start
            while curr != end:
                next_node = curr.right
                curr.right = prev
                prev = curr
                curr = next_node
            end.right = prev
        
        def print_reverse_path(start, end, result):
            """Print path in reverse order."""
            reverse_path(start, end)
            
            curr = end
            while True:
                result.append(curr.val)
                if curr == start:
                    break
                curr = curr.right
            
            reverse_path(end, start)
        
        dummy = TreeNode(0)
        dummy.left = root
        current = dummy
        result = []
        
        while current:
            if not current.left:
                current = current.right
            else:
                predecessor = current.left
                
                # Find rightmost node in left subtree
                while predecessor.right and predecessor.right != current:
                    predecessor = predecessor.right
                
                if not predecessor.right:
                    # Create thread
                    predecessor.right = current
                    current = current.left
                else:
                    # Remove thread and print
                    predecessor.right = None
                    print_reverse_path(current.left, predecessor, result)
                    current = current.right
        
        return result
    ```

---

## Problem 7: Serialize and Deserialize N-ary Tree

**Difficulty**: 🔴 Hard  
**Pattern**: Tree Encoding + Advanced Parsing  
**Time**: O(n), **Space**: O(n)

=== "Problem Statement"

    Design serialization for N-ary trees where each node can have any number of children.

=== "Optimal Solution"

    ```python
    def serialize(root):
        """
        Serialize N-ary tree with child count encoding.
        """
        def preorder(node):
            if not node:
                return
            
            result.append(str(node.val))
            result.append(str(len(node.children)))
            
            for child in node.children:
                preorder(child)
        
        result = []
        preorder(root)
        return ",".join(result)

    def deserialize(data):
        """
        Deserialize using child count information.
        """
        if not data:
            return None
        
        tokens = data.split(",")
        self.index = 0
        
        def build():
            if self.index >= len(tokens):
                return None
            
            val = int(tokens[self.index])
            self.index += 1
            
            child_count = int(tokens[self.index])
            self.index += 1
            
            node = Node(val)
            node.children = []
            
            for _ in range(child_count):
                child = build()
                if child:
                    node.children.append(child)
            
            return node
        
        return build()

    def serialize_level_order(root):
        """
        Alternative: Level-order serialization.
        """
        if not root:
            return ""
        
        from collections import deque
        queue = deque([root])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(str(node.val))
            result.append(str(len(node.children)))
            
            for child in node.children:
                queue.append(child)
        
        return ",".join(result)
    ```

---

## Problem 8: Tree Isomorphism

**Difficulty**: 🔴 Hard  
**Pattern**: Tree Canonical Form + Hashing  
**Time**: O(n), **Space**: O(n)

=== "Problem Statement"

    Determine if two trees are isomorphic (same structure, possibly different node arrangements).

=== "Optimal Solution"

    ```python
    def is_isomorphic(root1, root2):
        """
        Check isomorphism using canonical hash.
        """
        def get_canonical_hash(node):
            if not node:
                return "null"
            
            # Get hashes of all children
            child_hashes = []
            if node.left:
                child_hashes.append(get_canonical_hash(node.left))
            if node.right:
                child_hashes.append(get_canonical_hash(node.right))
            
            # Sort to make canonical
            child_hashes.sort()
            
            # Create hash for this subtree
            return f"({','.join(child_hashes)})"
        
        return get_canonical_hash(root1) == get_canonical_hash(root2)

    def is_isomorphic_with_values(root1, root2):
        """
        Isomorphism considering node values.
        """
        def get_structure_hash(node):
            if not node:
                return "null"
            
            left_hash = get_structure_hash(node.left)
            right_hash = get_structure_hash(node.right)
            
            # Create canonical representation
            children = sorted([left_hash, right_hash])
            return f"{node.val}({children[0]},{children[1]})"
        
        return get_structure_hash(root1) == get_structure_hash(root2)
    ```

---

## Problem 9: Minimum Height Trees

**Difficulty**: 🔴 Hard  
**Pattern**: Tree Topological Sort + Centroid Finding  
**Time**: O(n), **Space**: O(n)

=== "Problem Statement"

    Given a tree (connected acyclic graph), find all nodes that could be roots of minimum height trees.

=== "Optimal Solution"

    ```python
    def find_min_height_trees(n, edges):
        """
        Find centroids using topological sort approach.
        """
        if n == 1:
            return [0]
        
        # Build adjacency list
        from collections import defaultdict, deque
        graph = defaultdict(list)
        degree = [0] * n
        
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)
            degree[a] += 1
            degree[b] += 1
        
        # Start with leaf nodes (degree 1)
        queue = deque()
        for i in range(n):
            if degree[i] == 1:
                queue.append(i)
        
        remaining = n
        
        # Remove leaf nodes layer by layer
        while remaining > 2:
            leaf_count = len(queue)
            remaining -= leaf_count
            
            for _ in range(leaf_count):
                leaf = queue.popleft()
                
                # Remove leaf and update neighbors
                for neighbor in graph[leaf]:
                    degree[neighbor] -= 1
                    if degree[neighbor] == 1:
                        queue.append(neighbor)
        
        # Remaining nodes are centroids
        return list(queue)

    def find_min_height_trees_diameter(n, edges):
        """
        Alternative: Find diameter endpoints and take middle.
        """
        if n == 1:
            return [0]
        
        # Build adjacency list
        from collections import defaultdict, deque
        graph = defaultdict(list)
        
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)
        
        def bfs_farthest(start):
            """Find farthest node from start."""
            visited = set([start])
            queue = deque([(start, 0)])
            farthest_node, max_dist = start, 0
            
            while queue:
                node, dist = queue.popleft()
                
                if dist > max_dist:
                    max_dist = dist
                    farthest_node = node
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            
            return farthest_node, max_dist
        
        def get_path(start, end):
            """Get path from start to end."""
            parent = {start: None}
            queue = deque([start])
            
            while queue:
                node = queue.popleft()
                if node == end:
                    break
                
                for neighbor in graph[node]:
                    if neighbor not in parent:
                        parent[neighbor] = node
                        queue.append(neighbor)
            
            # Reconstruct path
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = parent[current]
            
            return path[::-1]
        
        # Find diameter endpoints
        end1, _ = bfs_farthest(0)
        end2, diameter = bfs_farthest(end1)
        
        # Get diameter path
        diameter_path = get_path(end1, end2)
        
        # Return middle node(s)
        mid = len(diameter_path) // 2
        if len(diameter_path) % 2 == 1:
            return [diameter_path[mid]]
        else:
            return [diameter_path[mid-1], diameter_path[mid]]
    ```

---

## Problem 10: Binary Tree Maximum Path Sum (All Variants)

**Difficulty**: 🔴 Hard  
**Pattern**: Tree DP + Multiple Constraints  
**Time**: O(n), **Space**: O(h)

=== "Problem Statement"

    Solve multiple variants of maximum path sum problem with different constraints.

=== "Optimal Solution"

    ```python
    def max_path_sum_any_to_any(root):
        """
        Classic: Maximum path sum between any two nodes.
        """
        def dfs(node):
            if not node:
                return 0
            
            left_max = max(0, dfs(node.left))
            right_max = max(0, dfs(node.right))
            
            # Path through current node
            path_through_node = node.val + left_max + right_max
            self.max_sum = max(self.max_sum, path_through_node)
            
            # Return max path ending at this node
            return node.val + max(left_max, right_max)
        
        self.max_sum = float('-inf')
        dfs(root)
        return self.max_sum

    def max_path_sum_leaf_to_leaf(root):
        """
        Variant: Maximum path sum from leaf to leaf.
        """
        def dfs(node):
            if not node:
                return float('-inf')
            
            # If leaf node
            if not node.left and not node.right:
                return node.val
            
            left_max = dfs(node.left)
            right_max = dfs(node.right)
            
            # If both children exist, consider path through node
            if node.left and node.right:
                path_through = node.val + left_max + right_max
                self.max_sum = max(self.max_sum, path_through)
                return node.val + max(left_max, right_max)
            
            # Only one child exists
            child_max = left_max if node.left else right_max
            return node.val + child_max
        
        self.max_sum = float('-inf')
        dfs(root)
        return self.max_sum

    def max_path_sum_root_to_leaf(root):
        """
        Variant: Maximum path sum from root to any leaf.
        """
        def dfs(node):
            if not node:
                return 0
            
            if not node.left and not node.right:
                return node.val
            
            left_max = dfs(node.left) if node.left else float('-inf')
            right_max = dfs(node.right) if node.right else float('-inf')
            
            return node.val + max(left_max, right_max)
        
        return dfs(root)
    ```

---

## 🎯 Expert Practice Summary

### Master-Level Tree Patterns

1. **Multi-State Tree DP**: Complex decision trees with multiple states per node
2. **Tree Flow Problems**: Calculating optimal resource distribution
3. **Geometric Tree Problems**: Coordinate-based tree analysis
4. **Tree Canonical Forms**: Unique representations for comparison
5. **Advanced Morris Traversal**: O(1) space complex operations
6. **Tree-Graph Hybrids**: Applying graph algorithms to tree structures
7. **Complete Tree Optimizations**: Exploiting perfect/complete tree properties

### Advanced Algorithmic Techniques

- **Multi-State DP**: Tracking multiple possibilities at each node
- **Flow Calculation**: Understanding resource movement in trees
- **Topological Sorting**: Finding centroids and important nodes
- **Binary Search on Trees**: Exploiting tree structure for efficiency
- **Canonical Hashing**: Creating unique tree representations

### Complexity Analysis Mastery

- **Time Optimization**: From O(n) to O(log n) using tree properties
- **Space Optimization**: Morris traversal for O(1) space solutions
- **Amortized Analysis**: Understanding average-case performance
- **Tree Height Impact**: Balanced vs unbalanced complexity differences

### Expert Interview Strategy

1. **Pattern Recognition**: Identify multi-constraint optimization problems
2. **State Design**: Choose appropriate state representation for DP
3. **Property Exploitation**: Use complete/perfect tree properties when available
4. **Space-Time Tradeoffs**: Consider Morris traversal for space constraints
5. **Edge Case Mastery**: Handle degenerate trees and special structures

### Real-World Applications

- **Database Indexing**: B-trees and balanced tree operations
- **Computer Graphics**: Spatial partitioning trees
- **Network Routing**: Spanning tree algorithms
- **Compiler Design**: Abstract syntax trees and optimization
- **Game Development**: Decision trees and pathfinding

---

*These hard tree problems represent the pinnacle of tree algorithmic thinking. Master these to handle any tree-based challenge in technical interviews or competitive programming!*
