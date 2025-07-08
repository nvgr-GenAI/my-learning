# Cartesian Trees 🌳📊

## Introduction

A Cartesian Tree is a binary tree derived from a sequence of values, combining properties of binary search trees and heaps. It's especially useful for range minimum/maximum query (RMQ) problems and priority-based data processing.

=== "Overview"
    **Core Concept**:
    
    - Binary tree constructed from an array using both value and position
    - Structure encodes both ordering and priority relationships
    - Inorder traversal yields the original array's indices
    - Tree height is related to the longest increasing/decreasing subsequence
    - Efficiently supports range minimum/maximum queries
    
    **When to Use**:
    
    - Range Minimum/Maximum Queries (RMQ)
    - Least Common Ancestor (LCA) problems
    - Building treaps with implicit keys
    - Solving problems requiring order statistics
    - When processing priority-based workflows
    
    **Time Complexity**:
    
    - Construction: O(n)
    - Range Minimum Query: O(log n) average, O(n) worst case
    - With additional preprocessing: O(1) for RMQ
    - Space: O(n)
    
    **Real-World Applications**:
    
    - Database query optimization
    - Time-series data analysis
    - Computational geometry problems
    - Text processing and searching
    - Network routing protocols

=== "Structure"
    **Key Properties**:
    
    - Every node's value is less than or equal to its children (min-heap property)
    - Inorder traversal of the tree yields the original array indices in order
    - The tree structure is uniquely determined by the input array
    - Root of the tree is the minimum element in the array
    - Parent-child relationship encodes "nearest smaller value" information
    
    **Construction Rules**:
    
    1. The root of the tree is the minimum element of the array
    2. The left subtree is recursively constructed from elements before the minimum
    3. The right subtree is recursively constructed from elements after the minimum
    
    **Visual Representation**:
    
    For array [9, 3, 7, 1, 8, 12, 10]:
    ```
         1
        / \
       /   \
      3     8
     / \     \
    9   7    12
              /
             10
    ```
    
    This tree maintains both the heap property (each node's value ≤ its children) and the inorder traversal property.

=== "Construction"
    **Linear-Time Construction Algorithm**:
    
    Cartesian trees can be constructed in O(n) time using a stack-based approach:
    
    ```python
    def build_cartesian_tree(arr):
        n = len(arr)
        parent = [-1] * n  # Parent indices
        left_child = [-1] * n  # Left child indices
        right_child = [-1] * n  # Right child indices
        
        stack = []  # Stack to track potential parents
        
        for i in range(n):
            last = -1
            
            # Pop elements from stack while they're greater than current element
            while stack and arr[stack[-1]] > arr[i]:
                last = stack.pop()
            
            # Connect current node to the tree
            if stack:
                # Current element becomes right child of the last element in stack
                right_child[stack[-1]] = i
            if last != -1:
                # Last popped element becomes left child of current element
                left_child[i] = last
            
            # Current element becomes parent of the last popped element
            if last != -1:
                parent[last] = i
                
            stack.append(i)
            
        # Identify the root (last element in the stack with no parent)
        root = -1
        for i in range(n):
            if parent[i] == -1:
                root = i
                break
                
        return root, parent, left_child, right_child
    ```
    
    **Object-Oriented Implementation**:
    
    ```python
    class CartesianTreeNode:
        def __init__(self, value, index):
            self.value = value
            self.index = index
            self.left = None
            self.right = None
    
    class CartesianTree:
        def __init__(self, arr):
            self.root = self._build_cartesian_tree(arr)
            
        def _build_cartesian_tree(self, arr):
            n = len(arr)
            if n == 0:
                return None
                
            nodes = [CartesianTreeNode(arr[i], i) for i in range(n)]
            stack = []
            
            for i in range(n):
                last = None
                
                while stack and arr[stack[-1]] > arr[i]:
                    last = stack.pop()
                
                if stack:
                    nodes[stack[-1]].right = nodes[i]
                    
                if last is not None:
                    nodes[i].left = nodes[last]
                    
                stack.append(i)
                
            return nodes[stack[0]] if stack else None
    ```

=== "Applications"
    **Range Minimum Query (RMQ)**:
    
    Cartesian trees can be used for efficient RMQ by:
    
    1. Building the Cartesian tree from the array
    2. Finding the Lowest Common Ancestor (LCA) of the two query indices
    
    ```python
    class RMQSolver:
        def __init__(self, arr):
            self.arr = arr
            self.cartesian_tree = CartesianTree(arr)
            # Preprocess for LCA queries (e.g., using sparse tables or binary lifting)
            self.lca_solver = LCASolver(self.cartesian_tree.root)
            
        def query(self, left, right):
            # Find LCA of nodes at indices left and right
            lca_index = self.lca_solver.find_lca(left, right).index
            return self.arr[lca_index]
    ```
    
    **Longest Increasing Subsequence**:
    
    The height of a Cartesian tree is directly related to the longest increasing subsequence in the array.
    
    **Nearest Smaller Element**:
    
    Finding the nearest smaller element to the left or right is equivalent to finding the parent in a Cartesian tree:
    
    ```python
    def nearest_smaller_element(arr):
        n = len(arr)
        result = [-1] * n  # Initialize with -1 (no smaller element)
        stack = []
        
        for i in range(n):
            while stack and arr[stack[-1]] >= arr[i]:
                stack.pop()
                
            if stack:
                result[i] = stack[-1]
                
            stack.append(i)
            
        return result
    ```

=== "RMQ Optimization"
    **Cartesian Tree + LCA for O(1) RMQ**:
    
    A powerful application of Cartesian trees is achieving O(1) range minimum queries:
    
    1. Build the Cartesian tree in O(n) time
    2. Preprocess the tree for O(1) LCA queries using the Sparse Table method
    3. Answer RMQ by finding the LCA in O(1) time
    
    ```python
    class OptimizedRMQ:
        def __init__(self, arr):
            self.arr = arr
            # Build Cartesian tree
            self.root, self.parent, self.left, self.right = build_cartesian_tree(arr)
            
            # Convert to Euler tour for LCA
            self.euler_tour = []
            self.first_occurrence = [-1] * len(arr)
            self.heights = []
            self._euler_tour(self.root, 0)
            
            # Preprocess with Sparse Table for O(1) LCA
            self.sparse_table = self._build_sparse_table()
            
        def _euler_tour(self, node, height):
            if node == -1:
                return
                
            self.euler_tour.append(node)
            self.heights.append(height)
            
            if self.first_occurrence[node] == -1:
                self.first_occurrence[node] = len(self.euler_tour) - 1
                
            if self.left[node] != -1:
                self._euler_tour(self.left[node], height + 1)
                self.euler_tour.append(node)
                self.heights.append(height)
                
            if self.right[node] != -1:
                self._euler_tour(self.right[node], height + 1)
                self.euler_tour.append(node)
                self.heights.append(height)
                
        def _build_sparse_table(self):
            # Build sparse table for RMQ on heights array
            # Implementation omitted for brevity
            pass
            
        def query(self, left, right):
            # Get first occurrences in Euler tour
            first_left = self.first_occurrence[left]
            first_right = self.first_occurrence[right]
            
            if first_left > first_right:
                first_left, first_right = first_right, first_left
                
            # Query sparse table for minimum height in Euler tour
            min_idx = self._query_sparse_table(first_left, first_right)
            
            # Return array value at the corresponding index
            return self.arr[self.euler_tour[min_idx]]
    ```

=== "Variants"
    **Treap (Tree + Heap)**:
    
    A treap is a randomized binary search tree where:
    - Keys follow BST ordering
    - Priorities (random values) follow heap ordering
    
    Unlike Cartesian trees, treaps use explicit priorities:
    
    ```python
    class TreapNode:
        def __init__(self, key):
            self.key = key
            self.priority = random.random()  # Random priority
            self.left = None
            self.right = None
            
    class Treap:
        def __init__(self):
            self.root = None
            
        # Insert, delete, and other operations...
    ```
    
    **Cartesian Tree Signature**:
    
    A sequence derived from the Cartesian tree structure that can be used for pattern matching:
    
    ```python
    def cartesian_tree_signature(arr):
        n = len(arr)
        parent = [-1] * n
        stack = []
        
        for i in range(n):
            while stack and arr[stack[-1]] > arr[i]:
                stack.pop()
                
            if stack:
                parent[i] = stack[-1]
                
            stack.append(i)
            
        return parent
    ```
    
    **Max Cartesian Tree**:
    
    A variant where the maximum element (not minimum) becomes the root:
    
    ```python
    def build_max_cartesian_tree(arr):
        # Similar to standard Cartesian tree but flip the comparison
        # arr[stack[-1]] < arr[i] instead of arr[stack[-1]] > arr[i]
        pass
    ```

=== "Examples"
    **Example 1: Building a Cartesian Tree**
    
    For the array [9, 3, 7, 1, 8, 12, 10]:
    
    1. Start with an empty tree
    2. Process 9: It becomes root
    3. Process 3: It's smaller than 9, so 9 becomes right child of 3, and 3 becomes new root
    4. Process 7: It's larger than 3, so it becomes right child of 3
    5. Process 1: It's smaller than 3, so 3 becomes right child of 1, and 1 becomes new root
    6. Process 8: It's larger than 1, becomes right child of the rightmost path (after 7)
    7. Continue similarly for 12 and 10
    
    Final tree:
    ```
         1
        / \
       /   \
      3     8
     / \     \
    9   7    12
              /
             10
    ```
    
    **Example 2: RMQ with Cartesian Tree**
    
    For array [9, 3, 7, 1, 8, 12, 10]:
    
    - Query(0, 2): LCA of indices 0 and 2 is node with value 3, so min is 3
    - Query(1, 5): LCA of indices 1 and 5 is node with value 1, so min is 1
    - Query(4, 6): LCA of indices 4 and 6 is node with value 8, so min is 8

=== "Tips"
    **Implementation Tips**:
    
    1. **Use Stack Efficiently**: The stack-based construction algorithm is most efficient
    
    2. **Parent Arrays vs. Nodes**: Choose based on your needs (parent arrays for RMQ, nodes for tree traversal)
    
    3. **Integrate with Other Algorithms**: Combine with LCA algorithms for optimal RMQ
    
    4. **Special Cases**: Handle empty arrays and arrays with duplicate values carefully
    
    5. **Memory Management**: Use parent/child arrays instead of node objects for better cache locality
    
    **Common Pitfalls**:
    
    1. **Incorrect Comparisons**: Be clear whether you're building a min-heap or max-heap variant
    
    2. **Index Confusion**: Keep track of array values vs. array indices carefully
    
    3. **Stack Management**: Ensure proper stack handling when building the tree
    
    4. **LCA Preprocessing**: Most RMQ applications need efficient LCA preprocessing
    
    **Optimization Strategies**:
    
    1. **Sparse Table for LCA**: Use sparse tables for O(1) LCA queries
    
    2. **Skip Euler Tour for Simple Applications**: If you just need basic RMQ, the full preprocessing may be overkill
    
    3. **Iterative vs. Recursive**: Prefer iterative approaches for better performance
    
    4. **Memory Optimization**: Use bit manipulation techniques for sparse tables to save memory
