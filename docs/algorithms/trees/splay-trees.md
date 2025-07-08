# Splay Trees 🌳↕️

## Introduction

Splay Trees are self-adjusting binary search trees that move recently accessed nodes closer to the root, providing efficient access to frequently used elements.

=== "Overview"
    **Core Concept**:
    
    - Self-adjusting binary search tree
    - Recently accessed elements are moved to the root via "splaying"
    - No explicit balance criteria, but achieves amortized O(log n) operations
    - Automatically adapts to access patterns
    
    **When to Use**:
    
    - When access patterns show temporal locality (same items accessed frequently)
    - For implementing caches or most-recently-used lists
    - When simpler implementation is preferred over AVL or Red-Black trees
    - For applications where recently accessed items are likely to be accessed again
    
    **Time Complexity**:
    
    - Access/Search: O(log n) amortized
    - Insert: O(log n) amortized
    - Delete: O(log n) amortized
    - Worst case for individual operation: O(n)
    
    **Real-World Applications**:
    
    - Cache implementations
    - Garbage collection algorithms
    - Network routing tables
    - Recently used file lists

=== "Structure"
    **Key Properties**:
    
    - Standard binary search tree property (left < node < right)
    - No explicit balance criteria or color properties
    - Tree shape adapts dynamically to access patterns
    - Every operation includes a "splay" step that brings the accessed node to the root
    
    **Splay Operation**:
    
    The core of splay trees is the splay operation, which uses three main steps:
    
    1. **Zig**: When the node is a direct child of the root
    2. **Zig-Zig**: When the node and its parent are both left/right children
    3. **Zig-Zag**: When the node is a left child and its parent is a right child (or vice versa)
    
    Through repeated application of these steps, the accessed node becomes the new root.

=== "Operations"
    **Splay Operation**:
    
    ```python
    def splay(self, node):
        while node.parent:
            parent = node.parent
            grandparent = parent.parent
            
            if not grandparent:  # Zig case
                if node == parent.left:
                    self._right_rotate(parent)
                else:
                    self._left_rotate(parent)
            
            elif parent == grandparent.left:
                if node == parent.left:  # Zig-Zig (left-left)
                    self._right_rotate(grandparent)
                    self._right_rotate(parent)
                else:  # Zig-Zag (left-right)
                    self._left_rotate(parent)
                    self._right_rotate(grandparent)
            
            else:  # parent is right child of grandparent
                if node == parent.right:  # Zig-Zig (right-right)
                    self._left_rotate(grandparent)
                    self._left_rotate(parent)
                else:  # Zig-Zag (right-left)
                    self._right_rotate(parent)
                    self._left_rotate(grandparent)
    ```
    
    **Search**:
    
    ```python
    def search(self, key):
        node = self._bst_search(self.root, key)
        if node:
            self.splay(node)
            return node
        return None
    ```
    
    **Insert**:
    
    ```python
    def insert(self, key):
        if not self.root:
            self.root = Node(key)
            return
        
        # Standard BST insert
        node = self._bst_insert(key)
        
        # Splay the newly inserted node to root
        self.splay(node)
    ```
    
    **Delete**:
    
    ```python
    def delete(self, key):
        if not self.root:
            return
        
        # First splay the node to delete to root
        self.search(key)
        
        if self.root.key != key:
            return  # Key not found
        
        if not self.root.left:
            self.root = self.root.right
            if self.root:
                self.root.parent = None
        elif not self.root.right:
            self.root = self.root.left
            self.root.parent = None
        else:
            # Find the largest in left subtree
            temp = self.root.left
            while temp.right:
                temp = temp.right
            
            # Splay this node to be root of left subtree
            self.splay(temp)
            
            # Connect right subtree
            temp.right = self.root.right
            self.root.right.parent = temp
            
            self.root = temp
            self.root.parent = None
    ```

=== "Rotations"
    **Left Rotation**:
    
    ```
        X                Y
       / \              / \
      α   Y     →      X   γ
         / \          / \
        β   γ        α   β
    ```
    
    ```python
    def _left_rotate(self, x):
        y = x.right
        x.right = y.left
        
        if y.left:
            y.left.parent = x
            
        y.parent = x.parent
        
        if not x.parent:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
            
        y.left = x
        x.parent = y
    ```
    
    **Right Rotation**:
    
    ```
          Y                X
         / \              / \
        X   γ     →      α   Y
       / \                  / \
      α   β                β   γ
    ```
    
    ```python
    def _right_rotate(self, y):
        x = y.left
        y.left = x.right
        
        if x.right:
            x.right.parent = y
            
        x.parent = y.parent
        
        if not y.parent:
            self.root = x
        elif y == y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x
            
        x.right = y
        y.parent = x
    ```

=== "Examples"
    **Example 1: Searching in a Splay Tree**
    
    Consider a splay tree with elements [10, 5, 15, 3, 7, 12, 17]:
    
    ```
          10
         /  \
        5    15
       / \   / \
      3   7 12  17
    ```
    
    When we search for element 7:
    
    1. Find node with key 7
    2. Since 7's parent is 5, and 5's parent is 10, we have a Zig-Zag case:
       - Right rotate 5 (7 becomes child of 10)
       - Left rotate 10
    3. After splaying, 7 is the new root:
    
    ```
          7
         / \
        5   10
       /     \
      3       15
             /  \
            12   17
    ```
    
    **Example 2: Repeated Access Pattern**
    
    If we repeatedly access elements 3 and 5:
    
    1. First access 3: After splaying, 3 becomes root
    2. Then access 5: After splaying, 5 becomes root
    3. Access 3 again: After splaying, 3 becomes root
    
    Notice that frequent accesses to 3 and 5 keep them near the top of the tree, making subsequent accesses faster.

=== "Comparison"
    **Splay Trees vs Red-Black Trees**:
    
    | Aspect | Splay Trees | Red-Black Trees |
    |--------|-------------|-----------------|
    | **Balance** | Self-adjusting, no explicit balance | Strictly balanced via color rules |
    | **Operation Time** | O(log n) amortized, O(n) worst case | O(log n) worst case |
    | **Memory** | No extra fields needed | Color bit per node |
    | **Access Pattern** | Adapts to usage patterns | Fixed structure regardless of usage |
    | **Implementation** | Simpler | More complex |
    
    **Splay Trees vs AVL Trees**:
    
    | Aspect | Splay Trees | AVL Trees |
    |--------|-------------|-----------|
    | **Balance** | Self-adjusting | Height-balanced (stricter) |
    | **Insertion/Deletion** | Simpler, adaptive | More rotations needed |
    | **Lookup** | Adapts to access patterns | Fixed O(log n) |
    | **Use Case** | Frequent repeated accesses | General-purpose balanced BST |

=== "Tips"
    **Implementation Tips**:
    
    1. Always track parent pointers for efficient splaying
    2. Consider a top-down splay implementation for better performance
    3. Use sentinel nodes to simplify boundary cases
    4. For frequently used items, splay trees often outperform other BSTs
    
    **Common Pitfalls**:
    
    1. Forgetting to update parent pointers during rotations
    2. Incorrect handling of the tree root during splaying
    3. Not considering the amortized cost in performance analysis
    4. Using splay trees when access patterns are random (no temporal locality)
    
    **When to Prefer Splay Trees**:
    
    1. When access patterns show strong locality of reference
    2. When recently accessed items are likely to be accessed again soon
    3. When simplicity is preferred over strict worst-case guarantees
    4. For implementing caches or MRU (Most Recently Used) data structures
