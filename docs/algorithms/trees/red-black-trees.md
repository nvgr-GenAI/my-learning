# Red-Black Trees 🌳🔴⚫

## Introduction

Red-Black Trees are self-balancing binary search trees with a set of properties that ensure a logarithmic height, making operations like insertion, deletion, and search efficient.

=== "Overview"
    **Core Concept**:
    
    - Self-balancing binary search tree with color properties
    - Each node is either red or black
    - Balance is maintained through a set of color rules and rotations
    - Guarantees O(log n) height for n nodes
    
    **When to Use**:
    
    - When you need guaranteed O(log n) operations
    - When frequent insertions and deletions occur
    - When memory overhead is a concern (compared to AVL trees)
    - For implementing maps and sets in many programming languages
    
    **Time Complexity**:
    
    - Search: O(log n)
    - Insert: O(log n)
    - Delete: O(log n)
    
    **Real-World Applications**:
    
    - Java's TreeMap and TreeSet
    - C++'s std::map and std::set
    - Linux kernel's completely fair scheduler
    - Database indexing systems

=== "Properties"
    **Red-Black Tree Rules**:
    
    1. Every node is either red or black
    2. The root is always black
    3. All leaves (NIL/NULL nodes) are black
    4. If a node is red, both its children must be black (no consecutive red nodes)
    5. For each node, all simple paths from the node to descendant leaves contain the same number of black nodes (black height)
    
    **Key Metrics**:
    
    - **Black Height**: The number of black nodes on any path from root to leaf (not counting the root if it's red)
    - **Height Guarantee**: A red-black tree with n nodes has height ≤ 2log₂(n+1)

=== "Operations"
    **Insertion**:
    
    1. Insert the node using standard BST insertion
    2. Color the new node red
    3. Fix any violations of red-black properties through recoloring and rotations
    
    ```python
    def insert(self, key):
        new_node = Node(key)
        new_node.color = RED  # Start with red
        
        # Standard BST insert
        self._bst_insert(new_node)
        
        # Fix red-black properties
        self._fix_insert(new_node)
    
    def _fix_insert(self, node):
        # Fix violations while node is not root and node's parent is red
        while node != self.root and node.parent.color == RED:
            # Cases based on whether parent is left or right child of grandparent
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                
                # Case 1: Uncle is red - recolor
                if uncle.color == RED:
                    node.parent.color = BLACK
                    uncle.color = BLACK
                    node.parent.parent.color = RED
                    node = node.parent.parent
                else:
                    # Case 2: Node is right child - left rotation
                    if node == node.parent.right:
                        node = node.parent
                        self._left_rotate(node)
                    
                    # Case 3: Node is left child - right rotation
                    node.parent.color = BLACK
                    node.parent.parent.color = RED
                    self._right_rotate(node.parent.parent)
            else:
                # Similar logic for right child case (symmetric)
                # ...
        
        # Ensure root is black
        self.root.color = BLACK
    ```
    
    **Deletion**:
    
    1. Delete the node using standard BST deletion
    2. If the removed node was black, fix the double-black violation
    
    **Searching**: Standard binary search tree search algorithm

=== "Rotations"
    **Left Rotation**:
    
    ```
        A                B
       / \              / \
      α   B     →      A   γ
         / \          / \
        β   γ        α   β
    ```
    
    ```python
    def _left_rotate(self, x):
        y = x.right
        x.right = y.left
        
        if y.left != self.NIL:
            y.left.parent = x
            
        y.parent = x.parent
        
        if x.parent is None:
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
          B                A
         / \              / \
        A   γ     →      α   B
       / \                  / \
      α   β                β   γ
    ```
    
    ```python
    def _right_rotate(self, y):
        x = y.left
        y.left = x.right
        
        if x.right != self.NIL:
            x.right.parent = y
            
        x.parent = y.parent
        
        if y.parent is None:
            self.root = x
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
            
        x.right = y
        y.parent = x
    ```

=== "Examples"
    **Example 1: Building a Red-Black Tree**
    
    Let's insert 7, 3, 18, 10, 22, 8, 11, 26 into a red-black tree:
    
    1. Insert 7: Root becomes black
    2. Insert 3: Add as red left child of 7
    3. Insert 18: Add as red right child of 7
    4. Insert 10: Add as red left child of 18
    5. Insert 22: Add as red right child of 18, causes recoloring
    6. Insert 8: Requires rotation and recoloring
    7. Insert 11: Requires rotation and recoloring
    8. Insert 26: Add as red right child of 22
    
    Final tree maintains all red-black properties with balanced structure.
    
    **Example 2: Deletion**
    
    Deleting 18 from the above tree:
    
    1. Find successor (20)
    2. Replace 18 with 20
    3. Fix any violations with rotations/recoloring
    4. Maintain black height property

=== "Comparison"
    **Red-Black Trees vs AVL Trees**:
    
    | Aspect | Red-Black Trees | AVL Trees |
    |--------|-----------------|-----------|
    | **Balance** | Less strict (height ≤ 2log n) | Stricter (heights differ by ≤1) |
    | **Insert/Delete** | Faster (fewer rotations) | Slower (more rotations) |
    | **Search** | Slightly slower | Slightly faster |
    | **Memory** | Color bit per node | Balance factor per node |
    | **Use Case** | Frequent updates | Frequent lookups |
    
    **Red-Black Trees vs B-Trees**:
    
    | Aspect | Red-Black Trees | B-Trees |
    |--------|-----------------|---------|
    | **Branching** | Binary (2) | Multi-way (≥2) |
    | **Memory Access** | Not optimized | Optimized for disk I/O |
    | **Use Case** | In-memory operations | Disk-based databases |
    | **Height** | O(log n) | O(logₘ n) where m is order |

=== "Tips"
    **Implementation Tips**:
    
    1. Use sentinel NIL nodes instead of NULL pointers
    2. Always track parent pointers for efficient rotations
    3. Consider using the left-leaning variant for simpler implementation
    4. Double-check all edge cases, especially around the root
    
    **Common Pitfalls**:
    
    1. Forgetting to update parent pointers during rotations
    2. Incorrect handling of the root node (should always be black)
    3. Missing edge cases during fixup procedures
    4. Not preserving the black height invariant after deletion
    
    **Optimization Techniques**:
    
    1. Use sentinel NIL node pattern for simpler boundary checks
    2. Consider bulk operations for batch insertions
    3. For lookups in specific patterns, consider augmenting the tree
