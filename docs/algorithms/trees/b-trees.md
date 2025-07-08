# B-Trees 🌳📊

## Introduction

B-Trees are self-balancing search trees designed to work efficiently with block storage devices such as disks. Unlike binary trees, each node can have multiple keys and children, making them ideal for systems that read/write large blocks of data.

=== "Overview"
    **Core Concept**:
    
    - Multi-way search tree (each node has multiple keys and children)
    - Designed for storage systems with large blocks (disks, SSDs)
    - All leaf nodes at the same level (perfect height balance)
    - Keys within each node are stored in sorted order
    - Optimized for reducing disk access operations
    
    **When to Use**:
    
    - Database indexing and file systems
    - When data is stored on disk rather than in memory
    - When minimizing I/O operations is critical
    - For handling large datasets that don't fit in memory
    
    **Time Complexity**:
    
    - Search: O(log_t n) where t is the minimum degree
    - Insert: O(log_t n)
    - Delete: O(log_t n)
    - Space: O(n)
    
    **Real-World Applications**:
    
    - Database management systems (MySQL, PostgreSQL, etc.)
    - File systems (NTFS, HFS+, ext4)
    - Key-value stores
    - Search indices in information retrieval systems

=== "Structure"
    **B-Tree Properties**:
    
    For a B-Tree of minimum degree t (t ≥ 2):
    
    1. Every node has at most 2t-1 keys
    2. Every non-leaf node (except root) has at least t children
    3. If the root is not a leaf, it has at least 2 children
    4. All leaves appear at the same level
    5. A non-leaf node with k keys has k+1 children
    
    **Node Structure**:
    
    Each node contains:
    - n keys (K₁, K₂, ..., Kₙ) stored in non-decreasing order
    - n+1 pointers to children (if non-leaf node)
    - Boolean indicating whether it's a leaf node
    
    **Example B-Tree Node (t=3)**:
    ```
    [P₁ | K₁ | P₂ | K₂ | P₃ | K₃ | P₄]
    ```
    Where K's are keys and P's are pointers to children.

=== "Operations"
    **Search**:
    
    ```python
    def search(self, k, x=None):
        if x is None:
            x = self.root
        i = 0
        while i < len(x.keys) and k > x.keys[i]:
            i += 1
        if i < len(x.keys) and k == x.keys[i]:
            return (x, i)  # Found
        if x.leaf:
            return None  # Not found
        return self.search(k, x.children[i])  # Recursively search appropriate child
    ```
    
    **Insert**:
    
    1. If tree is empty, create a new root node and insert the key
    2. If the root is full (has 2t-1 keys):
       - Split the root
       - Create a new root with the median key
       - The original root splits into two nodes
    3. Call insert_non_full recursively
    
    ```python
    def insert(self, k):
        root = self.root
        if len(root.keys) == (2 * self.t) - 1:  # Root is full
            # Split the root
            new_root = Node(leaf=False)
            self.root = new_root
            new_root.children.append(root)
            self._split_child(new_root, 0)
            self._insert_non_full(new_root, k)
        else:
            self._insert_non_full(root, k)
    
    def _insert_non_full(self, x, k):
        i = len(x.keys) - 1
        
        if x.leaf:  # If node is leaf, insert key at appropriate position
            # Find position for new key
            while i >= 0 and k < x.keys[i]:
                i -= 1
            x.keys.insert(i + 1, k)
        else:
            # Find child which is going to have the new key
            while i >= 0 and k < x.keys[i]:
                i -= 1
            i += 1
            
            # If child is full, split it
            if len(x.children[i].keys) == (2 * self.t) - 1:
                self._split_child(x, i)
                if k > x.keys[i]:
                    i += 1
            
            self._insert_non_full(x.children[i], k)
    
    def _split_child(self, x, i):
        t = self.t
        y = x.children[i]
        z = Node(leaf=y.leaf)
        
        # Move right half of y's keys to z
        z.keys = y.keys[t:]
        y.keys = y.keys[:t-1]
        
        # If not leaf, move right half of y's children to z
        if not y.leaf:
            z.children = y.children[t:]
            y.children = y.children[:t]
        
        # Insert a key from y to x
        x.keys.insert(i, y.keys[t-1])
        x.children.insert(i+1, z)
    ```
    
    **Delete**:
    
    Deletion is complex with multiple cases to handle:
    1. If the key is in a leaf node with enough keys, simply remove it
    2. If the key is in an internal node, replace it with its predecessor/successor
    3. If nodes don't have enough keys after removal, merge or redistribute keys

=== "Variations"
    **B+ Tree**:
    
    A variation of B-Tree commonly used in databases and file systems:
    
    - Only stores keys in internal nodes, data is stored in leaf nodes
    - Leaf nodes are linked together in a linked list for efficient range queries
    - All data entries appear in the leaf level, making sequential access faster
    - Internal nodes have more keys, leading to fewer levels and better performance
    
    **B* Tree**:
    
    Another variation of B-Tree:
    
    - Nodes are kept at least 2/3 full instead of 1/2 full
    - Delays splitting nodes by redistributing keys among siblings
    - Only splits when all adjacent siblings are full
    - Results in better space utilization but more complex rebalancing
    
    **2-3 Tree**:
    
    A special case of B-Tree where t=2:
    
    - Each internal node has either 2 or 3 children
    - Keys are stored only in leaf nodes
    - Simple implementation but less efficient for large datasets

=== "Examples"
    **Example 1: Building a B-Tree (t=2)**
    
    Let's insert keys [10, 20, 5, 6, 12, 30, 7, 17] into a B-Tree with minimum degree t=2:
    
    1. Insert 10: Create root node with key 10
    2. Insert 20: Add to root: [10, 20]
    3. Insert 5: Add to root: [5, 10, 20]
    4. Insert 6: Root is full (max 3 keys for t=2), split:
       - New root with key 10
       - Left child [5, 6], right child [20]
    5. Insert 12: Add to right child: [12, 20]
    6. Continue with other insertions...
    
    **Example 2: Range Query in a B-Tree**
    
    For a B-Tree with keys [3, 7, 10, 15, 20, 25, 30, 35, 40, 45]:
    
    1. To find all keys between 10 and 30:
       - Start search for key 10
       - Once found, traverse through nodes until reaching key 30
    2. B+ Trees are more efficient for this type of query due to linked leaf nodes

=== "Comparison"
    **B-Tree vs Binary Search Tree**:
    
    | Aspect | B-Tree | Binary Search Tree |
    |--------|--------|---------------------|
    | **Branching Factor** | Multiple (order t) | 2 (binary) |
    | **Height** | O(log_t n) | O(log₂ n) potentially unbalanced |
    | **I/O Operations** | Fewer (better locality) | More |
    | **Memory Usage** | Potentially better | Potentially worse per key |
    | **Use Case** | Disk-based storage | In-memory operations |
    
    **B-Tree vs Red-Black Tree**:
    
    | Aspect | B-Tree | Red-Black Tree |
    |--------|--------|----------------|
    | **Balance Method** | Node degree constraints | Color properties |
    | **Height** | O(log_t n) | O(log n) |
    | **Memory Access** | Optimized for blocks | Not optimized for blocks |
    | **Implementation** | More complex | Medium complexity |
    | **Use Case** | External memory | Internal memory |

=== "Tips"
    **Implementation Tips**:
    
    1. For database applications, consider B+ Trees instead of standard B-Trees
    2. Choose the degree (t) based on disk block size for optimal performance
    3. For in-memory applications, a lower degree might be more efficient
    4. Consider caching frequently accessed nodes to reduce disk I/O
    
    **Common Pitfalls**:
    
    1. Incorrectly handling the splitting and merging of nodes
    2. Not properly maintaining the B-Tree properties after modifications
    3. Inefficient implementations that don't consider disk block sizes
    4. Overcomplicating the deletion operation
    
    **Optimization Techniques**:
    
    1. Use bulk loading for initial construction with sorted data
    2. Implement proper caching strategies for frequently accessed nodes
    3. Consider compression techniques for keys in nodes
    4. For large datasets, implement efficient node serialization/deserialization
