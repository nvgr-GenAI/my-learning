# Disjoint Set Union (Union-Find) 🌳🔗

## Introduction

Disjoint Set Union (DSU), also known as Union-Find, is a data structure that tracks a set of elements partitioned into non-overlapping (disjoint) subsets. It provides near-constant time operations for adding new sets, merging sets, and determining whether elements are in the same set.

> **Note:** DSU is both a tree-based data structure and a graph algorithm. For its application in graph algorithms, see also [Union-Find in Graphs](../graphs/union-find.md).

=== "Overview"
    **Core Concept**:
    
    - Maintains disjoint (non-overlapping) sets of elements
    - Efficiently determines if two elements belong to the same set
    - Efficiently merges sets containing specified elements
    - Uses tree-like structures to represent each set
    - Optimized using path compression and union by rank/size
    
    **When to Use**:
    
    - Finding connected components in graphs
    - Detecting cycles in undirected graphs
    - Implementing Kruskal's algorithm for minimum spanning trees
    - Dynamic connectivity problems
    - Network/grid percolation problems
    
    **Time Complexity**:
    
    - MakeSet: O(1)
    - Find: O(α(n)) amortized (practically constant)
    - Union: O(α(n)) amortized (practically constant)
    
    Where α(n) is the inverse Ackermann function, which grows extremely slowly and is ≤ 4 for all practical values of n.
    
    **Real-World Applications**:
    
    - Network connectivity
    - Image processing (connected component labeling)
    - Social network analysis
    - Computational geometry (Voronoi diagrams)
    - Game development (dynamic connection handling)

=== "Structure"
    **Basic Components**:
    
    - **Parent Array**: Each element points to its parent in the tree
    - **Rank/Size Array**: Stores rank or size of each tree for union by rank/size
    
    **Tree Representation**:
    
    - Each set is represented as a tree
    - Elements point to their parent element
    - The root of the tree is the representative element of the set
    - Elements in the same set have the same root
    
    **Path Compression**:
    
    During Find operations, all traversed nodes are made to point directly to the root, flattening the tree structure and speeding up future operations.
    
    **Union by Rank/Size**:
    
    - **Union by Rank**: Attach the shorter tree under the root of the taller tree
    - **Union by Size**: Attach the smaller tree under the root of the larger tree
    
    Both strategies help keep the trees balanced and shallow.

=== "Operations"
    **MakeSet**:
    
    Creates a new set containing a single element.
    
    ```python
    def make_set(parent, rank, x):
        parent[x] = x    # Each element is initially its own parent
        rank[x] = 0      # Initial rank/height is 0
    ```
    
    **Find (with Path Compression)**:
    
    Returns the representative (root) element of the set containing x.
    
    ```python
    def find(parent, x):
        if parent[x] != x:
            parent[x] = find(parent, parent[x])  # Path compression
        return parent[x]
    ```
    
    **Union (by Rank)**:
    
    Merges the sets containing elements x and y.
    
    ```python
    def union(parent, rank, x, y):
        root_x = find(parent, x)
        root_y = find(parent, y)
        
        if root_x == root_y:
            return  # Already in the same set
            
        # Make root of smaller rank point to root of larger rank
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1  # Increase rank of root_x
    ```
    
    **Union (by Size)**:
    
    Alternative implementation using size instead of rank:
    
    ```python
    def union(parent, size, x, y):
        root_x = find(parent, x)
        root_y = find(parent, y)
        
        if root_x == root_y:
            return  # Already in the same set
            
        # Make smaller tree a subtree of the larger tree
        if size[root_x] < size[root_y]:
            parent[root_x] = root_y
            size[root_y] += size[root_x]
        else:
            parent[root_y] = root_x
            size[root_x] += size[root_y]
    ```

=== "Implementation"
    **Complete DSU Class Implementation**:
    
    ```python
    class DisjointSet:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.size = [1] * n
            self.count = n  # Number of distinct sets
            
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])  # Path compression
            return self.parent[x]
            
        def union(self, x, y):
            root_x = self.find(x)
            root_y = self.find(y)
            
            if root_x == root_y:
                return False  # Already in the same set
                
            # Union by rank
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
                self.size[root_y] += self.size[root_x]
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
                self.size[root_x] += self.size[root_y]
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
                self.size[root_x] += self.size[root_y]
                
            self.count -= 1  # Decrease count of distinct sets
            return True
            
        def connected(self, x, y):
            return self.find(x) == self.find(y)
            
        def get_size(self, x):
            return self.size[self.find(x)]
            
        def get_count(self):
            return self.count
    ```
    
    **C++ Implementation**:
    
    ```cpp
    class DisjointSet {
    private:
        vector<int> parent, rank, size;
        int count;
        
    public:
        DisjointSet(int n) {
            parent.resize(n);
            rank.resize(n, 0);
            size.resize(n, 1);
            count = n;
            
            for (int i = 0; i < n; i++) {
                parent[i] = i;
            }
        }
        
        int find(int x) {
            if (parent[x] != x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        }
        
        bool union_(int x, int y) {
            int root_x = find(x);
            int root_y = find(y);
            
            if (root_x == root_y) return false;
            
            if (rank[root_x] < rank[root_y]) {
                parent[root_x] = root_y;
                size[root_y] += size[root_x];
            } else if (rank[root_x] > rank[root_y]) {
                parent[root_y] = root_x;
                size[root_x] += size[root_y];
            } else {
                parent[root_y] = root_x;
                rank[root_x]++;
                size[root_x] += size[root_y];
            }
            
            count--;
            return true;
        }
        
        bool connected(int x, int y) {
            return find(x) == find(y);
        }
        
        int getSize(int x) {
            return size[find(x)];
        }
        
        int getCount() {
            return count;
        }
    };
    ```

=== "Examples"
    **Example 1: Connected Components**
    
    Finding connected components in an undirected graph:
    
    ```python
    def count_connected_components(n, edges):
        dsu = DisjointSet(n)
        
        for u, v in edges:
            dsu.union(u, v)
            
        return dsu.get_count()
    ```
    
    **Example 2: Kruskal's Algorithm for Minimum Spanning Tree**
    
    ```python
    def kruskal_mst(n, edges):
        # Sort edges by weight
        edges.sort(key=lambda x: x[2])
        
        dsu = DisjointSet(n)
        mst = []
        total_weight = 0
        
        for u, v, weight in edges:
            if not dsu.connected(u, v):
                dsu.union(u, v)
                mst.append((u, v, weight))
                total_weight += weight
                
        return mst, total_weight
    ```
    
    **Example 3: Detecting Cycles in an Undirected Graph**
    
    ```python
    def has_cycle(n, edges):
        dsu = DisjointSet(n)
        
        for u, v in edges:
            # If vertices are already in the same component, adding this edge creates a cycle
            if dsu.connected(u, v):
                return True
            dsu.union(u, v)
            
        return False
    ```
    
    **Example 4: Grid Percolation**
    
    ```python
    def percolates(grid):
        n = len(grid)
        dsu = DisjointSet(n * n + 2)  # +2 for virtual top and bottom nodes
        top = n * n
        bottom = n * n + 1
        
        # Connect top row to virtual top, bottom row to virtual bottom
        for i in range(n):
            if grid[0][i] == 1:  # Open cell
                dsu.union(i, top)
            if grid[n-1][i] == 1:  # Open cell
                dsu.union((n-1) * n + i, bottom)
                
        # Connect adjacent open cells
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:  # Open cell
                    for dx, dy in dirs:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < n and 0 <= nj < n and grid[ni][nj] == 1:
                            dsu.union(i * n + j, ni * n + nj)
                            
        # Check if top is connected to bottom
        return dsu.connected(top, bottom)
    ```

=== "Advanced Techniques"
    **Persistent Union-Find**:
    
    A variant that allows reverting to previous states:
    
    ```python
    class PersistentDSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.history = []
            
        def find(self, x):
            if self.parent[x] != x:
                return self.find(self.parent[x])
            return x
            
        def union(self, x, y):
            root_x = self.find(x)
            root_y = self.find(y)
            
            if root_x == root_y:
                self.history.append(None)  # No change
                return False
                
            if self.rank[root_x] < self.rank[root_y]:
                self.history.append((root_x, self.parent[root_x], root_y, self.rank[root_y]))
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.history.append((root_y, self.parent[root_y], root_x, self.rank[root_x]))
                self.parent[root_y] = root_x
            else:
                self.history.append((root_y, self.parent[root_y], root_x, self.rank[root_x]))
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
                
            return True
            
        def undo(self):
            if not self.history:
                return False
                
            last = self.history.pop()
            if last is None:
                return True
                
            node, old_parent, parent_node, old_rank = last
            self.parent[node] = old_parent
            self.rank[parent_node] = old_rank
            
            return True
    ```
    
    **Weighted Union-Find**:
    
    A variant that supports weighted edges:
    
    ```python
    class WeightedDSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.weight = [0] * n  # Weight represents distance from node to its parent
            
        def find(self, x):
            if self.parent[x] != x:
                root = self.find(self.parent[x])
                self.weight[x] += self.weight[self.parent[x]]  # Update weight
                self.parent[x] = root
            return self.parent[x]
            
        def union(self, x, y, w):
            # w: weight of edge from x to y
            root_x = self.find(x)
            root_y = self.find(y)
            
            if root_x == root_y:
                return False
                
            self.parent[root_x] = root_y
            self.weight[root_x] = self.weight[y] - self.weight[x] + w
            
            return True
            
        def difference(self, x, y):
            # Returns the difference in weight between x and y if they're connected
            if self.find(x) != self.find(y):
                return None  # Not in same component
            return self.weight[x] - self.weight[y]
    ```

=== "Tips"
    **Implementation Tips**:
    
    1. **Path Compression**: Always implement path compression in the find method to achieve near-constant time operations
    
    2. **Union by Rank/Size**: Use union by rank or size to keep trees balanced and shallow
    
    3. **1-Indexed vs 0-Indexed**: Be consistent with your indexing scheme; adjust your initialization accordingly
    
    4. **Error Handling**: Add checks for invalid indices to prevent out-of-bounds errors
    
    5. **Additional State**: Consider tracking component counts or sizes for specific applications
    
    **Common Pitfalls**:
    
    1. **Forgetting Path Compression**: Without path compression, operations can degrade to O(n) in the worst case
    
    2. **Incorrect Root Finding**: Always use the find method to get the root of an element, don't use parent[x] directly
    
    3. **Missing Union Optimization**: Without union by rank/size, trees can become highly imbalanced
    
    4. **Ignoring Return Values**: Check the return value of union to detect if a merge actually occurred
    
    **Application-Specific Tips**:
    
    1. **Graph Problems**: For graph connectivity, create DSU with nodes as elements
    
    2. **Grid Problems**: Convert 2D coordinates (i,j) to 1D indices (i*width + j)
    
    3. **Dynamic Connectivity**: Use DSU when connectivity information changes frequently
    
    4. **Offline Algorithms**: For queries that can be processed in any order, sort them to optimize DSU operations
