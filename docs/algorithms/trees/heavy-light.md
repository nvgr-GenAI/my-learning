# Heavy-Light Decomposition 🌲⚖️

## Introduction

Heavy-Light Decomposition (HLD) is a powerful technique for handling path queries in trees by decomposing the tree into a collection of paths that can be efficiently processed.

=== "Overview"
    **Core Concept**:
    
    - Decomposes a tree into a set of "heavy" paths and "light" edges
    - A "heavy" path is a path where each node follows its child with the largest subtree
    - Each node has at most one "heavy" edge to a child
    - Enables efficient path queries and updates in trees
    - Reduces the complexity of tree path operations to array operations
    
    **When to Use**:
    
    - Path queries in trees (sum, min, max, etc.)
    - Path updates in trees (range updates)
    - Lowest Common Ancestor (LCA) queries
    - Competitive programming problems involving trees
    - Problems requiring efficient tree traversal along paths
    
    **Time Complexity**:
    
    - Preprocessing: O(n)
    - Path Query/Update: O(log² n) or O(log n) with segment trees
    - LCA: O(log n)
    
    **Real-World Applications**:
    
    - Network routing problems
    - Hierarchical data processing
    - Game tree evaluation
    - Ancestry and relationship queries in hierarchies
    - File system operations (when modeled as trees)

=== "Key Properties"
    **1. Path Decomposition**
    
    - Tree is split into O(log n) distinct paths
    - Each path can be represented as a contiguous segment in an array
    - Each node belongs to exactly one heavy path
    - Paths are connected by light edges
    
    **2. Depth Properties**
    
    - Light edges on any root-to-leaf path: O(log n)
    - Maximum number of heavy paths on any root-to-leaf path: O(log n)
    - Moving from any node to the root crosses at most O(log n) heavy paths
    
    **3. Heavy and Light Edges**
    
    - **Heavy Edge**: Edge connecting a node to its child with the largest subtree
    - **Light Edge**: All other edges
    - Each node can have at most one outgoing heavy edge
    - A heavy path is a sequence of connected heavy edges
    
    **4. Path Representation**
    
    - Each heavy path can be stored as a linear array/segment tree
    - Enables efficient range operations on paths
    - Transforms tree path problems into array segment problems

=== "Implementation Steps"
    **1. Preprocessing**
    
    - Compute subtree sizes using DFS
    - Identify heavy edges based on subtree sizes
    - Decompose the tree into heavy paths
    - Assign positions in arrays for efficient access
    
    **2. Path Query Process**
    
    - Find the path between two nodes
    - Break the path into heavy path segments
    - Process each segment using efficient data structures (e.g., segment trees)
    - Combine results from all segments
    
    **3. Data Structures**
    
    - **Segment Trees**: For range queries/updates on heavy paths
    - **Parent Array**: To move up the tree
    - **Heavy Path Array**: To identify which path a node belongs to
    - **Position Arrays**: To map tree nodes to positions in segment trees

## 🔧 Implementation

### Basic Heavy-Light Decomposition

```cpp
const int MAXN = 100005;
vector<int> adj[MAXN]; // Adjacency list representation of the tree
int size[MAXN];       // Subtree size
int depth[MAXN];      // Depth of node
int parent[MAXN];     // Parent of node
int heavy[MAXN];      // Heavy child of node
int head[MAXN];       // Head of heavy path containing node
int pos[MAXN];        // Position of node in segment tree array
int segTreeArray[MAXN]; // Values for segment tree
int current_pos = 0;

// Calculate subtree sizes and identify heavy edges
void dfs(int node, int par = -1, int d = 0) {
    size[node] = 1;
    depth[node] = d;
    parent[node] = par;
    int max_size = 0;
    heavy[node] = -1;
    
    for (int child : adj[node]) {
        if (child != par) {
            dfs(child, node, d + 1);
            size[node] += size[child];
            
            if (size[child] > max_size) {
                max_size = size[child];
                heavy[node] = child;
            }
        }
    }
}

// Decompose the tree into heavy paths
void decompose(int node, int h) {
    head[node] = h;
    pos[node] = current_pos++;
    
    // Follow heavy path
    if (heavy[node] != -1) {
        decompose(heavy[node], h);
    }
    
    // Process light edges
    for (int child : adj[node]) {
        if (child != parent[node] && child != heavy[node]) {
            decompose(child, child);
        }
    }
}

// Initialize the decomposition
void init_hld(int root) {
    dfs(root);
    current_pos = 0;
    decompose(root, root);
    
    // Initialize segment tree with values
    // (code for segment tree initialization omitted)
}
```

### Path Query Implementation

```cpp
// Find the result of a query on the path from node a to node b
int query_path(int a, int b) {
    int res = /* identity element */;
    
    // Process until both nodes are on the same heavy path
    while (head[a] != head[b]) {
        if (depth[head[a]] < depth[head[b]]) {
            swap(a, b);
        }
        
        // Query the segment from position of head[a] to position of a
        int current_heavy_path_result = query_segment_tree(pos[head[a]], pos[a]);
        res = combine_results(res, current_heavy_path_result);
        
        // Move to the parent of the head
        a = parent[head[a]];
    }
    
    // Both nodes are on the same heavy path
    if (depth[a] > depth[b]) {
        swap(a, b);
    }
    
    // Query the final segment
    int final_segment_result = query_segment_tree(pos[a], pos[b]);
    res = combine_results(res, final_segment_result);
    
    return res;
}

// Update a value on the path from a to b
void update_path(int a, int b, int value) {
    // Similar structure to query_path, but with update operations
    while (head[a] != head[b]) {
        if (depth[head[a]] < depth[head[b]]) {
            swap(a, b);
        }
        
        update_segment_tree(pos[head[a]], pos[a], value);
        a = parent[head[a]];
    }
    
    if (depth[a] > depth[b]) {
        swap(a, b);
    }
    
    update_segment_tree(pos[a], pos[b], value);
}
```

## 📊 Applications

=== "Path Queries"
    **1. Path Sum Queries**
    
    - Find the sum of values on a path between two nodes
    - Update values along a path
    - Example: Total weight along a route in a network
    
    ```cpp
    // Example: Query sum of values on path from node a to node b
    int query_path_sum(int a, int b) {
        int sum = 0;
        
        while (head[a] != head[b]) {
            if (depth[head[a]] < depth[head[b]]) {
                swap(a, b);
            }
            
            // Sum values on current heavy path
            sum += query_sum_segment_tree(pos[head[a]], pos[a]);
            a = parent[head[a]];
        }
        
        // Process the last segment
        if (depth[a] > depth[b]) {
            swap(a, b);
        }
        sum += query_sum_segment_tree(pos[a], pos[b]);
        
        return sum;
    }
    ```
    
    **2. Path Min/Max Queries**
    
    - Find the minimum/maximum value on a path
    - Applications in network bottleneck problems
    - Resource allocation in hierarchical systems
    
    **3. Path Updates**
    
    - Update all values on a path (add, set, etc.)
    - Efficiently propagate changes through a hierarchy
    - Lazy propagation techniques can be applied

=== "Advanced Applications"
    **1. Lowest Common Ancestor (LCA)**
    
    - HLD provides an efficient way to find LCAs
    - Query the highest node on the path between two nodes
    - Time complexity: O(log n)
    
    ```cpp
    // Find LCA of nodes a and b using HLD
    int lca(int a, int b) {
        while (head[a] != head[b]) {
            if (depth[head[a]] < depth[head[b]]) {
                swap(a, b);
            }
            a = parent[head[a]];
        }
        
        // Now both nodes are on the same heavy path
        return depth[a] < depth[b] ? a : b;
    }
    ```
    
    **2. Subtree Queries**
    
    - Process queries on entire subtrees
    - Combine with Euler tour technique for more efficiency
    - Update all descendants of a given node
    
    **3. Distance Queries**
    
    - Find distance between any two nodes in the tree
    - Compute efficiently using LCA:
    
    ```cpp
    int distance(int a, int b) {
        int lca_node = lca(a, b);
        return depth[a] + depth[b] - 2 * depth[lca_node];
    }
    ```

=== "Specialized Use Cases"
    **1. Dynamic Connectivity**
    
    - Handle link-cut operations in trees
    - Efficiently restore connectivity after edge removals
    - Used in dynamic graph algorithms
    
    **2. Tree Compression**
    
    - Represent tree paths compactly for efficient operations
    - Preserve hierarchical relationships while enabling fast queries
    - Useful for in-memory processing of large trees
    
    **3. Virtual Tree Construction**
    
    - Build temporary trees connecting important nodes
    - Solve problems on subsets of vertices
    - Combine with HLD for powerful problem-solving techniques
    
    **4. Tree Separators**
    
    - Use heavy paths as natural tree separators
    - Divide-and-conquer algorithms on trees
    - Breaking complex problems into manageable pieces

## 🧩 Common Problems

1. **Path Queries on Trees**
   - Find sum/min/max on a path between two nodes
   - Update values along a path
   - Solution: HLD + Segment Trees

2. **Vertex-Weighted Paths**
   - Similar to edge-weighted paths but with values on vertices
   - Slight modification of the basic HLD approach

3. **Dynamic Tree Paths**
   - Handle path queries with dynamic tree structure
   - Combine HLD with Link-Cut Trees for advanced cases

4. **Tree Coloring**
   - Count distinct colors on a path
   - Update colors of nodes or paths
   - Solution: HLD with appropriate data structures

5. **Tree Modification Queries**
   - Handle operations like cutting subtrees
   - Reattaching subtrees elsewhere
   - Heavy path maintenance during modifications

## 📝 Summary

- **Heavy-Light Decomposition** breaks trees into O(log n) paths for efficient processing
- **Path queries/updates** can be performed in O(log² n) or O(log n) time
- **Each root-to-leaf path** crosses at most O(log n) heavy paths
- **Combines with segment trees** for powerful tree operations
- **Essential technique** for competitive programming and tree algorithms
- **Reduces tree path operations** to array segment operations
