# Segment Trees

## 🎯 Overview

Segment Trees are binary trees used for storing information about array intervals efficiently. They allow querying and updating ranges in O(log n) time, making them essential for competitive programming and range query problems.

## 🔑 Key Concepts

### Structure

- **Binary Tree**: Each node represents an array segment
- **Leaf Nodes**: Represent individual array elements  
- **Internal Nodes**: Store aggregated information about child segments
- **Height**: O(log n) for array of size n

### Properties

- **Range Representation**: Each node covers [left, right] interval
- **Recursive Structure**: Node [l,r] has children [l,mid] and [mid+1,r]
- **Space Efficient**: 4n space for array of size n
- **Update Propagation**: Changes bubble up from leaves to root

---

## 📚 Basic Implementation

### 1. Range Sum Segment Tree

```python
class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)  # 4n space
        self.build(arr, 0, 0, self.n - 1)
    
    def build(self, arr, node, start, end):
        """Build segment tree from array"""
        if start == end:
            # Leaf node
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            # Build left and right subtrees
            self.build(arr, left_child, start, mid)
            self.build(arr, right_child, mid + 1, end)
            
            # Internal node value = sum of children
            self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def query(self, node, start, end, l, r):
        """Query sum in range [l, r]"""
        if r < start or end < l:
            # No overlap
            return 0
        
        if l <= start and end <= r:
            # Complete overlap
            return self.tree[node]
        
        # Partial overlap
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        left_sum = self.query(left_child, start, mid, l, r)
        right_sum = self.query(right_child, mid + 1, end, l, r)
        
        return left_sum + right_sum
    
    def update(self, node, start, end, idx, val):
        """Update element at index idx to val"""
        if start == end:
            # Leaf node
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            if idx <= mid:
                self.update(left_child, start, mid, idx, val)
            else:
                self.update(right_child, mid + 1, end, idx, val)
            
            # Update internal node
            self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def range_sum(self, l, r):
        """Public method to query range sum"""
        return self.query(0, 0, self.n - 1, l, r)
    
    def point_update(self, idx, val):
        """Public method to update single element"""
        self.update(0, 0, self.n - 1, idx, val)

# Example usage
arr = [1, 3, 5, 7, 9, 11]
seg_tree = SegmentTree(arr)

print(seg_tree.range_sum(1, 3))  # Sum of arr[1:4] = 3+5+7 = 15
seg_tree.point_update(1, 10)     # Change arr[1] from 3 to 10
print(seg_tree.range_sum(1, 3))  # New sum = 10+5+7 = 22
```

### 2. Generic Segment Tree

```python
class GenericSegmentTree:
    def __init__(self, arr, combine_func, default_value):
        """
        arr: input array
        combine_func: function to combine two values (e.g., min, max, sum)
        default_value: identity element for combine_func
        """
        self.n = len(arr)
        self.combine = combine_func
        self.default = default_value
        self.tree = [default_value] * (4 * self.n)
        self.build(arr, 0, 0, self.n - 1)
    
    def build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            self.build(arr, left_child, start, mid)
            self.build(arr, right_child, mid + 1, end)
            
            self.tree[node] = self.combine(
                self.tree[left_child], 
                self.tree[right_child]
            )
    
    def query(self, node, start, end, l, r):
        if r < start or end < l:
            return self.default
        
        if l <= start and end <= r:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        left_result = self.query(left_child, start, mid, l, r)
        right_result = self.query(right_child, mid + 1, end, l, r)
        
        return self.combine(left_result, right_result)
    
    def update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            if idx <= mid:
                self.update(left_child, start, mid, idx, val)
            else:
                self.update(right_child, mid + 1, end, idx, val)
            
            self.tree[node] = self.combine(
                self.tree[left_child], 
                self.tree[right_child]
            )
    
    def range_query(self, l, r):
        return self.query(0, 0, self.n - 1, l, r)
    
    def point_update(self, idx, val):
        self.update(0, 0, self.n - 1, idx, val)

# Examples of different segment trees
arr = [2, 1, 4, 3, 5]

# Range Minimum Query
min_tree = GenericSegmentTree(arr, min, float('inf'))
print(min_tree.range_query(1, 3))  # min(1, 4, 3) = 1

# Range Maximum Query  
max_tree = GenericSegmentTree(arr, max, float('-inf'))
print(max_tree.range_query(1, 3))  # max(1, 4, 3) = 4

# Range Sum Query
sum_tree = GenericSegmentTree(arr, lambda x, y: x + y, 0)
print(sum_tree.range_query(1, 3))  # sum(1, 4, 3) = 8
```

---

## 🚀 Advanced Segment Trees

### 1. Lazy Propagation

```python
class LazySegmentTree:
    """Segment tree with lazy propagation for range updates"""
    
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)  # Lazy propagation array
        self.build(arr, 0, 0, self.n - 1)
    
    def build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            self.build(arr, left_child, start, mid)
            self.build(arr, right_child, mid + 1, end)
            
            self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def push(self, node, start, end):
        """Push lazy value down to children"""
        if self.lazy[node] != 0:
            # Apply lazy update to current node
            self.tree[node] += (end - start + 1) * self.lazy[node]
            
            # If not leaf, push to children
            if start != end:
                left_child = 2 * node + 1
                right_child = 2 * node + 2
                self.lazy[left_child] += self.lazy[node]
                self.lazy[right_child] += self.lazy[node]
            
            # Reset lazy value
            self.lazy[node] = 0
    
    def range_update(self, node, start, end, l, r, val):
        """Add val to all elements in range [l, r]"""
        self.push(node, start, end)
        
        if start > r or end < l:
            return
        
        if start >= l and end <= r:
            # Complete overlap - lazy update
            self.lazy[node] += val
            self.push(node, start, end)
            return
        
        # Partial overlap
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        self.range_update(left_child, start, mid, l, r, val)
        self.range_update(right_child, mid + 1, end, l, r, val)
        
        # Update current node after children updates
        self.push(left_child, start, mid)
        self.push(right_child, mid + 1, end)
        self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def range_query(self, node, start, end, l, r):
        """Query sum in range [l, r]"""
        if start > r or end < l:
            return 0
        
        self.push(node, start, end)
        
        if start >= l and end <= r:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        left_sum = self.range_query(left_child, start, mid, l, r)
        right_sum = self.range_query(right_child, mid + 1, end, l, r)
        
        return left_sum + right_sum
    
    def update_range(self, l, r, val):
        """Public method for range update"""
        self.range_update(0, 0, self.n - 1, l, r, val)
    
    def query_range(self, l, r):
        """Public method for range query"""
        return self.range_query(0, 0, self.n - 1, l, r)

# Example usage
arr = [1, 2, 3, 4, 5]
lazy_tree = LazySegmentTree(arr)

print(lazy_tree.query_range(1, 3))  # Sum of [2, 3, 4] = 9
lazy_tree.update_range(1, 3, 10)    # Add 10 to range [1, 3]
print(lazy_tree.query_range(1, 3))  # New sum = 9 + 3*10 = 39
```

### 2. 2D Segment Tree

```python
class SegmentTree2D:
    """2D segment tree for rectangle range queries"""
    
    def __init__(self, matrix):
        self.rows = len(matrix)
        self.cols = len(matrix[0]) if self.rows > 0 else 0
        self.tree = [[0] * (4 * self.cols) for _ in range(4 * self.rows)]
        
        if self.rows > 0 and self.cols > 0:
            self.build_y(matrix, 1, 0, self.rows - 1)
    
    def build_y(self, matrix, vx, lx, rx):
        """Build y-dimension segment tree"""
        if lx == rx:
            self.build_x(matrix[lx], vx, 1, 0, self.cols - 1)
        else:
            mid = (lx + rx) // 2
            self.build_y(matrix, 2 * vx, lx, mid)
            self.build_y(matrix, 2 * vx + 1, mid + 1, rx)
            self.merge_x(vx, 1, 0, self.cols - 1)
    
    def build_x(self, arr, vx, vy, ly, ry):
        """Build x-dimension segment tree"""
        if ly == ry:
            self.tree[vx][vy] = arr[ly]
        else:
            mid = (ly + ry) // 2
            self.build_x(arr, vx, 2 * vy, ly, mid)
            self.build_x(arr, vx, 2 * vy + 1, mid + 1, ry)
            self.tree[vx][vy] = (self.tree[vx][2 * vy] + 
                                self.tree[vx][2 * vy + 1])
    
    def merge_x(self, vx, vy, ly, ry):
        """Merge x-dimension values"""
        if ly == ry:
            self.tree[vx][vy] = (self.tree[2 * vx][vy] + 
                                self.tree[2 * vx + 1][vy])
        else:
            mid = (ly + ry) // 2
            self.merge_x(vx, 2 * vy, ly, mid)
            self.merge_x(vx, 2 * vy + 1, mid + 1, ry)
            self.tree[vx][vy] = (self.tree[vx][2 * vy] + 
                                self.tree[vx][2 * vy + 1])
    
    def query_2d(self, x1, y1, x2, y2):
        """Query rectangle sum from (x1,y1) to (x2,y2)"""
        return self.query_y(1, 0, self.rows - 1, x1, x2, y1, y2)
    
    def query_y(self, vx, lx, rx, x1, x2, y1, y2):
        """Query in y-dimension"""
        if x1 > rx or x2 < lx:
            return 0
        
        if x1 <= lx and rx <= x2:
            return self.query_x(vx, 1, 0, self.cols - 1, y1, y2)
        
        mid = (lx + rx) // 2
        return (self.query_y(2 * vx, lx, mid, x1, x2, y1, y2) +
                self.query_y(2 * vx + 1, mid + 1, rx, x1, x2, y1, y2))
    
    def query_x(self, vx, vy, ly, ry, y1, y2):
        """Query in x-dimension"""
        if y1 > ry or y2 < ly:
            return 0
        
        if y1 <= ly and ry <= y2:
            return self.tree[vx][vy]
        
        mid = (ly + ry) // 2
        return (self.query_x(vx, 2 * vy, ly, mid, y1, y2) +
                self.query_x(vx, 2 * vy + 1, mid + 1, ry, y1, y2))

# Example usage
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
tree_2d = SegmentTree2D(matrix)
print(tree_2d.query_2d(0, 0, 1, 1))  # Sum of top-left 2x2 = 1+2+4+5 = 12
```

---

## 🎛️ Specialized Segment Trees

### 1. Range Maximum Query with Index

```python
class RMQSegmentTree:
    """Range Maximum Query with index tracking"""
    
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [(0, -1)] * (4 * self.n)  # (value, index) pairs
        self.build(arr, 0, 0, self.n - 1)
    
    def build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = (arr[start], start)
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            self.build(arr, left_child, start, mid)
            self.build(arr, right_child, mid + 1, end)
            
            # Choose maximum value (and its index)
            left_val, left_idx = self.tree[left_child]
            right_val, right_idx = self.tree[right_child]
            
            if left_val >= right_val:
                self.tree[node] = (left_val, left_idx)
            else:
                self.tree[node] = (right_val, right_idx)
    
    def query(self, node, start, end, l, r):
        if r < start or end < l:
            return (float('-inf'), -1)
        
        if l <= start and end <= r:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        left_result = self.query(left_child, start, mid, l, r)
        right_result = self.query(right_child, mid + 1, end, l, r)
        
        # Return maximum
        if left_result[0] >= right_result[0]:
            return left_result
        else:
            return right_result
    
    def range_max_query(self, l, r):
        """Returns (max_value, index_of_max)"""
        return self.query(0, 0, self.n - 1, l, r)

# Example usage
arr = [3, 1, 4, 1, 5, 9, 2]
rmq_tree = RMQSegmentTree(arr)
max_val, max_idx = rmq_tree.range_max_query(2, 5)
print(f"Max value: {max_val}, Index: {max_idx}")  # Max value: 9, Index: 5
```

### 2. Persistent Segment Tree

```python
class PersistentSegmentTree:
    """Segment tree that maintains all historical versions"""
    
    class Node:
        def __init__(self, value=0, left=None, right=None):
            self.value = value
            self.left = left
            self.right = right
    
    def __init__(self, arr):
        self.n = len(arr)
        self.versions = []  # Store root of each version
        
        # Build initial version
        initial_root = self.build(arr, 0, self.n - 1)
        self.versions.append(initial_root)
    
    def build(self, arr, start, end):
        if start == end:
            return self.Node(arr[start])
        
        mid = (start + end) // 2
        left_child = self.build(arr, start, mid)
        right_child = self.build(arr, mid + 1, end)
        
        return self.Node(
            left_child.value + right_child.value,
            left_child,
            right_child
        )
    
    def update(self, old_root, start, end, idx, val):
        """Create new version with updated value"""
        if start == end:
            return self.Node(val)
        
        mid = (start + end) // 2
        
        if idx <= mid:
            new_left = self.update(old_root.left, start, mid, idx, val)
            new_right = old_root.right  # Reuse old right subtree
        else:
            new_left = old_root.left   # Reuse old left subtree
            new_right = self.update(old_root.right, mid + 1, end, idx, val)
        
        return self.Node(
            new_left.value + new_right.value,
            new_left,
            new_right
        )
    
    def query(self, root, start, end, l, r):
        if r < start or end < l:
            return 0
        
        if l <= start and end <= r:
            return root.value
        
        mid = (start + end) // 2
        left_sum = self.query(root.left, start, mid, l, r)
        right_sum = self.query(root.right, mid + 1, end, l, r)
        
        return left_sum + right_sum
    
    def point_update(self, idx, val):
        """Create new version with point update"""
        old_root = self.versions[-1]
        new_root = self.update(old_root, 0, self.n - 1, idx, val)
        self.versions.append(new_root)
    
    def range_sum(self, version, l, r):
        """Query range sum in specific version"""
        root = self.versions[version]
        return self.query(root, 0, self.n - 1, l, r)

# Example usage
arr = [1, 2, 3, 4, 5]
persistent_tree = PersistentSegmentTree(arr)

print(persistent_tree.range_sum(0, 1, 3))  # Version 0: sum(2,3,4) = 9

persistent_tree.point_update(2, 10)        # Update index 2 to 10
print(persistent_tree.range_sum(0, 1, 3))  # Version 0: still 9
print(persistent_tree.range_sum(1, 1, 3))  # Version 1: sum(2,10,4) = 16
```

---

## 📊 Complexity Analysis

| **Operation** | **Time Complexity** | **Space Complexity** | **Notes** |
|---------------|-------------------|---------------------|-----------|
| **Build** | O(n) | O(n) | Linear construction |
| **Point Query** | O(log n) | O(1) | Single element |
| **Range Query** | O(log n) | O(1) | Any range |
| **Point Update** | O(log n) | O(1) | Single element |
| **Range Update** | O(log n) | O(1) | With lazy propagation |

## 🎯 Applications

### Competitive Programming
- **Range Sum Queries**: Subarray sum problems
- **Range Minimum/Maximum**: RMQ problems
- **Range Updates**: Mass assignment problems
- **Counting Queries**: Frequency counting in ranges

### Real-World Applications
- **Database Indexing**: Range queries on sorted data
- **Graphics**: Range queries for collision detection
- **Finance**: Portfolio analysis over time ranges
- **Data Analytics**: Aggregations over time windows

## 🔥 Common Problems

1. **Range Sum Query**: Basic segment tree
2. **Range Minimum Query**: RMQ with sparse tables
3. **Range Update Queries**: Lazy propagation
4. **Count of Range Sum**: Coordinate compression + segment tree
5. **Rectangle Area Queries**: 2D segment trees

---

*Segment Trees are essential for competitive programming - master them for efficient range operations!*
