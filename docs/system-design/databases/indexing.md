# Database Indexing Algorithms 📊

Understanding database indexing algorithms is crucial for system design. This guide covers the fundamental data structures and algorithms that power database performance.

## 🎯 Index Fundamentals

### What is a Database Index?

**Definition**: A data structure that improves the speed of data retrieval operations on a database table at the cost of additional writes and storage space.

**Key Properties**:
- **Ordered structure**: Maintains sorted order for efficient searching
- **Pointer system**: Contains references to actual data rows
- **Balanced structure**: Ensures consistent performance
- **Selective access**: Allows direct access to specific rows

> **Real-World Analogy**: Think of an index like a book's table of contents. Instead of reading every page to find a topic, you look it up in the index and jump directly to the relevant page.

## 🌲 Tree-Based Indexing

### 1. **B-Tree (Balanced Tree)**

**Structure**: Self-balancing tree data structure that maintains sorted data and allows searches, insertions, and deletions in logarithmic time.

**Properties**:
- **Order (m)**: Maximum number of children per node
- **Balanced**: All leaf nodes at same level
- **Sorted**: Keys in non-decreasing order
- **Branching factor**: High branching factor reduces tree height

**Use Cases**:
- **Database indexes**: Primary and secondary indexes
- **File systems**: Directory structures
- **Range queries**: Efficient range scans

**Implementation**:
```python
class BTreeNode:
    def __init__(self, is_leaf=False):
        self.keys = []
        self.children = []
        self.is_leaf = is_leaf
        self.parent = None
    
    def __repr__(self):
        return f"BTreeNode(keys={self.keys}, is_leaf={self.is_leaf})"

class BTree:
    def __init__(self, min_degree=3):
        self.root = BTreeNode(is_leaf=True)
        self.min_degree = min_degree  # t in classic B-tree notation
        self.max_keys = 2 * min_degree - 1
        self.min_keys = min_degree - 1
    
    def search(self, key, node=None):
        """Search for a key in the B-tree"""
        if node is None:
            node = self.root
        
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        
        if i < len(node.keys) and key == node.keys[i]:
            return node, i  # Found the key
        
        if node.is_leaf:
            return None  # Key not found
        
        return self.search(key, node.children[i])
    
    def insert(self, key):
        """Insert a key into the B-tree"""
        if len(self.root.keys) == self.max_keys:
            # Root is full, need to split
            old_root = self.root
            self.root = BTreeNode()
            self.root.children.append(old_root)
            old_root.parent = self.root
            self._split_child(self.root, 0)
        
        self._insert_non_full(self.root, key)
    
    def _insert_non_full(self, node, key):
        """Insert key into a non-full node"""
        i = len(node.keys) - 1
        
        if node.is_leaf:
            # Insert into leaf node
            node.keys.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            node.keys[i + 1] = key
        else:
            # Find child to insert into
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            
            if len(node.children[i].keys) == self.max_keys:
                # Child is full, split it
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            
            self._insert_non_full(node.children[i], key)
    
    def _split_child(self, parent, index):
        """Split a full child node"""
        full_child = parent.children[index]
        new_child = BTreeNode(is_leaf=full_child.is_leaf)
        
        # Split keys
        mid_index = self.min_keys
        new_child.keys = full_child.keys[mid_index + 1:]
        full_child.keys = full_child.keys[:mid_index]
        
        # Split children if not leaf
        if not full_child.is_leaf:
            new_child.children = full_child.children[mid_index + 1:]
            full_child.children = full_child.children[:mid_index + 1]
            
            # Update parent pointers
            for child in new_child.children:
                child.parent = new_child
        
        # Move median key up to parent
        parent.keys.insert(index, full_child.keys[mid_index])
        parent.children.insert(index + 1, new_child)
        
        # Update parent pointers
        new_child.parent = parent
    
    def range_query(self, start_key, end_key):
        """Perform range query on B-tree"""
        results = []
        
        def traverse(node):
            if node.is_leaf:
                for key in node.keys:
                    if start_key <= key <= end_key:
                        results.append(key)
            else:
                for i, key in enumerate(node.keys):
                    if start_key <= key:
                        traverse(node.children[i])
                    if start_key <= key <= end_key:
                        results.append(key)
                    if key < end_key:
                        traverse(node.children[i + 1])
        
        traverse(self.root)
        return sorted(results)

# Usage example
btree = BTree(min_degree=3)
keys = [10, 20, 5, 6, 12, 30, 7, 17]

for key in keys:
    btree.insert(key)

# Search for a key
result = btree.search(12)
print(f"Found key 12: {result is not None}")

# Range query
range_results = btree.range_query(5, 15)
print(f"Keys in range [5, 15]: {range_results}")
```

**Time Complexity**:
- Search: O(log n)
- Insert: O(log n)
- Delete: O(log n)
- Range query: O(log n + k) where k is result size

**Space Complexity**: O(n)

### 2. **B+ Tree**

**Structure**: Variation of B-tree where all data is stored in leaf nodes and internal nodes only store keys for navigation.

**Key Differences from B-Tree**:
- **Data in leaves**: All actual data stored in leaf nodes
- **Linked leaves**: Leaf nodes linked for efficient range queries
- **Higher fanout**: Internal nodes can store more keys
- **Sequential access**: Optimal for range scans

**Implementation**:
```python
class BPlusTreeNode:
    def __init__(self, is_leaf=False):
        self.keys = []
        self.values = []  # Only used in leaf nodes
        self.children = []  # Only used in internal nodes
        self.is_leaf = is_leaf
        self.next_leaf = None  # Pointer to next leaf (for range queries)
        self.parent = None

class BPlusTree:
    def __init__(self, min_degree=3):
        self.root = BPlusTreeNode(is_leaf=True)
        self.min_degree = min_degree
        self.max_keys = 2 * min_degree - 1
        self.min_keys = min_degree - 1
    
    def search(self, key):
        """Search for a key in B+ tree"""
        node = self.root
        
        while not node.is_leaf:
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.children[i]
        
        # Search in leaf node
        try:
            index = node.keys.index(key)
            return node.values[index]
        except ValueError:
            return None
    
    def insert(self, key, value):
        """Insert key-value pair into B+ tree"""
        if len(self.root.keys) == self.max_keys:
            # Root is full, create new root
            old_root = self.root
            self.root = BPlusTreeNode()
            self.root.children.append(old_root)
            old_root.parent = self.root
            self._split_child(self.root, 0)
        
        self._insert_non_full(self.root, key, value)
    
    def _insert_non_full(self, node, key, value):
        """Insert into non-full node"""
        if node.is_leaf:
            # Insert into leaf node
            i = 0
            while i < len(node.keys) and key > node.keys[i]:
                i += 1
            
            node.keys.insert(i, key)
            node.values.insert(i, value)
        else:
            # Find child to insert into
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            
            if len(node.children[i].keys) == self.max_keys:
                self._split_child(node, i)
                if key >= node.keys[i]:
                    i += 1
            
            self._insert_non_full(node.children[i], key, value)
    
    def _split_child(self, parent, index):
        """Split a full child node"""
        full_child = parent.children[index]
        new_child = BPlusTreeNode(is_leaf=full_child.is_leaf)
        
        mid_index = self.min_keys
        
        if full_child.is_leaf:
            # Split leaf node
            new_child.keys = full_child.keys[mid_index:]
            new_child.values = full_child.values[mid_index:]
            full_child.keys = full_child.keys[:mid_index]
            full_child.values = full_child.values[:mid_index]
            
            # Update leaf pointers
            new_child.next_leaf = full_child.next_leaf
            full_child.next_leaf = new_child
            
            # Promote first key of new child
            promote_key = new_child.keys[0]
        else:
            # Split internal node
            new_child.keys = full_child.keys[mid_index + 1:]
            new_child.children = full_child.children[mid_index + 1:]
            
            promote_key = full_child.keys[mid_index]
            
            full_child.keys = full_child.keys[:mid_index]
            full_child.children = full_child.children[:mid_index + 1]
            
            # Update parent pointers
            for child in new_child.children:
                child.parent = new_child
        
        # Insert promoted key into parent
        parent.keys.insert(index, promote_key)
        parent.children.insert(index + 1, new_child)
        new_child.parent = parent
    
    def range_query(self, start_key, end_key):
        """Efficient range query using leaf pointers"""
        results = []
        
        # Find starting leaf
        node = self.root
        while not node.is_leaf:
            i = 0
            while i < len(node.keys) and start_key >= node.keys[i]:
                i += 1
            node = node.children[i]
        
        # Traverse leaf nodes
        while node:
            for i, key in enumerate(node.keys):
                if start_key <= key <= end_key:
                    results.append((key, node.values[i]))
                elif key > end_key:
                    return results
            node = node.next_leaf
        
        return results

# Usage example
bplus_tree = BPlusTree(min_degree=3)
data = [(10, "value10"), (20, "value20"), (5, "value5"), (15, "value15")]

for key, value in data:
    bplus_tree.insert(key, value)

# Range query
range_results = bplus_tree.range_query(5, 15)
print(f"Range query [5, 15]: {range_results}")
```

**Benefits over B-Tree**:
- **Better range queries**: Linked leaves enable efficient scanning
- **Higher fanout**: More keys per internal node
- **Predictable performance**: All data access requires same depth
- **Better caching**: Internal nodes cache better

### 3. **Hash Indexes**

**Structure**: Hash table-based indexing for exact match queries.

**Use Cases**:
- **Equality queries**: WHERE column = value
- **Primary key lookups**: Fast row access
- **Memory databases**: In-memory hash tables
- **Unique constraints**: Duplicate detection

**Implementation**:
```python
import hashlib

class HashIndex:
    def __init__(self, initial_size=16):
        self.size = initial_size
        self.buckets = [[] for _ in range(self.size)]
        self.count = 0
        self.load_factor_threshold = 0.75
    
    def _hash(self, key):
        """Hash function for key"""
        if isinstance(key, str):
            return hash(key) % self.size
        return key % self.size
    
    def _resize(self):
        """Resize hash table when load factor exceeds threshold"""
        old_buckets = self.buckets
        self.size *= 2
        self.buckets = [[] for _ in range(self.size)]
        self.count = 0
        
        # Rehash all elements
        for bucket in old_buckets:
            for key, value in bucket:
                self.insert(key, value)
    
    def insert(self, key, value):
        """Insert key-value pair"""
        # Check load factor
        if self.count >= self.size * self.load_factor_threshold:
            self._resize()
        
        hash_value = self._hash(key)
        bucket = self.buckets[hash_value]
        
        # Update existing key
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        # Insert new key
        bucket.append((key, value))
        self.count += 1
    
    def search(self, key):
        """Search for key"""
        hash_value = self._hash(key)
        bucket = self.buckets[hash_value]
        
        for k, v in bucket:
            if k == key:
                return v
        
        return None
    
    def delete(self, key):
        """Delete key"""
        hash_value = self._hash(key)
        bucket = self.buckets[hash_value]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.count -= 1
                return v
        
        return None
    
    def get_statistics(self):
        """Get hash table statistics"""
        used_buckets = sum(1 for bucket in self.buckets if bucket)
        max_bucket_size = max(len(bucket) for bucket in self.buckets)
        avg_bucket_size = self.count / used_buckets if used_buckets > 0 else 0
        
        return {
            'size': self.size,
            'count': self.count,
            'load_factor': self.count / self.size,
            'used_buckets': used_buckets,
            'max_bucket_size': max_bucket_size,
            'avg_bucket_size': avg_bucket_size
        }

# Usage example
hash_index = HashIndex()
data = [("user1", {"name": "Alice"}), ("user2", {"name": "Bob"}), ("user3", {"name": "Charlie"})]

for key, value in data:
    hash_index.insert(key, value)

# Search
result = hash_index.search("user2")
print(f"Found user2: {result}")

# Statistics
stats = hash_index.get_statistics()
print(f"Hash index statistics: {stats}")
```

**Time Complexity**:
- Search: O(1) average, O(n) worst case
- Insert: O(1) average, O(n) worst case
- Delete: O(1) average, O(n) worst case

**Limitations**:
- **No range queries**: Only exact matches
- **No ordering**: Cannot support ORDER BY
- **Hash collisions**: Performance degrades with poor hash function

## 🔍 Specialized Indexing

### 1. **Bitmap Indexes**

**Structure**: Uses bitmaps to represent the presence of values in columns with low cardinality.

**Use Cases**:
- **Data warehousing**: OLAP queries
- **Low cardinality columns**: Gender, status, boolean flags
- **Complex queries**: Multiple WHERE conditions
- **Compression**: Efficient storage for sparse data

**Implementation**:
```python
class BitmapIndex:
    def __init__(self):
        self.bitmaps = {}  # value -> bitmap
        self.row_count = 0
    
    def insert(self, row_id, value):
        """Insert value at row_id"""
        # Ensure row_id doesn't exceed current row count
        if row_id >= self.row_count:
            self._extend_bitmaps(row_id + 1)
        
        # Set bit for this value
        if value not in self.bitmaps:
            self.bitmaps[value] = [0] * self.row_count
        
        self.bitmaps[value][row_id] = 1
    
    def _extend_bitmaps(self, new_size):
        """Extend all bitmaps to new size"""
        for value in self.bitmaps:
            self.bitmaps[value].extend([0] * (new_size - len(self.bitmaps[value])))
        self.row_count = new_size
    
    def search(self, value):
        """Get bitmap for value"""
        return self.bitmaps.get(value, [0] * self.row_count)
    
    def and_operation(self, value1, value2):
        """Bitmap AND operation"""
        bitmap1 = self.search(value1)
        bitmap2 = self.search(value2)
        return [a & b for a, b in zip(bitmap1, bitmap2)]
    
    def or_operation(self, value1, value2):
        """Bitmap OR operation"""
        bitmap1 = self.search(value1)
        bitmap2 = self.search(value2)
        return [a | b for a, b in zip(bitmap1, bitmap2)]
    
    def not_operation(self, value):
        """Bitmap NOT operation"""
        bitmap = self.search(value)
        return [1 - b for b in bitmap]
    
    def count(self, value):
        """Count rows with value"""
        return sum(self.search(value))
    
    def get_row_ids(self, bitmap):
        """Get row IDs from bitmap"""
        return [i for i, bit in enumerate(bitmap) if bit]

# Usage example
bitmap_index = BitmapIndex()

# Insert data: gender column
gender_data = [(0, 'M'), (1, 'F'), (2, 'M'), (3, 'F'), (4, 'M')]
for row_id, gender in gender_data:
    bitmap_index.insert(row_id, gender)

# Query: Find all males
male_bitmap = bitmap_index.search('M')
male_rows = bitmap_index.get_row_ids(male_bitmap)
print(f"Male rows: {male_rows}")

# Complex query: Find all non-females
not_female_bitmap = bitmap_index.not_operation('F')
not_female_rows = bitmap_index.get_row_ids(not_female_bitmap)
print(f"Not female rows: {not_female_rows}")
```

### 2. **Spatial Indexes (R-Tree)**

**Structure**: Tree data structure for indexing spatial data (rectangles, points, polygons).

**Use Cases**:
- **GIS applications**: Geographic information systems
- **Location-based services**: Find nearby restaurants
- **Gaming**: Collision detection
- **CAD systems**: Spatial queries

**Implementation**:
```python
class Rectangle:
    def __init__(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
    
    def intersects(self, other):
        """Check if this rectangle intersects with another"""
        return (self.min_x <= other.max_x and self.max_x >= other.min_x and
                self.min_y <= other.max_y and self.max_y >= other.min_y)
    
    def contains(self, other):
        """Check if this rectangle contains another"""
        return (self.min_x <= other.min_x and self.max_x >= other.max_x and
                self.min_y <= other.min_y and self.max_y >= other.max_y)
    
    def area(self):
        """Calculate area of rectangle"""
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)
    
    def union(self, other):
        """Create union rectangle"""
        return Rectangle(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y)
        )

class RTreeNode:
    def __init__(self, is_leaf=True, max_entries=4):
        self.entries = []  # List of (rectangle, data/child)
        self.is_leaf = is_leaf
        self.max_entries = max_entries
    
    def is_full(self):
        return len(self.entries) >= self.max_entries

class RTree:
    def __init__(self, max_entries=4):
        self.root = RTreeNode(is_leaf=True, max_entries=max_entries)
        self.max_entries = max_entries
    
    def insert(self, rectangle, data):
        """Insert rectangle with associated data"""
        leaf = self._choose_leaf(rectangle)
        leaf.entries.append((rectangle, data))
        
        if leaf.is_full():
            self._split_node(leaf)
    
    def _choose_leaf(self, rectangle):
        """Choose leaf node to insert rectangle"""
        node = self.root
        
        while not node.is_leaf:
            best_child = None
            best_increase = float('inf')
            
            for rect, child in node.entries:
                if rect.contains(rectangle):
                    # If contained, choose this child
                    best_child = child
                    break
                
                # Calculate area increase
                union_rect = rect.union(rectangle)
                increase = union_rect.area() - rect.area()
                
                if increase < best_increase:
                    best_increase = increase
                    best_child = child
            
            node = best_child
        
        return node
    
    def _split_node(self, node):
        """Split a full node"""
        if node == self.root:
            # Create new root
            new_root = RTreeNode(is_leaf=False, max_entries=self.max_entries)
            self.root = new_root
            new_root.entries.append((self._calculate_mbr(node), node))
        
        # Simple split: divide entries roughly in half
        mid = len(node.entries) // 2
        new_node = RTreeNode(is_leaf=node.is_leaf, max_entries=self.max_entries)
        
        new_node.entries = node.entries[mid:]
        node.entries = node.entries[:mid]
        
        # Update parent (simplified)
        # In a full implementation, you'd need to handle parent updates properly
    
    def _calculate_mbr(self, node):
        """Calculate minimum bounding rectangle for node"""
        if not node.entries:
            return Rectangle(0, 0, 0, 0)
        
        min_x = min(rect.min_x for rect, _ in node.entries)
        min_y = min(rect.min_y for rect, _ in node.entries)
        max_x = max(rect.max_x for rect, _ in node.entries)
        max_y = max(rect.max_y for rect, _ in node.entries)
        
        return Rectangle(min_x, min_y, max_x, max_y)
    
    def search(self, query_rect):
        """Search for rectangles that intersect with query"""
        results = []
        self._search_recursive(self.root, query_rect, results)
        return results
    
    def _search_recursive(self, node, query_rect, results):
        """Recursive search helper"""
        for rect, data_or_child in node.entries:
            if rect.intersects(query_rect):
                if node.is_leaf:
                    results.append((rect, data_or_child))
                else:
                    self._search_recursive(data_or_child, query_rect, results)

# Usage example
rtree = RTree(max_entries=4)

# Insert some spatial data
locations = [
    (Rectangle(0, 0, 10, 10), "Restaurant A"),
    (Rectangle(5, 5, 15, 15), "Restaurant B"),
    (Rectangle(20, 20, 30, 30), "Restaurant C"),
    (Rectangle(25, 25, 35, 35), "Restaurant D")
]

for rect, name in locations:
    rtree.insert(rect, name)

# Search for restaurants in area
search_area = Rectangle(0, 0, 12, 12)
results = rtree.search(search_area)
print(f"Restaurants in search area: {[name for _, name in results]}")
```

## 🚀 Index Optimization Strategies

### 1. **Covering Indexes**

**Strategy**: Include all columns needed for a query in the index to avoid table lookups.

```python
class CoveringIndex:
    def __init__(self):
        self.index = {}  # key -> (indexed_columns, included_columns)
    
    def create_covering_index(self, key_columns, included_columns):
        """Create covering index"""
        index_key = tuple(key_columns)
        self.index[index_key] = {
            'key_columns': key_columns,
            'included_columns': included_columns,
            'data': {}
        }
    
    def insert(self, key_columns, included_columns, row_data):
        """Insert data into covering index"""
        index_key = tuple(key_columns)
        if index_key in self.index:
            data_key = tuple(row_data[col] for col in self.index[index_key]['key_columns'])
            self.index[index_key]['data'][data_key] = row_data
    
    def search(self, key_columns, search_values):
        """Search using covering index"""
        index_key = tuple(key_columns)
        if index_key in self.index:
            data_key = tuple(search_values)
            return self.index[index_key]['data'].get(data_key)
        return None

# Usage example
covering_index = CoveringIndex()
covering_index.create_covering_index(['user_id'], ['name', 'email', 'status'])

# Insert data
row_data = {'user_id': 1, 'name': 'Alice', 'email': 'alice@example.com', 'status': 'active'}
covering_index.insert(['user_id'], ['name', 'email', 'status'], row_data)

# Search (no table lookup needed)
result = covering_index.search(['user_id'], [1])
print(f"Covering index result: {result}")
```

### 2. **Composite Indexes**

**Strategy**: Multi-column indexes for queries with multiple WHERE conditions.

```python
class CompositeIndex:
    def __init__(self):
        self.indexes = {}  # (col1, col2, ...) -> nested dict structure
    
    def create_composite_index(self, columns):
        """Create composite index on multiple columns"""
        index_key = tuple(columns)
        self.indexes[index_key] = {}
    
    def insert(self, columns, values, row_id):
        """Insert into composite index"""
        index_key = tuple(columns)
        if index_key not in self.indexes:
            return
        
        current_dict = self.indexes[index_key]
        
        # Navigate/create nested structure
        for i, value in enumerate(values[:-1]):
            if value not in current_dict:
                current_dict[value] = {}
            current_dict = current_dict[value]
        
        # Insert at final level
        final_value = values[-1]
        if final_value not in current_dict:
            current_dict[final_value] = []
        current_dict[final_value].append(row_id)
    
    def search(self, columns, values):
        """Search using composite index"""
        index_key = tuple(columns)
        if index_key not in self.indexes:
            return []
        
        current_dict = self.indexes[index_key]
        
        # Navigate nested structure
        for value in values:
            if value not in current_dict:
                return []
            current_dict = current_dict[value]
        
        return current_dict if isinstance(current_dict, list) else []

# Usage example
composite_index = CompositeIndex()
composite_index.create_composite_index(['department', 'status', 'level'])

# Insert data
composite_index.insert(['department', 'status', 'level'], ['Engineering', 'Active', 'Senior'], 1)
composite_index.insert(['department', 'status', 'level'], ['Engineering', 'Active', 'Junior'], 2)
composite_index.insert(['department', 'status', 'level'], ['Sales', 'Active', 'Senior'], 3)

# Search
results = composite_index.search(['department', 'status', 'level'], ['Engineering', 'Active', 'Senior'])
print(f"Composite index results: {results}")
```

## 🎯 Index Selection Guide

| Query Pattern | Recommended Index | Reason |
|---------------|-------------------|---------|
| **Equality (=)** | Hash Index | O(1) lookup time |
| **Range (>, <, BETWEEN)** | B+ Tree | Efficient range scans |
| **Full-text search** | Inverted Index | Text search optimization |
| **Spatial queries** | R-Tree | Multidimensional data |
| **Low cardinality** | Bitmap Index | Efficient for few distinct values |
| **Complex WHERE** | Composite Index | Multiple column optimization |
| **SELECT * queries** | Covering Index | Avoid table lookups |

## 💡 Best Practices

### 1. **Index Design Principles**

- **Selectivity**: Choose columns with high selectivity (many unique values)
- **Query patterns**: Index based on actual query usage
- **Maintenance overhead**: Balance query performance with write performance
- **Storage cost**: Consider disk space and memory usage

### 2. **Common Anti-Patterns**

- **Over-indexing**: Too many indexes slow down writes
- **Wrong column order**: Composite index column order matters
- **Unused indexes**: Remove indexes that aren't used
- **Duplicate indexes**: Avoid redundant index definitions

### 3. **Monitoring and Optimization**

- **Query analysis**: Use EXPLAIN plans to understand index usage
- **Performance metrics**: Monitor query execution times
- **Index statistics**: Track index hit ratios and efficiency
- **Regular maintenance**: Update statistics and rebuild fragmented indexes

---

**💡 Key Takeaway**: The right indexing strategy can transform a slow system into a high-performance one. Choose indexes based on query patterns, monitor their effectiveness, and maintain them properly for optimal performance.
