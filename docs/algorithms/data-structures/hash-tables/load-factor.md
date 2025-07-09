# Load Factor and Rehashing

## What is Load Factor?

The **load factor** of a hash table is a measure of how full the table is. It's calculated as:

```
Load Factor = Number of Entries / Size of Hash Table
```

This important metric helps determine when to resize a hash table to maintain efficiency and performance.

## Significance of Load Factor

The load factor directly impacts the performance of hash table operations:

| Load Factor | Impact on Performance |
|-------------|----------------------|
| Very low    | Wasted memory space, but fast operations |
| Optimal     | Good balance between memory usage and performance |
| High        | Risk of increased collisions and degraded performance |

Most hash table implementations maintain a load factor between 0.5 and 0.75 for optimal performance.

## Critical Thresholds

Different implementations use different thresholds for rehashing:

- **Java HashMap**: Rehashes when load factor exceeds 0.75 (default)
- **Python dictionaries**: Maintains load factor around 0.66
- **C++ unordered_map**: Implementation-dependent, typically 1.0

## Rehashing Process

When the load factor exceeds a predefined threshold, **rehashing** occurs to maintain performance:

1. **Allocate a larger table**: Typically double the size of the original table
2. **Recompute hash codes**: For each entry in the original table
3. **Redistribute entries**: Place each entry in its new position in the larger table
4. **Replace the old table**: Use the new table for future operations

### Pseudocode for Rehashing

```python
def rehash():
    old_table = current_hash_table
    # Create a new table with double the size
    new_table = create_new_hash_table(current_size * 2)
    
    # Rehash all entries from the old table to the new table
    for entry in old_table:
        if entry is not None:
            new_position = hash_function(entry.key) % new_table.size
            place_in_new_table(new_table, new_position, entry)
    
    # Replace the old table with the new one
    current_hash_table = new_table
```

## Performance Considerations

### Time Complexity

- **Average Case**: O(n) where n is the number of entries
- **Amortized Cost**: O(1) per insertion when averaged over many operations

### Memory Considerations

- Each rehashing operation temporarily requires memory for both tables
- Doubling the size leads to exponential memory growth
- Some implementations use incremental rehashing to distribute cost

## Dynamic Resizing Strategies

### Growing Strategies

Most hash tables resize by doubling their capacity when the load factor exceeds a threshold:

```python
if (num_entries / table_size) > LOAD_FACTOR_THRESHOLD:
    resize_table(table_size * 2)
```

### Shrinking Strategies

Some implementations also shrink the hash table when the load factor becomes too small:

```python
if (num_entries / table_size) < SHRINK_THRESHOLD and table_size > MINIMUM_SIZE:
    resize_table(table_size / 2)
```

## Load Factor in Different Collision Resolution Methods

### Separate Chaining

With separate chaining, higher load factors are tolerable since collisions are managed through linked lists or other data structures.

### Open Addressing

Open addressing is more sensitive to load factor increases:

- **Linear Probing**: Performance degrades rapidly as load factor approaches 0.7
- **Quadratic Probing**: Better than linear probing at higher load factors
- **Double Hashing**: Most resilient to higher load factors among open addressing methods

## Advanced Techniques

### Robin Hood Hashing

Reduces variance in probe sequence lengths to maintain performance even at higher load factors.

### Cuckoo Hashing

Uses multiple hash functions and maintains a load factor below 0.5 to guarantee O(1) worst-case lookup.

## Implementation Examples

### Java Implementation

```java
public class CustomHashMap<K, V> {
    private static final float DEFAULT_LOAD_FACTOR = 0.75f;
    private Entry<K, V>[] table;
    private int size;
    private int threshold;
    private float loadFactor;
    
    public CustomHashMap(int initialCapacity, float loadFactor) {
        // Implementation details
        this.loadFactor = loadFactor;
        this.table = new Entry[initialCapacity];
        this.threshold = (int)(initialCapacity * loadFactor);
    }
    
    public V put(K key, V value) {
        // If adding this entry would exceed the threshold, rehash
        if (size >= threshold) {
            rehash();
        }
        // Add the entry
        // ...
    }
    
    private void rehash() {
        Entry<K, V>[] oldTable = table;
        int oldCapacity = oldTable.length;
        int newCapacity = oldCapacity * 2;
        
        Entry<K, V>[] newTable = new Entry[newCapacity];
        threshold = (int)(newCapacity * loadFactor);
        
        // Transfer all entries to the new table
        for (int i = 0; i < oldCapacity; i++) {
            Entry<K, V> e = oldTable[i];
            while (e != null) {
                Entry<K, V> next = e.next;
                int index = hash(e.key) % newCapacity;
                e.next = newTable[index];
                newTable[index] = e;
                e = next;
            }
        }
        
        table = newTable;
    }
}
```

### Python-like Implementation

```python
class HashMap:
    def __init__(self, initial_capacity=16, load_factor=0.75):
        self.capacity = initial_capacity
        self.load_factor = load_factor
        self.size = 0
        self.buckets = [None] * initial_capacity
        
    def put(self, key, value):
        # Check if rehashing is needed
        if (self.size + 1) / self.capacity > self.load_factor:
            self._rehash()
            
        # Insert the key-value pair
        # ...
        
    def _rehash(self):
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [None] * self.capacity
        self.size = 0
        
        # Reinsert all entries
        for bucket in old_buckets:
            if bucket:
                for key, value in bucket:
                    self.put(key, value)
```

## Common Pitfalls and Best Practices

### Pitfalls

- **Setting load factor too high**: Causes excessive collisions
- **Setting load factor too low**: Wastes memory
- **Forgetting to rehash**: Leads to performance degradation
- **Inefficient rehashing implementation**: Creates performance bottlenecks

### Best Practices

- Choose appropriate initial capacity to minimize early rehashing
- Use proven load factor thresholds (0.6-0.75 for most applications)
- Consider application access patterns when tuning load factors
- For read-heavy applications, lower load factors may be beneficial
- For memory-constrained systems, higher load factors may be necessary

## Conclusion

Load factor and rehashing are fundamental concepts in hash table implementation that directly impact performance and memory usage. Proper management of load factor through timely rehashing ensures that hash table operations maintain their expected O(1) time complexity, making them one of the most efficient data structures for key-value storage and retrieval.
