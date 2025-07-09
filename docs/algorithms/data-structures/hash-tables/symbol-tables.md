# Symbol Tables

## Introduction to Symbol Tables

A **symbol table** is a data structure that associates names (symbols) with values or data. It is one of the most fundamental and practical applications of hash tables. Symbol tables are essential components in many computing systems, including:

- Compilers and interpreters
- Assemblers
- Database indexing systems
- Network routing tables
- Operating system kernels
- File systems

## Structure and Operations

At its core, a symbol table implements the following key operations:

| Operation | Description |
|-----------|-------------|
| `put(key, value)` | Inserts a key-value pair into the table |
| `get(key)` | Retrieves the value associated with the key |
| `delete(key)` | Removes the key-value pair from the table |
| `contains(key)` | Checks if the key exists in the table |
| `size()` | Returns the number of key-value pairs |
| `isEmpty()` | Checks if the table is empty |
| `keys()` | Returns all keys in the table |

## Implementation Approaches

Symbol tables can be implemented using various underlying data structures, each with different performance characteristics:

### Hash Table-Based Implementation

```java
public class HashSymbolTable<Key, Value> {
    private int size;
    private int capacity;
    private Key[] keys;
    private Value[] values;
    private static final int INITIAL_CAPACITY = 16;
    
    public HashSymbolTable() {
        this(INITIAL_CAPACITY);
    }
    
    @SuppressWarnings("unchecked")
    public HashSymbolTable(int capacity) {
        this.capacity = capacity;
        keys = (Key[]) new Object[capacity];
        values = (Value[]) new Object[capacity];
    }
    
    private int hash(Key key) {
        return (key.hashCode() & 0x7fffffff) % capacity;
    }
    
    public void put(Key key, Value value) {
        if (key == null) throw new IllegalArgumentException("Key cannot be null");
        if (value == null) {
            delete(key);
            return;
        }
        
        // Resize if necessary
        if (size >= capacity / 2) resize(2 * capacity);
        
        int i = hash(key);
        while (keys[i] != null) {
            if (keys[i].equals(key)) {
                values[i] = value;
                return;
            }
            i = (i + 1) % capacity;
        }
        keys[i] = key;
        values[i] = value;
        size++;
    }
    
    public Value get(Key key) {
        if (key == null) throw new IllegalArgumentException("Key cannot be null");
        int i = hash(key);
        while (keys[i] != null) {
            if (keys[i].equals(key)) return values[i];
            i = (i + 1) % capacity;
        }
        return null;
    }
    
    public void delete(Key key) {
        // Implementation details
    }
    
    private void resize(int newCapacity) {
        // Implementation details
    }
}
```

### Binary Search Tree Implementation

For ordered symbol tables where keys have a natural order:

```java
public class BSTSymbolTable<Key extends Comparable<Key>, Value> {
    private Node root;
    
    private class Node {
        private Key key;
        private Value value;
        private Node left, right;
        private int size;
        
        public Node(Key key, Value value, int size) {
            this.key = key;
            this.value = value;
            this.size = size;
        }
    }
    
    public int size() {
        return size(root);
    }
    
    private int size(Node x) {
        if (x == null) return 0;
        return x.size;
    }
    
    public Value get(Key key) {
        return get(root, key);
    }
    
    private Value get(Node x, Key key) {
        if (x == null) return null;
        int cmp = key.compareTo(x.key);
        if (cmp < 0) return get(x.left, key);
        else if (cmp > 0) return get(x.right, key);
        else return x.value;
    }
    
    public void put(Key key, Value value) {
        root = put(root, key, value);
    }
    
    private Node put(Node x, Key key, Value value) {
        if (x == null) return new Node(key, value, 1);
        int cmp = key.compareTo(x.key);
        if (cmp < 0) x.left = put(x.left, key, value);
        else if (cmp > 0) x.right = put(x.right, key, value);
        else x.value = value;
        x.size = 1 + size(x.left) + size(x.right);
        return x;
    }
}
```

## Applications in Different Domains

### Compiler Symbol Tables

In compilers, symbol tables track various entities:

```
┌────────────────────────────────────────────┐
│            Compiler Symbol Table           │
├────────────┬───────────┬──────────┬────────┤
│ Identifier │   Type    │  Scope   │ Other  │
├────────────┼───────────┼──────────┼────────┤
│ counter    │ int       │ global   │ Line 5 │
│ max        │ constant  │ global   │ Line 3 │
│ sum        │ float     │ function │ Line 8 │
│ calculate  │ function  │ global   │ Line 7 │
└────────────┴───────────┴──────────┴────────┘
```

#### Key Features of Compiler Symbol Tables:

1. **Scope Management**: 
   - Global scope
   - Function scope
   - Block scope
   - Class scope (in OOP languages)

2. **Symbol Information**:
   - Data type
   - Memory location/offset
   - Line number
   - Parameter count (for functions)
   - Return type (for functions)

3. **Type Checking**:
   - Verify operations are performed on compatible types
   - Enforce language type rules

### Database Indexing

Symbol tables form the basis of database indexes:

```
┌────────────────────────────────────────┐
│          Database Index Table          │
├────────────┬─────────────────────┬─────┤
│ Primary Key│   Record Location   │ Meta│
├────────────┼─────────────────────┼─────┤
│ 1001       │ Block 54, Offset 12 │ ... │
│ 1002       │ Block 67, Offset 8  │ ... │
│ 1003       │ Block 54, Offset 40 │ ... │
└────────────┴─────────────────────┴─────┘
```

### Operating Systems

Symbol tables in operating systems track:

1. **Process Tables**: Map process IDs to process control blocks
2. **File Tables**: Map file descriptors to file objects
3. **Page Tables**: Map virtual addresses to physical memory addresses

## Performance Considerations

The choice of implementation affects performance characteristics:

| Implementation | Search | Insert | Delete | Ordered Iteration |
|---------------|--------|--------|--------|-------------------|
| Hash Table    | O(1)   | O(1)   | O(1)   | O(N log N)        |
| BST (unbalanced)| O(N) | O(N)   | O(N)   | O(N)              |
| BST (balanced)| O(log N) | O(log N) | O(log N) | O(N)        |
| Red-Black Tree| O(log N) | O(log N) | O(log N) | O(N)        |

## Advanced Symbol Table Concepts

### Chaining vs. Open Addressing

Symbol tables implemented with hash tables must handle collisions:

- **Chaining**: Store colliding entries in linked lists
- **Open Addressing**: Find another slot in the table when collision occurs

### Deletion Challenges

Deletion in open-addressing schemes requires special handling to maintain proper lookup:

- **Tombstone Approach**: Mark deleted slots as "deleted" rather than empty
- **Rehashing Approach**: Rehash entries after deletion

### Thread Safety

For multi-threaded environments, symbol tables need thread-safety mechanisms:

```java
public class ConcurrentSymbolTable<K, V> {
    private final ConcurrentHashMap<K, V> map = new ConcurrentHashMap<>();
    
    public V get(K key) {
        return map.get(key);
    }
    
    public void put(K key, V value) {
        map.put(key, value);
    }
    
    public boolean containsKey(K key) {
        return map.containsKey(key);
    }
    
    // Additional methods
}
```

## Symbol Tables in Programming Languages

### Java Implementation

Java's `HashMap` and `TreeMap` serve as symbol table implementations:

```java
// Hash-based symbol table
HashMap<String, Integer> hashSymbolTable = new HashMap<>();
hashSymbolTable.put("counter", 10);
hashSymbolTable.put("max", 100);
System.out.println(hashSymbolTable.get("counter")); // 10

// Order-based symbol table
TreeMap<String, Integer> orderedSymbolTable = new TreeMap<>();
orderedSymbolTable.put("counter", 10);
orderedSymbolTable.put("max", 100);
orderedSymbolTable.put("min", 1);
// Keys are returned in sorted order
for (String key : orderedSymbolTable.keySet()) {
    System.out.println(key);  // "counter", "max", "min"
}
```

### Python Implementation

Python dictionaries serve as symbol tables:

```python
# Python dictionary as a symbol table
symbol_table = {}
symbol_table["counter"] = 10
symbol_table["max"] = 100

print(symbol_table["counter"])  # 10

# Ordered dictionary maintains insertion order
from collections import OrderedDict
ordered_symbols = OrderedDict()
ordered_symbols["counter"] = 10
ordered_symbols["max"] = 100
ordered_symbols["min"] = 1

for key in ordered_symbols:
    print(key)  # "counter", "max", "min"
```

## Best Practices and Optimization Techniques

1. **Choose the Right Implementation**:
   - Need ordered operations? Use BST or balanced tree
   - Need fastest lookups? Use hash table
   - Need predictable performance? Use balanced trees

2. **Sizing and Capacity Planning**:
   - Estimate maximum expected entries
   - Set initial capacity to reduce rehashing
   - Set appropriate load factor

3. **Key Design**:
   - Ensure proper `equals()` and `hashCode()` implementation
   - Immutable keys preferred
   - Use simple, fast hash functions

4. **Handling Edge Cases**:
   - Null keys
   - Duplicate keys
   - Very large symbol tables
   - String interning for string keys

## Conclusion

Symbol tables represent one of the most powerful applications of hash tables and other key-value data structures. Their efficient implementation is crucial for the performance of many computing systems. Whether implemented using hash tables for raw speed or balanced trees for ordered operations, symbol tables demonstrate the importance of choosing the right data structure for specific application needs.
