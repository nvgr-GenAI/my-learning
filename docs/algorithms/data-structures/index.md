# Data Structures

Master the fundamental building blocks of efficient algorithms. The right data structure can transform an O(n²) solution into O(n) or O(log n).

---

## What Are Data Structures?

Data structures are specialized formats for organizing, processing, and storing data in a computer's memory. They define:

- **How data is arranged** - Sequential, hierarchical, or associative
- **What operations are efficient** - Access, search, insert, delete
- **Memory and performance trade-offs** - Space vs time complexity

**Why they matter:** An algorithm's efficiency depends heavily on the underlying data structure. Searching an unsorted array takes O(n) time, but searching a hash table takes O(1). Choosing the right structure is often more important than optimizing the algorithm itself.

**Real-world analogy:**

- **Array** = Parking lot with numbered spaces (find by number instantly)
- **Linked List** = Train cars connected by couplers (add/remove cars easily)
- **Stack** = Stack of plates (only access top plate)
- **Hash Table** = Library card catalog (look up by title instantly)
- **Heap** = Priority queue at hospital (most urgent patient seen first)

---

## Linear Data Structures

Elements arranged in **sequential order**, where each element has a predecessor and successor (except first and last).

**Characteristic:** Data flows in a line - you traverse from one element to the next.

**Use when:** Your data naturally has an order, you need to process items sequentially, or you're implementing other structures (stacks/queues often use arrays or linked lists internally).

| Structure | Access | Insert | Delete | Space | Memory Layout | Best Use Case |
|-----------|--------|--------|--------|-------|---------------|---------------|
| **[Arrays](arrays/index.md)** | O(1) | O(n) | O(n) | O(n) | Contiguous block | Random access, iteration, fixed size |
| **[Linked Lists](linked-lists/index.md)** | O(n) | O(1)* | O(1)* | O(n) | Scattered nodes | Frequent insert/delete, unknown size |
| **[Stacks](stacks/index.md)** | O(n) | O(1) | O(1) | O(n) | Array or List | LIFO: undo, backtracking, DFS |
| **[Queues](queues/index.md)** | O(n) | O(1) | O(1) | O(n) | Array or List | FIFO: BFS, task scheduling, buffering |

`*` When you have direct reference to the node

### Key Differences

**Arrays vs Linked Lists:**

- **Array:** Fast access by index O(1), slow insert/delete O(n) due to shifting
- **Linked List:** Slow access O(n) traversal, fast insert/delete O(1) with pointer update

**Stacks vs Queues:**

- **Stack (LIFO):** Last item added is first removed - like browser back button
- **Queue (FIFO):** First item added is first removed - like checkout line

---

## Associative Data Structures

Elements accessed by **key** rather than position. No inherent order (in hash-based) or ordered by key (in tree-based).

**Characteristic:** Direct lookup using a key - no need to search sequentially.

**Use when:** You need to find data by identifier (not position), check membership, or ensure uniqueness.

| Structure | Search | Insert | Delete | Space | Internal Storage | Best Use Case |
|-----------|--------|--------|--------|-------|------------------|---------------|
| **[Hash Tables](hash-tables/index.md)** | O(1)* | O(1)* | O(1)* | O(n) | Array + hash function | Key-value mapping, counting, caching |
| **[Sets](sets/index.md)** | O(1)* | O(1)* | O(1)* | O(n) | Hash table or tree | Unique elements, membership, deduplication |

`*` Average case; worst case O(n) with poor hash function or many collisions

### Key Characteristics

**Hash Tables:**

- Store key-value pairs
- Use hash function to compute array index from key
- Handle collisions with chaining (linked lists) or open addressing (probing)
- No ordering - iteration order is unpredictable
- **Trade-off:** Fast operations but extra memory for load factor

**Sets:**

- Store only keys (no values) - enforce uniqueness
- Same performance as hash tables
- Hash Set: unordered, O(1) operations
- Tree Set: ordered, O(log n) operations
- **Trade-off:** Can't store duplicates or counts

---

## Hierarchical Data Structures

Elements arranged in **parent-child relationships** forming tree-like structures. Each node can have multiple children but only one parent.

**Characteristic:** Data organized in levels - root at top, leaves at bottom.

**Use when:** Your data naturally has hierarchy (file systems, org charts) or you need to maintain ordering with efficient updates.

| Structure | Search | Insert | Delete | Space | Internal Storage | Best Use Case |
|-----------|--------|--------|--------|-------|------------------|---------------|
| **Heaps** | O(n) | O(log n) | O(log n) | O(n) | Array (complete binary tree) | Priority queue, top K elements, merge K sorted |
| **BST (balanced)** | O(log n) | O(log n) | O(log n) | O(n) | Nodes with pointers | Ordered data, range queries, sorted iteration |
| **Trie (Prefix Tree)** | O(m)* | O(m)* | O(m)* | O(ALPHABET × N × M) | Tree with character edges | Autocomplete, prefix search, dictionary, spell check |

`*` m = length of string/word

### Key Characteristics

**Heaps:**

- Complete binary tree (all levels filled except possibly last)
- **Min Heap:** Parent ≤ children (root is minimum)
- **Max Heap:** Parent ≥ children (root is maximum)
- Stored in array: parent at i, children at 2i+1 and 2i+2
- **Trade-off:** Can only efficiently access min/max, not arbitrary elements

**Binary Search Trees (BST):**

- Left child < parent < right child (for all nodes)
- In-order traversal gives sorted sequence
- Must be balanced (AVL, Red-Black) for O(log n) guarantees
- **Trade-off:** More complex than arrays, but maintains order efficiently

**Trie (Prefix Tree):**

- Tree where each node represents a character/prefix
- Words share common prefixes (memory efficient for many words)
- Root is empty, path from root to node forms prefix
- Each node has up to ALPHABET_SIZE children (26 for lowercase English)
- **Trade-off:** Fast prefix operations but high space for sparse data

**See [Trees section](../trees/index.md) for comprehensive tree coverage.**

---

## Advanced & Specialized Data Structures

Sophisticated structures for specific use cases and advanced algorithms.

**Use when:** Standard structures aren't efficient enough for your specific problem domain.

| Structure | Primary Operation | Complexity | Space | Best Use Case |
|-----------|------------------|------------|-------|---------------|
| **Graphs** | Various | Varies | O(V + E) | Networks, relationships, paths, connectivity |
| **Disjoint Set (Union-Find)** | Union, Find | O(α(n))* | O(n) | Connected components, cycle detection, MST |
| **Segment Tree** | Range query/update | O(log n) | O(n) | Range sum/min/max with updates, interval queries |
| **Fenwick Tree (BIT)** | Prefix sum/update | O(log n) | O(n) | Cumulative frequency, range sums, inversion count |
| **Bloom Filter** | Membership test | O(k)** | O(m) | Probabilistic membership, space-efficient sets |

`*` α(n) = Inverse Ackermann function (effectively constant)
`**` k = number of hash functions

### Key Characteristics

**Graphs:**

- Collection of vertices (nodes) connected by edges
- **Directed:** Edges have direction (A→B)
- **Undirected:** Edges bidirectional (A—B)
- **Weighted:** Edges have costs/distances
- Representations: Adjacency List O(V+E), Adjacency Matrix O(V²)
- **Trade-off:** Flexible for relationships but complex algorithms (BFS, DFS, Dijkstra, etc.)

**Disjoint Set (Union-Find):**

- Tracks elements partitioned into disjoint sets
- Two operations: Union (merge sets), Find (which set does element belong to)
- Path compression + union by rank → nearly O(1) operations
- **Trade-off:** Only tracks connectivity, not actual paths

**Segment Tree:**

- Binary tree for range queries (sum, min, max) with updates
- Each node stores info about a range [L, R]
- Build: O(n), Query: O(log n), Update: O(log n)
- **Trade-off:** 4× space overhead, complex implementation

**Fenwick Tree (Binary Indexed Tree):**

- Array-based structure for prefix sums and range queries
- More space-efficient than Segment Tree (just array)
- Only works for reversible operations (sum, XOR)
- **Trade-off:** Less intuitive than Segment Tree, limited to certain operations

**Bloom Filter:**

- Probabilistic data structure for membership testing
- Can say "definitely not in set" or "probably in set"
- False positives possible, false negatives impossible
- **Trade-off:** Space-efficient but allows false positives, can't delete elements

### Interview Relevance

| Structure | Interview Frequency | Typical Level | Example Problems |
|-----------|-------------------|---------------|------------------|
| **Graphs** | Very Common | All levels | Number of Islands, Course Schedule, Clone Graph |
| **Trie** | Common | Medium-Hard | Implement Trie, Word Search II, Autocomplete System |
| **Disjoint Set** | Common | Medium-Hard | Number of Provinces, Redundant Connection, Accounts Merge |
| **Segment Tree** | Rare | Hard/Competitive | Range Sum Query 2D Mutable, Count of Smaller After Self |
| **Fenwick Tree** | Rare | Hard/Competitive | Range Sum Query, Count of Smaller After Self |
| **Bloom Filter** | Very Rare | System Design | Cache design, Distributed systems |

---

## Decision Guide

### When to Use Each Structure

| Scenario | Use This Structure | Why | Complexity |
|----------|-------------------|-----|------------|
| **Access element by index/position** | Array | Direct memory offset calculation | O(1) |
| **Search by key/identifier** | Hash Table | Hash function computes index instantly | O(1) avg |
| **Frequent insert/delete at beginning/end** | Linked List or Deque | Just update pointers, no shifting | O(1) |
| **Frequent insert/delete in middle** | Linked List | Update pointers at location | O(1) if you have reference |
| **Need LIFO (last-in-first-out)** | Stack | Top element always accessible | O(1) push/pop |
| **Need FIFO (first-in-first-out)** | Queue | Front/rear pointers | O(1) enqueue/dequeue |
| **Check if element exists** | Set or Hash Table | Hash-based lookup | O(1) avg |
| **Eliminate duplicates** | Set | Automatically rejects duplicates | O(1) per element |
| **Count frequencies** | Hash Table | Map key → count | O(1) per lookup/update |
| **Repeatedly find minimum/maximum** | Heap | Root always holds min/max | O(1) peek, O(log n) insert/delete |
| **Find K largest/smallest elements** | Heap (size K) | Maintain K elements in heap | O(n log k) |
| **Maintain sorted order with updates** | BST (balanced) or Heap | Logarithmic insert maintains order | O(log n) |
| **Range queries on sorted data** | BST (balanced) | Binary search on subtrees | O(log n) |
| **Iterate in sorted order** | Array (sorted) or Tree Set | Natural ordering | O(n) iteration |
| **Autocomplete/prefix search** | Trie | Words share prefixes, instant prefix lookup | O(m) m=word length |
| **Spell checking/dictionary** | Trie | Fast word lookup and prefix matching | O(m) |
| **Model relationships/networks** | Graph | Vertices and edges represent connections | Varies by algorithm |
| **Track connected components** | Disjoint Set (Union-Find) | Near O(1) union and find operations | O(α(n)) ≈ O(1) |
| **Range sum/min/max with updates** | Segment Tree or Fenwick Tree | Logarithmic query and update | O(log n) |
| **Space-efficient membership (approximate)** | Bloom Filter | Probabilistic, very space-efficient | O(k) k=hash functions |
| **Find shortest path** | Graph (BFS/Dijkstra) | Models network/map as graph | O(V+E) or O(E log V) |
| **Detect cycles** | Graph (DFS) or Union-Find | Track visited nodes or components | O(V+E) or O(α(n)) |

---

## Selection Based on Constraints

| Primary Constraint | Recommended Structure | Avoid | Reason |
|-------------------|----------------------|-------|--------|
| **Memory is tight** | Array, Bit Set | Linked List | Pointers add 8-16 bytes overhead per node |
| **Size unknown at creation** | Dynamic Array, Linked List | Static Array | Can't resize fixed arrays |
| **Need specific ordering** | Array, Tree Set, BST | Hash Table, Hash Set | Hash structures don't maintain order |
| **Many lookup operations** | Hash Table, Set | Array, Linked List | O(1) vs O(n) for lookups |
| **Many insert/delete operations** | Linked List, Heap | Array | Array requires O(n) shifting |
| **Need access from both ends** | Deque, Doubly Linked List | Stack, Queue, Singly Linked | Single-ended structures restrict access |
| **Need index-based access** | Array | Linked List | Array O(1), Linked List O(n) |
| **Need fast min/max repeatedly** | Heap | Sorted Array | Heap O(1) peek, Array O(1) but O(n) updates |
| **Cache performance critical** | Array | Linked List | Arrays have better spatial locality |

---

## Common Mistakes & Solutions

| ❌ Common Mistake | Problem | ✅ Solution | Why Better |
|------------------|---------|------------|------------|
| Using array for frequent middle insertions | O(n) to shift elements | Use Linked List | O(1) pointer updates |
| Using linked list for index access | O(n) to traverse | Use Array | O(1) direct access |
| Using array/list for membership checks | O(n) linear search | Use Set or Hash Table | O(1) hash lookup |
| Using hash table when need ordering | No guaranteed order | Use Tree Set or sorted array | Tree Set maintains order |
| Searching unsorted array | O(n) every time | Sort once + Binary Search, or use Hash Table | O(log n) or O(1) |
| Not using stack for backtracking | Manual tracking complex | Use Stack | LIFO naturally handles backtracking |
| Using stack when need both ends | Can't access bottom | Use Deque | Access both ends O(1) |
| Using single-ended queue for sliding window | Can't remove from both sides | Use Deque | Remove from both ends |
| Using array for priority queue | O(n) to find min/max | Use Heap | O(1) to peek, O(log n) to update |
| Linear search on sorted array | O(n) when could be O(log n) | Use Binary Search | Exploit sorted property |

---

## Quick Comparison

### Lookup Performance

| Structure | By Index | By Key | By Value | Sorted Iteration |
|-----------|----------|--------|----------|------------------|
| Array | O(1) ✓ | - | O(n) | O(n log n) must sort first |
| Linked List | O(n) | - | O(n) | O(n log n) must sort first |
| Hash Table | - | O(1) ✓ | - | Not possible |
| Hash Set | - | O(1) ✓ | O(1) ✓ | Not possible |
| Tree Set | - | O(log n) | O(log n) | O(n) ✓ |
| Heap | - | - | O(n) | O(n log n) heap sort |

### Modification Performance

| Structure | Insert Beginning | Insert End | Insert Middle | Delete | Resize |
|-----------|-----------------|------------|---------------|--------|--------|
| Static Array | O(n) shift | O(n) if full | O(n) shift | O(n) shift | Not possible |
| Dynamic Array | O(n) shift | O(1) amortized | O(n) shift | O(n) shift | O(n) copy |
| Linked List | O(1) ✓ | O(1) ✓ | O(1)* ✓ | O(1)* ✓ | Not needed |
| Stack | - | O(1) (push) ✓ | - | O(1) (pop) ✓ | - |
| Queue | O(1) (enqueue) ✓ | - | - | O(1) (dequeue) ✓ | - |
| Hash Table | - | - | O(1) avg ✓ | O(1) avg ✓ | O(n) rehash |
| Heap | - | O(log n) | - | O(log n) | - |

`*` When you have reference to the node

---

## Explore Data Structures

| Structure | Learn | Key Use Cases |
|-----------|-------|---------------|
| **[Arrays](arrays/index.md)** | Static, Dynamic, Multidimensional | Two Pointers, Sliding Window, Binary Search, Matrix problems |
| **[Linked Lists](linked-lists/index.md)** | Singly, Doubly, Circular | Fast & Slow Pointers, Reversal, Cycle Detection, Merge operations |
| **[Stacks](stacks/index.md)** | Array-based, Linked List-based | Monotonic Stack, Expression Evaluation, Parentheses Matching, DFS |
| **[Queues](queues/index.md)** | Simple, Circular, Deque, Priority | BFS, Sliding Window Maximum, Level-order traversal, Task scheduling |
| **[Hash Tables](hash-tables/index.md)** | Chaining, Open Addressing | Two Sum pattern, Frequency counting, Caching, Grouping |
| **[Sets](sets/index.md)** | Hash Set, Tree Set, Bit Set | Deduplication, Set operations (union, intersection), Membership testing |
| **[Heaps](../trees/heaps.md)** | Min Heap, Max Heap | Top K elements, Merge K sorted lists, Priority queue, Running median |
| **[Trie](../trees/trie.md)** | Prefix Tree, Suffix Trie | Autocomplete, Dictionary, Prefix search, Spell checker, Word Search II |
| **[Graphs](../trees/graphs.md)** | Directed, Undirected, Weighted | Shortest path, Network flow, Social networks, Topological sort, MST |
| **Disjoint Set** | Union-Find, Path Compression | Connected components, Kruskal's MST, Cycle detection, Network connectivity |
| **Segment Tree** | Range queries, Lazy propagation | Range sum/min/max with updates, Interval queries, Competitive programming |
| **Fenwick Tree** | Binary Indexed Tree | Prefix sums, Range updates, Inversion count, Cumulative frequency |
| **Bloom Filter** | Probabilistic set | Cache filtering, Spell checking, Database query optimization, Deduplication

---

**Remember:** The best data structure is the one that makes your key operations efficient. Always:

1. Identify the operations you'll perform most frequently
2. Choose the structure that optimizes those operations
3. Consider space-time trade-offs for your specific constraints
