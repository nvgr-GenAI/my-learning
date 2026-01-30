# Time & Space Complexity Analysis

Understanding algorithmic complexity is fundamental to writing efficient code. This guide covers how to analyze and optimize runtime and memory usage.

!!! tip "Advanced Topics"
    Looking for deeper analysis? Check out [Advanced Complexity Analysis](complexity-analysis-advanced.md) covering:

    - **Analyzing Recursive Algorithms** (Master Theorem, recursion trees)
    - **Amortized Analysis** (dynamic arrays, union-find)
    - **Space Complexity Deep Dive** (call stack, in-place algorithms)
    - **Hidden Complexity Traps** (string concatenation, nested library calls)
    - **Interview Communication** (how to explain complexity)
    - **Code Examples** by complexity class

---

## 📊 Complexity Fundamentals

=== "Big O Basics"
    **What is Big O?** Describes how runtime/memory grows as input size increases.

    **Key Principle:** Focus on **rate of growth**, not exact operations.

    **Growth Order:** O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(n³) < O(2ⁿ) < O(n!)

    | Complexity | Name | Performance | Example Operations |
    |------------|------|-------------|-------------------|
    | O(1) | Constant | 🟢 Instant | Array access, hash lookup, stack push/pop |
    | O(log n) | Logarithmic | 🟢 Super Fast | Binary search, BST operations |
    | O(n) | Linear | 🟡 Fast | Linear search, array traversal |
    | O(n log n) | Linearithmic | 🟡 Moderate | Merge sort, heap sort |
    | O(n²) | Quadratic | 🔴 Slow | Bubble sort, nested loops |
    | O(n³) | Cubic | 🔴 Very Slow | Matrix multiply, triple nested loops |
    | O(2ⁿ) | Exponential | ⚫ Terrible | Recursive Fibonacci, subsets |
    | O(n!) | Factorial | ⚫ Impossible | Permutations, TSP |

=== "Growth Comparison"
    **Relative Performance by Input Size:**

    | Input Size | O(1) | O(log n) | O(n) | O(n log n) | O(n²) | O(2ⁿ) | O(n!) |
    |------------|------|----------|------|------------|-------|-------|-------|
    | n=10 | 1 | 3 | 10 | 33 | 100 | 1K | 3.6M |
    | n=100 | 1 | 7 | 100 | 664 | 10K | 10³⁰ | 10¹⁵⁸ |
    | n=1,000 | 1 | 10 | 1K | 10K | 1M | 10³⁰¹ | 10²⁵⁶⁸ |

    **⏰ Practical Limits:**
    - **n=10:** All algorithms acceptable
    - **n=100:** O(n²) starts slowing down
    - **n=1,000:** O(n³) becomes impractical
    - **n=1M:** Only O(1), O(log n), O(n) feasible

=== "Analysis Types"
    | Type | Description | When to Use |
    |------|-------------|-------------|
    | **Best Case** | Optimal conditions | Theoretical understanding |
    | **Average Case** | Normal conditions | Practical everyday use |
    | **Worst Case** | Most unfavorable | Reliability guarantees |
    | **Amortized** | Average over many ops | Occasional expensive operations |

---

## 🗂️ Data Structures Complexity

=== "Arrays & Lists"
    | Operation | Array | Dynamic Array | Linked List | Doubly-Linked |
    |-----------|-------|---------------|-------------|---------------|
    | Access | O(1) 🟢 | O(1) 🟢 | O(n) 🟡 | O(n) 🟡 |
    | Search | O(n) 🟡 | O(n) 🟡 | O(n) 🟡 | O(n) 🟡 |
    | Insert (start) | O(n) 🟡 | O(n) 🟡 | O(1) 🟢 | O(1) 🟢 |
    | Insert (end) | O(1)* 🟢 | O(1)** 🟢 | O(n) 🟡 | O(1)*** 🟢 |
    | Delete (start) | O(n) 🟡 | O(n) 🟡 | O(1) 🟢 | O(1) 🟢 |
    | Delete (end) | O(1) 🟢 | O(1) 🟢 | O(n) 🟡 | O(1) 🟢 |

    *If size known | **Amortized | ***With tail pointer

=== "Trees & Hash Tables"
    | Operation | BST (Balanced) | BST (Unbalanced) | Hash Table | AVL/Red-Black |
    |-----------|----------------|------------------|------------|---------------|
    | Access | O(log n) 🟢 | O(n) 🟡 | N/A | O(log n) 🟢 |
    | Search | O(log n) 🟢 | O(n) 🟡 | O(1)* 🟢 | O(log n) 🟢 |
    | Insert | O(log n) 🟢 | O(n) 🟡 | O(1)* 🟢 | O(log n) 🟢 |
    | Delete | O(log n) 🟢 | O(n) 🟡 | O(1)* 🟢 | O(log n) 🟢 |

    *Average case with good hash function

=== "Heaps & Tries"
    | Operation | Min/Max Heap | Priority Queue | Trie |
    |-----------|--------------|----------------|------|
    | Find Min/Max | O(1) 🟢 | O(1) 🟢 | N/A |
    | Insert | O(log n) 🟢 | O(log n) 🟢 | O(m)* 🟢 |
    | Delete | O(log n) 🟢 | O(log n) 🟢 | O(m)* 🟢 |
    | Search | O(n) 🟡 | O(n) 🟡 | O(m)* 🟢 |
    | Prefix Search | N/A | N/A | O(m)* 🟢 |

    *Where m = key length

=== "Graphs"
    | Representation | Space | Add Vertex | Add Edge | Remove Vertex | Remove Edge | Query Edge |
    |----------------|-------|------------|----------|---------------|-------------|------------|
    | Adjacency List | O(V+E) | O(1) | O(1) | O(V+E) | O(E) | O(V) |
    | Adjacency Matrix | O(V²) | O(V²) | O(1) | O(V²) | O(1) | O(1) |

    **Graph Algorithms:**

    | Algorithm | Time | Space | Optimal | Use Case |
    |-----------|------|-------|---------|----------|
    | DFS | O(V+E) | O(V) | ❌ | Topological sort, cycle detection |
    | BFS | O(V+E) | O(V) | ✅* | Shortest path (unweighted) |
    | Dijkstra | O((V+E)log V) | O(V) | ✅** | Shortest path (non-negative weights) |
    | Bellman-Ford | O(VE) | O(V) | ✅*** | Shortest path (negative weights) |
    | A* | O(E) | O(V) | ✅** | Heuristic-guided shortest path |

    *Unweighted graphs | **Non-negative weights | ***Can detect negative cycles

---

## 🔄 Algorithms Complexity

=== "Sorting Algorithms"
    **Comparison-Based:**

    | Algorithm | Best | Average | Worst | Space | Stable | Use Case |
    |-----------|------|---------|-------|-------|--------|----------|
    | Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | ✅ | Nearly sorted, small data |
    | Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | ✅ | Small data, online sorting |
    | Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | ❌ | Minimize swaps |
    | Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | ✅ | Stable sorting, linked lists |
    | Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | ❌ | General purpose (best avg) |
    | Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | ❌ | Guaranteed performance |

    **Non-Comparison:**

    | Algorithm | Time | Space | Stable | Use Case |
    |-----------|------|-------|--------|----------|
    | Counting Sort | O(n+k) | O(n+k) | ✅ | Small integer range |
    | Radix Sort | O(nk) | O(n+k) | ✅ | Fixed-length integers/strings |
    | Bucket Sort | O(n+k) | O(n+k) | ✅ | Uniformly distributed data |

=== "Search Algorithms"
    **Array Search:**

    | Algorithm | Best | Average | Worst | Space | Requirement |
    |-----------|------|---------|-------|-------|-------------|
    | Linear Search | O(1) | O(n) | O(n) | O(1) | None |
    | Binary Search | O(1) | O(log n) | O(log n) | O(1) | Sorted array |
    | Jump Search | O(1) | O(√n) | O(√n) | O(1) | Sorted array |
    | Interpolation | O(1) | O(log log n) | O(n) | O(1) | Sorted, uniform distribution |

    **String Search:**

    | Algorithm | Preprocessing | Search | Space | Use Case |
    |-----------|--------------|--------|-------|----------|
    | Naive | O(1) | O(mn) | O(1) | Simple, short patterns |
    | KMP | O(m) | O(n) | O(m) | Pattern matching |
    | Boyer-Moore | O(m+k) | O(n/m) best | O(k) | Large alphabets |
    | Rabin-Karp | O(m) | O(n+m) | O(1) | Multiple pattern search |

=== "Dynamic Programming"
    | Problem Type | Time | Space | Optimization Technique |
    |--------------|------|-------|----------------------|
    | Fibonacci | O(n) | O(n) → O(1) | Space optimization |
    | Longest Common Subsequence | O(mn) | O(mn) → O(min(m,n)) | Rolling array |
    | Knapsack (0/1) | O(nW) | O(nW) → O(W) | 1D DP |
    | Matrix Chain Multiplication | O(n³) | O(n²) | Memoization |
    | Edit Distance | O(mn) | O(mn) → O(min(m,n)) | Space optimization |

---

## ⚡ Optimization Strategies

=== "Time-Space Tradeoffs"
    | Technique | Time Gain | Space Cost | Example |
    |-----------|-----------|------------|---------|
    | Hash Tables | O(n) → O(1) | +O(n) | Two-sum problem |
    | Memoization | Exponential → Polynomial | +O(n) or more | DP problems |
    | Precomputation | Runtime → Compile time | +Storage | Lookup tables |
    | Indexing | O(n) → O(log n) or O(1) | +O(n) | Database indices |
    | Caching | Repeated → O(1) | +O(cache size) | Web caching |

=== "Algorithm Improvement"
    **Key Techniques:**

    1. **Choose Better Algorithm:** Bubble sort O(n²) → Quick sort O(n log n)
    2. **Optimize Data Structure:** Linear search → Hash table lookup
    3. **Early Termination:** Break when condition met
    4. **Avoid Redundant Work:** Cache results, avoid recalculation
    5. **Divide & Conquer:** Break into smaller subproblems
    6. **Two Pointers:** Reduce nested loops from O(n²) to O(n)
    7. **Sliding Window:** Optimize subarray problems
    8. **Binary Search:** O(n) → O(log n) on sorted data

=== "Real-World Factors"
    **Beyond Big O:**

    | Factor | Impact |
    |--------|--------|
    | **Constant Factors** | Small inputs: O(n²) with tiny constants may beat O(n log n) |
    | **Cache Locality** | Sequential access faster than random access |
    | **Memory Hierarchy** | CPU cache > RAM > Disk (1x vs 100x vs 100,000x) |
    | **Input Distribution** | Quick sort excellent on random, poor on sorted |
    | **Hardware** | SIMD, multi-core, GPU opportunities |
    | **I/O Bounds** | Disk/network often bottleneck, not CPU |

---

## 🎯 Quick Reference Guide

=== "Algorithm Selection"
    | Need | Use | Avoid | Complexity |
    |------|-----|-------|------------|
    | Sort small data (<50) | Insertion sort | Quick/merge sort | O(n²) acceptable |
    | Sort large data | Quick/merge sort | Bubble/insertion | O(n log n) |
    | Search sorted data | Binary search | Linear search | O(log n) |
    | Frequent lookups | Hash table | Array search | O(1) |
    | Ordered iteration | BST | Hash table | O(log n) |
    | Priority processing | Heap | Sorted array | O(log n) |
    | Prefix matching | Trie | Linear string search | O(m) |
    | Shortest path | Dijkstra/A* | DFS | O((V+E)log V) |

=== "Data Structure Selection"
    | Problem Pattern | Data Structure | Why |
    |----------------|----------------|-----|
    | Fast access by key | Hash table | O(1) lookup |
    | Maintain sorted order | BST/AVL | O(log n) operations |
    | Find min/max frequently | Heap | O(1) peek, O(log n) insert/delete |
    | Prefix/suffix queries | Trie | O(m) string operations |
    | FIFO order | Queue | O(1) enqueue/dequeue |
    | LIFO order | Stack | O(1) push/pop |
    | Range queries | Segment tree | O(log n) query/update |
    | Dynamic median | Two heaps | O(log n) insert, O(1) median |

=== "Common Patterns"
    | Pattern | Complexity Reduction | Example |
    |---------|---------------------|---------|
    | Two Pointers | O(n²) → O(n) | Two sum on sorted array |
    | Sliding Window | O(n²) → O(n) | Max subarray of size k |
    | Binary Search | O(n) → O(log n) | Search in rotated array |
    | Hash Map | O(n²) → O(n) | Two sum on unsorted array |
    | Prefix Sum | O(n²) → O(n) | Subarray sum queries |
    | Monotonic Stack | O(n²) → O(n) | Next greater element |
    | Union Find | O(n²) → O(n·α(n)) | Connected components |

---

## 💡 Best Practices

!!! success "Engineering Principles"
    1. **Make it work first, then optimize** - Correctness before performance
    2. **Measure before optimizing** - Use profilers to find bottlenecks
    3. **Focus on hot paths** - 80% time spent in 20% of code
    4. **Consider readability** - Maintainability often > minor performance gains
    5. **Know your constraints** - Optimize for speed, memory, or both based on needs
    6. **Start with simple** - Use O(n²) if n<100 and it's clearer

!!! warning "Common Mistakes"
    - Premature optimization without profiling
    - Ignoring constant factors for small inputs
    - Over-engineering when simple solution suffices
    - Not considering average vs worst case
    - Forgetting space complexity
    - Optimizing already-fast code

!!! quote "Remember"
    > "Premature optimization is the root of all evil" — Donald Knuth

    The best algorithm is the one that meets your requirements while remaining maintainable—not necessarily the one with optimal theoretical complexity.
