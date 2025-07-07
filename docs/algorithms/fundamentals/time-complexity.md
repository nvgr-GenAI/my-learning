# Time & Space Complexity Analysis 📊

Understanding algorithmic complexity is fundamental to writing efficient code. This guide explains how to analyze and optimize the runtime and memory usage of your algorithms.

!!! tip "Quick Reference"
    | Complexity | Name | Speed for large inputs | Example |
    |------------|------|------------------------|---------|
    | **O(1)** | Constant | ⚡ Instant | Array access |
    | **O(log n)** | Logarithmic | 🚀 Super Fast | Binary search |
    | **O(n)** | Linear | 🚗 Fast | Linear search |
    | **O(n log n)** | Linearithmic | 🚲 Moderate | Merge sort |
    | **O(n²)** | Quadratic | 🚶 Slow | Bubble sort |
    | **O(2ⁿ)** | Exponential | 🐢 Very slow | Recursive Fibonacci |

## Big O Notation Explained

Big O notation describes the performance or complexity of an algorithm in terms of:

- **Time complexity**: How runtime grows as input size increases
- **Space complexity**: How memory usage grows as input size increases

!!! info "Remember"
    Big O is concerned with the **rate of growth** as input size increases, not the exact number of operations.

## Complexity Classes

!!! abstract "Big O Cheat Sheet"
    === "Efficient Algorithms"
        | Complexity | Name | Performance | Example 1 | Example 2 | Example 3 |
        |------------|------|-------------|-----------|-----------|-----------|
        | **O(1)** | Constant | 🟢 Excellent | Array access | Hash lookup | Stack push/pop |
        | **O(log n)** | Logarithmic | 🟢 Excellent | Binary search | BST operations | Heap operations |
        | **O(n)** | Linear | 🟡 Good | Linear search | Array traversal | Counting elements |
        | **O(n log n)** | Linearithmic | 🟡 Good | Merge sort | Heap sort | Quick sort (avg) |

    === "Inefficient Algorithms"
        | Complexity | Name | Performance | Example 1 | Example 2 | Example 3 |
        |------------|------|-------------|-----------|-----------|-----------|
        | **O(n²)** | Quadratic | 🔴 Poor | Bubble sort | Selection sort | Nested loops |
        | **O(n³)** | Cubic | 🔴 Very Poor | Matrix multiply | Floyd-Warshall | Triple nested loops |
        | **O(2ⁿ)** | Exponential | ⚫ Terrible | Recursive Fibonacci | Subset generation | Tower of Hanoi |
        | **O(n!)** | Factorial | ⚫ Impossible | Permutations | Traveling Salesman | NP-complete problems |

## Visual Complexity Comparison

!!! example "Relative Growth Rates"
    ```
    O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(n³) < O(2ⁿ) < O(n!)
    ```

    | Input Size | O(1) | O(log n) | O(n) | O(n log n) | O(n²) | O(n³) | O(2ⁿ) | O(n!) |
    |------------|------|----------|------|------------|-------|-------|-------|-------|
    | **n=10**   | 1    | 3.3      | 10   | 33         | 100   | 1,000  | 1,024  | 3.6M  |
    | **n=100**  | 1    | 6.6      | 100  | 664        | 10K   | 1M     | 10^30  | 10^158 |
    | **n=1000** | 1    | 10       | 1K   | 10K        | 1M    | 1B     | 10^301 | 10^2568 |

    ⏰ **Runtime Translation**:
    
    - **n = 10**: All algorithms finish quickly
    - **n = 100**: O(n²) starts to slow down significantly 
    - **n = 1,000**: O(n³) becomes impractical
    - **n = 1,000,000**: Only O(1), O(log n), O(n) remain feasible

## Common Data Structure Operations

!!! info "Data Structure Complexity Cheat Sheet"
    === "Arrays & Lists"
        | Operation | Array | Dynamic Array | Singly-Linked List | Doubly-Linked List |
        |-----------|-------|---------------|-------------------|-------------------|
        | **Access** | O(1) 🟢 | O(1) 🟢 | O(n) 🟡 | O(n) 🟡 |
        | **Search** | O(n) 🟡 | O(n) 🟡 | O(n) 🟡 | O(n) 🟡 |
        | **Insertion (beginning)** | O(n) 🟡 | O(n) 🟡 | O(1) 🟢 | O(1) 🟢 |
        | **Insertion (end)** | O(1)\* 🟢 | O(1)\*\* 🟢 | O(n)\*\*\* 🟡 | O(1)\*\*\* 🟢 |
        | **Insertion (middle)** | O(n) 🟡 | O(n) 🟡 | O(n) 🟡 | O(n) 🟡 |
        | **Deletion (beginning)** | O(n) 🟡 | O(n) 🟡 | O(1) 🟢 | O(1) 🟢 |
        | **Deletion (end)** | O(1) 🟢 | O(1) 🟢 | O(n) 🟡 | O(1) 🟢 |
        | **Deletion (middle)** | O(n) 🟡 | O(n) 🟡 | O(n) 🟡 | O(n) 🟡 |

        *If size is known, **Amortized, ***With tail pointer
    
    === "Trees & Hash Tables"
        | Operation | BST (balanced) | BST (unbalanced) | Hash Table |
        |-----------|----------------|------------------|------------|
        | **Access** | O(log n) 🟢 | O(n) 🟡 | N/A |
        | **Search** | O(log n) 🟢 | O(n) 🟡 | O(1)* 🟢 |
        | **Insertion** | O(log n) 🟢 | O(n) 🟡 | O(1)* 🟢 |
        | **Deletion** | O(log n) 🟢 | O(n) 🟡 | O(1)* 🟢 |

        *Assuming good hash function with minimal collisions
    
    === "Heaps & Tries"
        | Operation | Min/Max Heap | Priority Queue | Trie |
        |-----------|--------------|----------------|------|
        | **Find Min/Max** | O(1) 🟢 | O(1) 🟢 | N/A |
        | **Insert** | O(log n) 🟢 | O(log n) 🟢 | O(m)* 🟢 |
        | **Delete** | O(log n) 🟢 | O(log n) 🟢 | O(m)* 🟢 |
        | **Search** | O(n) 🟡 | O(n) 🟡 | O(m)* 🟢 |
        | **Prefix Search** | N/A | N/A | O(m)* 🟢 |

        *Where m is the key length

## Sorting Algorithm Complexities

!!! tip "Sorting Algorithm Selection Guide"
    === "Comparison-Based"
        | Algorithm | Time (Best) | Time (Avg) | Time (Worst) | Space | Stable | When to Use |
        |-----------|-------------|------------|--------------|-------|--------|-------------|
        | **Bubble Sort** | O(n) | O(n²) | O(n²) | O(1) | ✅ | Small datasets, nearly sorted data |
        | **Selection Sort** | O(n²) | O(n²) | O(n²) | O(1) | ❌ | Small datasets, minimizing swaps |
        | **Insertion Sort** | O(n) | O(n²) | O(n²) | O(1) | ✅ | Small datasets, online sorting |
        | **Merge Sort** | O(n log n) | O(n log n) | O(n log n) | O(n) | ✅ | Stable sorting required, linked lists |
        | **Quick Sort** | O(n log n) | O(n log n) | O(n²) | O(log n) | ❌ | General purpose, average case performance |
        | **Heap Sort** | O(n log n) | O(n log n) | O(n log n) | O(1) | ❌ | Limited memory, guaranteed performance |

    === "Non-Comparison"
        | Algorithm | Time (Best) | Time (Avg) | Time (Worst) | Space | Stable | When to Use |
        |-----------|-------------|------------|--------------|-------|--------|-------------|
        | **Counting Sort** | O(n+k) | O(n+k) | O(n+k) | O(n+k) | ✅ | Small range of integers |
        | **Radix Sort** | O(nk) | O(nk) | O(nk) | O(n+k) | ✅ | Fixed-length integers or strings |
        | **Bucket Sort** | O(n+k) | O(n+k) | O(n²) | O(n+k) | ✅ | Uniformly distributed data |

    Where n is input size, k is range of values

## Search Algorithm Complexities

!!! example "Search Algorithms Compared"
    | Algorithm | Time (Best) | Time (Avg) | Time (Worst) | Space | Use Case |
    |-----------|-------------|------------|--------------|-------|----------|
    | **Linear Search** | O(1) | O(n) | O(n) | O(1) | Unsorted data, small datasets |
    | **Binary Search** | O(1) | O(log n) | O(log n) | O(1) | Sorted arrays |
    | **Jump Search** | O(1) | O(√n) | O(√n) | O(1) | Sorted arrays, better than linear but simpler than binary |
    | **Interpolation Search** | O(1) | O(log log n) | O(n) | O(1) | Sorted, uniformly distributed arrays |

    **Graph Search**

    | Algorithm | Time | Space | Complete | Optimal | Use Case |
    |-----------|------|-------|----------|---------|----------|
    | **Depth-First Search** | O(V+E) | O(V) | ✅ | ❌ | Maze solving, topological sorting, cycle detection |
    | **Breadth-First Search** | O(V+E) | O(V) | ✅ | ✅* | Shortest path (unweighted), connected components |
    | **Dijkstra's Algorithm** | O((V+E)log V) | O(V) | ✅ | ✅** | Shortest path (non-negative weights) |
    | **A* Search** | O(E) | O(V) | ✅ | ✅** | Shortest path with heuristic guidance |

    \* For unweighted graphs only  
    \** For graphs with non-negative edge weights  
    Where V = number of vertices, E = number of edges

## Common Trade-offs & Optimizations

!!! info "Space-Time Tradeoffs"
    | Technique | Time Benefit | Space Cost | Example |
    |-----------|-------------|------------|---------|
    | **Hash Tables** | O(n) → O(1) lookup | O(1) → O(n) space | Dictionary implementation |
    | **Memoization** | Exponential → Polynomial | O(1) → O(n) or more | Dynamic programming |
    | **Precomputation** | Runtime → Compile time | Increased storage | Lookup tables, precomputed results |
    | **Index/Cache** | O(n) → O(1) or O(log n) | Extra storage structures | Database indices, web caching |

## Analyzing Algorithms

!!! tip "Multiple Perspectives"
    === "Analysis Types"
        | Analysis Type | Description | When to Use |
        |---------------|-------------|------------|
        | **Best Case** | Performance under optimal conditions | Rarely useful except for theoretical understanding |
        | **Average Case** | Expected performance under normal conditions | Most practical for everyday use cases |
        | **Worst Case** | Performance under the most unfavorable conditions | Critical for reliability guarantees |
        | **Amortized** | Average cost over many operations | Data structures with occasional expensive operations |
    
    === "Advanced Concepts"
        | Concept | Description | Example |
        |---------|-------------|---------|
        | **Loop Invariants** | Conditions that remain true throughout loop execution | Binary search correctness proof |
        | **Recurrence Relations** | Equations describing recursive algorithm complexity | Master theorem applications |
        | **Asymptotic Notation** | O, Ω, and Θ notation for algorithm growth rates | Comparing algorithm efficiency |
        | **NP-Completeness** | Problems for which no known polynomial solution exists | Traveling Salesman Problem |

## Optimization Techniques

!!! success "Algorithm Improvement Strategies"
    | Category | Key Techniques | Examples |
    |----------|------------|----------|
    | **Algorithm Choice** | Choose appropriate complexity for your constraints | Replace bubble sort with quick sort |
    | **Data Structures** | Select optimal structures for your access patterns | HashMap instead of linear search |
    | **Code Optimization** | Use early termination and avoid redundant work | Break when found; lazy evaluation |
    | **Memory Management** | Implement caching and reduce allocations | Object pooling; result memoization |
    | **Concurrency** | Utilize parallelization where appropriate | Multi-threaded sorting; async I/O |

    **Additional Techniques**:
    
    1. **Algorithmic**: Dynamic programming, greedy algorithms, divide and conquer
    2. **Data Structure**: Using specialized structures (Bloom filters, tries, etc.)
    3. **System-Level**: I/O batching, memory mapping, locality optimization

## Real-World Considerations

!!! warning "Beyond Big O"
    | Consideration | Description |
    |---------------|-------------|
    | **Constant Factors** | For small inputs, O(n²) with tiny constants might outperform O(n log n) with large constants |
    | **Cache Performance** | Algorithms with good locality often perform better in practice |
    | **Memory Hierarchy** | Consider CPU cache misses, page faults, and disk access |
    | **Input Distribution** | Some algorithms excel with specific data patterns |
    | **Hardware Architecture** | SIMD, multi-core, GPU acceleration opportunities |

## Practical Advice

!!! quote "Engineering Wisdom"
    > "Premature optimization is the root of all evil" — Donald Knuth

    1. **Make it work first**, then make it fast (if needed)
    2. **Measure before optimizing** — use profilers to identify bottlenecks
    3. **Focus on hot spots** — 80% of time is often spent in 20% of code
    4. **Consider maintenance** — sometimes readability trumps small performance gains
    5. **Know your constraints** — are you optimizing for speed, memory, or something else?

---

## Quick Decision Guide

!!! tip "Algorithm Selection Reference"
    | If you need to... | Consider using... | Avoid... |
    |-------------------|-------------------|----------|
    | **Search sorted data** | Binary search O(log n) | Linear search O(n) |
    | **Sort small datasets** | Insertion sort | Quick/merge sort (overhead) |
    | **Sort large datasets** | Quick sort, merge sort | Bubble, insertion sort |
    | **Queue with priority** | Heap/priority queue | Regular queue + resorting |
    | **Frequent lookups** | Hash table | Array/list searches |
    | **Ordered map operations** | Balanced BST | Hash tables |
    | **Prefix matching** | Trie | Linear string searches |
    | **Shortest path in graph** | Dijkstra's/A* | DFS (not optimal) |

Remember that the best algorithm is often the one that's good enough for your specific requirements and constraints, not necessarily the one with the best theoretical complexity.
