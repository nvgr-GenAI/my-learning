# Sorting Algorithms

Welcome to the Sorting Algorithms section. Learn about different algorithms to sort data efficiently and understand when to apply each technique.

=== "📊 Overview"

    ## Introduction
    
    Sorting algorithms are fundamental building blocks in computer science that arrange data elements in a specific order. Understanding various sorting techniques is essential for efficient data manipulation and serves as the foundation for many other algorithms.
    
    ## [Key Concepts](fundamentals.md)
    
    - **Comparison vs. Non-comparison sorting**
    - **Stability** in sorting algorithms
    - **In-place** vs. auxiliary space requirements
    - **Adaptive** algorithms that perform better with partially sorted data
    - **Internal** vs. **External** sorting methods
    
    ## Choosing the Right Algorithm
    
    The optimal sorting algorithm depends on various factors:
    
    - **Data size**: Small collections may benefit from simple sorts like insertion sort
    - **Data distribution**: Nearly sorted data favors adaptive algorithms
    - **Memory constraints**: Limited memory may require in-place algorithms
    - **Stability requirements**: Some applications need original order preservation
    - **Time complexity needs**: Real-time applications have strict performance requirements

=== "🔄 Algorithm Categories"

    ## Basic Sorting Algorithms
    
    Simple algorithms with O(n²) average complexity:
    
    - **[Bubble Sort](bubble-sort.md)** - Compare adjacent elements and swap if needed
    - **[Selection Sort](selection-sort.md)** - Find minimum and place it at beginning
    - **[Insertion Sort](insertion-sort.md)** - Build sorted array incrementally
    
    ## Advanced Sorting Algorithms
    
    Efficient algorithms with O(n log n) average complexity:
    
    - **[Quick Sort](quick-sort.md)** - Divide and conquer with pivot partitioning
    - **[Merge Sort](merge-sort.md)** - Divide, sort, and merge approach
    - **[Heap Sort](heap-sort.md)** - Uses binary heap data structure
    
    ## Linear-time Sorting
    
    Special-case algorithms with O(n) complexity under specific conditions:
    
    - **[Counting Sort](counting-sort.md)** - For small range of integer values
    - **[Radix Sort](radix-sort.md)** - Sort by individual digits/characters
    - **[Bucket Sort](bucket-sort.md)** - Distribute elements into buckets
    
    ## Hybrid Algorithms
    
    Modern algorithms that combine multiple techniques:
    
    - **[Timsort](timsort.md)** - Combines merge sort and insertion sort
    - **[Introsort](introsort.md)** - Combines quicksort, heapsort and insertion sort

=== "📈 Complexity Comparison"

    | Algorithm | Best | Average | Worst | Space | Stable | In-Place |
    |-----------|------|---------|-------|-------|--------|----------|
    | [Bubble Sort](bubble-sort.md) | O(n) | O(n²) | O(n²) | O(1) | Yes | Yes |
    | [Selection Sort](selection-sort.md) | O(n²) | O(n²) | O(n²) | O(1) | No | Yes |
    | [Insertion Sort](insertion-sort.md) | O(n) | O(n²) | O(n²) | O(1) | Yes | Yes |
    | [Quick Sort](quick-sort.md) | O(n log n) | O(n log n) | O(n²) | O(log n) | No | Yes |
    | [Merge Sort](merge-sort.md) | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | No |
    | [Heap Sort](heap-sort.md) | O(n log n) | O(n log n) | O(n log n) | O(1) | No | Yes |
    | [Counting Sort](counting-sort.md) | O(n+k) | O(n+k) | O(n+k) | O(n+k) | Yes | No |
    | [Radix Sort](radix-sort.md) | O(nk) | O(nk) | O(nk) | O(n+k) | Yes | No |
    | [Bucket Sort](bucket-sort.md) | O(n+k) | O(n+k) | O(n²) | O(n+k) | Yes | No |
    | [Timsort](timsort.md) | O(n) | O(n log n) | O(n log n) | O(n) | Yes | No |
    | [Introsort](introsort.md) | O(n log n) | O(n log n) | O(n log n) | O(log n) | No | Yes |

=== "🎯 Practice"

    ## Problem Categories
    
    - **[Easy Problems](easy-problems.md)** - Foundational sorting problems
    - **[Medium Problems](medium-problems.md)** - Intermediate challenges
    - **[Hard Problems](hard-problems.md)** - Advanced sorting techniques
    
    ## Real-world Applications
    
    - **Database indexing and query optimization**
    - **File system organization**
    - **Priority scheduling in operating systems**
    - **Computational biology sequence alignment**
    - **Computer graphics rendering pipelines**
