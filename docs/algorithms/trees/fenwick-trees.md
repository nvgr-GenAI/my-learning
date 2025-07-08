# Fenwick Trees (Binary Indexed Trees) 🌳🔢

## Introduction

Fenwick Trees, also known as Binary Indexed Trees (BIT), are specialized data structures that efficiently support dynamic cumulative frequency queries and updates in logarithmic time.

=== "Overview"
    **Core Concept**:
    
    - Space-efficient tree structure for maintaining cumulative frequencies
    - Efficiently handles point updates and range sum queries
    - Uses a clever binary representation approach for operations
    - More memory efficient than segment trees for certain operations
    
    **When to Use**:
    
    - Dynamic range sum queries with frequent updates
    - Calculating cumulative frequencies or partial sums
    - When memory efficiency is important
    - Computing inversions in an array
    - Range update, point query operations (with modifications)
    
    **Time Complexity**:
    
    - Query (prefix sum): O(log n)
    - Update: O(log n)
    - Construction: O(n) or O(n log n)
    - Space: O(n)
    
    **Real-World Applications**:
    
    - Statistical computations with frequent updates
    - Counting inversions in arrays
    - Range sum queries in databases
    - Dynamic histogram maintenance
    - Competitive programming problems involving range operations

=== "Structure"
    **Key Insight**:
    
    Fenwick Trees store partial sum information in an implicit tree structure using a clever binary indexing scheme.
    
    **Binary Representation**:
    
    - Each index i in the tree is responsible for a specific range of elements
    - The range size depends on the least significant bit (LSB) of i
    - For example, if i's binary form ends with k 1-bits, it's responsible for 2ᵏ elements
    
    **Parent-Child Relationship**:
    
    - Parent of node i: i - (i & -i) [removing LSB]
    - Next node after i: i + (i & -i) [adding LSB]
    
    **Array Representation**:
    
    The tree is typically stored as a simple array where:
    - BIT[i] stores the cumulative sum of a specific range ending at i
    - The range covered by BIT[i] is determined by the binary representation of i

=== "Operations"
    **Query (Prefix Sum)**:
    
    Computes the sum of elements from 1 to index:
    
    ```python
    def prefix_sum(bit, index):
        result = 0
        while index > 0:
            result += bit[index]
            # Remove least significant bit
            index -= index & -index
        return result
    ```
    
    **Update**:
    
    Updates an element at a given index by a delta:
    
    ```python
    def update(bit, index, delta, n):
        while index <= n:
            bit[index] += delta
            # Add least significant bit
            index += index & -index
    ```
    
    **Range Sum**:
    
    Sum of elements from index l to index r:
    
    ```python
    def range_sum(bit, left, right):
        return prefix_sum(bit, right) - prefix_sum(bit, left - 1)
    ```
    
    **Construction**:
    
    Build a Fenwick Tree from an array:
    
    ```python
    def build(arr):
        n = len(arr)
        bit = [0] * (n + 1)  # 1-based indexing
        
        # Initialize with values
        for i in range(n):
            update(bit, i+1, arr[i], n)
            
        return bit
    ```

=== "Advanced Operations"
    **Range Update, Point Query**:
    
    With a clever transformation, Fenwick Trees can support range updates and point queries:
    
    1. Create two BITs: BIT1 and BIT2
    2. To add val to range [l, r]:
       - update(BIT1, l, val)
       - update(BIT1, r+1, -val)
       - update(BIT2, l, val * (l-1))
       - update(BIT2, r+1, -val * r)
    3. To query the value at index i:
       - result = original_value + prefix_sum(BIT1, i) * i - prefix_sum(BIT2, i)
    
    **Finding Kth Element**:
    
    If the BIT represents frequencies, we can find the kth element:
    
    ```python
    def find_kth(bit, k, n):
        pos = 0
        for i in range(log2(n), -1, -1):
            next_pos = pos + (1 << i)
            if next_pos <= n and bit[next_pos] < k:
                pos = next_pos
                k -= bit[next_pos]
        return pos + 1  # kth element
    ```
    
    **2D Fenwick Tree**:
    
    For 2D range queries:
    
    ```python
    def update_2d(bit, x, y, delta, n, m):
        i = x
        while i <= n:
            j = y
            while j <= m:
                bit[i][j] += delta
                j += j & -j
            i += i & -i
    
    def query_2d(bit, x, y):
        result = 0
        i = x
        while i > 0:
            j = y
            while j > 0:
                result += bit[i][j]
                j -= j & -j
            i -= i & -i
        return result
    ```

=== "Examples"
    **Example 1: Sum Queries**
    
    Consider the array [3, 2, 5, 4, 1, 6, 3]:
    
    1. Build the Fenwick Tree:
       - BIT[1] = 3
       - BIT[2] = 3 + 2 = 5
       - BIT[3] = 5
       - BIT[4] = 3 + 2 + 5 + 4 = 14
       - ... and so on
    
    2. Query the sum of elements from index 2 to 5:
       - range_sum(bit, 2, 5) = prefix_sum(5) - prefix_sum(1)
       - prefix_sum(5) = BIT[5] + BIT[4] = 1 + 14 = 15
       - prefix_sum(1) = BIT[1] = 3
       - Result: 15 - 3 = 12
    
    3. Update element at index 3 by +2:
       - update(bit, 3, 2)
       - This updates BIT[3], BIT[4], BIT[7], BIT[8], ...
    
    **Example 2: Counting Inversions**
    
    For array [8, 4, 2, 1]:
    
    1. Scan from right to left
    2. For each element, count smaller elements to its right using a BIT
    3. Inversion count = 0 + 0 + 0 + 0 = 0 + 0 + 1 + 3 = 6

=== "Comparison"
    **Fenwick Tree vs Segment Tree**:
    
    | Aspect | Fenwick Tree | Segment Tree |
    |--------|--------------|--------------|
    | **Space** | O(n) | O(n) but with larger constant |
    | **Functionality** | Limited (mainly prefix sums) | More versatile (min/max/custom operations) |
    | **Implementation** | Simpler, more compact | More complex |
    | **Operations** | Prefix sums, point updates | Range queries, range updates |
    | **Memory Access** | Better locality | More scattered memory access |
    
    **Fenwick Tree vs Prefix Sum Array**:
    
    | Aspect | Fenwick Tree | Prefix Sum Array |
    |--------|--------------|------------------|
    | **Updates** | O(log n) | O(n) |
    | **Queries** | O(log n) | O(1) |
    | **Use Case** | Dynamic data | Static data |

=== "Tips"
    **Implementation Tips**:
    
    1. Use 1-based indexing for simplicity (makes bit operations cleaner)
    2. The operation (i & -i) efficiently extracts the least significant bit
    3. For range updates, consider using the difference array technique
    4. Pre-compute logs for finding kth element operations
    
    **Common Pitfalls**:
    
    1. Using 0-based indexing without adjusting the algorithms
    2. Not accounting for array bounds when updating
    3. Mixing up the update and query logic
    4. Forgetting that Fenwick Trees work on cumulative operations
    
    **Optimizations**:
    
    1. Cache-friendly implementation by using sequential memory
    2. Batch updates for better performance
    3. Use compressed Fenwick Trees for sparse data
    4. Combine with other data structures for complex problems
