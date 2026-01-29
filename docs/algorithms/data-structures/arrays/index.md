# Arrays

Contiguous memory blocks storing elements of the same type. The foundation of efficient algorithms and most data structures.

---

## What is an Array?

An array is a collection of elements stored in **contiguous memory locations**, where each element is accessed directly using its index. Think of it like a row of numbered mailboxes - you can instantly go to mailbox #5 without checking mailboxes 1-4 first.

**Visual representation:**

```
Array: [10, 20, 30, 40, 50]
Index:  0   1   2   3   4

Memory Layout:
┌────┬────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ 40 │ 50 │
└────┴────┴────┴────┴────┘
Address: base+0  +4  +8  +12  +16
```

**Key characteristics:**

- **O(1) access** - Direct memory offset: `address = base + (index × size)`
- **Contiguous memory** - Elements stored consecutively for cache efficiency
- **Fixed or dynamic size** - Static (fixed) or dynamic (growable)
- **Homogeneous** - All elements same type (typically)

---

## Why Arrays Matter

Arrays are simple but powerful. In interviews, you need to understand:

**✅ Strengths:**

- **O(1) random access** - Get any element instantly by index
- **Cache-friendly** - Elements stored together in memory
- **Simple and universal** - Every language has arrays

**❌ Limitations:**

- **O(n) insert/delete** - Must shift elements (except at end)
- **Fixed capacity** - Static arrays can't grow; dynamic arrays waste space

**Interview reality:** Most problems use arrays as input. Your job is to manipulate them efficiently using patterns like two pointers, sliding window, or binary search - not to implement array data structures.

---

## Complexity Analysis

### Time Complexity

| Operation | Static Array | Dynamic Array | Sorted Array | Notes |
|-----------|--------------|---------------|--------------|-------|
| **Access by index** | O(1) | O(1) | O(1) | Direct memory calculation |
| **Search (unsorted)** | O(n) | O(n) | O(log n) | Linear scan vs binary search |
| **Insert at end** | N/A | O(1)* | O(n) | *Amortized for dynamic |
| **Insert at beginning** | O(n) | O(n) | O(n) | Shift all elements right |
| **Insert at middle** | O(n) | O(n) | O(n) | Shift elements |
| **Delete at end** | N/A | O(1) | O(1) | Just decrease size |
| **Delete at beginning** | O(n) | O(n) | O(n) | Shift all elements left |
| **Delete at middle** | O(n) | O(n) | O(n) | Shift elements |

### Space Complexity

| Array Type | Space | Overhead | Notes |
|------------|-------|----------|-------|
| **Static** | O(n) | None | Exact allocation |
| **Dynamic** | O(n) | 25-50% | Extra capacity for growth |
| **Multidimensional (2D)** | O(m × n) | None | m rows × n columns |

---

## Core Patterns

=== "Two Pointers"

    **Use opposite or same-direction pointers to eliminate nested loops**

    **When to use:**

    - Sorted arrays (find pairs, triplets)
    - Palindrome checking
    - Remove duplicates
    - Partition problems

    **Time:** O(n) | **Space:** O(1)

    **Example: Find pair with target sum**
    ```python
    def two_sum_sorted(arr, target):
        left, right = 0, len(arr) - 1

        while left < right:
            current_sum = arr[left] + arr[right]

            if current_sum == target:
                return [left, right]
            elif current_sum < target:
                left += 1  # Need larger sum
            else:
                right -= 1  # Need smaller sum

        return []
    ```

    **Common problems:**

    - Two Sum (sorted array)
    - Remove Duplicates from Sorted Array
    - Valid Palindrome
    - Container With Most Water
    - 3Sum, 4Sum

=== "Sliding Window"

    **Maintain a window of elements and slide it efficiently**

    **When to use:**

    - Subarray/substring problems
    - Maximum/minimum in window
    - Optimization problems
    - Constraints on contiguous elements

    **Time:** O(n) | **Space:** O(1) or O(k)

    **Fixed window:**
    ```python
    def max_sum_subarray(arr, k):
        # Find max sum of k consecutive elements
        window_sum = sum(arr[:k])
        max_sum = window_sum

        for i in range(k, len(arr)):
            window_sum = window_sum - arr[i-k] + arr[i]
            max_sum = max(max_sum, window_sum)

        return max_sum
    ```

    **Variable window:**
    ```python
    def longest_subarray_with_sum(arr, target):
        left = 0
        current_sum = 0
        max_length = 0

        for right in range(len(arr)):
            current_sum += arr[right]

            # Shrink window while sum > target
            while current_sum > target and left <= right:
                current_sum -= arr[left]
                left += 1

            if current_sum == target:
                max_length = max(max_length, right - left + 1)

        return max_length
    ```

    **Common problems:**

    - Maximum Sum Subarray of Size K
    - Longest Substring Without Repeating Characters
    - Minimum Size Subarray Sum
    - Sliding Window Maximum

=== "Prefix Sum"

    **Precompute cumulative sums for O(1) range queries**

    **When to use:**

    - Range sum queries
    - Subarray sum problems
    - Multiple queries on same array
    - Need cumulative information

    **Time:** O(n) build, O(1) query | **Space:** O(n)

    **Basic prefix sum:**
    ```python
    def build_prefix_sum(arr):
        prefix = [0] * (len(arr) + 1)
        for i in range(len(arr)):
            prefix[i+1] = prefix[i] + arr[i]
        return prefix

    def range_sum(prefix, left, right):
        # Sum from left to right (inclusive)
        return prefix[right+1] - prefix[left]
    ```

    **Subarray sum equals K:**
    ```python
    def subarray_sum_equals_k(arr, k):
        count = 0
        prefix_sum = 0
        sum_map = {0: 1}

        for num in arr:
            prefix_sum += num

            # If (prefix_sum - k) exists, found subarray
            if prefix_sum - k in sum_map:
                count += sum_map[prefix_sum - k]

            sum_map[prefix_sum] = sum_map.get(prefix_sum, 0) + 1

        return count
    ```

    **Common problems:**

    - Range Sum Query
    - Subarray Sum Equals K
    - Contiguous Array
    - Product of Array Except Self

=== "Binary Search"

    **Divide search space in half each iteration**

    **When to use:**

    - Sorted arrays
    - Search space reduction
    - Find first/last occurrence
    - Search in rotated sorted array

    **Time:** O(log n) | **Space:** O(1)

    **Template:**
    ```python
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1

        while left <= right:
            mid = left + (right - left) // 2

            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return -1  # Not found
    ```

    **Find first occurrence:**
    ```python
    def find_first(arr, target):
        left, right = 0, len(arr) - 1
        result = -1

        while left <= right:
            mid = left + (right - left) // 2

            if arr[mid] == target:
                result = mid
                right = mid - 1  # Continue searching left
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return result
    ```

    **Common problems:**

    - Binary Search
    - Find First and Last Position
    - Search in Rotated Sorted Array
    - Find Minimum in Rotated Sorted Array

---

## Common Algorithms

### Searching

| Algorithm | Time | Space | When to Use |
|-----------|------|-------|-------------|
| **Linear Search** | O(n) | O(1) | Unsorted array, small dataset |
| **Binary Search** | O(log n) | O(1) | Sorted array |
| **Two Pointers** | O(n) | O(1) | Sorted array, find pairs |

### Sorting

| Algorithm | Time (Avg) | Time (Worst) | Space | Stable | When to Use |
|-----------|-----------|--------------|-------|--------|-------------|
| **Quick Sort** | O(n log n) | O(n²) | O(log n) | No | General purpose, good cache |
| **Merge Sort** | O(n log n) | O(n log n) | O(n) | Yes | Need stability, linked lists |
| **Heap Sort** | O(n log n) | O(n log n) | O(1) | No | Limited memory |
| **Counting Sort** | O(n+k) | O(n+k) | O(k) | Yes | Small range integers |

### Classic Problems

| Algorithm | Problem | Time | Key Insight |
|-----------|---------|------|-------------|
| **Kadane's** | Maximum subarray sum | O(n) | Track current/global max |
| **Boyer-Moore** | Majority element | O(n) | Cancel out pairs |
| **Dutch Flag** | Sort 0s, 1s, 2s | O(n) | Three-way partition |
| **Fisher-Yates** | Random shuffle | O(n) | Swap with random index |

---

## Practice Problems

### By Difficulty

| Level | Count | Focus | Start Here |
|-------|-------|-------|------------|
| 🟢 **Easy** | 15+ | Basic operations, simple patterns | [Easy Problems](easy-problems.md) |
| 🟡 **Medium** | 20+ | Complex algorithms, optimization | [Medium Problems](medium-problems.md) |
| 🔴 **Hard** | 10+ | Advanced techniques, edge cases | [Hard Problems](hard-problems.md) |

### By Pattern

| Pattern | Problems | Key Technique |
|---------|----------|---------------|
| **Two Pointers** | 8+ | Opposite/same direction pointers |
| **Sliding Window** | 6+ | Fixed/variable window |
| **Prefix Sum** | 5+ | Cumulative sums, hash map |
| **Binary Search** | 5+ | Divide search space |
| **Dynamic Programming** | 4+ | Build from subproblems |

---

## When to Use Arrays

### ✅ Use Arrays When

| Scenario | Why Array is Good |
|----------|-------------------|
| Need O(1) random access | Direct index-based access |
| Sequential processing | Excellent cache locality |
| Fixed or predictable size | Memory efficient (static) or convenient (dynamic) |
| Numerical computations | Vectorized operations, SIMD |
| Implementing other structures | Foundation for stacks, queues, heaps |

### ❌ Avoid Arrays When

| Scenario | Better Alternative |
|----------|-------------------|
| Frequent middle insert/delete | Linked List (O(1) with reference) |
| Unknown size with many operations | Linked List or specialized structure |
| Need fast lookup by key | Hash Table (O(1) by key) |
| Sparse data | Hash Map or Sparse Matrix |
| Need ordering with updates | BST or Heap |

---

## Quick Reference

### Pattern Decision

| Problem Type | Pattern | Complexity |
|--------------|---------|------------|
| Find pair in sorted array | Two Pointers | O(n), O(1) space |
| Max/min in subarrays | Sliding Window | O(n), O(1) space |
| Range sum queries | Prefix Sum | O(n) build, O(1) query |
| Search in sorted array | Binary Search | O(log n) |
| Subarray with condition | Sliding Window + Hash | O(n), O(k) space |

---

## Start Practicing

| Level | Focus | Problems | Link |
|-------|-------|----------|------|
| 🟢 **Easy** | Basic operations, two pointers, simple patterns | 15+ | [Start Here](easy-problems.md) |
| 🟡 **Medium** | Sliding window, prefix sum, optimization | 20+ | [Practice](medium-problems.md) |
| 🔴 **Hard** | Advanced techniques, DP, complex algorithms | 10+ | [Challenge](hard-problems.md) |

---

**New to arrays?** Start with [Easy Problems](easy-problems.md) - master the fundamentals before moving to harder challenges.
