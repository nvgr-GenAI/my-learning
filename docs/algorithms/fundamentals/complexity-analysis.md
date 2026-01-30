# Complexity Analysis

**Master time and space complexity for technical interviews** | ⏱️ 45 min read

## Quick Reference

| Complexity | Name | Example | When to Use |
|------------|------|---------|-------------|
| O(1) | Constant | Array access, hash lookup | Direct access operations |
| O(log n) | Logarithmic | Binary search, balanced BST | Divide and conquer |
| O(n) | Linear | Array traversal, linear search | Single pass required |
| O(n log n) | Linearithmic | Merge sort, heap sort | Efficient sorting |
| O(n²) | Quadratic | Bubble sort, nested loops | Small inputs, simple solutions |
| O(2ⁿ) | Exponential | Recursive Fibonacci, subsets | Brute force enumeration |
| O(n!) | Factorial | Permutations | Complete search problems |

---

=== "📊 Big O Fundamentals"

    ## What is Big O?

    Big O notation describes the **worst-case** time or space complexity as input size grows. It focuses on the dominant term and ignores constants.

    ### Growth Comparison

    **From Best to Worst:**

    ```
    O(1) < O(log n) < O(√n) < O(n) < O(n log n) < O(n²) < O(n³) < O(2ⁿ) < O(n!)
    ```

    | Input Size (n) | O(1) | O(log n) | O(n) | O(n log n) | O(n²) | O(2ⁿ) |
    |----------------|------|----------|------|------------|-------|-------|
    | 10 | 1 | 3 | 10 | 33 | 100 | 1,024 |
    | 100 | 1 | 7 | 100 | 664 | 10,000 | 1.26×10³⁰ |
    | 1,000 | 1 | 10 | 1,000 | 9,966 | 1,000,000 | ∞ |
    | 10,000 | 1 | 13 | 10,000 | 132,877 | 100,000,000 | ∞ |

    ### Common Patterns

    === "O(1) - Constant"

        **Operations that take fixed time regardless of input size**

        ```python
        # Array access
        def get_element(arr, index):
            return arr[index]  # O(1)

        # Hash table operations
        def lookup(hash_map, key):
            return hash_map.get(key)  # O(1) average

        # Mathematical operations
        def is_even(n):
            return n % 2 == 0  # O(1)
        ```

        **Interview Tip:** Always mention "average case" for hash operations.

    === "O(log n) - Logarithmic"

        **Halves the search space each iteration**

        ```python
        # Binary search
        def binary_search(arr, target):
            left, right = 0, len(arr) - 1

            while left <= right:  # O(log n) iterations
                mid = (left + right) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1

        # Binary tree height
        # For balanced tree: height = O(log n)
        ```

        **Key Insight:** Each step eliminates half the possibilities.

    === "O(n) - Linear"

        **Single pass through input**

        ```python
        # Array traversal
        def find_max(arr):
            max_val = arr[0]
            for num in arr:  # O(n)
                max_val = max(max_val, num)
            return max_val

        # Two pointers (still O(n))
        def reverse_array(arr):
            left, right = 0, len(arr) - 1
            while left < right:  # O(n/2) = O(n)
                arr[left], arr[right] = arr[right], arr[left]
                left += 1
                right -= 1
        ```

        **Common Mistake:** O(n/2) simplifies to O(n), not O(log n).

    === "O(n log n) - Linearithmic"

        **Efficient sorting algorithms**

        ```python
        # Merge sort
        def merge_sort(arr):
            if len(arr) <= 1:
                return arr

            mid = len(arr) // 2
            left = merge_sort(arr[:mid])   # O(log n) levels
            right = merge_sort(arr[mid:])  # O(log n) levels

            return merge(left, right)       # O(n) per level

        # Quick sort (average case)
        # Heap operations in loop
        def k_largest(arr, k):
            heap = []
            for num in arr:  # O(n)
                heapq.heappush(heap, num)  # O(log n)
            # Total: O(n log n)
        ```

        **Interview Insight:** Best comparison-based sorting complexity.

    === "O(n²) - Quadratic"

        **Nested loops over input**

        ```python
        # Bubble sort
        def bubble_sort(arr):
            n = len(arr)
            for i in range(n):           # O(n)
                for j in range(n - i - 1):  # O(n)
                    if arr[j] > arr[j + 1]:
                        arr[j], arr[j + 1] = arr[j + 1], arr[j]

        # Finding all pairs
        def find_pairs(arr):
            pairs = []
            for i in range(len(arr)):      # O(n)
                for j in range(i + 1, len(arr)):  # O(n)
                    pairs.append((arr[i], arr[j]))
            return pairs
        ```

        **Optimization Tip:** Often can be reduced with hash maps or sorting.

    === "O(2ⁿ) - Exponential"

        **Recursive branching**

        ```python
        # Naive Fibonacci
        def fib(n):
            if n <= 1:
                return n
            return fib(n - 1) + fib(n - 2)  # 2 branches each call

        # Generate all subsets
        def subsets(nums):
            if not nums:
                return [[]]

            rest = subsets(nums[1:])  # 2^(n-1) subsets
            return rest + [nums[:1] + sub for sub in rest]
        ```

        **Red Flag:** Usually needs optimization (DP, memoization).

    === "O(n!) - Factorial"

        **All permutations/arrangements**

        ```python
        # Generate permutations
        def permute(nums):
            if len(nums) <= 1:
                return [nums]

            result = []
            for i in range(len(nums)):
                rest = nums[:i] + nums[i+1:]
                for perm in permute(rest):  # (n-1)! permutations
                    result.append([nums[i]] + perm)
            return result

        # Traveling Salesman (brute force)
        ```

        **Interview Note:** Mention if this is brute force baseline.

=== "⏱️ Time Complexity"

    ## Data Structures Complexity

    ### Arrays & Lists

    | Operation | Array | Dynamic Array | Linked List |
    |-----------|-------|---------------|-------------|
    | **Access** | O(1) | O(1) | O(n) |
    | **Search** | O(n) | O(n) | O(n) |
    | **Insert (end)** | N/A | O(1)* | O(1) |
    | **Insert (beginning)** | N/A | O(n) | O(1) |
    | **Insert (middle)** | N/A | O(n) | O(n) |
    | **Delete (end)** | N/A | O(1) | O(1)** |
    | **Delete (beginning)** | N/A | O(n) | O(1) |
    | **Delete (middle)** | N/A | O(n) | O(n) |

    *Amortized, **With tail pointer

    ### Hash Tables

    | Operation | Average | Worst Case |
    |-----------|---------|------------|
    | **Search** | O(1) | O(n) |
    | **Insert** | O(1) | O(n) |
    | **Delete** | O(1) | O(n) |
    | **Space** | O(n) | O(n) |

    **Worst case:** All keys hash to same bucket (poor hash function or adversarial input).

    ### Trees

    | Operation | BST (Balanced) | BST (Unbalanced) | Heap |
    |-----------|----------------|------------------|------|
    | **Search** | O(log n) | O(n) | O(n) |
    | **Insert** | O(log n) | O(n) | O(log n) |
    | **Delete** | O(log n) | O(n) | O(log n) |
    | **Find Min/Max** | O(log n) | O(n) | O(1) |
    | **Space** | O(n) | O(n) | O(n) |

    **AVL/Red-Black Trees:** Guarantee O(log n) operations through self-balancing.

    ### Graphs

    | Operation | Adjacency List | Adjacency Matrix |
    |-----------|----------------|------------------|
    | **Add Vertex** | O(1) | O(V²) |
    | **Add Edge** | O(1) | O(1) |
    | **Remove Vertex** | O(V + E) | O(V²) |
    | **Remove Edge** | O(E) | O(1) |
    | **Query Edge** | O(V) | O(1) |
    | **Space** | O(V + E) | O(V²) |

    **Dense graphs (E ≈ V²):** Matrix is more space-efficient
    **Sparse graphs (E ≪ V²):** List is more space-efficient

    ---

    ## Algorithm Complexity

    ### Sorting Algorithms

    | Algorithm | Best | Average | Worst | Space | Stable |
    |-----------|------|---------|-------|-------|--------|
    | **Bubble Sort** | O(n) | O(n²) | O(n²) | O(1) | ✅ |
    | **Insertion Sort** | O(n) | O(n²) | O(n²) | O(1) | ✅ |
    | **Selection Sort** | O(n²) | O(n²) | O(n²) | O(1) | ❌ |
    | **Merge Sort** | O(n log n) | O(n log n) | O(n log n) | O(n) | ✅ |
    | **Quick Sort** | O(n log n) | O(n log n) | O(n²) | O(log n) | ❌ |
    | **Heap Sort** | O(n log n) | O(n log n) | O(n log n) | O(1) | ❌ |
    | **Counting Sort** | O(n + k) | O(n + k) | O(n + k) | O(k) | ✅ |
    | **Radix Sort** | O(d(n + k)) | O(d(n + k)) | O(d(n + k)) | O(n + k) | ✅ |

    **k** = range of input, **d** = number of digits

    ### Searching Algorithms

    | Algorithm | Time | Space | Requirements |
    |-----------|------|-------|--------------|
    | **Linear Search** | O(n) | O(1) | None |
    | **Binary Search** | O(log n) | O(1) | Sorted array |
    | **BFS** | O(V + E) | O(V) | Graph traversal |
    | **DFS** | O(V + E) | O(V) | Graph traversal |
    | **Dijkstra's** | O((V + E) log V) | O(V) | Weighted graph |
    | **A*** | O((V + E) log V) | O(V) | Heuristic available |

    ### Dynamic Programming

    ```python
    # Fibonacci - Memoization
    def fib_memo(n, memo={}):
        if n in memo:
            return memo[n]
        if n <= 1:
            return n
        memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
        return memo[n]
    # Time: O(n), Space: O(n)

    # Fibonacci - Tabulation
    def fib_tab(n):
        if n <= 1:
            return n
        dp = [0] * (n + 1)
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]
    # Time: O(n), Space: O(n)

    # Fibonacci - Space Optimized
    def fib_optimized(n):
        if n <= 1:
            return n
        prev, curr = 0, 1
        for _ in range(2, n + 1):
            prev, curr = curr, prev + curr
        return curr
    # Time: O(n), Space: O(1)
    ```

=== "💾 Space Complexity"

    ## Understanding Space Complexity

    Space complexity measures **extra memory** used by an algorithm (excluding input).

    ### What Counts?

    | Include ✅ | Exclude ❌ |
    |-----------|------------|
    | Variables created | Input array |
    | Data structures (arrays, hash maps) | Output (if required) |
    | Recursion call stack | Code itself |
    | Temporary arrays/objects | Primitive constants |

    ---

    ## Space Complexity Patterns

    === "O(1) - Constant Space"

        **Fixed number of variables**

        ```python
        def reverse_array(arr):
            left, right = 0, len(arr) - 1
            while left < right:
                arr[left], arr[right] = arr[right], arr[left]
                left += 1
                right -= 1
        # Space: O(1) - only 2 variables

        def find_max(arr):
            max_val = float('-inf')
            for num in arr:
                max_val = max(max_val, num)
            return max_val
        # Space: O(1) - only 1 variable
        ```

    === "O(log n) - Logarithmic Space"

        **Recursive call stack for divide-and-conquer**

        ```python
        def binary_search(arr, target, left, right):
            if left > right:
                return -1

            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                return binary_search(arr, target, mid + 1, right)
            else:
                return binary_search(arr, target, left, mid - 1)
        # Space: O(log n) - recursion depth

        # Balanced BST traversal
        def inorder(root):
            if not root:
                return
            inorder(root.left)
            print(root.val)
            inorder(root.right)
        # Space: O(log n) for balanced tree
        ```

    === "O(n) - Linear Space"

        **New data structure proportional to input**

        ```python
        # Creating new array
        def double_values(arr):
            result = []
            for num in arr:
                result.append(num * 2)
            return result
        # Space: O(n)

        # Hash map for frequency
        def count_frequency(arr):
            freq = {}
            for num in arr:
                freq[num] = freq.get(num, 0) + 1
            return freq
        # Space: O(n) worst case (all unique)

        # Unbalanced tree recursion
        def dfs(root):
            if not root:
                return
            dfs(root.left)
            dfs(root.right)
        # Space: O(n) for skewed tree
        ```

    === "O(n²) - Quadratic Space"

        **2D arrays or nested structures**

        ```python
        # Dynamic programming table
        def lcs(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])

            return dp[m][n]
        # Space: O(m × n)

        # Adjacency matrix
        graph = [[0] * n for _ in range(n)]
        # Space: O(n²)
        ```

    ---

    ## Space Optimization Techniques

    ### 1. In-Place Algorithms

    **Modify input instead of creating new structures**

    ```python
    # Bad: O(n) space
    def remove_duplicates_bad(arr):
        return list(set(arr))  # Creates new list

    # Good: O(1) space (if sorted)
    def remove_duplicates_good(arr):
        if not arr:
            return 0

        write_idx = 1
        for i in range(1, len(arr)):
            if arr[i] != arr[write_idx - 1]:
                arr[write_idx] = arr[i]
                write_idx += 1

        return write_idx
    ```

    ### 2. Reuse Input Array

    ```python
    # Two Sum - using input array for visited tracking
    def two_sum(nums, target):
        seen = {}  # O(n) space
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return [seen[complement], i]
            seen[num] = i

    # If we can modify input and values are in range [0, n-1]
    def two_sum_inplace(nums, target):
        # Mark visited by negating
        for i, num in enumerate(nums):
            actual = abs(num)
            complement = target - actual
            if 0 <= complement < len(nums) and nums[complement] < 0:
                return [complement, i]
            if 0 <= actual < len(nums):
                nums[actual] = -nums[actual]
    ```

    ### 3. State Space Reduction in DP

    ```python
    # Fibonacci: O(n) space
    def fib(n):
        dp = [0] * (n + 1)
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]

    # Fibonacci: O(1) space
    def fib_optimized(n):
        if n <= 1:
            return n
        prev, curr = 0, 1
        for _ in range(2, n + 1):
            prev, curr = curr, prev + curr
        return curr
    ```

    ### 4. Iterative vs Recursive

    ```python
    # Recursive: O(n) space (call stack)
    def sum_recursive(n):
        if n == 0:
            return 0
        return n + sum_recursive(n - 1)

    # Iterative: O(1) space
    def sum_iterative(n):
        total = 0
        for i in range(n + 1):
            total += i
        return total
    ```

=== "🔄 Recursive Analysis"

    ## Analyzing Recursive Algorithms

    ### Master Theorem

    For recurrences of the form: **T(n) = aT(n/b) + f(n)**

    Where:
    - **a** = number of subproblems
    - **b** = factor by which problem size is divided
    - **f(n)** = work done outside recursion

    | Case | Condition | Solution |
    |------|-----------|----------|
    | **1** | f(n) = O(n^c) where c < log_b(a) | T(n) = Θ(n^(log_b(a))) |
    | **2** | f(n) = Θ(n^c log^k(n)) where c = log_b(a) | T(n) = Θ(n^c log^(k+1)(n)) |
    | **3** | f(n) = Ω(n^c) where c > log_b(a) | T(n) = Θ(f(n)) |

    ### Common Examples

    === "Binary Search"

        ```python
        def binary_search(arr, target, left, right):
            if left > right:
                return -1
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                return binary_search(arr, target, mid + 1, right)
            else:
                return binary_search(arr, target, left, mid - 1)
        ```

        **Recurrence:** T(n) = T(n/2) + O(1)
        - a = 1, b = 2, f(n) = O(1)
        - log₂(1) = 0, f(n) = O(n⁰)
        - **Case 2:** T(n) = O(log n)

    === "Merge Sort"

        ```python
        def merge_sort(arr):
            if len(arr) <= 1:
                return arr

            mid = len(arr) // 2
            left = merge_sort(arr[:mid])
            right = merge_sort(arr[mid:])

            return merge(left, right)  # O(n)
        ```

        **Recurrence:** T(n) = 2T(n/2) + O(n)
        - a = 2, b = 2, f(n) = O(n)
        - log₂(2) = 1, f(n) = O(n¹)
        - **Case 2:** T(n) = O(n log n)

    === "Binary Tree Recursion"

        ```python
        def tree_sum(root):
            if not root:
                return 0
            return root.val + tree_sum(root.left) + tree_sum(root.right)
        ```

        **Recurrence:** T(n) = 2T(n/2) + O(1)
        - a = 2, b = 2, f(n) = O(1)
        - log₂(2) = 1, f(n) = O(n⁰)
        - **Case 1:** T(n) = O(n)

    === "Fibonacci (Naive)"

        ```python
        def fib(n):
            if n <= 1:
                return n
            return fib(n-1) + fib(n-2)
        ```

        **Recurrence:** T(n) = T(n-1) + T(n-2) + O(1)
        - Not in Master Theorem form
        - **Solution:** T(n) = O(2ⁿ) (exponential branching)

    ---

    ## Recursion Tree Method

    **Visual approach to analyze recursive time complexity**

    ### Example: T(n) = 2T(n/2) + n

    ```
    Level 0:                n                    → n
                          /   \
    Level 1:            n/2   n/2                → n
                       / \   / \
    Level 2:         n/4 n/4 n/4 n/4             → n
                     ...
    Level log n:    1 1 1 1 1 1 1 1              → n

    Height: log n
    Work per level: n
    Total: n × log n = O(n log n)
    ```

    ### Example: T(n) = T(n-1) + n (Sum 1 to n)

    ```
    Level 0:    n
    Level 1:    n-1
    Level 2:    n-2
    ...
    Level n:    1

    Total: n + (n-1) + (n-2) + ... + 1 = n(n+1)/2 = O(n²)
    ```

    ### Example: T(n) = 2T(n-1) + 1 (Binary branching)

    ```
    Level 0:                1              → 1 = 2⁰
    Level 1:              1   1            → 2 = 2¹
    Level 2:            1 1  1 1           → 4 = 2²
    Level 3:          1111  1111           → 8 = 2³
    ...
    Level n:          2ⁿ nodes             → 2ⁿ

    Total: 1 + 2 + 4 + 8 + ... + 2ⁿ = 2^(n+1) - 1 = O(2ⁿ)
    ```

=== "⚡ Amortized Analysis"

    ## What is Amortized Analysis?

    **Average cost per operation over a sequence of operations**, even if individual operations may be expensive.

    ---

    ## Common Examples

    ### Dynamic Array (ArrayList)

    ```python
    class DynamicArray:
        def __init__(self):
            self.capacity = 1
            self.size = 0
            self.array = [None] * self.capacity

        def append(self, val):
            if self.size == self.capacity:
                self._resize()  # Expensive: O(n)

            self.array[self.size] = val
            self.size += 1

        def _resize(self):
            self.capacity *= 2
            new_array = [None] * self.capacity
            for i in range(self.size):
                new_array[i] = self.array[i]
            self.array = new_array
    ```

    **Analysis:**
    - Most appends: O(1)
    - Resize at powers of 2: O(n)
    - Sequence of n appends:
      - Resize costs: 1 + 2 + 4 + 8 + ... + n = 2n - 1
      - Total cost: n + (2n - 1) = 3n - 1
      - **Amortized per append: O(1)**

    ### Hash Table with Chaining

    ```python
    class HashTable:
        def __init__(self):
            self.capacity = 8
            self.size = 0
            self.buckets = [[] for _ in range(self.capacity)]

        def insert(self, key, value):
            if self.size / self.capacity > 0.75:  # Load factor
                self._rehash()  # Expensive: O(n)

            bucket = hash(key) % self.capacity
            self.buckets[bucket].append((key, value))
            self.size += 1

        def _rehash(self):
            old_buckets = self.buckets
            self.capacity *= 2
            self.buckets = [[] for _ in range(self.capacity)]
            self.size = 0

            for bucket in old_buckets:
                for key, value in bucket:
                    self.insert(key, value)
    ```

    **Amortized O(1) per insert** (rehashing happens infrequently)

    ### Union-Find (Disjoint Set)

    ```python
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n

        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])  # Path compression
            return self.parent[x]

        def union(self, x, y):
            root_x = self.find(x)
            root_y = self.find(y)

            if root_x == root_y:
                return

            # Union by rank
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
    ```

    **With path compression + union by rank:**
    - Individual operation: O(log n) worst case
    - **Amortized: O(α(n))** where α is inverse Ackermann (≈ constant)

    ### Stack with Min Operation

    ```python
    class MinStack:
        def __init__(self):
            self.stack = []
            self.min_stack = []

        def push(self, val):
            self.stack.append(val)
            if not self.min_stack or val <= self.min_stack[-1]:
                self.min_stack.append(val)

        def pop(self):
            if self.stack.pop() == self.min_stack[-1]:
                self.min_stack.pop()

        def get_min(self):
            return self.min_stack[-1]
    ```

    **All operations O(1)** (no amortization needed, but demonstrates constant-time design)

=== "⚠️ Hidden Complexity"

    ## Common Gotchas in Complexity Analysis

    ### 1. String Operations in Loops

    ```python
    # Looks O(n), actually O(n²) in Python
    def concatenate(words):
        result = ""
        for word in words:           # O(n)
            result += word           # O(k) where k = current length
        return result
    # Total: O(n²) because strings are immutable

    # Correct: O(n) using list
    def concatenate_efficient(words):
        parts = []
        for word in words:
            parts.append(word)       # O(1)
        return "".join(parts)        # O(n)
    ```

    ### 2. Array Slicing

    ```python
    # Looks O(log n), actually O(n log n)
    def binary_search_slice(arr, target):
        if not arr:
            return -1

        mid = len(arr) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            return binary_search_slice(arr[mid+1:], target)  # O(n) slice!
        else:
            return binary_search_slice(arr[:mid], target)     # O(n) slice!
    # Each recursion creates O(n) copy

    # Correct: O(log n) using indices
    def binary_search_correct(arr, target, left=0, right=None):
        if right is None:
            right = len(arr) - 1

        if left > right:
            return -1

        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            return binary_search_correct(arr, target, mid+1, right)
        else:
            return binary_search_correct(arr, target, left, mid-1)
    ```

    ### 3. Nested Data Structure Operations

    ```python
    # Looks O(n), actually O(n²)
    def process_lists(matrix):
        for row in matrix:           # O(n)
            row.sort()               # O(m log m) per row
    # If matrix is n×m: O(n × m log m)

    # Dictionary in loop
    def count_all(lists):
        result = {}
        for lst in lists:            # O(n)
            for item in lst:         # O(m)
                result[item] = result.get(item, 0) + 1  # O(1) average
        return result
    # Total: O(n × m)
    ```

    ### 4. Python Built-in Functions

    | Function | Complexity | Notes |
    |----------|------------|-------|
    | `list.sort()` | O(n log n) | Timsort |
    | `sorted(list)` | O(n log n) | Creates new list |
    | `min(list)`, `max(list)` | O(n) | Linear scan |
    | `list.reverse()` | O(n) | In-place |
    | `reversed(list)` | O(1) | Returns iterator |
    | `''.join(list)` | O(n) | Linear in total length |
    | `list.copy()` | O(n) | Shallow copy |
    | `set(list)` | O(n) | Average case |
    | `x in set` | O(1) | Average case |
    | `x in list` | O(n) | Linear search |

    ```python
    # Looks O(n), actually O(n²)
    def remove_duplicates_bad(arr):
        result = []
        for num in arr:                  # O(n)
            if num not in result:        # O(n) for list!
                result.append(num)
        return result

    # Correct: O(n) using set
    def remove_duplicates_good(arr):
        seen = set()
        result = []
        for num in arr:                  # O(n)
            if num not in seen:          # O(1) for set!
                seen.add(num)
                result.append(num)
        return result
    ```

    ### 5. Recursion Depth vs Work

    ```python
    # Linear recursion depth, but exponential work!
    def bad_fibonacci(n):
        if n <= 1:
            return n
        return bad_fibonacci(n-1) + bad_fibonacci(n-2)
    # Time: O(2ⁿ), Space: O(n) call stack

    # Same recursion depth, linear work
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n-1)
    # Time: O(n), Space: O(n) call stack
    ```

    ### 6. Multiple Loops (Not Always Multiplicative)

    ```python
    # Two separate loops: O(n + m), NOT O(n × m)
    def process_two_arrays(arr1, arr2):
        for x in arr1:       # O(n)
            print(x)
        for y in arr2:       # O(m)
            print(y)
        # Total: O(n + m)

    # Nested loops: O(n × m)
    def cartesian_product(arr1, arr2):
        for x in arr1:       # O(n)
            for y in arr2:   # O(m)
                print(x, y)
        # Total: O(n × m)
    ```

=== "🎯 Interview Tips"

    ## Communicating Complexity

    ### The 4-Step Framework

    **1. Identify Operations**
    - "I'm iterating through the array once..."
    - "For each element, I'm doing a hash lookup..."

    **2. Count Iterations**
    - "The outer loop runs n times..."
    - "The inner loop runs k times for each outer iteration..."

    **3. Combine**
    - "So that's n × k operations total..."
    - "But k is at most n, so worst case is n²..."

    **4. Simplify**
    - "Dropping constants and lower terms: O(n²)"

    ---

    ## Common Interview Questions

    === "Time vs Space Tradeoff"

        **Q:** "Can we optimize the space complexity?"

        ```python
        # O(n) time, O(n) space
        def two_sum_hash(nums, target):
            seen = {}
            for i, num in enumerate(nums):
                if target - num in seen:
                    return [seen[target - num], i]
                seen[num] = i

        # O(n log n) time, O(1) space
        def two_sum_sort(nums, target):
            # Only if we can modify input
            indexed = sorted(enumerate(nums), key=lambda x: x[1])
            left, right = 0, len(nums) - 1

            while left < right:
                curr_sum = indexed[left][1] + indexed[right][1]
                if curr_sum == target:
                    return [indexed[left][0], indexed[right][0]]
                elif curr_sum < target:
                    left += 1
                else:
                    right -= 1
        ```

        **Answer:** "Yes, we can sort and use two pointers for O(1) space, but time becomes O(n log n). The hash map approach is faster at O(n) time but uses O(n) space."

    === "Best/Average/Worst Case"

        **Q:** "What's the difference between best, average, and worst case?"

        ```python
        def linear_search(arr, target):
            for i, val in enumerate(arr):
                if val == target:
                    return i
            return -1
        ```

        **Answer:**
        - **Best case O(1):** Target is first element
        - **Average case O(n):** Target is in middle on average
        - **Worst case O(n):** Target is last element or not present
        - **Big O uses worst case:** O(n)

    === "Amortized Complexity"

        **Q:** "Why is ArrayList append O(1)?"

        **Answer:** "Individual appends can be O(n) when resizing, but this happens rarely. Over a sequence of n appends:
        - Normal appends: n × O(1) = O(n)
        - Resizes: O(1 + 2 + 4 + ... + n) = O(2n)
        - Total: O(3n) = O(n)
        - Amortized per operation: O(n)/n = O(1)"

    === "Space Complexity"

        **Q:** "Does recursion always use O(n) space?"

        ```python
        # O(n) space - linear recursion
        def sum_recursive(n):
            if n == 0:
                return 0
            return n + sum_recursive(n-1)

        # O(log n) space - binary tree
        def binary_search(arr, target, left, right):
            if left > right:
                return -1
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            # Only one recursive call proceeds
            return binary_search(...)

        # O(1) space - tail recursion (optimized)
        def factorial_tail(n, acc=1):
            if n <= 1:
                return acc
            return factorial_tail(n-1, n * acc)
        ```

        **Answer:** "No, it depends on recursion depth and whether the language optimizes tail calls. Binary search is O(log n), tail recursion can be O(1)."

    ---

    ## Red Flags in Interviews

    | Statement | Problem | Better Answer |
    |-----------|---------|---------------|
    | "It's pretty fast" | Vague | "Time complexity is O(n log n)" |
    | "Nested loops are always O(n²)" | Incorrect | "Depends on inner loop bounds" |
    | "Hash maps are O(1)" | Incomplete | "Average O(1), worst O(n)" |
    | "Space complexity is O(n)" | Missing details | "O(n) auxiliary space for the hash map" |
    | "This is optimal" | Unproven | "This is optimal for comparison-based sorting" |

    ---

    ## Quick Checklist

    **Before finishing analysis:**

    - [ ] Identified all loops and recursion
    - [ ] Counted iterations correctly
    - [ ] Considered hidden costs (string ops, slicing, built-ins)
    - [ ] Specified best/average/worst if relevant
    - [ ] Analyzed both time AND space
    - [ ] Mentioned any assumptions (sorted input, etc.)
    - [ ] Discussed potential optimizations
    - [ ] Used Big O notation correctly (no constants, dropped lower terms)

    ---

    ## Practice Problems by Complexity

    | Complexity | Classic Problems |
    |------------|------------------|
    | **O(1)** | Two Sum (with hash), Valid Parentheses (stack) |
    | **O(log n)** | Binary Search, Search in Rotated Sorted Array |
    | **O(n)** | Two Sum, Max Subarray (Kadane's), Linked List Cycle |
    | **O(n log n)** | Merge Sort, K Closest Points, Meeting Rooms II |
    | **O(n²)** | Longest Palindromic Substring, Valid Sudoku |
    | **O(2ⁿ)** | Subsets, Permutations, Combination Sum |

---

## Summary

**Key Takeaways:**

1. **Big O focuses on growth rate** - ignore constants and lower-order terms
2. **Analyze both time and space** - they're equally important
3. **Hidden costs matter** - string operations, slicing, built-ins
4. **Master Theorem** - quick way to solve divide-and-conquer recurrences
5. **Amortized analysis** - consider cost over sequence of operations
6. **Communicate clearly** - walk through your analysis step-by-step

**Most Common in Interviews:**
- O(1), O(log n), O(n), O(n log n), O(n²)
- Know when to trade time for space (hash maps vs sorting)
- Recognize when optimization is needed (O(2ⁿ) → O(n) with DP)
