# Advanced Complexity Analysis

Master advanced techniques for analyzing algorithm complexity, including recursive algorithms, amortized analysis, space complexity, and common pitfalls.

---

## 🔍 Space Complexity Analysis

=== "Fundamentals"
    **What is Space Complexity?** Memory required by an algorithm as input size grows.

    **Types of Space:**
    - **Auxiliary Space:** Extra space used by algorithm (excluding input)
    - **Total Space:** Input space + Auxiliary space
    - **In-place:** O(1) auxiliary space (modifies input directly)

    | Category | Space Usage | Examples |
    |----------|-------------|----------|
    | **O(1)** | Constant | Swap two variables, iterative algorithms |
    | **O(log n)** | Logarithmic | Recursive binary search (call stack) |
    | **O(n)** | Linear | Recursive algorithms, hash tables, additional arrays |
    | **O(n²)** | Quadratic | 2D arrays, graph adjacency matrix |

=== "Common Patterns"
    **Stack Space in Recursion:**

    ```python
    # O(n) space due to call stack
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)  # n recursive calls

    # O(1) space - iterative
    def factorial_iterative(n):
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
    ```

    **In-place vs Additional Space:**

    ```python
    # O(n) space - creates new array
    def reverse_new_array(arr):
        return arr[::-1]

    # O(1) space - in-place
    def reverse_in_place(arr):
        left, right = 0, len(arr) - 1
        while left < right:
            arr[left], arr[right] = arr[right], arr[left]
            left += 1
            right -= 1
    ```

=== "Space-Time Tradeoffs"
    | Problem | Time-First Approach | Space-First Approach |
    |---------|-------------------|---------------------|
    | **Two Sum** | O(n²) time, O(1) space | O(n) time, O(n) space (hash map) |
    | **Fibonacci** | O(2ⁿ) time, O(n) space | O(n) time, O(n) space (memoization) → O(n) time, O(1) space (iterative) |
    | **String Reversal** | O(n) time, O(n) space (new string) | O(n) time, O(1) space (in-place for arrays) |
    | **Sorting** | Merge sort: O(n log n), O(n) space | Heap sort: O(n log n), O(1) space |

    **When to Optimize Space:**
    - Memory-constrained environments (embedded systems, mobile)
    - Large datasets that don't fit in RAM
    - Streaming data processing
    - Recursive algorithms causing stack overflow

---

## 🌲 Analyzing Recursive Algorithms

=== "Recursion Tree Method"
    **Visualize recursive calls as a tree to calculate total work.**

    **Example: Merge Sort**
    ```
    T(n) = 2T(n/2) + n

                    n                    Level 0: n work
                  /   \
                n/2   n/2                Level 1: n work (2 × n/2)
               / \   / \
             n/4 n/4 n/4 n/4             Level 2: n work (4 × n/4)
             ...

    Height: log n levels
    Work per level: n
    Total: O(n log n)
    ```

    **Example: Binary Search**
    ```
    T(n) = T(n/2) + O(1)

                    n                    Level 0: 1 work
                    |
                   n/2                   Level 1: 1 work
                    |
                   n/4                   Level 2: 1 work
                    ...

    Height: log n levels
    Work per level: 1
    Total: O(log n)
    ```

=== "Master Theorem"
    **For recurrences:** T(n) = aT(n/b) + f(n), where a ≥ 1, b > 1

    Let c = log_b(a)

    | Case | Condition | Result |
    |------|-----------|--------|
    | **Case 1** | f(n) = O(n^c - ε) for ε > 0 | T(n) = Θ(n^c) |
    | **Case 2** | f(n) = Θ(n^c) | T(n) = Θ(n^c log n) |
    | **Case 3** | f(n) = Ω(n^c + ε) and regularity condition | T(n) = Θ(f(n)) |

    **Common Examples:**

    | Recurrence | a | b | f(n) | c = log_b(a) | Case | Result |
    |------------|---|---|------|--------------|------|--------|
    | T(n) = 2T(n/2) + n | 2 | 2 | n | 1 | 2 | O(n log n) - Merge Sort |
    | T(n) = T(n/2) + O(1) | 1 | 2 | 1 | 0 | 2 | O(log n) - Binary Search |
    | T(n) = 2T(n/2) + O(1) | 2 | 2 | 1 | 1 | 1 | O(n) - Tree Traversal |
    | T(n) = 4T(n/2) + n | 4 | 2 | n | 2 | 1 | O(n²) - Bad Divide & Conquer |
    | T(n) = T(n-1) + O(1) | N/A | N/A | 1 | N/A | N/A | O(n) - Linear Recursion |
    | T(n) = 2T(n-1) + O(1) | N/A | N/A | 1 | N/A | N/A | O(2ⁿ) - Fibonacci |

=== "Solving Recurrences"
    **Method 1: Substitution Method**

    Guess the solution, then prove by induction.

    ```
    Example: T(n) = T(n-1) + n

    Guess: T(n) = O(n²)

    Prove:
    T(n) = T(n-1) + n
         ≤ c(n-1)² + n      [by inductive hypothesis]
         = cn² - 2cn + c + n
         ≤ cn²              [for large enough c and n]
    ```

    **Method 2: Iteration Method**

    Expand recurrence until pattern emerges.

    ```
    Example: T(n) = T(n-1) + n

    T(n) = T(n-1) + n
         = [T(n-2) + (n-1)] + n
         = [T(n-3) + (n-2)] + (n-1) + n
         = T(0) + 1 + 2 + ... + n
         = n(n+1)/2
         = O(n²)
    ```

---

## ⚖️ Amortized Analysis

=== "What is Amortized Analysis?"
    **Average cost per operation over a worst-case sequence of operations.**

    **Not the same as average-case analysis:**
    - Average-case: depends on input distribution
    - Amortized: worst-case sequence, but averaged over all operations

    **When to Use:**
    - Occasional expensive operations
    - Data structures with variable operation costs
    - Analyzing sequences of operations

=== "Dynamic Array Example"
    **Problem:** Array doubling when full

    ```python
    class DynamicArray:
        def append(self, item):
            if self.size == self.capacity:
                # Double capacity - O(n) operation
                self._resize(self.capacity * 2)
            self.arr[self.size] = item
            self.size += 1
    ```

    **Analysis:**
    - Most appends: O(1)
    - Resize operations at sizes: 1, 2, 4, 8, 16, 32, ..., n
    - Total cost for n operations:
      - Regular appends: n
      - Copies during resizing: 1 + 2 + 4 + 8 + ... + n/2 ≈ n
      - Total: 2n = O(n)
    - **Amortized cost per append: O(n)/n = O(1)**

=== "Union-Find Example"
    **Operations:** Union and Find with path compression + union by rank

    ```python
    class UnionFind:
        def find(self, x):
            if self.parent[x] != x:
                # Path compression - flatten tree
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x, y):
            # Union by rank - keep tree balanced
            root_x, root_y = self.find(x), self.find(y)
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
    ```

    **Amortized Complexity:** O(α(n)) ≈ O(1) in practice
    - α(n) is the inverse Ackermann function
    - Grows incredibly slowly: α(2^65536) < 5

=== "Analysis Techniques"
    **1. Aggregate Method:** Calculate total cost for n operations, divide by n

    **2. Accounting Method:** Assign different charges to operations, store credit for expensive ops

    **3. Potential Method:** Define potential function that measures "stored energy"

    | Example | Worst Single Op | Amortized | Technique |
    |---------|----------------|-----------|-----------|
    | Dynamic Array append | O(n) | O(1) | Aggregate |
    | Stack with multipop | O(n) | O(1) | Accounting |
    | Splay Tree operations | O(n) | O(log n) | Potential |
    | Fibonacci Heap decrease-key | O(log n) | O(1) | Potential |

---

## ⚠️ Hidden Complexity Traps

=== "String Operations"
    **String Concatenation in Loops:**

    ```python
    # ❌ BAD: O(n²) - creates new string each iteration
    def join_bad(strings):
        result = ""
        for s in strings:
            result += s  # O(len(result)) each time
        return result

    # ✅ GOOD: O(n) - uses efficient join
    def join_good(strings):
        return "".join(strings)  # Single allocation
    ```

    **Substring Operations:**

    ```python
    # Python strings are immutable
    text[1:]      # O(n) - creates new string
    text[:100]    # O(100) - copies characters

    # Use indices instead when possible
    start_idx = 1  # O(1)
    ```

=== "Collection Operations"
    **List Operations:**

    ```python
    # Python list complexity traps
    arr.insert(0, x)    # O(n) - shifts all elements
    arr.pop(0)          # O(n) - shifts all elements
    arr.remove(x)       # O(n) - search + shift
    x in arr            # O(n) - linear search

    # Better alternatives
    from collections import deque
    d = deque()
    d.appendleft(x)     # O(1)
    d.popleft()         # O(1)

    # Or use set for membership
    s = set(arr)
    x in s              # O(1) average
    ```

    **Dictionary Operations:**

    ```python
    # Usually O(1), but watch out:
    dict.copy()         # O(n)
    dict.items()        # O(n) to iterate
    dict.clear()        # O(n)

    # Worst case O(n) due to hash collisions
    # Use proper hash functions
    ```

=== "Nested Library Calls"
    **Hidden O(n) in O(n) loops:**

    ```python
    # ❌ O(n²) - sort is O(n log n) but in O(n) loop
    for sublist in data:
        sublist.sort()  # If sublists are size n, this is O(n²)

    # ❌ O(n²) - list slicing in loop
    for i in range(len(arr)):
        process(arr[:i])  # Creates new list of size i each time

    # ❌ O(n³) - list copying in nested loops
    for i in range(n):
        for j in range(n):
            new_list = old_list.copy()  # O(n) inside O(n²) loop
    ```

=== "Language-Specific Gotchas"
    | Language | Operation | Apparent | Actual | Solution |
    |----------|-----------|----------|--------|----------|
    | **Python** | `list.insert(0, x)` | O(1)? | O(n) | Use `deque.appendleft()` |
    | **Python** | `str += str` in loop | O(n) | O(n²) | Use `"".join(list)` |
    | **Java** | `String +` in loop | O(n) | O(n²) | Use `StringBuilder` |
    | **JavaScript** | `arr.unshift(x)` | O(1)? | O(n) | Use push + reverse, or linked list |
    | **C++** | `vector.insert(begin)` | O(1)? | O(n) | Use `deque` or `list` |
    | **Python** | `list * n` | O(n)? | O(1) shallow | Deep copy elements if needed |

---

## 🎤 Interview Complexity Communication

=== "How to Explain Complexity"
    **Step-by-Step Framework:**

    1. **Identify input size:** "Let n be the array length"
    2. **Analyze loops:** "We iterate through the array once - O(n)"
    3. **Analyze nested operations:** "For each element, we search the hash map - O(1)"
    4. **Calculate total:** "O(n) × O(1) = O(n) total"
    5. **Space analysis:** "We use a hash map storing up to n elements - O(n) space"
    6. **State final answer:** "Time: O(n), Space: O(n)"

    **Example Walkthrough:**
    ```python
    def two_sum(nums, target):
        seen = {}                    # O(1) space allocation
        for i, num in enumerate(nums):  # O(n) iterations
            complement = target - num    # O(1)
            if complement in seen:       # O(1) hash lookup
                return [seen[complement], i]
            seen[num] = i               # O(1) hash insert
        return []
    ```

    **Your explanation:**
    > "We iterate through the array once - that's O(n). For each element, we do constant time hash map operations: lookup and insert, both O(1). So the total time complexity is O(n). For space, we store at most n elements in the hash map, so space complexity is also O(n)."

=== "Common Interview Questions"
    **Q: "Can you optimize this?"**
    - Always state current complexity first
    - Discuss theoretical lower bound if known
    - Consider time-space tradeoffs
    - Mention if current solution is optimal

    **Q: "What's the time complexity?"**
    - State both best and worst case if different
    - Mention average case if relevant
    - Don't forget space complexity

    **Q: "Why is this O(n log n)?"**
    - Walk through the recursion tree or iterations
    - Explain work per level × number of levels
    - Use concrete examples (n=8, n=16)

=== "Red Flags Interviewers Watch For"
    ❌ **Mistakes to Avoid:**

    1. **Confusing nested loops with sequential:**
       - Two loops: O(n) + O(n) = O(n), not O(n²)
       - Nested loops: O(n) × O(n) = O(n²)

    2. **Ignoring hidden complexity:**
       - "It's just one sort" - but sort is O(n log n)!
       - String concatenation in loop - O(n²), not O(n)

    3. **Dropping important terms:**
       - O(n + m) ≠ O(n) when m could be much larger than n
       - Graph problems: keep O(V + E), don't simplify

    4. **Wrong recurrence analysis:**
       - T(n) = 2T(n/2) + n is O(n log n), not O(n²)

    5. **Forgetting space complexity:**
       - Recursion uses call stack - O(depth) space
       - "In-place" but creates temp arrays

---

## 📝 Code Examples by Complexity

=== "O(1) - Constant"
    ```python
    # Array access
    def get_element(arr, index):
        return arr[index]  # O(1)

    # Hash map operations
    def hash_lookup(hash_map, key):
        return hash_map.get(key)  # O(1) average

    # Math operations
    def is_even(n):
        return n % 2 == 0  # O(1)
    ```

=== "O(log n) - Logarithmic"
    ```python
    # Binary search
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1  # O(log n)

    # Balanced BST operations
    # TreeMap/TreeSet in Java, balanced trees
    ```

=== "O(n) - Linear"
    ```python
    # Array traversal
    def find_max(arr):
        max_val = arr[0]
        for num in arr:
            max_val = max(max_val, num)
        return max_val  # O(n)

    # Two pointers
    def two_sum_sorted(arr, target):
        left, right = 0, len(arr) - 1
        while left < right:
            current = arr[left] + arr[right]
            if current == target:
                return [left, right]
            elif current < target:
                left += 1
            else:
                right -= 1
        return []  # O(n)
    ```

=== "O(n log n) - Linearithmic"
    ```python
    # Merge sort
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])

        return merge(left, right)  # O(n log n)

    # Sorting for preprocessing
    def find_closest_pair(points):
        points.sort()  # O(n log n)
        min_diff = float('inf')
        for i in range(len(points) - 1):
            min_diff = min(min_diff, points[i+1] - points[i])
        return min_diff  # Total: O(n log n)
    ```

=== "O(n²) - Quadratic"
    ```python
    # Bubble sort
    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            for j in range(n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr  # O(n²)

    # Nested iteration
    def count_pairs(arr):
        count = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[i] + arr[j] == target:
                    count += 1
        return count  # O(n²)
    ```

=== "O(2ⁿ) - Exponential"
    ```python
    # Recursive Fibonacci (naive)
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)  # O(2ⁿ)

    # Generate all subsets
    def subsets(nums):
        result = []
        def backtrack(index, current):
            if index == len(nums):
                result.append(current[:])
                return
            # Include nums[index]
            current.append(nums[index])
            backtrack(index + 1, current)
            # Exclude nums[index]
            current.pop()
            backtrack(index + 1, current)
        backtrack(0, [])
        return result  # O(2ⁿ)
    ```

---

## 🎯 Practice Problems by Pattern

=== "Two Pointers O(n)"
    **Problems:**
    1. [Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/) - Easy
    2. [Container With Most Water](https://leetcode.com/problems/container-with-most-water/) - Medium
    3. [3Sum](https://leetcode.com/problems/3sum/) - Medium
    4. [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/) - Hard

    **Key Insight:** Reduce O(n²) nested loops to O(n) by moving pointers strategically

=== "Sliding Window O(n)"
    **Problems:**
    1. [Maximum Average Subarray I](https://leetcode.com/problems/maximum-average-subarray-i/) - Easy
    2. [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/) - Medium
    3. [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/) - Hard

    **Key Insight:** Maintain a window and slide it instead of recalculating each subarray

=== "Binary Search O(log n)"
    **Problems:**
    1. [Binary Search](https://leetcode.com/problems/binary-search/) - Easy
    2. [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/) - Medium
    3. [Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/) - Hard

    **Key Insight:** Reduce search space by half each iteration

=== "Hash Map O(n)"
    **Problems:**
    1. [Two Sum](https://leetcode.com/problems/two-sum/) - Easy
    2. [Group Anagrams](https://leetcode.com/problems/group-anagrams/) - Medium
    3. [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/) - Medium

    **Key Insight:** Trade space for time to achieve O(1) lookups

=== "Dynamic Programming"
    **Problems:**
    1. [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/) - Easy, O(n)
    2. [Coin Change](https://leetcode.com/problems/coin-change/) - Medium, O(n × amount)
    3. [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/) - Medium, O(n log n) optimal
    4. [Edit Distance](https://leetcode.com/problems/edit-distance/) - Hard, O(m × n)

    **Key Insight:** Memoize overlapping subproblems to avoid exponential time

---

## 🔗 Related Resources

**Time Complexity Main Guide:** [Time & Space Complexity](time-complexity.md) - Quick reference tables and fundamentals

**Algorithm Patterns:** [Problem-Solving Patterns](problem-solving-patterns.md) - Common patterns with complexity analysis

**Interview Strategy:** [Interview Strategy](interview-strategy.md) - How to discuss complexity in interviews

**External Resources:**
- [Big-O Cheat Sheet](https://www.bigocheatsheet.com/) - Visual complexity reference
- [Master Theorem Calculator](https://www.geogebra.org/m/mDDzmgSQ) - Verify recurrence relations
- [VisuAlgo](https://visualgo.net/) - Visualize algorithm complexity
