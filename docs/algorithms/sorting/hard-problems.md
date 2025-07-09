# Sorting & Searching Algorithms - Hard Problems

## 🎯 Learning Objectives

Master advanced sorting and searching techniques:

- External sorting and memory-efficient algorithms
- Advanced data structures for dynamic sorting
- Complex search problems and optimizations
- Hybrid algorithms combining sorting and searching
- Real-world applications with multiple constraints

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Count of Smaller Numbers After Self | BIT/Merge Sort | Hard | O(n log n) | O(n) |
    | 2 | Reverse Pairs | Modified Merge Sort | Hard | O(n log n) | O(n) |
    | 3 | Maximum Gap | Bucket Sort/Radix | Hard | O(n) | O(n) |
    | 4 | Median from Data Stream | Two Heaps | Hard | O(log n) | O(n) |
    | 5 | Sliding Window Maximum | Deque/Segment Tree | Hard | O(n) | O(k) |
    | 6 | Count of Range Sum | Prefix Sum + Sort | Hard | O(n log n) | O(n) |
    | 7 | Russian Doll Envelopes | DP + Binary Search | Hard | O(n log n) | O(n) |
    | 8 | Smallest Range Covering K Lists | Multi-pointer + Heap | Hard | O(n log k) | O(k) |
    | 9 | Median of Two Sorted Arrays | Binary Search | Hard | O(log(min(m,n))) | O(1) |
    | 10 | Count Inversions | Merge Sort Variant | Hard | O(n log n) | O(n) |
    | 11 | Shortest Subarray with Sum ≥ K | Deque + Prefix Sum | Hard | O(n) | O(n) |
    | 12 | Split Array Largest Sum | Binary Search + Greedy | Hard | O(n log(sum)) | O(1) |
    | 13 | Kth Smallest in Sorted Matrix | Binary Search/Heap | Hard | O(n log(max-min)) | O(n) |
    | 14 | Count Good Triplets | Enhanced Merge Sort | Hard | O(n log n) | O(n) |
    | 15 | Minimum Number of Taps | DP + Greedy | Hard | O(n log n) | O(n) |

=== "🎯 Advanced Patterns"

    **🔢 Counting with Sort:**
    - Merge sort with inversion counting
    - Binary Indexed Trees for dynamic counting
    - Range sum queries with sorting
    
    **🎛️ Multi-dimensional Sorting:**
    - Sorting by multiple criteria
    - Coordinate compression techniques
    - K-way merge problems
    
    **🔍 Binary Search on Answer:**
    - Search for optimal value in range
    - Verification function design
    - Monotonic property exploitation
    
    **📊 Stream Processing:**
    - Maintaining sorted order in streams
    - Sliding window optimizations
    - Real-time statistics computation

## Count Smaller Numbers After Self

=== "🔍 Problem Statement"
    
        Given an integer array `nums`, return an integer array `counts` where `counts[i]` is the number of smaller elements to the right of `nums[i]`.
        
        **Example 1:**
        ```
        Input: nums = [5,2,6,1]
        Output: [2,1,1,0]
        Explanation:
        To the right of 5 there are 2 smaller elements (2 and 1).
        To the right of 2 there is only 1 smaller element (1).
        To the right of 6 there is 1 smaller element (1).
        To the right of 1 there is 0 smaller element.
        ```
        
        **Example 2:**
        ```
        Input: nums = [-1]
        Output: [0]
        ```
        
        **Example 3:**
        ```
        Input: nums = [-1,-1]
        Output: [0,0]
        ```
        
        **Constraints:**
        - 1 <= nums.length <= 10^5
        - -10^4 <= nums[i] <= 10^4
    
    === "💡 Solution Approach"
    
        **Key Insight:**
        
        This problem requires counting elements smaller than a given element but only considering elements that appear later in the array. This is a perfect use case for a Binary Indexed Tree (BIT) or Fenwick Tree, which efficiently handles prefix sum operations.
        
        **Step-by-step:**
        
        1. **Coordinate Compression:**
           - Sort the unique values in the array to create a mapping from values to ranks
           - This transforms the problem to work with ranks instead of actual values
        
        2. **Traversal and BIT Operations:**
           - Process the array from right to left
           - For each element:
             - Query the BIT for the count of elements with rank less than current element
             - Update the BIT to include the current element
           - Reverse the result to match the original order
        
        3. **Why BIT?**
           - BIT efficiently handles range queries and point updates in O(log n) time
           - It's more space-efficient than segment trees
        
        **Why it works:**
        
        By processing elements from right to left, we can dynamically count smaller elements seen so far. The BIT helps us efficiently count elements with ranks less than the current element's rank.
    
    === "💻 Implementation"
    
        ```python
        class BIT:
            def __init__(self, n):
                self.n = n
                self.tree = [0] * (n + 1)
            
            def update(self, i, delta):
                while i <= self.n:
                    self.tree[i] += delta
                    i += i & (-i)  # Add the least significant bit
            
            def query(self, i):
                res = 0
                while i > 0:
                    res += self.tree[i]
                    i -= i & (-i)  # Remove the least significant bit
                return res
        
        def countSmaller(nums):
            """
            Count smaller numbers after self using Binary Indexed Tree (BIT)
            
            Time: O(n log n) - n elements with log n operations each
            Space: O(n) - for the BIT and result array
            """
            # Coordinate compression
            sorted_nums = sorted(set(nums))
            rank = {num: i + 1 for i, num in enumerate(sorted_nums)}
            
            bit = BIT(len(sorted_nums))
            result = []
            
            # Process from right to left
            for i in range(len(nums) - 1, -1, -1):
                # Count numbers smaller than current
                count = bit.query(rank[nums[i]] - 1)
                result.append(count)
                
                # Add current number to BIT
                bit.update(rank[nums[i]], 1)
            
            return result[::-1]  # Reverse to match original order
        ```
    
    === "🔄 Alternative Approaches"
    
        **Approach 1: Merge Sort Based Approach**
        
        ```python
        def countSmaller(nums):
            """
            Time: O(n log n) - merge sort complexity
            Space: O(n) - auxiliary arrays for merge sort
            """
            # Pair each number with its original index
            indexed_nums = [(nums[i], i) for i in range(len(nums))]
            result = [0] * len(nums)
            
            def merge_sort(arr, start, end):
                if end - start <= 1:
                    return arr[start:end]
                
                mid = (start + end) // 2
                left = merge_sort(arr, start, mid)
                right = merge_sort(arr, mid, end)
                
                # Count smaller elements on the right for each element on the left
                j = 0
                for i in range(len(left)):
                    while j < len(right) and left[i][0] > right[j][0]:
                        j += 1
                    result[left[i][1]] += j
                
                # Merge the two sorted arrays
                merged = []
                i = j = 0
                while i < len(left) and j < len(right):
                    if left[i][0] <= right[j][0]:
                        merged.append(left[i])
                        i += 1
                    else:
                        merged.append(right[j])
                        j += 1
                
                merged.extend(left[i:])
                merged.extend(right[j:])
                return merged
            
            merge_sort(indexed_nums, 0, len(nums))
            return result
        ```
        
        **Approach 2: Segment Tree Approach**
        
        ```python
        class SegmentTree:
            def __init__(self, n):
                self.n = n
                self.tree = [0] * (4 * n)
                
            def update(self, i, val, node=0, start=0, end=None):
                if end is None:
                    end = self.n - 1
                
                if start == end:
                    self.tree[node] += val
                    return
                    
                mid = (start + end) // 2
                if i <= mid:
                    self.update(i, val, 2*node+1, start, mid)
                else:
                    self.update(i, val, 2*node+2, mid+1, end)
                    
                self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]
                
            def query(self, i, j, node=0, start=0, end=None):
                if end is None:
                    end = self.n - 1
                    
                if i > end or j < start:
                    return 0
                    
                if i <= start and j >= end:
                    return self.tree[node]
                    
                mid = (start + end) // 2
                return (self.query(i, j, 2*node+1, start, mid) + 
                        self.query(i, j, 2*node+2, mid+1, end))
                        
        def countSmaller(nums):
            # Same coordinate compression as BIT approach
            sorted_nums = sorted(set(nums))
            rank = {num: i for i, num in enumerate(sorted_nums)}
            
            seg_tree = SegmentTree(len(sorted_nums))
            result = []
            
            for i in range(len(nums)-1, -1, -1):
                smaller_count = seg_tree.query(0, rank[nums[i]]-1) if rank[nums[i]] > 0 else 0
                result.append(smaller_count)
                seg_tree.update(rank[nums[i]], 1)
                
            return result[::-1]
        ```
    
    === "💭 Tips & Insights"
    
        **Interview Tips:**
        
        1. **Recognize the Pattern:** This problem is a variation of the "count inversions" problem in an array, but with the constraint that we only count inversions with elements to the right.
        
        2. **Data Structure Selection:**
           - BIT is optimal for this problem due to its efficient updates and queries
           - Segment trees are more powerful but have more overhead
           - BSTs can also be used but may lead to unbalanced trees in worst cases
        
        3. **Coordinate Compression:**
           - Essential technique when dealing with large value ranges
           - Maps original values to ranks (1 to n) to reduce space requirements
           - Preserves relative ordering which is all we need for counting
        
        4. **Common Mistakes:**
           - Not handling duplicates correctly
           - Confusing the direction of counting (smaller elements to the right, not left)
           - Incorrectly implementing BIT update/query operations
        
        **Follow-up Questions:**
        
        - How would you modify the solution to count larger elements to the left?
        - Can you solve it with a self-balancing BST like AVL or Red-Black tree?
        - How would you handle very large arrays that don't fit in memory?
        
        **Real-world Applications:**
        
        - Statistical analysis of inversions in data
        - Measuring disorder in sequences
        - Used in adaptive sorting algorithms
        - Analyzing ranking correlations in recommendations systems
    
## Reverse Pairs

=== "🔍 Problem Statement"
    
        Given an integer array `nums`, return the number of reverse pairs in the array.
        
        A reverse pair is a pair (i, j) where:
        - 0 <= i < j < nums.length and
        - nums[i] > 2 * nums[j]
        
        **Example 1:**
        ```
        Input: nums = [1,3,2,3,1]
        Output: 2
        Explanation: The reverse pairs are:
        (1, 4) --> nums[1] = 3, nums[4] = 1, 3 > 2 * 1
        (3, 4) --> nums[3] = 3, nums[4] = 1, 3 > 2 * 1
        ```
        
        **Example 2:**
        ```
        Input: nums = [2,4,3,5,1]
        Output: 3
        Explanation: The reverse pairs are:
        (0, 4) --> nums[0] = 2, nums[4] = 1, 2 > 2 * 1
        (1, 4) --> nums[1] = 4, nums[4] = 1, 4 > 2 * 1
        (2, 4) --> nums[2] = 3, nums[4] = 1, 3 > 2 * 1
        ```
        
        **Constraints:**
        - 1 <= nums.length <= 5 * 10^4
        - -2^31 <= nums[i] <= 2^31 - 1
    
    === "💡 Solution Approach"
    
        **Key Insight:**
        
        The key insight is to use a modified merge sort algorithm that counts reverse pairs during the merge step. This allows us to solve the problem in O(n log n) time instead of O(n²).
        
        **Step-by-step:**
        
        1. Implement a merge sort algorithm with an additional step to count reverse pairs
        2. Before merging two halves, count reverse pairs by comparing each element from the left half with elements from the right half
        3. The condition to check is nums[i] > 2 * nums[j]
        4. Sum up all counts during the recursive merge sort process
        
        **Code Implementation:**
        
        ```python
        def reversePairs(nums):
            """
            Count reverse pairs: i < j and nums[i] > 2 * nums[j]
            """
            def mergeSort(left, right):
                if left >= right:
                    return 0
                
                mid = (left + right) // 2
                count = mergeSort(left, mid) + mergeSort(mid + 1, right)
                
                # Count reverse pairs between left and right halves
                j = mid + 1
                for i in range(left, mid + 1):
                    while j <= right and nums[i] > 2 * nums[j]:
                        j += 1
                    count += j - (mid + 1)
                
                # Merge the two halves
                temp = []
                i, j = left, mid + 1
                while i <= mid and j <= right:
                    if nums[i] <= nums[j]:
                        temp.append(nums[i])
                        i += 1
                    else:
                        temp.append(nums[j])
                        j += 1
                
                temp.extend(nums[i:mid + 1])
                temp.extend(nums[j:right + 1])
                
                for i, val in enumerate(temp):
                    nums[left + i] = val
                
                return count
            
            return mergeSort(0, len(nums) - 1)
        ```
    
    === "🔄 Alternative Approaches"
    
        **Approach: Binary Indexed Tree (Fenwick Tree)**
        
        ```python
        def reversePairs(nums):
            """
            Time: O(n log n)
            Space: O(n)
            """
            # Discretize the array for BIT
            sorted_vals = sorted(nums + [2 * num for num in nums])
            rank = {val: i + 1 for i, val in enumerate(sorted_vals)}
            
            # BIT implementation
            def update(bit, i, val):
                while i < len(bit):
                    bit[i] += val
                    i += i & -i
            
            def query(bit, i):
                res = 0
                while i > 0:
                    res += bit[i]
                    i -= i & -i
                return res
            
            # Count reverse pairs
            bit = [0] * (len(rank) + 1)
            count = 0
            
            for num in reversed(nums):
                count += query(bit, rank[num] - 1)
                update(bit, rank[2 * num], 1)
                
            return count
        ```
        
        **Approach: Binary Search Tree**
        
        ```python
        class TreeNode:
            def __init__(self, val):
                self.val = val
                self.left = None
                self.right = None
                self.count = 1  # Count of this node
                self.less = 0   # Count of nodes in left subtree
        
        def reversePairs(nums):
            """
            Time: O(n log n) average, O(n²) worst case with unbalanced tree
            Space: O(n)
            """
            root = None
            count = 0
            
            # Insert and count function
            def insert_and_count(root, val):
                if not root:
                    return TreeNode(val), 0
                
                count = 0
                if val <= root.val:
                    root.less += 1
                    root.left, count = insert_and_count(root.left, val)
                else:
                    count = root.less + root.count
                    root.right, right_count = insert_and_count(root.right, val)
                    count += right_count
                    
                return root, count
            
            for num in reversed(nums):
                root, pairs = insert_and_count(root, 2 * num)
                count += pairs
                
            return count
        ```
    
    === "💭 Tips & Insights"
    
        **Interview Tips:**
        
        1. **Algorithm Selection:**
           - Merge Sort-based approach is the most straightforward
           - BIT works well for large ranges but requires discretization
           - BST provides another perspective but may suffer from unbalanced trees
        
        2. **Critical Edge Cases:**
           - Handling integer overflow when calculating 2 * nums[j]
           - Arrays with duplicate elements
           - Negative numbers in the array
           - Very large or very small numbers that might cause overflow
        
        3. **Efficiency Concerns:**
           - The naive O(n²) approach will time out for large inputs
           - In-place merge sort modifications are tricky but possible
           - Balancing the tree is critical if using a BST approach
        
        4. **Follow-up Questions:**
           - How would you handle streaming data?
           - Can you solve it with O(n) space complexity?
           - What if we're looking for nums[i] > k * nums[j] for any k?
        
        **Real-world Applications:**
        
        - Analyzing market trends where significant price differences matter
        - Detecting anomalies in time series data
        - Statistical analysis of disproportionate values
        - Data integrity checking in distributed systems
    
## Median from Data Stream

=== "🔍 Problem Statement"

    Design a data structure that supports the following two operations:
    
    - `addNum(int num)`: Add an integer number to the data structure.
    - `findMedian()`: Find the median of all elements so far.
    
    **Example:**
    ```
    addNum(1)
    addNum(2)
    findMedian() -> 1.5
    addNum(3) 
    findMedian() -> 2
    ```
    
    **Constraints:**
    - `-10^5 <= num <= 10^5`
    - There will be at least one element before calling `findMedian`.
    - At most 5 * 10^4 calls will be made to `addNum` and `findMedian`.
    
    **Follow up:**
    - If all integer numbers from the stream are between 0 and 100, how would you optimize it?
    - If 99% of all integer numbers from the stream are between 0 and 100, how would you optimize it?

=== "💡 Solution Approach"

    **Key Insight:**
    
    To efficiently find the median of a stream of numbers, we need a data structure that can:
    1. Keep the elements sorted as new numbers come in
    2. Provide quick access to the middle element(s)
    
    We can use two heaps:
    - A max heap for the smaller half of the numbers
    - A min heap for the larger half of the numbers
    
    By maintaining these two heaps with a balanced size (or with max heap having at most one more element), we can find the median in O(1) time.
    
    **Step-by-step:**
    
    1. Add numbers to either the max heap (small) or min heap (large) based on their value
    2. Balance the heaps so their sizes differ by at most 1
    3. Find the median by:
       - If heaps have equal size: average of the tops of both heaps
       - Otherwise: top of the larger heap
    
    **Code Implementation:**
    
    ```python
        import heapq
        
        class MedianFinder:
            def __init__(self):
                self.small = []  # max heap (negative values)
                self.large = []  # min heap
            
            def addNum(self, num):
                # Add to appropriate heap
                if not self.small or num <= -self.small[0]:
                    heapq.heappush(self.small, -num)
                else:
                    heapq.heappush(self.large, num)
                
                # Balance heaps
                if len(self.small) > len(self.large) + 1:
                    val = -heapq.heappop(self.small)
                    heapq.heappush(self.large, val)
                elif len(self.large) > len(self.small) + 1:
                    val = heapq.heappop(self.large)
                    heapq.heappush(self.small, -val)
            
            def findMedian(self):
                if len(self.small) == len(self.large):
                    return (-self.small[0] + self.large[0]) / 2
                elif len(self.small) > len(self.large):
                    return -self.small[0]
                else:
                    return self.large[0]
        ```

=== "🔄 Alternative Approaches"

    **Approach: Ordered Multiset**
    
    ```python
    from sortedcontainers import SortedList
    
    class MedianFinder:
        def __init__(self):
            self.nums = SortedList()
        
        def addNum(self, num):
            """
            Time: O(log n)
            """
            self.nums.add(num)
        
        def findMedian(self):
            """
            Time: O(1)
            """
            n = len(self.nums)
            if n % 2 == 1:
                return self.nums[n // 2]
            else:
                return (self.nums[n // 2 - 1] + self.nums[n // 2]) / 2
    ```
    
    **Approach: Binary Search Tree**
    
    ```python
    class TreeNode:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None
            self.count = 1
            self.left_count = 0  # Count of nodes in left subtree
    
    class MedianFinder:
        def __init__(self):
            self.root = None
            self.size = 0
        
        def addNum(self, num):
            """
            Time: O(log n) average, O(n) worst case
            """
            self.size += 1
            if not self.root:
                self.root = TreeNode(num)
                return
                
            node = self.root
            while True:
                if num <= node.val:
                    node.left_count += 1
                    if not node.left:
                        node.left = TreeNode(num)
                        break
                    node = node.left
                else:
                    if not node.right:
                        node.right = TreeNode(num)
                        break
                    node = node.right
        
        def findMedian(self):
            """
            Time: O(log n)
            """
            if self.size % 2 == 1:
                return self._findKth((self.size + 1) // 2)
            else:
                return (self._findKth(self.size // 2) + self._findKth(self.size // 2 + 1)) / 2
        
        def _findKth(self, k):
            node = self.root
            while node:
                left_size = node.left_count
                if k == left_size + 1:
                    return node.val
                elif k <= left_size:
                    node = node.left
                else:
                    k -= (left_size + 1)
                    node = node.right
            return None
    ```

=== "💭 Tips & Insights"

    **Interview Tips:**
    
    1. **Data Structure Choice:** Two heaps approach is elegant and efficient, with both operations being O(log n).
    
    2. **Balance Maintenance:** The key challenge is maintaining the balance between the two heaps so that their size difference is at most 1.
    
    3. **Follow-up Optimization:**
       - For integers between 0 and 100, you can use a counting sort approach with an array of size 101
       - For 99% of numbers between 0 and 100, combine a counting array for 0-100 with a heap for outliers
    
    4. **Edge Cases:**
       - Empty data stream
       - Adding the same number multiple times
       - Very large or small numbers that might cause overflow
    
    **Common Mistakes:**
    
    - Incorrectly balancing the heaps
    - Not handling the case where both heaps have the same size
    - Using the wrong heap type (min vs max)
    
    **Real-world Applications:**
    
    - Real-time analytics on streaming data
    - Monitoring systems that need to track the "middle" value
    - Statistical analysis of ongoing experiments
    - Signal processing with continuous input
    
## Median of Two Sorted Arrays

=== "🔍 Problem Statement"

    Given two sorted arrays `nums1` and `nums2` of size `m` and `n` respectively, return the median of the two sorted arrays.
    
    The overall run time complexity should be O(log (m+n)).
    
    **Example 1:**
    ```
    Input: nums1 = [1,3], nums2 = [2]
    Output: 2.0
    Explanation: Merged array = [1,2,3] and median is 2.
    ```
    
    **Example 2:**
    ```
    Input: nums1 = [1,2], nums2 = [3,4]
    Output: 2.5
    Explanation: Merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.
    ```
    
    **Constraints:**
    - `nums1.length == m`
    - `nums2.length == n`
    - `0 <= m <= 1000`
    - `0 <= n <= 1000`
    - `1 <= m + n <= 2000`
    - `-10^6 <= nums1[i], nums2[i] <= 10^6`

=== "💡 Solution Approach"

    **Key Insight:**
    
    The key insight is to use binary search to find the correct partitioning of both arrays such that:
    1. Left parts of both arrays combined have the same number of elements as right parts combined
    2. Every element in the left parts is smaller than or equal to every element in the right parts
    
    When we find such a partitioning, the median is either the max of the left parts (for odd total length) or the average of the max of left parts and min of right parts (for even total length).
    
    **Step-by-step:**
    
    1. Ensure the first array is smaller than or equal to the second array in length
    2. Do binary search on the smaller array to find the right partitioning point
    3. Calculate the corresponding partition point in the second array
    4. Check if the partitioning satisfies our conditions:
       - maxLeft1 ≤ minRight2 and maxLeft2 ≤ minRight1
    5. If not, adjust the partition and continue the binary search
    6. Once found, compute the median based on total length (odd or even)
    
    **Code Implementation:**
    
    ```python
        def findMedianSortedArrays(nums1, nums2):
            """
            Find median of two sorted arrays in O(log(min(m,n)))
            """
            # Ensure nums1 is the smaller array
            if len(nums1) > len(nums2):
                nums1, nums2 = nums2, nums1
            
            m, n = len(nums1), len(nums2)
            left, right = 0, m
            
            while left <= right:
                partition1 = (left + right) // 2
                partition2 = (m + n + 1) // 2 - partition1
                
                # Get max/min elements around partitions
                maxLeft1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
                minRight1 = float('inf') if partition1 == m else nums1[partition1]
                
                maxLeft2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
                minRight2 = float('inf') if partition2 == n else nums2[partition2]
                
                # Check if we found the right partition
                if maxLeft1 <= minRight2 and maxLeft2 <= minRight1:
                    if (m + n) % 2 == 0:
                        return (max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2
                    else:
                        return max(maxLeft1, maxLeft2)
                elif maxLeft1 > minRight2:
                    right = partition1 - 1
                else:
                    left = partition1 + 1
        ```

=== "🔄 Alternative Approaches"

    **Approach: Merge and Find Middle**
    
    ```python
    def findMedianSortedArrays(nums1, nums2):
        """
        Time: O(m+n)
        Space: O(m+n)
        """
        # Merge the arrays
        merged = []
        i = j = 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                merged.append(nums1[i])
                i += 1
            else:
                merged.append(nums2[j])
                j += 1
        
        # Add remaining elements
        merged.extend(nums1[i:])
        merged.extend(nums2[j:])
        
        # Find median
        n = len(merged)
        if n % 2 == 0:
            return (merged[n//2 - 1] + merged[n//2]) / 2
        else:
            return merged[n//2]
    ```
    
    **Approach: Kth Element**
    
    ```python
    def findMedianSortedArrays(nums1, nums2):
        """
        Time: O(log(m+n))
        Space: O(1)
        """
        def findKthElement(arr1, start1, arr2, start2, k):
            # Base cases
            if start1 >= len(arr1):
                return arr2[start2 + k - 1]
            if start2 >= len(arr2):
                return arr1[start1 + k - 1]
            if k == 1:
                return min(arr1[start1], arr2[start2])
            
            # Compare middle elements
            mid1 = start1 + k//2 - 1 if start1 + k//2 - 1 < len(arr1) else len(arr1) - 1
            mid2 = start2 + k//2 - 1 if start2 + k//2 - 1 < len(arr2) else len(arr2) - 1
            
            if arr1[mid1] > arr2[mid2]:
                # Discard first k//2 elements of arr2
                return findKthElement(arr1, start1, arr2, mid2 + 1, k - (mid2 - start2 + 1))
            else:
                # Discard first k//2 elements of arr1
                return findKthElement(arr1, mid1 + 1, arr2, start2, k - (mid1 - start1 + 1))
        
        total = len(nums1) + len(nums2)
        if total % 2 == 1:
            return findKthElement(nums1, 0, nums2, 0, total//2 + 1)
        else:
            return (findKthElement(nums1, 0, nums2, 0, total//2) + 
                   findKthElement(nums1, 0, nums2, 0, total//2 + 1)) / 2
    ```

=== "💭 Tips & Insights"

    **Interview Tips:**
    
    1. **Logarithmic Time Requirement:** This is one of the few problems that explicitly asks for O(log(m+n)) time complexity, which rules out the naive merge approach.
    
    2. **Binary Search Intuition:** The key insight is to partition both arrays in a way that:
       - Left parts combined have exactly (m+n+1)/2 elements
       - Every element in left parts is ≤ every element in right parts
    
    3. **Edge Cases:**
       - Empty arrays
       - Arrays of significantly different sizes
       - Duplicate elements
       - Single-element arrays
    
    4. **Common Pitfalls:**
       - Integer overflow when calculating indices
       - Off-by-one errors in partition calculations
       - Not handling boundary conditions properly
    
    **Real-world Applications:**
    
    - Statistical data analysis
    - Database query optimization
    - Streaming algorithms for large datasets
    - Signal processing with multiple data sources
    
## Sliding Window Maximum

=== "🔍 Problem Statement"

    You are given an array of integers `nums`, there is a sliding window of size `k` which is moving from the very left of the array to the very right. You can only see the `k` numbers in the window. Each time the sliding window moves right by one position.
    
    Return an array of the maximum element in each sliding window.
    
    **Example 1:**
    ```
    Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
    Output: [3,3,5,5,6,7]
    Explanation: 
    Window position                Max
    ---------------               -----
    [1  3  -1] -3  5  3  6  7       3
     1 [3  -1  -3] 5  3  6  7       3
     1  3 [-1  -3  5] 3  6  7       5
     1  3  -1 [-3  5  3] 6  7       5
     1  3  -1  -3 [5  3  6] 7       6
     1  3  -1  -3  5 [3  6  7]      7
    ```
    
    **Example 2:**
    ```
    Input: nums = [1], k = 1
    Output: [1]
    ```
    
    **Constraints:**
    - `1 <= nums.length <= 10^5`
    - `-10^4 <= nums[i] <= 10^4`
    - `1 <= k <= nums.length`

=== "💡 Solution Approach"

    **Key Insight:**
    
    The key insight is to use a deque (double-ended queue) to maintain a decreasing sequence of element indices. The deque will store indices of elements such that the corresponding elements are in decreasing order.
    
    **Step-by-step:**
    
    1. Use a deque to store indices of potential maximum values
    2. For each new element:
       - Remove indices that are outside the current window
       - Remove indices of smaller elements from the back (as they can't be maximum)
       - Add the current index
       - The front of the deque will contain the index of the maximum element
    3. Once the window is full (i >= k-1), add the maximum to the result
    
    This approach is efficient because each element is pushed and popped at most once, leading to O(n) time complexity.
    
    **Code Implementation:**
    
    ```python
        from collections import deque
        
        def maxSlidingWindow(nums, k):
            """
            Find maximum in each sliding window using deque
            """
            dq = deque()  # Store indices
            result = []
            
            for i in range(len(nums)):
                # Remove indices outside current window
                while dq and dq[0] <= i - k:
                    dq.popleft()
                
                # Remove smaller elements from back
                while dq and nums[dq[-1]] < nums[i]:
                    dq.pop()
                
                dq.append(i)
                
                # Add maximum to result if window is complete
                if i >= k - 1:
                    result.append(nums[dq[0]])
            
            return result
        ```

=== "🔄 Alternative Approaches"

    **Approach: Segment Tree**
    
    ```python
    class SegmentTree:
        def __init__(self, nums):
            n = len(nums)
            self.tree = [0] * (4 * n)
            self._build(nums, 0, 0, n-1)
            
        def _build(self, nums, node, start, end):
            if start == end:
                self.tree[node] = nums[start]
                return
                
            mid = (start + end) // 2
            self._build(nums, 2*node+1, start, mid)
            self._build(nums, 2*node+2, mid+1, end)
            self.tree[node] = max(self.tree[2*node+1], self.tree[2*node+2])
            
        def query(self, node, start, end, left, right):
            if left > end or right < start:
                return float('-inf')
                
            if left <= start and end <= right:
                return self.tree[node]
                
            mid = (start + end) // 2
            left_max = self.query(2*node+1, start, mid, left, right)
            right_max = self.query(2*node+2, mid+1, end, left, right)
            return max(left_max, right_max)
    
    def maxSlidingWindow(nums, k):
        """
        Time: O(n log n)
        Space: O(n)
        """
        if not nums or k == 0:
            return []
            
        n = len(nums)
        if k == 1:
            return nums
            
        segment_tree = SegmentTree(nums)
        result = []
        
        for i in range(n-k+1):
            max_val = segment_tree.query(0, 0, n-1, i, i+k-1)
            result.append(max_val)
            
        return result
    ```
    
    **Approach: Brute Force (inefficient)**
    
    ```python
    def maxSlidingWindow(nums, k):
        """
        Time: O(n*k)
        Space: O(n-k+1)
        """
        n = len(nums)
        if n * k == 0:
            return []
            
        result = []
        for i in range(n-k+1):
            max_val = max(nums[i:i+k])
            result.append(max_val)
            
        return result
    ```

=== "💭 Tips & Insights"

    **Interview Tips:**
    
    1. **Optimal Approach:** The deque approach is optimal with O(n) time complexity, where each element is processed exactly twice (pushed and popped).
    
    2. **Data Structure Selection:**
       - Deque is ideal for this problem as it allows O(1) operations at both ends
       - Segment tree is more versatile but has higher overhead
       - Heap-based approaches typically result in O(n log k) complexity
    
    3. **Monotonic Queue Pattern:**
       - This is an example of the "monotonic queue" pattern
       - Useful for finding minimums/maximums in sliding windows
       - Can be adapted for other sliding window problems
    
    4. **Edge Cases:**
       - Empty array
       - k = 1 (each element is its own maximum)
       - k = n (entire array)
    
    **Common Mistakes:**
    
    - Not removing elements outside the current window
    - Storing elements instead of indices (makes it harder to check window boundaries)
    - Inefficient implementation leading to O(n*k) time complexity
    
    **Real-world Applications:**
    
    - Stock price analysis (maximum in a time window)
    - Network traffic monitoring
    - Image processing (maximum filtering)
    - Streaming data analytics
    
## Russian Doll Envelopes

=== "🔍 Problem Statement"

    You are given a 2D array of integers `envelopes` where `envelopes[i] = [wi, hi]` represents the width and the height of an envelope.
    
    One envelope can fit into another if and only if both the width and height of one envelope are greater than the other envelope's width and height.
    
    Return the maximum number of envelopes you can Russian doll (i.e., put one inside the other).
    
    **Note**: You cannot rotate an envelope.
    
    **Example 1:**
    ```
    Input: envelopes = [[5,4],[6,4],[6,7],[2,3]]
    Output: 3
    Explanation: The maximum number of envelopes you can Russian doll is 3 ([2,3] => [5,4] => [6,7]).
    ```
    
    **Example 2:**
    ```
    Input: envelopes = [[1,1],[1,1],[1,1]]
    Output: 1
    ```
    
    **Constraints:**
    - `1 <= envelopes.length <= 10^5`
    - `envelopes[i].length == 2`
    - `1 <= wi, hi <= 10^5`

=== "💡 Solution Approach"

    **Key Insight:**
    
    This problem can be reduced to finding the Longest Increasing Subsequence (LIS) after sorting the envelopes properly. The key insight is to:
    
    1. Sort envelopes by width in ascending order
    2. When widths are the same, sort by height in descending order (to prevent nesting envelopes of the same width)
    3. Find the LIS based on heights
    
    **Step-by-step:**
    
    1. Sort the envelopes by width (ascending) and then by height (descending when widths are equal)
    2. Extract the heights after sorting
    3. Apply an efficient algorithm to find the LIS on the heights
    4. Return the length of the LIS
    
    The sorting step ensures that we only need to consider the height dimension for the LIS, since the width dimension is already taken care of by sorting.
    
    **Code Implementation:**
    
    ```python
        def maxEnvelopes(envelopes):
            """
            Maximum number of nested envelopes
            Sort by width, then find LIS by height
            """
            # Sort by width ascending, height descending
            envelopes.sort(key=lambda x: (x[0], -x[1]))
            
            # Find LIS on heights using binary search
            def lis(heights):
                dp = []
                for h in heights:
                    left, right = 0, len(dp)
                    while left < right:
                        mid = (left + right) // 2
                        if dp[mid] < h:
                            left = mid + 1
                        else:
                            right = mid
                    
                    if left == len(dp):
                        dp.append(h)
                    else:
                        dp[left] = h
                
                return len(dp)
            
            heights = [envelope[1] for envelope in envelopes]
            return lis(heights)
        ```

=== "🔄 Alternative Approaches"

    **Approach: Dynamic Programming**
    
    ```python
    def maxEnvelopes(envelopes):
        """
        Time: O(n²)
        Space: O(n)
        """
        if not envelopes:
            return 0
            
        # Sort by width
        envelopes.sort(key=lambda x: x[0])
        n = len(envelopes)
        dp = [1] * n
        
        for i in range(n):
            for j in range(i):
                if envelopes[i][0] > envelopes[j][0] and envelopes[i][1] > envelopes[j][1]:
                    dp[i] = max(dp[i], dp[j] + 1)
                    
        return max(dp)
    ```

=== "💭 Tips & Insights"

    **Interview Tips:**
    
    1. **Problem Recognition:** This is a classic "box stacking" or "nested objects" problem that can be reduced to the LIS problem.
    
    2. **Sorting Trick:** The key insight is in the special sorting approach. Sorting by width ascending and height descending prevents considering envelopes of the same width.
    
    3. **Binary Search LIS:** Using binary search to find the LIS is much more efficient (O(n log n)) than the traditional dynamic programming approach (O(n²)).
    
    4. **Similar Problems:**
       - Longest Increasing Subsequence
       - Box Stacking Problem
       - Nested Rectangles/Circles
    
    **Common Mistakes:**
    
    - Not handling the case where envelopes have the same width correctly
    - Using inefficient LIS algorithm (quadratic time complexity)
    - Forgetting that both width and height must be strictly greater
    
    **Real-world Applications:**
    
    - Packaging optimization
    - Layout planning for nested containers
    - Resource allocation with hierarchical constraints

=== "📊 Advanced Techniques"

    **🔧 Algorithm Design:**
    - **Divide and Conquer**: Merge sort variants for counting
    - **Binary Search**: Search on answer space, not array
    - **Data Structure Augmentation**: BIT, Segment Tree for dynamic queries
    - **Two Pointers**: Multi-dimensional problems
    - **Coordinate Compression**: Handle large value ranges
    
    **⚡ Optimization Strategies:**
    - **Space-Time Trade-offs**: Memory vs computation
    - **Preprocessing**: Sort once, query multiple times
    - **Incremental Processing**: Stream algorithms
    - **Approximation**: When exact solution is too expensive
    
    **🎯 Pattern Recognition:**
    - **Inversion Counting**: Modified merge sort
    - **Range Queries**: Segment trees, BIT
    - **Multi-dimensional**: Sort by one dimension, process others
    - **Streaming**: Maintain data structure invariants

=== "🚀 Expert Tips"

    **💡 Problem-Solving Approach:**
    1. **Identify Core Operation**: What needs to be optimized?
    2. **Choose Data Structure**: Based on query patterns
    3. **Consider Constraints**: Time, space, update frequency
    4. **Handle Edge Cases**: Empty inputs, single elements
    5. **Verify Complexity**: Ensure it meets requirements
    
    **🔍 Common Pitfalls:**
    - **Integer Overflow**: In counting and sum problems
    - **Index Management**: Off-by-one errors in complex algorithms
    - **Stability Requirements**: When order matters
    - **Memory Limits**: For large datasets
    
    **🏆 Advanced Techniques:**
    - **Persistent Data Structures**: For historical queries
    - **Parallel Algorithms**: For multi-core processing
    - **External Sorting**: For datasets larger than memory
    - **Cache-Aware Algorithms**: Optimize for memory hierarchy

## 📝 Summary

These hard sorting/searching problems demonstrate:

- **Advanced Counting** with merge sort and data structures
- **Multi-dimensional Optimization** with sophisticated algorithms
- **Stream Processing** for real-time applications
- **Binary Search** on complex answer spaces
- **Hybrid Approaches** combining multiple techniques

These algorithms form the foundation for:

- **Database Query Optimization**
- **Distributed Systems** data processing
- **Real-time Analytics** platforms
- **Competitive Programming** advanced problems
- **System Design** scalable solutions

Master these techniques to tackle the most challenging sorting and searching problems in technical interviews and production systems!
