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

=== "💡 Solutions"

    === "Count Smaller Numbers After Self"
        ```python
        class BIT:
            def __init__(self, n):
                self.n = n
                self.tree = [0] * (n + 1)
            
            def update(self, i, delta):
                while i <= self.n:
                    self.tree[i] += delta
                    i += i & (-i)
            
            def query(self, i):
                res = 0
                while i > 0:
                    res += self.tree[i]
                    i -= i & (-i)
                return res
        
        def countSmaller(nums):
            """
            Count smaller numbers after self using BIT
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
            
            return result[::-1]
        ```
    
    === "Reverse Pairs"
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
    
    === "Median from Data Stream"
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
    
    === "Median of Two Sorted Arrays"
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
    
    === "Sliding Window Maximum"
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
    
    === "Russian Doll Envelopes"
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
