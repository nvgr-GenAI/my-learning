# Arrays: Hard Problems

## 🔥 Advanced Array Challenges

These problems require sophisticated algorithms, complex optimizations, and deep understanding of array manipulation techniques.

---

## Problem 1: Median of Two Sorted Arrays

**Difficulty:** Hard  
**Pattern:** Binary Search  
**Time:** O(log(min(m,n))) | **Space:** O(1)

### Problem Statement

Given two sorted arrays `nums1` and `nums2`, return the median of the two sorted arrays combined.

**Example:**
```text
Input: nums1 = [1,3], nums2 = [2]
Output: 2.0
Explanation: merged array = [1,2,3] and median is 2
```

### Solution

```python
def find_median_sorted_arrays(nums1, nums2):
    """
    Find median using binary search on smaller array.
    
    Ensure nums1 is the smaller array for optimization.
    """
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    
    while left <= right:
        partition1 = (left + right) // 2
        partition2 = (m + n + 1) // 2 - partition1
        
        # Handle edge cases
        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float('inf') if partition1 == m else nums1[partition1]
        
        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]
        
        # Check if we found the correct partition
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            # Found the correct partition
            if (m + n) % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            else:
                return max(max_left1, max_left2)
        elif max_left1 > min_right2:
            # Too far right in nums1
            right = partition1 - 1
        else:
            # Too far left in nums1
            left = partition1 + 1
    
    raise ValueError("Arrays are not sorted")

# Alternative: Merge approach (less efficient)
def find_median_merge(nums1, nums2):
    """O((m+n) log(m+n)) approach by merging and sorting."""
    merged = sorted(nums1 + nums2)
    n = len(merged)
    
    if n % 2 == 0:
        return (merged[n // 2 - 1] + merged[n // 2]) / 2
    else:
        return merged[n // 2]

# Test
test_cases = [
    ([1, 3], [2]),
    ([1, 2], [3, 4]),
    ([0, 0], [0, 0]),
    ([], [1]),
    ([2], [])
]

for nums1, nums2 in test_cases:
    median_binary = find_median_sorted_arrays(nums1, nums2)
    median_merge = find_median_merge(nums1, nums2)
    print(f"Arrays: {nums1}, {nums2}")
    print(f"  Binary search: {median_binary}")
    print(f"  Merge approach: {median_merge}")
```

---

## Problem 2: First Missing Positive

**Difficulty:** Hard  
**Pattern:** Array as Hash Set  
**Time:** O(n) | **Space:** O(1)

### Problem Statement

Given an unsorted integer array `nums`, return the smallest missing positive integer.

**Example:**
```text
Input: nums = [3,4,-1,1]
Output: 2
```

### Solution

```python
def first_missing_positive(nums):
    """
    Find first missing positive using array as hash set.
    
    Place each positive number at its correct index position.
    """
    n = len(nums)
    
    # Step 1: Replace non-positive numbers with n+1
    for i in range(n):
        if nums[i] <= 0:
            nums[i] = n + 1
    
    # Step 2: Mark presence using sign
    for i in range(n):
        num = abs(nums[i])
        if 1 <= num <= n:
            # Mark as negative (present)
            if nums[num - 1] > 0:
                nums[num - 1] = -nums[num - 1]
    
    # Step 3: Find first positive number
    for i in range(n):
        if nums[i] > 0:
            return i + 1
    
    return n + 1

# Alternative: Using cyclic sort
def first_missing_positive_cyclic_sort(nums):
    """Using cyclic sort to place elements at correct positions."""
    n = len(nums)
    
    # Place each number at its correct position
    i = 0
    while i < n:
        # If number is in range [1, n] and not at correct position
        if 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            # Swap to correct position
            correct_pos = nums[i] - 1
            nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
        else:
            i += 1
    
    # Find first missing positive
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    
    return n + 1

# Alternative: Using set (extra space)
def first_missing_positive_set(nums):
    """Using set - O(n) space but simpler logic."""
    num_set = set(nums)
    
    for i in range(1, len(nums) + 2):
        if i not in num_set:
            return i

# Test
test_cases = [
    [3, 4, -1, 1],
    [1, 2, 0],
    [7, 8, 9, 11, 12],
    [1],
    [-1, 4, 2, 1, 9, 10]
]

for nums in test_cases:
    nums_copy1 = nums.copy()
    nums_copy2 = nums.copy()
    nums_copy3 = nums.copy()
    
    result1 = first_missing_positive(nums_copy1)
    result2 = first_missing_positive_cyclic_sort(nums_copy2)
    result3 = first_missing_positive_set(nums_copy3)
    
    print(f"Array: {nums}")
    print(f"  Sign marking: {result1}")
    print(f"  Cyclic sort: {result2}")
    print(f"  Set approach: {result3}")
```

---

## Problem 3: Trapping Rain Water

**Difficulty:** Hard  
**Pattern:** Two Pointers / Stack  
**Time:** O(n) | **Space:** O(1)

### Problem Statement

Given an array representing heights of bars, calculate how much water can be trapped after raining.

**Example:**
```text
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
```

### Solution

```python
def trap_rain_water(height):
    """
    Calculate trapped water using two pointers.
    
    Water level at any position is min(max_left, max_right) - height.
    """
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0
    
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    
    return water

# Alternative: Using dynamic programming
def trap_rain_water_dp(height):
    """Using left and right max arrays."""
    if not height:
        return 0
    
    n = len(height)
    left_max = [0] * n
    right_max = [0] * n
    
    # Fill left_max array
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i - 1], height[i])
    
    # Fill right_max array
    right_max[n - 1] = height[n - 1]
    for i in range(n - 2, -1, -1):
        right_max[i] = max(right_max[i + 1], height[i])
    
    # Calculate trapped water
    water = 0
    for i in range(n):
        water_level = min(left_max[i], right_max[i])
        if water_level > height[i]:
            water += water_level - height[i]
    
    return water

# Alternative: Using stack
def trap_rain_water_stack(height):
    """Using stack to find water pockets."""
    stack = []
    water = 0
    
    for i, h in enumerate(height):
        while stack and height[stack[-1]] < h:
            bottom = stack.pop()
            
            if not stack:
                break
            
            distance = i - stack[-1] - 1
            bounded_height = min(height[stack[-1]], h) - height[bottom]
            water += distance * bounded_height
        
        stack.append(i)
    
    return water

# Test
height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
print(f"Height array: {height}")
print(f"Trapped water (two pointers): {trap_rain_water(height)}")
print(f"Trapped water (DP): {trap_rain_water_dp(height)}")
print(f"Trapped water (stack): {trap_rain_water_stack(height)}")
```

---

## Problem 4: Sliding Window Maximum

**Difficulty:** Hard  
**Pattern:** Monotonic Deque  
**Time:** O(n) | **Space:** O(k)

### Problem Statement

Given an array and a sliding window of size k, return the maximum element in each window position.

**Example:**
```text
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
```

### Solution

```python
from collections import deque

def max_sliding_window(nums, k):
    """
    Find maximum in each sliding window using monotonic deque.
    
    Deque stores indices in decreasing order of values.
    """
    if not nums or k == 0:
        return []
    
    dq = deque()  # Store indices
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Maintain decreasing order in deque
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add result when window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

# Alternative: Using heap (less efficient)
import heapq

def max_sliding_window_heap(nums, k):
    """Using max heap - O(n log k) approach."""
    if not nums or k == 0:
        return []
    
    result = []
    heap = []  # Max heap using negative values
    
    # Process first window
    for i in range(k):
        heapq.heappush(heap, (-nums[i], i))
    
    result.append(-heap[0][0])
    
    # Process remaining elements
    for i in range(k, len(nums)):
        heapq.heappush(heap, (-nums[i], i))
        
        # Remove elements outside current window
        while heap[0][1] <= i - k:
            heapq.heappop(heap)
        
        result.append(-heap[0][0])
    
    return result

# Alternative: Brute force (inefficient)
def max_sliding_window_brute(nums, k):
    """Brute force - O(nk) approach."""
    if not nums or k == 0:
        return []
    
    result = []
    for i in range(len(nums) - k + 1):
        window_max = max(nums[i:i + k])
        result.append(window_max)
    
    return result

# Test
nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3

print(f"Array: {nums}, k = {k}")
print(f"Deque approach: {max_sliding_window(nums, k)}")
print(f"Heap approach: {max_sliding_window_heap(nums, k)}")
print(f"Brute force: {max_sliding_window_brute(nums, k)}")
```

---

## Problem 5: Longest Consecutive Sequence

**Difficulty:** Hard  
**Pattern:** Hash Set  
**Time:** O(n) | **Space:** O(n)

### Problem Statement

Given an unsorted array, find the length of the longest consecutive elements sequence.

**Example:**
```text
Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: [1, 2, 3, 4] is the longest consecutive sequence
```

### Solution

```python
def longest_consecutive(nums):
    """
    Find longest consecutive sequence using hash set.
    
    Only start counting from numbers that don't have a predecessor.
    """
    if not nums:
        return 0
    
    num_set = set(nums)
    longest = 0
    
    for num in num_set:
        # Only start sequence if num-1 is not in set
        if num - 1 not in num_set:
            current_num = num
            current_length = 1
            
            # Count consecutive numbers
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
            
            longest = max(longest, current_length)
    
    return longest

# Alternative: Using Union-Find
class UnionFind:
    def __init__(self, nums):
        self.parent = {num: num for num in nums}
        self.size = {num: 1 for num in nums}
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            # Union by size
            if self.size[root_x] < self.size[root_y]:
                root_x, root_y = root_y, root_x
            
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
    
    def get_max_size(self):
        return max(self.size.values()) if self.size else 0

def longest_consecutive_union_find(nums):
    """Using Union-Find data structure."""
    if not nums:
        return 0
    
    unique_nums = list(set(nums))
    uf = UnionFind(unique_nums)
    num_set = set(unique_nums)
    
    for num in unique_nums:
        if num + 1 in num_set:
            uf.union(num, num + 1)
    
    return uf.get_max_size()

# Alternative: Sorting approach (less efficient)
def longest_consecutive_sort(nums):
    """Using sorting - O(n log n) approach."""
    if not nums:
        return 0
    
    nums = sorted(set(nums))  # Remove duplicates and sort
    longest = 1
    current_length = 1
    
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            current_length += 1
        else:
            longest = max(longest, current_length)
            current_length = 1
    
    return max(longest, current_length)

# Test
test_cases = [
    [100, 4, 200, 1, 3, 2],
    [0, 3, 7, 2, 5, 8, 4, 6, 0, 1],
    [1, 2, 0, 1],
    [],
    [1]
]

for nums in test_cases:
    result1 = longest_consecutive(nums)
    result2 = longest_consecutive_union_find(nums)
    result3 = longest_consecutive_sort(nums)
    
    print(f"Array: {nums}")
    print(f"  Hash set: {result1}")
    print(f"  Union-Find: {result2}")
    print(f"  Sorting: {result3}")
```

---

## Problem 6: Minimum Window Substring

**Difficulty:** Hard  
**Pattern:** Sliding Window + Hash Map  
**Time:** O(m+n) | **Space:** O(k)

### Problem Statement

Given strings `s` and `t`, return the minimum window substring of `s` such that every character in `t` is included in the window.

**Example:**
```text
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
```

### Solution

```python
from collections import defaultdict, Counter

def min_window(s, t):
    """
    Find minimum window substring using sliding window.
    
    Expand window until valid, then contract to find minimum.
    """
    if not s or not t or len(s) < len(t):
        return ""
    
    # Count characters in t
    t_count = Counter(t)
    required = len(t_count)
    
    # Sliding window variables
    left = right = 0
    formed = 0  # Number of unique chars in window with desired frequency
    window_counts = defaultdict(int)
    
    # Result
    min_len = float('inf')
    min_left = 0
    
    while right < len(s):
        # Expand window
        char = s[right]
        window_counts[char] += 1
        
        # Check if char frequency matches requirement
        if char in t_count and window_counts[char] == t_count[char]:
            formed += 1
        
        # Contract window
        while left <= right and formed == required:
            # Update result if current window is smaller
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_left = left
            
            # Remove leftmost character
            char = s[left]
            window_counts[char] -= 1
            if char in t_count and window_counts[char] < t_count[char]:
                formed -= 1
            
            left += 1
        
        right += 1
    
    return "" if min_len == float('inf') else s[min_left:min_left + min_len]

# Alternative: Optimized with filtered string
def min_window_optimized(s, t):
    """Optimized version with filtered string."""
    if not s or not t:
        return ""
    
    t_count = Counter(t)
    required = len(t_count)
    
    # Filter s to only include characters in t
    filtered_s = []
    for i, char in enumerate(s):
        if char in t_count:
            filtered_s.append((i, char))
    
    left = right = 0
    formed = 0
    window_counts = defaultdict(int)
    
    min_len = float('inf')
    min_left = 0
    
    while right < len(filtered_s):
        char = filtered_s[right][1]
        window_counts[char] += 1
        
        if window_counts[char] == t_count[char]:
            formed += 1
        
        while left <= right and formed == required:
            start = filtered_s[left][0]
            end = filtered_s[right][0]
            
            if end - start + 1 < min_len:
                min_len = end - start + 1
                min_left = start
            
            char = filtered_s[left][1]
            window_counts[char] -= 1
            if window_counts[char] < t_count[char]:
                formed -= 1
            
            left += 1
        
        right += 1
    
    return "" if min_len == float('inf') else s[min_left:min_left + min_len]

# Test
test_cases = [
    ("ADOBECODEBANC", "ABC"),
    ("a", "a"),
    ("a", "aa"),
    ("ab", "b"),
    ("abc", "cba")
]

for s, t in test_cases:
    result1 = min_window(s, t)
    result2 = min_window_optimized(s, t)
    
    print(f"s = '{s}', t = '{t}'")
    print(f"  Standard: '{result1}'")
    print(f"  Optimized: '{result2}'")
```

---

## 🎯 Advanced Problem-Solving Patterns

### 1. Binary Search on Answer Space
- **Use when:** Looking for optimal value in sorted space
- **Pattern:** Binary search on possible answers
- **Examples:** Median of sorted arrays, allocate minimum pages

### 2. Array as Hash Map/Set
- **Use when:** Array elements are in specific range
- **Pattern:** Use indices to store information
- **Examples:** First missing positive, find duplicates

### 3. Monotonic Deque/Stack
- **Use when:** Need to maintain order while processing
- **Pattern:** Remove elements that can't be optimal
- **Examples:** Sliding window maximum, largest rectangle

### 4. Advanced Sliding Window
- **Use when:** Complex substring/subarray problems
- **Pattern:** Expand and contract with multiple conditions
- **Examples:** Minimum window substring, substring with k distinct

## 💡 Expert Tips

!!! tip "Time-Space Tradeoffs"
    Many hard problems offer multiple solutions with different time-space complexities. Choose based on constraints.

!!! warning "Edge Case Mastery"
    Hard problems often have tricky edge cases. Always test with empty inputs, single elements, and boundary conditions.

!!! success "Pattern Recognition"
    Master recognizing when to apply binary search, two pointers, sliding window, or stack/deque patterns.

## 🚀 Mastery Achieved

Congratulations! You've completed the hardest array problems. You now have mastery over:

- **Binary Search Variants**: Complex search space problems
- **Advanced Two Pointers**: Multi-constraint optimization
- **Sophisticated Sliding Window**: Complex substring problems  
- **Array Manipulation**: In-place algorithms with O(1) space
- **Data Structure Integration**: Combining arrays with stacks, deques, heaps

---

*🏆 Outstanding achievement! You've mastered the most challenging array problems and are ready for advanced algorithmic challenges!*
