# Divide and Conquer - Medium Problems

## Problem Categories

### 1. Advanced Sorting and Searching
- Quick sort variations
- Order statistics
- Matrix searching

### 2. Geometric Algorithms
- Closest pair of points
- Convex hull
- Line intersection

### 3. String Algorithms
- Longest common subsequence
- Edit distance
- Pattern matching

---

## 1. Kth Largest Element (QuickSelect)

**Problem**: Find the kth largest element in an unsorted array.

**Example**:
```
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5
```

**Solution**:
```python
def find_kth_largest(nums, k):
    """
    Find kth largest element using QuickSelect (divide and conquer).
    
    Average Time: O(n)
    Worst Time: O(n²)  
    Space: O(log n) - recursion stack
    """
    def quickselect(left, right, k_smallest):
        # Base case: only one element
        if left == right:
            return nums[left]
        
        # Choose random pivot to avoid worst case
        import random
        pivot_index = random.randint(left, right)
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
        
        # Partition around pivot
        pivot_index = partition(left, right)
        
        # Check which part contains kth smallest
        if k_smallest == pivot_index:
            return nums[k_smallest]
        elif k_smallest < pivot_index:
            return quickselect(left, pivot_index - 1, k_smallest)
        else:
            return quickselect(pivot_index + 1, right, k_smallest)
    
    def partition(left, right):
        pivot = nums[right]
        i = left
        
        for j in range(left, right):
            if nums[j] <= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        
        nums[i], nums[right] = nums[right], nums[i]
        return i
    
    # Convert kth largest to (n-k)th smallest
    return quickselect(0, len(nums) - 1, len(nums) - k)

# Optimized version with median-of-medians for guaranteed O(n)
def find_kth_largest_guaranteed(nums, k):
    """
    Guaranteed O(n) using median-of-medians pivot selection.
    """
    def median_of_medians(arr, left, right):
        if right - left < 5:
            return sorted(arr[left:right+1])[len(arr[left:right+1])//2]
        
        medians = []
        for i in range(left, right + 1, 5):
            group_right = min(i + 4, right)
            group_median = sorted(arr[i:group_right+1])[len(arr[i:group_right+1])//2]
            medians.append(group_median)
        
        return median_of_medians(medians, 0, len(medians) - 1)
    
    def quickselect_guaranteed(left, right, k_smallest):
        if left == right:
            return nums[left]
        
        # Use median-of-medians as pivot
        pivot_value = median_of_medians(nums, left, right)
        
        # Find pivot index
        pivot_index = left
        for i in range(left, right + 1):
            if nums[i] == pivot_value:
                pivot_index = i
                break
        
        # Move pivot to end
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
        
        # Partition
        pivot_index = partition(left, right)
        
        if k_smallest == pivot_index:
            return nums[k_smallest]
        elif k_smallest < pivot_index:
            return quickselect_guaranteed(left, pivot_index - 1, k_smallest)
        else:
            return quickselect_guaranteed(pivot_index + 1, right, k_smallest)
    
    return quickselect_guaranteed(0, len(nums) - 1, len(nums) - k)
```

**Key Points**:
- Average O(n) time complexity due to eliminating half the elements each time
- Randomized pivot selection helps avoid worst-case performance
- Median-of-medians guarantees O(n) worst-case performance

---

## 2. Closest Pair of Points

**Problem**: Given n points in a 2D plane, find the pair of points with the smallest distance.

**Example**:
```
Input: points = [[0,0],[3,4],[1,1],[2,2]]
Output: 1.41421 (distance between [1,1] and [2,2])
```

**Solution**:
```python
import math

def closest_pair_distance(points):
    """
    Find closest pair of points using divide and conquer.
    
    Time Complexity: O(n log²n)
    Space Complexity: O(n)
    """
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def brute_force(px):
        """Brute force for small arrays."""
        min_dist = float('inf')
        n = len(px)
        
        for i in range(n):
            for j in range(i + 1, n):
                min_dist = min(min_dist, distance(px[i], px[j]))
        
        return min_dist
    
    def closest_pair_rec(px, py):
        n = len(px)
        
        # Base case: use brute force for small arrays
        if n <= 3:
            return brute_force(px)
        
        # Divide: find middle point
        mid = n // 2
        midpoint = px[mid]
        
        pyl = [point for point in py if point[0] <= midpoint[0]]
        pyr = [point for point in py if point[0] > midpoint[0]]
        
        # Conquer: find minimum distances in both halves
        dl = closest_pair_rec(px[:mid], pyl)
        dr = closest_pair_rec(px[mid:], pyr)
        
        # Find minimum of the two
        d = min(dl, dr)
        
        # Combine: check points near the dividing line
        strip = [point for point in py if abs(point[0] - midpoint[0]) < d]
        
        # Find closest points in strip
        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and (strip[j][1] - strip[i][1]) < d:
                d = min(d, distance(strip[i], strip[j]))
                j += 1
        
        return d
    
    # Sort points by x and y coordinates
    px = sorted(points, key=lambda p: p[0])
    py = sorted(points, key=lambda p: p[1])
    
    return closest_pair_rec(px, py)

# Test
points = [[0, 0], [3, 4], [1, 1], [2, 2]]
print(f"Closest distance: {closest_pair_distance(points):.5f}")
```

**Key Points**:
- Divide points by x-coordinate
- Recursively find closest pairs in left and right halves
- Check strip near dividing line for cross-boundary closest pairs
- Sort by y-coordinate to optimize strip searching

---

## 3. Majority Element II

**Problem**: Find all elements that appear more than ⌊n/3⌋ times in the array.

**Example**:
```
Input: nums = [3,2,3]
Output: [3]
```

**Solution**:
```python
def majority_element_ii(nums):
    """
    Find majority elements using divide and conquer.
    
    Time Complexity: O(n log n)
    Space Complexity: O(log n)
    """
    def majority_helper(left, right):
        # Base case: single element
        if left == right:
            return [nums[left]]
        
        # Divide
        mid = left + (right - left) // 2
        left_majority = majority_helper(left, mid)
        right_majority = majority_helper(mid + 1, right)
        
        # Combine: merge candidates and validate
        candidates = list(set(left_majority + right_majority))
        result = []
        
        # Count occurrences in current range
        for candidate in candidates:
            count = 0
            for i in range(left, right + 1):
                if nums[i] == candidate:
                    count += 1
            
            # Check if it's a majority in current range
            if count > (right - left + 1) // 3:
                result.append(candidate)
        
        return result
    
    # Get candidates from divide and conquer
    candidates = majority_helper(0, len(nums) - 1)
    
    # Final validation: check if candidates appear > n/3 times globally
    result = []
    n = len(nums)
    
    for candidate in set(candidates):
        count = nums.count(candidate)
        if count > n // 3:
            result.append(candidate)
    
    return result

# More efficient Boyer-Moore approach for comparison
def majority_element_ii_boyer_moore(nums):
    """
    Boyer-Moore majority vote algorithm (more efficient).
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not nums:
        return []
    
    # Phase 1: Find candidates
    candidate1 = candidate2 = None
    count1 = count2 = 0
    
    for num in nums:
        if candidate1 is not None and num == candidate1:
            count1 += 1
        elif candidate2 is not None and num == candidate2:
            count2 += 1
        elif count1 == 0:
            candidate1, count1 = num, 1
        elif count2 == 0:
            candidate2, count2 = num, 1
        else:
            count1 -= 1
            count2 -= 1
    
    # Phase 2: Validate candidates
    result = []
    n = len(nums)
    
    for candidate in [candidate1, candidate2]:
        if candidate is not None and nums.count(candidate) > n // 3:
            result.append(candidate)
    
    return result
```

**Key Points**:
- At most 2 elements can appear more than n/3 times
- Divide and conquer finds local majorities, then validates globally
- Boyer-Moore algorithm is more efficient for this specific problem

---

## 4. Search in Rotated Sorted Array

**Problem**: Search for a target value in a rotated sorted array.

**Example**:
```
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
```

**Solution**:
```python
def search_rotated_array(nums, target):
    """
    Search in rotated sorted array using modified binary search.
    
    Time Complexity: O(log n)
    Space Complexity: O(log n) - recursion stack
    """
    def search_helper(left, right):
        if left > right:
            return -1
        
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Determine which half is sorted
        if nums[left] <= nums[mid]:  # Left half is sorted
            if nums[left] <= target < nums[mid]:
                return search_helper(left, mid - 1)
            else:
                return search_helper(mid + 1, right)
        else:  # Right half is sorted
            if nums[mid] < target <= nums[right]:
                return search_helper(mid + 1, right)
            else:
                return search_helper(left, mid - 1)
    
    return search_helper(0, len(nums) - 1)

# Iterative version
def search_rotated_array_iterative(nums, target):
    """
    Iterative approach to avoid recursion overhead.
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

# Handle array with duplicates
def search_rotated_array_duplicates(nums, target):
    """
    Handle rotated array with duplicates.
    
    Time Complexity: O(n) worst case, O(log n) average
    Space Complexity: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return True
        
        # Handle duplicates
        if nums[left] == nums[mid] == nums[right]:
            left += 1
            right -= 1
        elif nums[left] <= nums[mid]:  # Left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return False
```

**Key Points**:
- One half of the array is always sorted
- Compare target with sorted half boundaries to decide search direction
- Handle duplicates by shrinking search space when pivot equals boundaries

---

## 5. Maximum Subarray Product

**Problem**: Find the contiguous subarray with the largest product.

**Example**:
```
Input: nums = [2,3,-2,4]
Output: 6 (subarray [2,3])
```

**Solution**:
```python
def max_product_subarray_divide_conquer(nums):
    """
    Find maximum product subarray using divide and conquer.
    
    Time Complexity: O(n log n)
    Space Complexity: O(log n)
    """
    def max_product_helper(left, right):
        if left == right:
            return nums[left]
        
        mid = left + (right - left) // 2
        
        # Get max products from left and right halves
        left_max = max_product_helper(left, mid)
        right_max = max_product_helper(mid + 1, right)
        
        # Find max product crossing the middle
        cross_max = max_crossing_product(left, mid, right)
        
        return max(left_max, right_max, cross_max)
    
    def max_crossing_product(left, mid, right):
        # Find maximum product ending at mid (going left)
        left_product = nums[mid]
        max_left = left_product
        
        for i in range(mid - 1, left - 1, -1):
            left_product *= nums[i]
            max_left = max(max_left, left_product)
        
        # Find maximum product starting at mid+1 (going right)
        right_product = nums[mid + 1]
        max_right = right_product
        
        for i in range(mid + 2, right + 1):
            right_product *= nums[i]
            max_right = max(max_right, right_product)
        
        # Return maximum of crossing products
        return max(max_left * max_right, max_left, max_right)
    
    return max_product_helper(0, len(nums) - 1)

# More efficient dynamic programming approach
def max_product_subarray_dp(nums):
    """
    Dynamic programming approach (more efficient).
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not nums:
        return 0
    
    max_product = min_product = result = nums[0]
    
    for i in range(1, len(nums)):
        num = nums[i]
        
        # Store current max before updating
        temp_max = max_product
        
        # Update max and min products
        max_product = max(num, max_product * num, min_product * num)
        min_product = min(num, temp_max * num, min_product * num)
        
        # Update global result
        result = max(result, max_product)
    
    return result
```

**Key Points**:
- Handle negative numbers by tracking both maximum and minimum products
- Crossing case considers products extending left and right from middle
- DP approach is more efficient for this problem

---

## 6. Merge k Sorted Lists

**Problem**: Merge k sorted linked lists and return it as one sorted list.

**Solution**:
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_k_lists(lists):
    """
    Merge k sorted lists using divide and conquer.
    
    Time Complexity: O(n log k) where n is total nodes, k is number of lists  
    Space Complexity: O(log k) - recursion stack
    """
    if not lists:
        return None
    
    def merge_two_lists(l1, l2):
        dummy = ListNode(0)
        current = dummy
        
        while l1 and l2:
            if l1.val <= l2.val:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next
            current = current.next
        
        current.next = l1 or l2
        return dummy.next
    
    def merge_helper(lists, start, end):
        if start == end:
            return lists[start]
        
        if start + 1 == end:
            return merge_two_lists(lists[start], lists[end])
        
        mid = start + (end - start) // 2
        left = merge_helper(lists, start, mid)
        right = merge_helper(lists, mid + 1, end)
        
        return merge_two_lists(left, right)
    
    return merge_helper(lists, 0, len(lists) - 1)

# Alternative using min-heap
import heapq

def merge_k_lists_heap(lists):
    """
    Using min-heap approach.
    
    Time Complexity: O(n log k)
    Space Complexity: O(k)
    """
    heap = []
    
    # Initialize heap with first node from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    
    return dummy.next
```

**Key Points**:
- Divide lists into pairs and merge recursively
- Each merge operation takes O(n) time where n is total nodes in two lists
- Total time complexity is O(n log k) due to log k levels of merging

These medium problems demonstrate more sophisticated applications of divide and conquer, involving advanced data structures, optimization techniques, and complex combining strategies.
