# Search Problems

A comprehensive collection of searching problems ranging from basic binary search applications to advanced optimization problems.

## Easy Problems

### 1. Binary Search Implementation

**Problem**: Implement binary search to find a target value in a sorted array.

```python
def search(nums, target):
    """
    LeetCode 704: Binary Search
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

### 2. First Bad Version

**Problem**: Find the first bad version using minimum API calls.

```python
def first_bad_version(n):
    """
    LeetCode 278: First Bad Version
    Binary search on version numbers
    """
    left, right = 1, n
    
    while left < right:
        mid = left + (right - left) // 2
        
        if isBadVersion(mid):
            right = mid
        else:
            left = mid + 1
    
    return left
```

### 3. Square Root

**Problem**: Find the integer square root of a number without using built-in functions.

```python
def my_sqrt(x):
    """
    LeetCode 69: Sqrt(x)
    Binary search on answer space
    """
    if x < 2:
        return x
    
    left, right = 2, x // 2
    
    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid
        
        if square == x:
            return mid
        elif square < x:
            left = mid + 1
        else:
            right = mid - 1
    
    return right
```

### 4. Search Insert Position

**Problem**: Find the index where target should be inserted in a sorted array.

```python
def search_insert(nums, target):
    """
    LeetCode 35: Search Insert Position
    Find insertion point in sorted array
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return left
```

### 5. Two Sum II - Input Array is Sorted

**Problem**: Find two numbers in sorted array that add up to target.

```python
def two_sum(numbers, target):
    """
    LeetCode 167: Two Sum II
    Two pointers on sorted array
    """
    left, right = 0, len(numbers) - 1
    
    while left < right:
        current_sum = numbers[left] + numbers[right]
        
        if current_sum == target:
            return [left + 1, right + 1]  # 1-indexed
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []
```

## Medium Problems

### 1. Search in Rotated Sorted Array

**Problem**: Search in a rotated sorted array where elements are distinct.

```python
def search_rotated(nums, target):
    """
    LeetCode 33: Search in Rotated Sorted Array
    Modified binary search handling rotation
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Check if left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1
```

### 2. Find First and Last Position

**Problem**: Find the starting and ending position of a target value in sorted array.

```python
def search_range(nums, target):
    """
    LeetCode 34: Find First and Last Position of Element
    Binary search for bounds
    """
    def find_first():
        left, right = 0, len(nums) - 1
        first_pos = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                first_pos = mid
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return first_pos
    
    def find_last():
        left, right = 0, len(nums) - 1
        last_pos = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                last_pos = mid
                left = mid + 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return last_pos
    
    first = find_first()
    if first == -1:
        return [-1, -1]
    
    last = find_last()
    return [first, last]
```

### 3. Search a 2D Matrix

**Problem**: Search for a value in a 2D matrix with sorted properties.

```python
def search_matrix(matrix, target):
    """
    LeetCode 74: Search a 2D Matrix
    Treat 2D matrix as 1D sorted array
    """
    if not matrix or not matrix[0]:
        return False
    
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        row, col = divmod(mid, n)
        
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False
```

### 4. Find Peak Element

**Problem**: Find a peak element in an array where peak is greater than its neighbors.

```python
def find_peak_element(nums):
    """
    LeetCode 162: Find Peak Element
    Binary search on unsorted array
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    
    return left
```

### 5. Find Minimum in Rotated Sorted Array

**Problem**: Find the minimum element in a rotated sorted array.

```python
def find_min(nums):
    """
    LeetCode 153: Find Minimum in Rotated Sorted Array
    Binary search to find rotation point
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    
    return nums[left]
```

### 6. Koko Eating Bananas

**Problem**: Find minimum eating speed to finish all bananas within given hours.

```python
def min_eating_speed(piles, h):
    """
    LeetCode 875: Koko Eating Bananas
    Binary search on answer space
    """
    def can_finish(speed):
        hours = 0
        for pile in piles:
            hours += (pile + speed - 1) // speed  # Ceiling division
        return hours <= h
    
    left, right = 1, max(piles)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_finish(mid):
            right = mid
        else:
            left = mid + 1
    
    return left
```

## Hard Problems

### 1. Median of Two Sorted Arrays

**Problem**: Find the median of two sorted arrays in O(log(min(m,n))) time.

```python
def find_median_sorted_arrays(nums1, nums2):
    """
    LeetCode 4: Median of Two Sorted Arrays
    Binary search on smaller array
    """
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    
    while left <= right:
        partition1 = (left + right) // 2
        partition2 = (m + n + 1) // 2 - partition1
        
        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float('inf') if partition1 == m else nums1[partition1]
        
        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]
        
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            if (m + n) % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            else:
                return max(max_left1, max_left2)
        elif max_left1 > min_right2:
            right = partition1 - 1
        else:
            left = partition1 + 1
    
    raise ValueError("Input arrays are not sorted")
```

### 2. Split Array Largest Sum

**Problem**: Split array into m subarrays to minimize the largest sum among subarrays.

```python
def split_array(nums, m):
    """
    LeetCode 410: Split Array Largest Sum
    Binary search on answer space
    """
    def can_split(max_sum):
        count = 1
        current_sum = 0
        
        for num in nums:
            if current_sum + num > max_sum:
                count += 1
                current_sum = num
            else:
                current_sum += num
        
        return count <= m
    
    left, right = max(nums), sum(nums)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_split(mid):
            right = mid
        else:
            left = mid + 1
    
    return left
```

### 3. Capacity to Ship Packages Within D Days

**Problem**: Find minimum ship capacity to ship all packages within D days.

```python
def ship_within_days(weights, days):
    """
    LeetCode 1011: Capacity to Ship Packages Within D Days
    Binary search on capacity
    """
    def can_ship(capacity):
        current_weight = 0
        days_needed = 1
        
        for weight in weights:
            if current_weight + weight > capacity:
                days_needed += 1
                current_weight = weight
            else:
                current_weight += weight
        
        return days_needed <= days
    
    left, right = max(weights), sum(weights)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_ship(mid):
            right = mid
        else:
            left = mid + 1
    
    return left
```

### 4. Find K-th Smallest Element in Sorted Matrix

**Problem**: Find the k-th smallest element in an n×n matrix where each row and column is sorted.

```python
def kth_smallest(matrix, k):
    """
    LeetCode 378: Kth Smallest Element in Sorted Matrix
    Binary search on value range
    """
    def count_less_equal(target):
        count = 0
        row, col = len(matrix) - 1, 0
        
        while row >= 0 and col < len(matrix[0]):
            if matrix[row][col] <= target:
                count += row + 1
                col += 1
            else:
                row -= 1
        
        return count
    
    left, right = matrix[0][0], matrix[-1][-1]
    
    while left < right:
        mid = left + (right - left) // 2
        
        if count_less_equal(mid) >= k:
            right = mid
        else:
            left = mid + 1
    
    return left
```

### 5. Minimum Number of Days to Make m Bouquets

**Problem**: Find minimum days to make m bouquets where each bouquet needs k adjacent flowers.

```python
def min_days(bloom_day, m, k):
    """
    LeetCode 1482: Minimum Number of Days to Make m Bouquets
    Binary search on days
    """
    if m * k > len(bloom_day):
        return -1
    
    def can_make_bouquets(days):
        bouquets = 0
        consecutive = 0
        
        for bloom in bloom_day:
            if bloom <= days:
                consecutive += 1
                if consecutive == k:
                    bouquets += 1
                    consecutive = 0
            else:
                consecutive = 0
        
        return bouquets >= m
    
    left, right = min(bloom_day), max(bloom_day)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_make_bouquets(mid):
            right = mid
        else:
            left = mid + 1
    
    return left
```

## Advanced String Search Problems

### 1. Implement strStr()

**Problem**: Find the index of the first occurrence of needle in haystack.

```python
def str_str(haystack, needle):
    """
    LeetCode 28: Implement strStr()
    Using KMP algorithm for efficient string matching
    """
    if not needle:
        return 0
    
    # Build LPS array
    def build_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1
        
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps
    
    lps = build_lps(needle)
    i = j = 0
    
    while i < len(haystack):
        if haystack[i] == needle[j]:
            i += 1
            j += 1
        
        if j == len(needle):
            return i - j
        elif i < len(haystack) and haystack[i] != needle[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return -1
```

## Problem-Solving Patterns

### 1. Binary Search Template Selection

- **Template 1**: Use when you can determine the target immediately
- **Template 2**: Use when you need access to the element's immediate right neighbor
- **Template 3**: Use when you need access to both neighbors

### 2. Binary Search on Answer Space

Common pattern for optimization problems:
1. Define the search space (min and max possible answers)
2. Write a feasibility function
3. Apply binary search to find the optimal answer

### 3. Two Pointers Patterns

- **Opposite Direction**: Start from both ends, move towards center
- **Same Direction**: Both pointers move forward, maintain window properties
- **Fast-Slow**: Different speeds for cycle detection or finding middle

## Practice Recommendations

1. **Start with Easy**: Master basic binary search implementation
2. **Understand Templates**: Learn when to use each binary search template
3. **Practice Bounds**: Get comfortable with first/last occurrence problems
4. **Master Answer Space**: Practice optimization problems with binary search
5. **String Algorithms**: Learn KMP and Rabin-Karp for string matching

These problems cover the essential patterns in searching algorithms and provide a solid foundation for tackling more complex challenges in competitive programming and technical interviews.
