# Search Algorithms 🔍

## 🎯 Overview

Search algorithms are fundamental techniques for finding specific elements or solving optimization problems. This section covers linear search, binary search, and advanced searching techniques used in competitive programming and real-world applications.

## 📋 Categories of Search

### Basic Search
- **Linear Search**: Sequential scanning
- **Binary Search**: Divide and conquer on sorted data
- **Jump Search**: Block-based searching
- **Interpolation Search**: Position estimation

### Advanced Search
- **Exponential Search**: Unbounded binary search
- **Ternary Search**: Optimization problems
- **Fibonacci Search**: Golden ratio based
- **Subarray Search**: Pattern matching

### Specialized Search
- **Search in Rotated Arrays**: Modified binary search
- **Search in 2D Matrices**: Multi-dimensional binary search
- **Search in Infinite Arrays**: Unbounded search
- **Peak Finding**: Local maxima/minima

## 🔧 Basic Search Algorithms

### Linear Search

```python
def linear_search(arr, target):
    """
    Search for target in unsorted array
    Time: O(n), Space: O(1)
    """
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

def linear_search_all_occurrences(arr, target):
    """
    Find all occurrences of target
    Time: O(n), Space: O(k) where k = number of occurrences
    """
    indices = []
    for i, val in enumerate(arr):
        if val == target:
            indices.append(i)
    return indices

def linear_search_with_sentinel(arr, target):
    """
    Linear search with sentinel optimization
    Reduces comparisons by half
    """
    n = len(arr)
    if n == 0:
        return -1
    
    # Store last element and put target at end
    last = arr[n - 1]
    arr[n - 1] = target
    
    i = 0
    while arr[i] != target:
        i += 1
    
    # Restore last element
    arr[n - 1] = last
    
    # Check if target was found or it was the sentinel
    if i < n - 1 or arr[n - 1] == target:
        return i
    return -1

# Test
arr = [64, 34, 25, 12, 22, 11, 90]
print(linear_search(arr, 22))  # 4
```

### Binary Search

```python
def binary_search(arr, target):
    """
    Binary search in sorted array
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def binary_search_recursive(arr, target, left=0, right=None):
    """
    Recursive binary search
    Time: O(log n), Space: O(log n)
    """
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = left + (right - left) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

def binary_search_leftmost(arr, target):
    """
    Find leftmost occurrence of target
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left if left < len(arr) and arr[left] == target else -1

def binary_search_rightmost(arr, target):
    """
    Find rightmost occurrence of target
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    
    return left - 1 if left > 0 and arr[left - 1] == target else -1

# Test
arr = [1, 2, 3, 4, 4, 4, 5, 6, 7]
print(binary_search(arr, 4))           # 3, 4, or 5 (any occurrence)
print(binary_search_leftmost(arr, 4))  # 3
print(binary_search_rightmost(arr, 4)) # 5
```

### Advanced Search Variants

```python
import math

def jump_search(arr, target):
    """
    Jump search algorithm
    Time: O(√n), Space: O(1)
    """
    n = len(arr)
    step = int(math.sqrt(n))
    prev = 0
    
    # Jump through blocks
    while arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1
    
    # Linear search in the block
    while arr[prev] < target:
        prev += 1
        if prev == min(step, n):
            return -1
    
    return prev if arr[prev] == target else -1

def interpolation_search(arr, target):
    """
    Interpolation search for uniformly distributed data
    Time: O(log log n) average, O(n) worst, Space: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right and arr[left] <= target <= arr[right]:
        # If uniform distribution
        if arr[left] == arr[right]:
            return left if arr[left] == target else -1
        
        # Estimate position
        pos = left + ((target - arr[left]) * (right - left)) // (arr[right] - arr[left])
        
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            left = pos + 1
        else:
            right = pos - 1
    
    return -1

def exponential_search(arr, target):
    """
    Exponential search for unbounded arrays
    Time: O(log n), Space: O(1)
    """
    if arr[0] == target:
        return 0
    
    # Find range for binary search
    bound = 1
    while bound < len(arr) and arr[bound] < target:
        bound *= 2
    
    # Binary search in found range
    left = bound // 2
    right = min(bound, len(arr) - 1)
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Test
arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
print(jump_search(arr, 11))         # 5
print(interpolation_search(arr, 11)) # 5
print(exponential_search(arr, 11))   # 5
```

## 🎨 Search Problems & Patterns

### Pattern 1: Search in Rotated Arrays

```python
def search_rotated_array(nums, target):
    """
    Search in rotated sorted array
    Time: O(log n), Space: O(1)
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

def search_rotated_array_with_duplicates(nums, target):
    """
    Search in rotated sorted array with duplicates
    Time: O(log n) average, O(n) worst case, Space: O(1)
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
        elif nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return False

def find_minimum_rotated_array(nums):
    """
    Find minimum in rotated sorted array
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    
    return nums[left]

# Test
nums = [4, 5, 6, 7, 0, 1, 2]
print(search_rotated_array(nums, 0))      # 4
print(find_minimum_rotated_array(nums))   # 0
```

### Pattern 2: Search in 2D Matrices

```python
def search_matrix(matrix, target):
    """
    Search in row-wise and column-wise sorted matrix
    Time: O(m + n), Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False
    
    rows, cols = len(matrix), len(matrix[0])
    row, col = 0, cols - 1
    
    while row < rows and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1
        else:
            row += 1
    
    return False

def search_matrix_binary(matrix, target):
    """
    Search in matrix where each row is sorted and 
    first element of each row > last element of previous row
    Time: O(log(m*n)), Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False
    
    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        mid_val = matrix[mid // cols][mid % cols]
        
        if mid_val == target:
            return True
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False

def search_matrix_range(matrix, target):
    """
    Count occurrences of target in sorted matrix
    Time: O(m + n), Space: O(1)
    """
    if not matrix or not matrix[0]:
        return 0
    
    rows, cols = len(matrix), len(matrix[0])
    row, col = 0, cols - 1
    count = 0
    
    while row < rows and col >= 0:
        if matrix[row][col] == target:
            count += 1
            # Move both row and col to count all occurrences
            row += 1
            col -= 1
        elif matrix[row][col] > target:
            col -= 1
        else:
            row += 1
    
    return count

# Test
matrix = [
    [1,  3,  5,  7],
    [10, 11, 16, 20],
    [23, 30, 34, 60]
]
print(search_matrix_binary(matrix, 3))  # True
```

### Pattern 3: Peak Finding

```python
def find_peak_element(nums):
    """
    Find a peak element (element greater than neighbors)
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    
    return left

def find_peak_2d(matrix):
    """
    Find peak in 2D matrix
    Time: O(m log n), Space: O(1)
    """
    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, cols - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        # Find max in current column
        max_row = 0
        for i in range(rows):
            if matrix[i][mid] > matrix[max_row][mid]:
                max_row = i
        
        # Check if it's a peak
        left_val = matrix[max_row][mid - 1] if mid > 0 else -1
        right_val = matrix[max_row][mid + 1] if mid < cols - 1 else -1
        
        if matrix[max_row][mid] > left_val and matrix[max_row][mid] > right_val:
            return (max_row, mid)
        elif mid > 0 and matrix[max_row][mid - 1] > matrix[max_row][mid]:
            right = mid - 1
        else:
            left = mid + 1
    
    return (-1, -1)

def find_local_minima(arr):
    """
    Find all local minima in array
    Time: O(n), Space: O(k) where k = number of minima
    """
    n = len(arr)
    minima = []
    
    # Check first element
    if n > 0 and (n == 1 or arr[0] < arr[1]):
        minima.append(0)
    
    # Check middle elements
    for i in range(1, n - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            minima.append(i)
    
    # Check last element
    if n > 1 and arr[n - 1] < arr[n - 2]:
        minima.append(n - 1)
    
    return minima

# Test
nums = [1, 2, 3, 1]
print(find_peak_element(nums))  # 2 (element 3 is peak)
```

### Pattern 4: Ternary Search

```python
def ternary_search(arr, target):
    """
    Ternary search in sorted array
    Time: O(log₃ n), Space: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        # Divide into three parts
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3
        
        if arr[mid1] == target:
            return mid1
        if arr[mid2] == target:
            return mid2
        
        if target < arr[mid1]:
            right = mid1 - 1
        elif target > arr[mid2]:
            left = mid2 + 1
        else:
            left = mid1 + 1
            right = mid2 - 1
    
    return -1

def ternary_search_max(func, left, right, precision=1e-9):
    """
    Find maximum of unimodal function using ternary search
    Time: O(log n), Space: O(1)
    """
    while right - left > precision:
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        
        if func(mid1) < func(mid2):
            left = mid1
        else:
            right = mid2
    
    return (left + right) / 2

def minimize_max_distance(stations, k):
    """
    Minimize maximum distance between gas stations
    Time: O(n log(max_distance)), Space: O(1)
    """
    def possible(max_dist):
        count = 0
        for i in range(len(stations) - 1):
            count += int((stations[i + 1] - stations[i]) / max_dist)
        return count <= k
    
    left, right = 0, stations[-1] - stations[0]
    
    while right - left > 1e-6:
        mid = (left + right) / 2
        if possible(mid):
            right = mid
        else:
            left = mid
    
    return right

# Test function for ternary search
def quadratic(x):
    return -(x - 3) ** 2 + 10  # Maximum at x = 3

print(ternary_search_max(quadratic, 0, 6))  # Should be close to 3
```

## 🚀 Advanced Search Techniques

### Pattern Matching Algorithms

```python
def naive_string_search(text, pattern):
    """
    Naive string matching algorithm
    Time: O(nm), Space: O(1)
    """
    n, m = len(text), len(pattern)
    matches = []
    
    for i in range(n - m + 1):
        j = 0
        while j < m and text[i + j] == pattern[j]:
            j += 1
        if j == m:
            matches.append(i)
    
    return matches

def kmp_search(text, pattern):
    """
    Knuth-Morris-Pratt string matching
    Time: O(n + m), Space: O(m)
    """
    def compute_lps(pattern):
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1
        
        while i < m:
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
    
    n, m = len(text), len(pattern)
    lps = compute_lps(pattern)
    matches = []
    
    i = j = 0
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches

def rabin_karp_search(text, pattern, prime=101):
    """
    Rabin-Karp string matching using rolling hash
    Time: O(nm) worst case, O(n + m) average, Space: O(1)
    """
    n, m = len(text), len(pattern)
    d = 256  # Number of characters in alphabet
    matches = []
    
    # Calculate hash of pattern and first window
    pattern_hash = text_hash = 0
    h = 1
    
    for i in range(m - 1):
        h = (h * d) % prime
    
    for i in range(m):
        pattern_hash = (d * pattern_hash + ord(pattern[i])) % prime
        text_hash = (d * text_hash + ord(text[i])) % prime
    
    # Slide the pattern over text
    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            # Check characters one by one
            if text[i:i + m] == pattern:
                matches.append(i)
        
        # Calculate hash for next window
        if i < n - m:
            text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
            if text_hash < 0:
                text_hash += prime
    
    return matches

# Test
text = "ABABDABACDABABCABCABCABCABC"
pattern = "ABABCAB"
print(naive_string_search(text, pattern))  # [15]
print(kmp_search(text, pattern))           # [15]
print(rabin_karp_search(text, pattern))    # [15]
```

### Binary Search Applications

```python
def search_range(nums, target):
    """
    Find first and last position of target
    Time: O(log n), Space: O(1)
    """
    def find_first():
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] >= target:
                right = mid - 1
            else:
                left = mid + 1
        return left
    
    def find_last():
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        return right
    
    first = find_first()
    if first >= len(nums) or nums[first] != target:
        return [-1, -1]
    
    last = find_last()
    return [first, last]

def sqrt_binary_search(x):
    """
    Calculate square root using binary search
    Time: O(log x), Space: O(1)
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

def search_insert_position(nums, target):
    """
    Find position where target should be inserted
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums)
    
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left

# Test
nums = [5, 7, 7, 8, 8, 10]
print(search_range(nums, 8))        # [3, 4]
print(sqrt_binary_search(8))        # 2
print(search_insert_position(nums, 6))  # 1
```

## 📊 Complexity Analysis

| **Algorithm** | **Time Complexity** | **Space Complexity** | **Best Use Case** |
|---------------|-------------------|--------------------|--------------------|
| **Linear Search** | O(n) | O(1) | Unsorted data, small arrays |
| **Binary Search** | O(log n) | O(1) | Sorted arrays |
| **Jump Search** | O(√n) | O(1) | Large sorted arrays |
| **Interpolation Search** | O(log log n) avg | O(1) | Uniformly distributed data |
| **Exponential Search** | O(log n) | O(1) | Unbounded/infinite arrays |
| **Ternary Search** | O(log₃ n) | O(1) | Unimodal functions |

## 🏆 Practice Problems

### Easy
- [Binary Search](https://leetcode.com/problems/binary-search/)
- [Search Insert Position](https://leetcode.com/problems/search-insert-position/)
- [First Bad Version](https://leetcode.com/problems/first-bad-version/)
- [Sqrt(x)](https://leetcode.com/problems/sqrtx/)

### Medium
- [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
- [Find First and Last Position](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
- [Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)
- [Find Peak Element](https://leetcode.com/problems/find-peak-element/)
- [Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)

### Hard
- [Search in Rotated Sorted Array II](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/)
- [Find Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/)
- [Kth Smallest Element in Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)
- [Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/)

## 🎯 Key Search Patterns

### 1. **Template I - Exact Match**
```python
while left <= right:
    mid = left + (right - left) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        left = mid + 1
    else:
        right = mid - 1
```

### 2. **Template II - Left Boundary**
```python
while left < right:
    mid = left + (right - left) // 2
    if condition(mid):
        right = mid
    else:
        left = mid + 1
```

### 3. **Template III - Right Boundary**
```python
while left < right:
    mid = left + (right - left + 1) // 2
    if condition(mid):
        left = mid
    else:
        right = mid - 1
```

## 📚 Key Takeaways

1. **Choose the right search algorithm** based on data characteristics
2. **Binary search variants** solve many optimization problems
3. **Avoid integer overflow** using `left + (right - left) // 2`
4. **Template-based approach** helps handle boundary conditions
5. **Practice edge cases** - empty arrays, single elements, duplicates
