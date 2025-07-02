# Divide and Conquer - Easy Problems

## Problem Categories

### 1. Search Problems
- Binary search variations
- Finding elements in arrays
- Peak finding

### 2. Mathematical Operations
- Power calculation
- Factorial computation
- GCD calculation

### 3. Array Processing
- Maximum/minimum finding
- Sum calculations
- Counting elements

---

## 1. Binary Search

**Problem**: Given a sorted array and a target value, return the index of the target or -1 if not found.

**Example**:
```
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
```

**Solution**:
```python
def binary_search(nums, target):
    """
    Classic binary search using divide and conquer.
    
    Time Complexity: O(log n)
    Space Complexity: O(log n) - recursion stack
    """
    def search_helper(left, right):
        # Base case: element not found
        if left > right:
            return -1
        
        # Divide: find middle point
        mid = left + (right - left) // 2
        
        # Base case: found target
        if nums[mid] == target:
            return mid
        
        # Conquer: search in appropriate half
        if nums[mid] > target:
            return search_helper(left, mid - 1)  # Search left half
        else:
            return search_helper(mid + 1, right)  # Search right half
    
    return search_helper(0, len(nums) - 1)

# Iterative version (space-optimized)
def binary_search_iterative(nums, target):
    """
    Iterative binary search to avoid recursion overhead.
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    
    return -1

# Test
print(binary_search([-1, 0, 3, 5, 9, 12], 9))  # Output: 4
print(binary_search([-1, 0, 3, 5, 9, 12], 2))  # Output: -1
```

**Key Points**:
- Always use `left + (right - left) // 2` to avoid overflow
- Base case handles when element is not found
- Divide problem by eliminating half of search space

---

## 2. Find Peak Element

**Problem**: A peak element is an element that is strictly greater than its neighbors. Find any peak element and return its index.

**Example**:
```
Input: nums = [1,2,3,1]
Output: 2 (element 3 is a peak)
```

**Solution**:
```python
def find_peak_element(nums):
    """
    Find peak element using divide and conquer.
    
    Time Complexity: O(log n)
    Space Complexity: O(log n) - recursion stack
    """
    def find_peak_helper(left, right):
        # Base case: single element
        if left == right:
            return left
        
        # Divide: find middle point
        mid = left + (right - left) // 2
        
        # Check if mid is peak
        if nums[mid] > nums[mid + 1]:
            # Peak is in left half (including mid)
            return find_peak_helper(left, mid)
        else:
            # Peak is in right half
            return find_peak_helper(mid + 1, right)
    
    return find_peak_helper(0, len(nums) - 1)

# Alternative iterative approach
def find_peak_element_iterative(nums):
    """
    Iterative approach to find peak element.
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[mid + 1]:
            right = mid  # Peak is in left half (including mid)
        else:
            left = mid + 1  # Peak is in right half
    
    return left

# Test
print(find_peak_element([1, 2, 3, 1]))     # Output: 2
print(find_peak_element([1, 2, 1, 3, 5, 6, 4]))  # Output: 1 or 5
```

**Key Points**:
- Compare middle element with its neighbor to decide direction
- Always guaranteed to find a peak due to problem constraints
- No need to check both neighbors in binary search approach

---

## 3. Power Function (Fast Exponentiation)

**Problem**: Implement `pow(x, n)` which calculates `x` raised to the power `n`.

**Example**:
```
Input: x = 2.0, n = 10
Output: 1024.0
```

**Solution**:
```python
def my_pow(x, n):
    """
    Calculate x^n using fast exponentiation (divide and conquer).
    
    Time Complexity: O(log n)
    Space Complexity: O(log n) - recursion stack
    """
    def power_helper(base, exp):
        # Base case: any number to power 0 is 1
        if exp == 0:
            return 1
        
        # Base case: any number to power 1 is itself
        if exp == 1:
            return base
        
        # Divide: calculate power for half the exponent
        half_power = power_helper(base, exp // 2)
        
        # Combine: square the half power
        if exp % 2 == 0:
            return half_power * half_power
        else:
            return half_power * half_power * base
    
    # Handle negative exponents
    if n < 0:
        return 1 / power_helper(x, -n)
    else:
        return power_helper(x, n)

# Iterative version
def my_pow_iterative(x, n):
    """
    Iterative fast exponentiation.
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    if n == 0:
        return 1
    
    # Handle negative exponent
    if n < 0:
        x = 1 / x
        n = -n
    
    result = 1
    current_power = x
    
    while n > 0:
        # If n is odd, multiply result by current power
        if n % 2 == 1:
            result *= current_power
        
        # Square the current power and halve n
        current_power *= current_power
        n //= 2
    
    return result

# Test
print(my_pow(2.0, 10))   # Output: 1024.0
print(my_pow(2.1, 3))    # Output: 9.261
print(my_pow(2.0, -2))   # Output: 0.25
```

**Key Points**:
- Use the property: `x^n = (x^(n/2))^2` for even n
- For odd n: `x^n = x * (x^(n/2))^2`
- Handle negative exponents by inverting base and making exponent positive

---

## 4. Maximum Subarray (Divide and Conquer Approach)

**Problem**: Given an integer array, find the contiguous subarray with the largest sum.

**Example**:
```
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6 (subarray [4,-1,2,1])
```

**Solution**:
```python
def max_subarray_divide_conquer(nums):
    """
    Find maximum subarray sum using divide and conquer.
    
    Time Complexity: O(n log n)
    Space Complexity: O(log n) - recursion stack
    """
    def max_subarray_helper(left, right):
        # Base case: single element
        if left == right:
            return nums[left]
        
        # Divide: find middle point
        mid = left + (right - left) // 2
        
        # Conquer: find max sum in left and right halves
        left_max = max_subarray_helper(left, mid)
        right_max = max_subarray_helper(mid + 1, right)
        
        # Combine: find max sum crossing the middle
        cross_max = max_crossing_sum(nums, left, mid, right)
        
        # Return maximum of the three
        return max(left_max, right_max, cross_max)
    
    def max_crossing_sum(arr, left, mid, right):
        """Find maximum sum of subarray crossing the middle point."""
        # Find max sum for left side (including mid)
        left_sum = float('-inf')
        total = 0
        for i in range(mid, left - 1, -1):
            total += arr[i]
            left_sum = max(left_sum, total)
        
        # Find max sum for right side (excluding mid)
        right_sum = float('-inf')
        total = 0
        for i in range(mid + 1, right + 1):
            total += arr[i]
            right_sum = max(right_sum, total)
        
        return left_sum + right_sum
    
    return max_subarray_helper(0, len(nums) - 1)

# Note: Kadane's algorithm is O(n) and simpler for this problem
def max_subarray_kadane(nums):
    """
    Kadane's algorithm for comparison (more efficient).
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    max_sum = current_sum = nums[0]
    
    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Test
print(max_subarray_divide_conquer([-2, 1, -3, 4, -1, 2, 1, -5, 4]))  # Output: 6
```

**Key Points**:
- Consider three cases: max in left, max in right, max crossing middle
- Crossing case requires finding best left and right extensions from middle
- This demonstrates divide and conquer but Kadane's algorithm is more efficient

---

## 5. Merge Two Sorted Arrays

**Problem**: Given two sorted arrays, merge them into a single sorted array.

**Example**:
```
Input: nums1 = [1,2,3], nums2 = [2,5,6]
Output: [1,2,2,3,5,6]
```

**Solution**:
```python
def merge_sorted_arrays(nums1, nums2):
    """
    Merge two sorted arrays using divide and conquer principle.
    
    Time Complexity: O(m + n)
    Space Complexity: O(m + n)
    """
    result = []
    i = j = 0
    
    # Merge elements while both arrays have elements
    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            result.append(nums1[i])
            i += 1
        else:
            result.append(nums2[j])
            j += 1
    
    # Add remaining elements from nums1
    while i < len(nums1):
        result.append(nums1[i])
        i += 1
    
    # Add remaining elements from nums2
    while j < len(nums2):
        result.append(nums2[j])
        j += 1
    
    return result

# In-place merge for merge sort
def merge_in_place(arr, left, mid, right):
    """
    Merge two sorted subarrays arr[left:mid+1] and arr[mid+1:right+1].
    """
    # Create temporary arrays
    left_arr = arr[left:mid+1]
    right_arr = arr[mid+1:right+1]
    
    i = j = 0
    k = left
    
    # Merge back into original array
    while i < len(left_arr) and j < len(right_arr):
        if left_arr[i] <= right_arr[j]:
            arr[k] = left_arr[i]
            i += 1
        else:
            arr[k] = right_arr[j]
            j += 1
        k += 1
    
    # Copy remaining elements
    while i < len(left_arr):
        arr[k] = left_arr[i]
        i += 1
        k += 1
    
    while j < len(right_arr):
        arr[k] = right_arr[j]
        j += 1
        k += 1

# Test
print(merge_sorted_arrays([1, 2, 3], [2, 5, 6]))  # Output: [1, 2, 2, 3, 5, 6]
```

**Key Points**:
- Core operation of merge sort algorithm
- Use two pointers to traverse both arrays simultaneously
- Handle remaining elements after one array is exhausted

---

## 6. Count Inversions

**Problem**: Count the number of inversions in an array. An inversion is when `i < j` but `arr[i] > arr[j]`.

**Example**:
```
Input: arr = [2, 3, 8, 6, 1]
Output: 5 (inversions: (2,1), (3,1), (8,6), (8,1), (6,1))
```

**Solution**:
```python
def count_inversions(arr):
    """
    Count inversions using divide and conquer (modified merge sort).
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    def merge_and_count(arr, temp, left, mid, right):
        i, j, k = left, mid + 1, left
        inv_count = 0
        
        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp[k] = arr[i]
                i += 1
            else:
                temp[k] = arr[j]
                # All elements from i to mid are greater than arr[j]
                inv_count += (mid - i + 1)
                j += 1
            k += 1
        
        # Copy remaining elements
        while i <= mid:
            temp[k] = arr[i]
            i += 1
            k += 1
        
        while j <= right:
            temp[k] = arr[j]
            j += 1
            k += 1
        
        # Copy back to original array
        for i in range(left, right + 1):
            arr[i] = temp[i]
        
        return inv_count
    
    def merge_sort_and_count(arr, temp, left, right):
        inv_count = 0
        if left < right:
            mid = left + (right - left) // 2
            
            inv_count += merge_sort_and_count(arr, temp, left, mid)
            inv_count += merge_sort_and_count(arr, temp, mid + 1, right)
            inv_count += merge_and_count(arr, temp, left, mid, right)
        
        return inv_count
    
    temp = [0] * len(arr)
    arr_copy = arr.copy()  # Don't modify original array
    return merge_sort_and_count(arr_copy, temp, 0, len(arr) - 1)

# Brute force approach for comparison
def count_inversions_brute_force(arr):
    """
    Brute force approach (less efficient).
    
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    count = 0
    n = len(arr)
    
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                count += 1
    
    return count

# Test
print(count_inversions([2, 3, 8, 6, 1]))  # Output: 5
```

**Key Points**:
- Modify merge sort to count inversions during merge step
- When element from right array is smaller, count inversions
- Divide and conquer reduces time complexity from O(n²) to O(n log n)

---

## Common Patterns in Easy D&C Problems

### 1. Binary Elimination Pattern
```python
def binary_elimination_template(arr, target):
    def helper(left, right):
        if left > right:
            return -1  # Not found
        
        mid = left + (right - left) // 2
        
        if condition_met(arr[mid], target):
            return mid
        elif should_go_left(arr[mid], target):
            return helper(left, mid - 1)
        else:
            return helper(mid + 1, right)
    
    return helper(0, len(arr) - 1)
```

### 2. Mathematical Divide Pattern
```python
def mathematical_divide_template(x, n):
    if n == 0:
        return 1
    if n == 1:
        return x
    
    half = mathematical_divide_template(x, n // 2)
    
    if n % 2 == 0:
        return half * half
    else:
        return half * half * x
```

### 3. Array Processing Pattern
```python
def array_processing_template(arr, left, right):
    if left == right:
        return arr[left]  # Base case
    
    mid = left + (right - left) // 2
    
    left_result = array_processing_template(arr, left, mid)
    right_result = array_processing_template(arr, mid + 1, right)
    
    return combine_results(left_result, right_result)
```

## Practice Tips

### 1. Identify the Pattern
- Can the problem be split into smaller similar problems?
- Is there a natural midpoint or division strategy?
- Can solutions be combined efficiently?

### 2. Handle Base Cases
- What's the smallest problem size?
- What should be returned for trivial cases?
- Are there edge cases to consider?

### 3. Optimize Recursion
- Consider iterative alternatives for simple cases
- Use tail recursion when possible
- Be mindful of stack overflow for large inputs

### 4. Time Complexity Analysis
- Apply Master Theorem when applicable
- Count recursive calls and work per call
- Consider space complexity of recursion stack

These easy problems provide a solid foundation for understanding divide and conquer principles. Master these before moving to more complex algorithms!
