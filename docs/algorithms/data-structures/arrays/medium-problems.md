# Arrays: Medium Problems

## ⚡ Intermediate Array Challenges

These problems require more sophisticated techniques and combining multiple patterns.

---

## Problem 1: Three Sum

**Difficulty:** Medium  
**Pattern:** Two Pointers + Sorting  
**Time:** O(n²) | **Space:** O(1)

### Problem Statement

Given an integer array `nums`, return all unique triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.

**Example:**
```text
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
```

### Solution

```python
def three_sum(nums):
    """
    Find all unique triplets that sum to zero.
    
    Use sorting + two pointers to avoid duplicates and reduce complexity.
    """
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 2):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        left, right = i + 1, n - 1
        target = -nums[i]
        
        while left < right:
            current_sum = nums[left] + nums[right]
            
            if current_sum == target:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates for second element
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                # Skip duplicates for third element
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1
    
    return result

# Test
nums = [-1, 0, 1, 2, -1, -4]
print(f"Three sum triplets: {three_sum(nums)}")
```

---

## Problem 2: Container With Most Water

**Difficulty:** Medium  
**Pattern:** Two Pointers  
**Time:** O(n) | **Space:** O(1)

### Problem Statement

Given an array of heights, find two lines that together with the x-axis form a container that holds the most water.

**Example:**
```text
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
```

### Solution

```python
def max_area(height):
    """
    Find container with most water using two pointers.
    
    Move the pointer with smaller height to potentially find larger area.
    """
    left, right = 0, len(height) - 1
    max_water = 0
    
    while left < right:
        # Calculate current water area
        width = right - left
        current_height = min(height[left], height[right])
        current_area = width * current_height
        max_water = max(max_water, current_area)
        
        # Move pointer with smaller height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water

# Alternative: Brute force approach (less efficient)
def max_area_brute_force(height):
    """O(n²) brute force solution."""
    max_water = 0
    n = len(height)
    
    for i in range(n):
        for j in range(i + 1, n):
            width = j - i
            current_height = min(height[i], height[j])
            current_area = width * current_height
            max_water = max(max_water, current_area)
    
    return max_water

# Test
height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
print(f"Max area (two pointers): {max_area(height)}")
print(f"Max area (brute force): {max_area_brute_force(height)}")
```

---

## Problem 3: Product of Array Except Self

**Difficulty:** Medium  
**Pattern:** Prefix/Suffix Product  
**Time:** O(n) | **Space:** O(1)

### Problem Statement

Given an array `nums`, return an array `answer` such that `answer[i]` is equal to the product of all elements except `nums[i]`. You cannot use division.

**Example:**
```text
Input: nums = [1,2,3,4]
Output: [24,12,8,6]
```

### Solution

```python
def product_except_self(nums):
    """
    Calculate product of all elements except self without division.
    
    Use left and right pass to calculate prefix and suffix products.
    """
    n = len(nums)
    result = [1] * n
    
    # Left pass: store prefix products
    for i in range(1, n):
        result[i] = result[i - 1] * nums[i - 1]
    
    # Right pass: multiply with suffix products
    right_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right_product
        right_product *= nums[i]
    
    return result

# Alternative: Using separate arrays (more space)
def product_except_self_verbose(nums):
    """Using separate left and right arrays for clarity."""
    n = len(nums)
    left = [1] * n
    right = [1] * n
    result = [1] * n
    
    # Build left products
    for i in range(1, n):
        left[i] = left[i - 1] * nums[i - 1]
    
    # Build right products
    for i in range(n - 2, -1, -1):
        right[i] = right[i + 1] * nums[i + 1]
    
    # Combine results
    for i in range(n):
        result[i] = left[i] * right[i]
    
    return result

# Test
nums = [1, 2, 3, 4]
print(f"Product except self (optimized): {product_except_self(nums)}")
print(f"Product except self (verbose): {product_except_self_verbose(nums)}")
```

---

## Problem 4: Find All Duplicates in Array

**Difficulty:** Medium  
**Pattern:** Array as Hash Map  
**Time:** O(n) | **Space:** O(1)

### Problem Statement

Given an array where elements are in range [1, n], find all elements that appear twice. Do this without extra space and in O(n) time.

**Example:**
```text
Input: nums = [4,3,2,7,8,2,3,1]
Output: [2,3]
```

### Solution

```python
def find_duplicates(nums):
    """
    Find duplicates using array indices as hash map.
    
    Mark visited elements by negating values at corresponding indices.
    """
    result = []
    
    for num in nums:
        index = abs(num) - 1  # Convert to 0-based index
        
        if nums[index] < 0:
            # Already visited, this is a duplicate
            result.append(abs(num))
        else:
            # Mark as visited by negating
            nums[index] = -nums[index]
    
    # Restore original array (optional)
    for i in range(len(nums)):
        nums[i] = abs(nums[i])
    
    return result

# Alternative: Using set (extra space)
def find_duplicates_set(nums):
    """Using set to track seen elements."""
    seen = set()
    duplicates = set()
    
    for num in nums:
        if num in seen:
            duplicates.add(num)
        else:
            seen.add(num)
    
    return list(duplicates)

# Alternative: Cycle detection approach
def find_duplicates_cycle(nums):
    """Using Floyd's cycle detection on array as linked list."""
    duplicates = []
    
    for i in range(len(nums)):
        index = abs(nums[i]) - 1
        
        if nums[index] > 0:
            nums[index] = -nums[index]
        else:
            duplicates.append(abs(nums[i]))
    
    # Restore array
    for i in range(len(nums)):
        nums[i] = abs(nums[i])
    
    return duplicates

# Test
nums = [4, 3, 2, 7, 8, 2, 3, 1]
nums_copy1 = nums.copy()
nums_copy2 = nums.copy()

print(f"Duplicates (index marking): {find_duplicates(nums_copy1)}")
print(f"Duplicates (set): {find_duplicates_set(nums_copy2)}")
```

---

## Problem 5: Spiral Matrix

**Difficulty:** Medium  
**Pattern:** Matrix Traversal  
**Time:** O(m×n) | **Space:** O(1)

### Problem Statement

Given an `m x n` matrix, return all elements in spiral order (clockwise from outside to inside).

**Example:**
```text
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]
```

### Solution

```python
def spiral_order(matrix):
    """
    Traverse matrix in spiral order using boundary tracking.
    
    Maintain four boundaries and move them inward after each direction.
    """
    if not matrix or not matrix[0]:
        return []
    
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        # Traverse right along top row
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1
        
        # Traverse down along right column
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1
        
        # Traverse left along bottom row (if still valid)
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1
        
        # Traverse up along left column (if still valid)
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1
    
    return result

# Alternative: Using direction vectors
def spiral_order_direction_vectors(matrix):
    """Using direction vectors for cleaner code."""
    if not matrix or not matrix[0]:
        return []
    
    rows, cols = len(matrix), len(matrix[0])
    visited = [[False] * cols for _ in range(rows)]
    result = []
    
    # Direction vectors: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    direction_idx = 0
    
    row = col = 0
    
    for _ in range(rows * cols):
        result.append(matrix[row][col])
        visited[row][col] = True
        
        # Calculate next position
        next_row = row + directions[direction_idx][0]
        next_col = col + directions[direction_idx][1]
        
        # Check if we need to turn
        if (next_row < 0 or next_row >= rows or 
            next_col < 0 or next_col >= cols or 
            visited[next_row][next_col]):
            direction_idx = (direction_idx + 1) % 4
            next_row = row + directions[direction_idx][0]
            next_col = col + directions[direction_idx][1]
        
        row, col = next_row, next_col
    
    return result

# Test
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

print(f"Spiral order (boundary): {spiral_order(matrix)}")
print(f"Spiral order (direction): {spiral_order_direction_vectors(matrix)}")
```

---

## Problem 6: Rotate Array

**Difficulty:** Medium  
**Pattern:** Array Manipulation  
**Time:** O(n) | **Space:** O(1)

### Problem Statement

Given an array, rotate it to the right by `k` steps, where `k` is non-negative.

**Example:**
```text
Input: nums = [1,2,3,4,5,6,7], k = 3
Output: [5,6,7,1,2,3,4]
```

### Solution

```python
def rotate(nums, k):
    """
    Rotate array right by k positions using reversal technique.
    
    1. Reverse entire array
    2. Reverse first k elements
    3. Reverse remaining elements
    """
    n = len(nums)
    k = k % n  # Handle k > n
    
    if k == 0:
        return
    
    # Helper function to reverse subarray
    def reverse(start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
    
    # Step 1: Reverse entire array
    reverse(0, n - 1)
    
    # Step 2: Reverse first k elements
    reverse(0, k - 1)
    
    # Step 3: Reverse remaining elements
    reverse(k, n - 1)

# Alternative: Using extra space
def rotate_extra_space(nums, k):
    """Rotate using extra array - O(n) space."""
    n = len(nums)
    k = k % n
    
    rotated = [0] * n
    
    for i in range(n):
        rotated[(i + k) % n] = nums[i]
    
    # Copy back to original array
    for i in range(n):
        nums[i] = rotated[i]

# Alternative: Cyclic replacement
def rotate_cyclic(nums, k):
    """Rotate using cyclic replacements."""
    n = len(nums)
    k = k % n
    count = 0
    
    start = 0
    while count < n:
        current = start
        prev = nums[start]
        
        while True:
            next_idx = (current + k) % n
            nums[next_idx], prev = prev, nums[next_idx]
            current = next_idx
            count += 1
            
            if start == current:
                break
        
        start += 1

# Test all approaches
nums1 = [1, 2, 3, 4, 5, 6, 7]
nums2 = nums1.copy()
nums3 = nums1.copy()
k = 3

print(f"Original: {nums1}")

rotate(nums1, k)
print(f"Rotated (reversal): {nums1}")

rotate_extra_space(nums2, k)
print(f"Rotated (extra space): {nums2}")

rotate_cyclic(nums3, k)
print(f"Rotated (cyclic): {nums3}")
```

---

## 🎯 Problem-Solving Patterns

### 1. Two Pointers with Sorting
- **Use when:** Need to find pairs/triplets with specific sum
- **Pattern:** Sort array, use two pointers to navigate
- **Examples:** Three sum, two sum II

### 2. Prefix/Suffix Arrays
- **Use when:** Need cumulative information from left/right
- **Pattern:** Build prefix array, then use it for queries
- **Examples:** Product except self, range sum queries

### 3. Array as Hash Map
- **Use when:** Array elements are in range [1, n]
- **Pattern:** Use indices to mark visited elements
- **Examples:** Find duplicates, missing numbers

### 4. Matrix Traversal
- **Use when:** Need to visit matrix elements in specific order
- **Pattern:** Use boundary tracking or direction vectors
- **Examples:** Spiral matrix, rotate matrix

## 💡 Advanced Tips

!!! tip "Space Optimization"
    Many array problems can be solved in O(1) space using clever index manipulation or the array itself as storage.

!!! note "Edge Cases"
    Always consider: empty arrays, single elements, all same elements, and arrays with maximum/minimum constraints.

!!! success "Pattern Recognition"
    Learn to quickly identify which pattern applies: two pointers, sliding window, prefix sum, or matrix manipulation.

## 🚀 Next Steps

Ready for the ultimate challenge? Try:
- [Hard Array Problems](hard-problems.md)
- Practice with 2D arrays and matrices
- Advanced sliding window techniques

---

*🎉 Great progress! You've mastered medium array problems. Ready for [Hard Problems](hard-problems.md)?*
