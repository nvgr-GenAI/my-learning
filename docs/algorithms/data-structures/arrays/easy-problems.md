# Arrays: Easy Problems

## 🚀 Foundation Array Challenges

Perfect for building your array manipulation skills and understanding core patterns.

---

## Problem 1: Two Sum

**Difficulty:** Easy  
**Pattern:** Hash Map Lookup  
**Time:** O(n) | **Space:** O(n)

### Problem Statement

Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

**Example:**

```text
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: nums[0] + nums[1] = 2 + 7 = 9
```

### Solution

```python
def two_sum(nums, target):
    """
    Find two numbers that add up to target using hash map.
    
    Store each number and its index in a hash map.
    For each number, check if target - num exists in map.
    """
    num_map = {}  # value -> index
    
    for i, num in enumerate(nums):
        complement = target - num
        
        if complement in num_map:
            return [num_map[complement], i]
        
        num_map[num] = i
    
    return []  # No solution found

# Alternative: Brute force approach (less efficient)
def two_sum_brute_force(nums, target):
    """O(n²) solution using nested loops."""
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []

# Test both approaches
nums = [2, 7, 11, 15]
target = 9
print(f"Hash map approach: {two_sum(nums, target)}")
print(f"Brute force approach: {two_sum_brute_force(nums, target)}")
```

### Explanation

1. **Hash Map Strategy**: Store each number with its index as we iterate
2. **Complement Search**: For each number, look for `target - num` in the map
3. **One Pass**: Find the solution in a single pass through the array

---

## Problem 2: Best Time to Buy and Sell Stock

**Difficulty:** Easy  
**Pattern:** Sliding Window / Two Pointers  
**Time:** O(n) | **Space:** O(1)

### Problem Statement

You are given an array `prices` where `prices[i]` is the price of a given stock on the `i`th day. Find the maximum profit you can achieve by buying and selling the stock once.

**Example:**

```text
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5
```

### Solution

```python
def max_profit(prices):
    """
    Find maximum profit using single pass tracking.
    
    Keep track of minimum price seen so far and maximum profit.
    """
    if not prices:
        return 0
    
    min_price = prices[0]
    max_profit = 0
    
    for price in prices:
        # Update minimum price
        min_price = min(min_price, price)
        
        # Calculate profit if we sell today
        current_profit = price - min_price
        max_profit = max(max_profit, current_profit)
    
    return max_profit

# Alternative: Two pointers approach
def max_profit_two_pointers(prices):
    """Two pointers approach for clarity."""
    if not prices:
        return 0
    
    left = 0  # Buy day
    right = 1  # Sell day
    max_profit = 0
    
    while right < len(prices):
        # Check if we can make profit
        if prices[left] < prices[right]:
            profit = prices[right] - prices[left]
            max_profit = max(max_profit, profit)
        else:
            # Move buy day to current day (lower price)
            left = right
        
        right += 1
    
    return max_profit

# Test both approaches
prices = [7, 1, 5, 3, 6, 4]
print(f"Single pass: {max_profit(prices)}")
print(f"Two pointers: {max_profit_two_pointers(prices)}")
```

---

## Problem 3: Contains Duplicate

**Difficulty:** Easy  
**Pattern:** Hash Set  
**Time:** O(n) | **Space:** O(n)

### Problem Statement

Given an integer array `nums`, return `true` if any value appears at least twice in the array, and return `false` if every element is distinct.

**Example:**

```text
Input: nums = [1,2,3,1]
Output: true
```

### Solution

```python
def contains_duplicate(nums):
    """
    Check for duplicates using hash set.
    
    Add elements to set one by one.
    If element already exists, we found a duplicate.
    """
    seen = set()
    
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    
    return False

# Alternative: Using set length comparison
def contains_duplicate_set_length(nums):
    """Compare original length with set length."""
    return len(nums) != len(set(nums))

# Alternative: Sorting approach
def contains_duplicate_sorting(nums):
    """Sort and check adjacent elements."""
    nums.sort()
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1]:
            return True
    return False

# Test all approaches
test_arrays = [
    [1, 2, 3, 1],
    [1, 2, 3, 4],
    [1, 1, 1, 3, 3, 4, 3, 2, 4, 2]
]

for nums in test_arrays:
    nums_copy = nums.copy()  # For sorting approach
    print(f"Array: {nums}")
    print(f"  Hash set: {contains_duplicate(nums)}")
    print(f"  Set length: {contains_duplicate_set_length(nums)}")
    print(f"  Sorting: {contains_duplicate_sorting(nums_copy)}")
```

---

## Problem 4: Maximum Subarray (Kadane's Algorithm)

**Difficulty:** Easy  
**Pattern:** Dynamic Programming  
**Time:** O(n) | **Space:** O(1)

### Problem Statement

Given an integer array `nums`, find the contiguous subarray which has the largest sum and return its sum.

**Example:**

```text
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6
```

### Solution

```python
def max_subarray(nums):
    """
    Kadane's algorithm for maximum subarray sum.
    
    Keep track of current sum and maximum sum seen so far.
    """
    if not nums:
        return 0
    
    current_sum = nums[0]
    max_sum = nums[0]
    
    for i in range(1, len(nums)):
        # Either extend current subarray or start new one
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Alternative: Track subarray indices
def max_subarray_with_indices(nums):
    """Return max sum and the subarray indices."""
    if not nums:
        return 0, -1, -1
    
    current_sum = nums[0]
    max_sum = nums[0]
    start = 0
    end = 0
    temp_start = 0
    
    for i in range(1, len(nums)):
        if current_sum < 0:
            current_sum = nums[i]
            temp_start = i
        else:
            current_sum += nums[i]
        
        if current_sum > max_sum:
            max_sum = current_sum
            start = temp_start
            end = i
    
    return max_sum, start, end

# Alternative: Divide and conquer approach
def max_subarray_divide_conquer(nums):
    """Divide and conquer approach - O(n log n)."""
    def max_crossing_sum(nums, left, mid, right):
        """Find max sum crossing the midpoint."""
        left_sum = float('-inf')
        current_sum = 0
        for i in range(mid, left - 1, -1):
            current_sum += nums[i]
            left_sum = max(left_sum, current_sum)
        
        right_sum = float('-inf')
        current_sum = 0
        for i in range(mid + 1, right + 1):
            current_sum += nums[i]
            right_sum = max(right_sum, current_sum)
        
        return left_sum + right_sum
    
    def max_subarray_util(nums, left, right):
        if left == right:
            return nums[left]
        
        mid = (left + right) // 2
        
        left_max = max_subarray_util(nums, left, mid)
        right_max = max_subarray_util(nums, mid + 1, right)
        cross_max = max_crossing_sum(nums, left, mid, right)
        
        return max(left_max, right_max, cross_max)
    
    return max_subarray_util(nums, 0, len(nums) - 1)

# Test all approaches
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(f"Kadane's algorithm: {max_subarray(nums)}")

max_sum, start, end = max_subarray_with_indices(nums)
print(f"With indices: sum={max_sum}, subarray={nums[start:end+1]}")

print(f"Divide and conquer: {max_subarray_divide_conquer(nums)}")
```

---

## Problem 5: Merge Sorted Arrays

**Difficulty:** Easy  
**Pattern:** Two Pointers  
**Time:** O(m+n) | **Space:** O(1)

### Problem Statement

You are given two integer arrays `nums1` and `nums2`, sorted in non-decreasing order, and two integers `m` and `n`. Merge `nums2` into `nums1` as one sorted array.

**Example:**

```text
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
```

### Solution

```python
def merge(nums1, m, nums2, n):
    """
    Merge two sorted arrays in-place.
    
    Start from the end to avoid overwriting elements.
    """
    # Pointers for nums1, nums2, and merged array
    i = m - 1  # Last element in nums1
    j = n - 1  # Last element in nums2
    k = m + n - 1  # Last position in merged array
    
    # Merge from the end
    while i >= 0 and j >= 0:
        if nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
    
    # Copy remaining elements from nums2
    while j >= 0:
        nums1[k] = nums2[j]
        j -= 1
        k -= 1
    
    # No need to copy remaining from nums1 (already in place)

# Alternative: Create new array (if in-place not required)
def merge_new_array(nums1, m, nums2, n):
    """Create new array instead of in-place merge."""
    result = []
    i = j = 0
    
    # Merge elements
    while i < m and j < n:
        if nums1[i] <= nums2[j]:
            result.append(nums1[i])
            i += 1
        else:
            result.append(nums2[j])
            j += 1
    
    # Add remaining elements
    result.extend(nums1[i:m])
    result.extend(nums2[j:n])
    
    return result

# Test both approaches
nums1 = [1, 2, 3, 0, 0, 0]
m = 3
nums2 = [2, 5, 6]
n = 3

# In-place merge
nums1_copy = nums1.copy()
merge(nums1_copy, m, nums2, n)
print(f"In-place merge: {nums1_copy}")

# New array approach
result = merge_new_array(nums1[:m], m, nums2, n)
print(f"New array: {result}")
```

---

## Problem 6: Remove Duplicates from Sorted Array

**Difficulty:** Easy  
**Pattern:** Two Pointers  
**Time:** O(n) | **Space:** O(1)

### Problem Statement

Given an integer array `nums` sorted in non-decreasing order, remove duplicates in-place such that each unique element appears only once. Return the number of unique elements.

**Example:**

```text
Input: nums = [1,1,2]
Output: 2, nums = [1,2,_]
```

### Solution

```python
def remove_duplicates(nums):
    """
    Remove duplicates using two pointers.
    
    Keep a slow pointer for unique elements.
    Fast pointer scans through the array.
    """
    if not nums:
        return 0
    
    # Slow pointer for unique elements
    slow = 0
    
    # Fast pointer scans the array
    for fast in range(1, len(nums)):
        # Found a new unique element
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    
    return slow + 1  # Length of unique array

# Alternative: Using set (but loses order and uses extra space)
def remove_duplicates_set(nums):
    """Remove duplicates using set (loses original order)."""
    unique_nums = list(set(nums))
    unique_nums.sort()
    
    # Copy back to original array
    for i in range(len(unique_nums)):
        nums[i] = unique_nums[i]
    
    return len(unique_nums)

# Test both approaches
nums1 = [1, 1, 2]
nums2 = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]

print("Two pointers approach:")
length1 = remove_duplicates(nums1)
print(f"Length: {length1}, Array: {nums1[:length1]}")

length2 = remove_duplicates(nums2)
print(f"Length: {length2}, Array: {nums2[:length2]}")

# Test set approach
nums3 = [1, 1, 2]
print("\nSet approach:")
length3 = remove_duplicates_set(nums3)
print(f"Length: {length3}, Array: {nums3[:length3]}")
```

---

## 🎯 Problem-Solving Patterns

### 1. Hash Map/Set Lookup

- **Use when:** Need to check existence or count occurrences
- **Pattern:** Store elements in hash structure for O(1) lookup
- **Examples:** Two sum, contains duplicate

### 2. Two Pointers

- **Use when:** Need to compare elements or merge sorted arrays
- **Pattern:** Use two pointers moving at different speeds
- **Examples:** Merge sorted arrays, remove duplicates

### 3. Sliding Window

- **Use when:** Need to track a window of elements
- **Pattern:** Expand and contract window based on conditions
- **Examples:** Best time to buy/sell stock, maximum subarray

### 4. Single Pass Optimization

- **Use when:** Can solve in one pass with state tracking
- **Pattern:** Maintain state variables while iterating
- **Examples:** Kadane's algorithm, running maximum/minimum

## 💡 Key Tips

!!! tip "Hash Map Efficiency"
    Hash maps provide O(1) average-case lookup, making them perfect for existence checks and counting.

!!! note "Two Pointers Technique"
    Two pointers are especially useful for sorted arrays and when you need to avoid nested loops.

!!! success "In-Place Modifications"
    Many array problems can be solved in-place using clever pointer management, saving space complexity.

## 🚀 Next Steps

Ready for more challenging problems? Try:
- [Medium Array Problems](medium-problems.md)
- [Hard Array Problems](hard-problems.md)
- Practice with different array patterns

---

*🎉 Great job! You've mastered easy array problems. Ready for [Medium Problems](medium-problems.md)?*
