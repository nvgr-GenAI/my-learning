# Sorting and Searching - Easy Problems

## 🎯 Learning Objectives

Master fundamental sorting and searching algorithms through hands-on problem solving. These 15 problems cover the essential patterns needed for technical interviews.

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Binary Search | Binary Search | Easy | O(log n) | O(1) |
    | 2 | First Bad Version | Binary Search | Easy | O(log n) | O(1) |
    | 3 | Search Insert Position | Binary Search | Easy | O(log n) | O(1) |
    | 4 | Sqrt(x) | Binary Search | Easy | O(log n) | O(1) |
    | 5 | Sort Array by Parity | Two Pointers | Easy | O(n) | O(1) |
    | 6 | Sort Colors | Counting Sort/3-Way Partition | Easy | O(n) | O(1) |
    | 7 | Merge Sorted Array | Two Pointers | Easy | O(m+n) | O(1) |
    | 8 | Intersection of Two Arrays | Hash Set/Two Pointers | Easy | O(n+m) | O(n) |
    | 9 | Contains Duplicate | Sorting/Hash Set | Easy | O(n log n) | O(1) |
    | 10 | Missing Number | Math/XOR/Sorting | Easy | O(n) | O(1) |
    | 11 | Find All Duplicates | Cyclic Sort Pattern | Easy | O(n) | O(1) |
    | 12 | Peak Index in Mountain Array | Binary Search | Easy | O(log n) | O(1) |
    | 13 | Two Sum II (Sorted Array) | Two Pointers | Easy | O(n) | O(1) |
    | 14 | Valid Anagram | Sorting/Counting | Easy | O(n log n) | O(1) |
    | 15 | Relative Sort Array | Counting Sort | Easy | O(n+m) | O(k) |

=== "🎯 Core Algorithm Patterns"

    **🔍 Binary Search:**
    - Template for finding exact/approximate matches
    - Search space reduction by half each iteration
    
    **🔄 Two Pointers:**
    - Merge operations on sorted arrays
    - Partitioning and element rearrangement
    
    **📊 Sorting Algorithms:**
    - Understanding when to use different sorting approaches
    - In-place sorting and stability considerations
    
    **🎯 Search Optimization:**
    - Hash-based lookups for O(1) search
    - Leveraging sorted array properties

=== "⚡ Interview Strategy"

    **💡 Pattern Recognition:**
    
    - **Sorted array given**: Think binary search first
    - **Merge two sorted**: Use two pointers technique
    - **Find missing/duplicate**: Consider cyclic sort or math
    - **Counting elements**: Counting sort might be optimal
    
    **🔄 Common Templates:**
    
    1. **Binary Search**: `while left <= right: mid = left + (right - left) // 2`
    2. **Two Pointers**: `left = 0, right = n-1` with convergence logic
    3. **Cyclic Sort**: Place each element at its correct index
    4. **Counting Sort**: When range is small, count frequencies

---

## Problem 1: Binary Search

**Difficulty**: 🟢 Easy  
**Pattern**: Binary Search Template  
**Time**: O(log n), **Space**: O(1)

=== "Problem Statement"

    Given an array of integers `nums` which is sorted in ascending order, and an integer `target`, write a function to search `target` in `nums`. If `target` exists, return its index. Otherwise, return `-1`.

    **Example:**
    ```text
    Input: nums = [-1,0,3,5,9,12], target = 9
    Output: 4
    Explanation: 9 exists in nums and its index is 4
    ```

=== "Optimal Solution"

    ```python
    def search(nums, target):
        """
        Standard binary search implementation.
        """
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2  # Prevent overflow
            
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1  # Search right half
            else:
                right = mid - 1  # Search left half
        
        return -1  # Target not found

    def search_recursive(nums, target):
        """
        Recursive implementation for understanding.
        """
        def binary_search(left, right):
            if left > right:
                return -1
            
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                return binary_search(mid + 1, right)
            else:
                return binary_search(left, mid - 1)
        
        return binary_search(0, len(nums) - 1)
    ```

=== "Binary Search Template"

    ```python
    # Universal binary search template for various problems
    def binary_search_template(nums, target):
        """
        Template that works for most binary search problems.
        Adjust comparison logic based on specific requirements.
        """
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid  # Found target
            elif nums[mid] < target:
                left = mid + 1  # Search right
            else:
                right = mid - 1  # Search left
        
        # At this point: left = right + 1
        # left is the insertion point for target
        return -1  # Or return left for insertion point
    ```

---

## Problem 2: First Bad Version

**Difficulty**: 🟢 Easy  
**Pattern**: Binary Search with API Calls  
**Time**: O(log n), **Space**: O(1)

=== "Problem Statement"

    You are given an API `bool isBadVersion(version)` which returns whether `version` is bad. Find the first bad version.

=== "Optimal Solution"

    ```python
    def first_bad_version(n):
        """
        Binary search to find first bad version.
        """
        left, right = 1, n
        
        while left < right:
            mid = left + (right - left) // 2
            
            if isBadVersion(mid):
                right = mid  # First bad might be mid or earlier
            else:
                left = mid + 1  # First bad is after mid
        
        return left  # left == right at this point
    ```

---

## Problem 3: Search Insert Position

**Difficulty**: 🟢 Easy  
**Pattern**: Binary Search for Insertion Point  
**Time**: O(log n), **Space**: O(1)

=== "Problem Statement"

    Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be inserted.

=== "Optimal Solution"

    ```python
    def search_insert(nums, target):
        """
        Find insertion position using binary search.
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
        
        return left  # Insertion point
    ```

---

## Problem 4: Sqrt(x)

**Difficulty**: 🟢 Easy  
**Pattern**: Binary Search on Answer Space  
**Time**: O(log n), **Space**: O(1)

=== "Problem Statement"

    Given a non-negative integer `x`, compute and return the square root of `x`. Return only the integer part.

=== "Optimal Solution"

    ```python
    def my_sqrt(x):
        """
        Binary search on the answer space [0, x].
        """
        if x < 2:
            return x
        
        left, right = 1, x // 2
        
        while left <= right:
            mid = left + (right - left) // 2
            square = mid * mid
            
            if square == x:
                return mid
            elif square < x:
                left = mid + 1
            else:
                right = mid - 1
        
        return right  # Largest integer whose square <= x

    def my_sqrt_newton(x):
        """
        Newton's method for comparison (faster convergence).
        """
        if x < 2:
            return x
        
        r = x
        while r * r > x:
            r = (r + x // r) // 2
        
        return r
    ```

---

## Problem 5: Sort Array by Parity

**Difficulty**: 🟢 Easy  
**Pattern**: Two Pointers Partitioning  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Given an array of integers, move all even integers to the beginning followed by all odd integers.

=== "Optimal Solution"

    ```python
    def sort_array_by_parity(nums):
        """
        Two pointers to partition array in-place.
        """
        left, right = 0, len(nums) - 1
        
        while left < right:
            # If left is odd and right is even, swap
            if nums[left] % 2 > nums[right] % 2:
                nums[left], nums[right] = nums[right], nums[left]
            
            # Move pointers
            if nums[left] % 2 == 0:
                left += 1
            if nums[right] % 2 == 1:
                right -= 1
        
        return nums

    def sort_array_by_parity_simple(nums):
        """
        Single pass with write pointer.
        """
        write_idx = 0
        
        # First pass: place all evens at the beginning
        for i in range(len(nums)):
            if nums[i] % 2 == 0:
                nums[write_idx], nums[i] = nums[i], nums[write_idx]
                write_idx += 1
        
        return nums
    ```

---

## Problem 6: Sort Colors

**Difficulty**: 🟢 Easy  
**Pattern**: Dutch National Flag (3-Way Partitioning)  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Given an array with `n` objects colored red, white, or blue (represented by integers 0, 1, and 2), sort them in-place.

=== "Optimal Solution"

    ```python
    def sort_colors(nums):
        """
        Dutch National Flag algorithm (3-way partitioning).
        """
        # Three pointers: red (0s), white (1s), blue (2s)
        red = 0      # Next position for 0
        white = 0    # Current position
        blue = len(nums) - 1  # Next position for 2 (from right)
        
        while white <= blue:
            if nums[white] == 0:
                # Place 0 in red section
                nums[red], nums[white] = nums[white], nums[red]
                red += 1
                white += 1
            elif nums[white] == 1:
                # 1 is already in correct section
                white += 1
            else:  # nums[white] == 2
                # Place 2 in blue section
                nums[white], nums[blue] = nums[blue], nums[white]
                blue -= 1
                # Don't increment white (need to check swapped element)

    def sort_colors_counting(nums):
        """
        Counting sort approach for comparison.
        """
        # Count occurrences
        counts = [0, 0, 0]
        for num in nums:
            counts[num] += 1
        
        # Reconstruct array
        idx = 0
        for color in range(3):
            for _ in range(counts[color]):
                nums[idx] = color
                idx += 1
    ```

---

## Problem 7: Merge Sorted Array

**Difficulty**: 🟢 Easy  
**Pattern**: Two Pointers (Backward Merge)  
**Time**: O(m+n), **Space**: O(1)

=== "Problem Statement"

    Merge two sorted arrays `nums1` and `nums2` into `nums1`. `nums1` has enough space to hold elements from both arrays.

=== "Optimal Solution"

    ```python
    def merge(nums1, m, nums2, n):
        """
        Merge from the end to avoid overwriting elements.
        """
        # Start from the end of both arrays
        i = m - 1      # Last element in nums1
        j = n - 1      # Last element in nums2
        k = m + n - 1  # Last position in merged array
        
        while i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1
        
        # Copy remaining elements from nums2 (if any)
        while j >= 0:
            nums1[k] = nums2[j]
            j -= 1
            k -= 1
        
        # nums1 elements are already in place if i >= 0
    ```

---

## Problem 8: Intersection of Two Arrays

**Difficulty**: 🟢 Easy  
**Pattern**: Hash Set / Two Pointers on Sorted Arrays  
**Time**: O(n+m), **Space**: O(n)

=== "Problem Statement"

    Given two integer arrays, return an array of their intersection. Each element must be unique.

=== "Optimal Solution"

    ```python
    def intersection(nums1, nums2):
        """
        Hash set approach for unsorted arrays.
        """
        set1 = set(nums1)
        result = set()
        
        for num in nums2:
            if num in set1:
                result.add(num)
        
        return list(result)

    def intersection_sorted(nums1, nums2):
        """
        Two pointers approach when arrays are sorted.
        """
        nums1.sort()
        nums2.sort()
        
        i = j = 0
        result = []
        
        while i < len(nums1) and j < len(nums2):
            if nums1[i] == nums2[j]:
                if not result or result[-1] != nums1[i]:
                    result.append(nums1[i])
                i += 1
                j += 1
            elif nums1[i] < nums2[j]:
                i += 1
            else:
                j += 1
        
        return result
    ```

---

## Problem 9: Contains Duplicate

**Difficulty**: 🟢 Easy  
**Pattern**: Hash Set / Sorting  
**Time**: O(n), **Space**: O(n)

=== "Problem Statement"

    Given an integer array, return `true` if any value appears at least twice.

=== "Optimal Solution"

    ```python
    def contains_duplicate(nums):
        """
        Hash set approach - optimal for most cases.
        """
        seen = set()
        
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        
        return False

    def contains_duplicate_sorting(nums):
        """
        Sorting approach - O(1) extra space.
        """
        nums.sort()
        
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                return True
        
        return False
    ```

---

## Problem 10: Missing Number

**Difficulty**: 🟢 Easy  
**Pattern**: Math / XOR / Cyclic Sort  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Given an array containing `n` distinct numbers in the range `[0, n]`, find the missing number.

=== "Optimal Solution"

    ```python
    def missing_number_math(nums):
        """
        Mathematical approach using sum formula.
        """
        n = len(nums)
        expected_sum = n * (n + 1) // 2
        actual_sum = sum(nums)
        return expected_sum - actual_sum

    def missing_number_xor(nums):
        """
        XOR approach - very elegant and avoids overflow.
        """
        missing = len(nums)
        
        for i, num in enumerate(nums):
            missing ^= i ^ num
        
        return missing

    def missing_number_cyclic_sort(nums):
        """
        Cyclic sort approach - places each number at correct index.
        """
        i = 0
        while i < len(nums):
            correct_idx = nums[i]
            if correct_idx < len(nums) and nums[i] != nums[correct_idx]:
                nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
            else:
                i += 1
        
        # Find the missing number
        for i in range(len(nums)):
            if nums[i] != i:
                return i
        
        return len(nums)
    ```

---

## Problem 11: Find All Duplicates in Array

**Difficulty**: 🟢 Easy  
**Pattern**: Cyclic Sort / Index Marking  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Given an array where `1 ≤ a[i] ≤ n`, some elements appear twice and others appear once. Find all elements that appear twice.

=== "Optimal Solution"

    ```python
    def find_duplicates_index_marking(nums):
        """
        Mark visited indices by negating values.
        """
        result = []
        
        for num in nums:
            index = abs(num) - 1  # Convert to 0-based index
            
            if nums[index] < 0:
                # Already visited - this is a duplicate
                result.append(abs(num))
            else:
                # Mark as visited
                nums[index] = -nums[index]
        
        # Restore original array (optional)
        for i in range(len(nums)):
            nums[i] = abs(nums[i])
        
        return result

    def find_duplicates_cyclic_sort(nums):
        """
        Cyclic sort approach.
        """
        i = 0
        while i < len(nums):
            correct_idx = nums[i] - 1
            if nums[i] != nums[correct_idx]:
                nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
            else:
                i += 1
        
        # Find duplicates
        result = []
        for i in range(len(nums)):
            if nums[i] != i + 1:
                result.append(nums[i])
        
        return result
    ```

---

## Problem 12: Peak Index in Mountain Array

**Difficulty**: 🟢 Easy  
**Pattern**: Binary Search on Unimodal Function  
**Time**: O(log n), **Space**: O(1)

=== "Problem Statement"

    An array is a mountain if it increases then decreases. Find the peak index.

=== "Optimal Solution"

    ```python
    def peak_index_in_mountain_array(arr):
        """
        Binary search to find the peak.
        """
        left, right = 0, len(arr) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            if arr[mid] < arr[mid + 1]:
                # We're on the increasing part, peak is to the right
                left = mid + 1
            else:
                # We're on the decreasing part, peak is at mid or to the left
                right = mid
        
        return left  # left == right at the peak
    ```

---

## Problem 13: Two Sum II - Input Array is Sorted

**Difficulty**: 🟢 Easy  
**Pattern**: Two Pointers on Sorted Array  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Given a sorted array, find two numbers that add up to a target. Return their indices (1-indexed).

=== "Optimal Solution"

    ```python
    def two_sum(numbers, target):
        """
        Two pointers approach leveraging sorted property.
        """
        left, right = 0, len(numbers) - 1
        
        while left < right:
            current_sum = numbers[left] + numbers[right]
            
            if current_sum == target:
                return [left + 1, right + 1]  # 1-indexed
            elif current_sum < target:
                left += 1  # Need larger sum
            else:
                right -= 1  # Need smaller sum
        
        return []  # No solution found
    ```

---

## Problem 14: Valid Anagram

**Difficulty**: 🟢 Easy  
**Pattern**: Sorting / Character Counting  
**Time**: O(n log n), **Space**: O(1)

=== "Problem Statement"

    Given two strings `s` and `t`, return `true` if `t` is an anagram of `s`.

=== "Optimal Solution"

    ```python
    def is_anagram_sorting(s, t):
        """
        Sorting approach - simple and clean.
        """
        return sorted(s) == sorted(t)

    def is_anagram_counting(s, t):
        """
        Character counting approach - O(n) time.
        """
        if len(s) != len(t):
            return False
        
        char_count = {}
        
        # Count characters in s
        for char in s:
            char_count[char] = char_count.get(char, 0) + 1
        
        # Subtract character counts from t
        for char in t:
            if char not in char_count:
                return False
            char_count[char] -= 1
            if char_count[char] == 0:
                del char_count[char]
        
        return len(char_count) == 0
    ```

---

## Problem 15: Relative Sort Array

**Difficulty**: 🟢 Easy  
**Pattern**: Counting Sort with Custom Order  
**Time**: O(n+m), **Space**: O(k)

=== "Problem Statement"

    Sort `arr1` so that the relative ordering of items in `arr1` are the same as in `arr2`. Elements not in `arr2` should be placed at the end in ascending order.

=== "Optimal Solution"

    ```python
    def relative_sort_array(arr1, arr2):
        """
        Custom sorting with counting sort approach.
        """
        # Count occurrences in arr1
        count = {}
        for num in arr1:
            count[num] = count.get(num, 0) + 1
        
        result = []
        
        # Add elements in arr2 order
        for num in arr2:
            if num in count:
                result.extend([num] * count[num])
                del count[num]
        
        # Add remaining elements in sorted order
        remaining = []
        for num in count:
            remaining.extend([num] * count[num])
        
        remaining.sort()
        result.extend(remaining)
        
        return result
    ```

---

## 🎯 Practice Summary

### Algorithm Patterns Mastered

1. **Binary Search**: Template for logarithmic search operations
2. **Two Pointers**: Efficient array manipulation and merging
3. **Cyclic Sort**: O(n) sorting for arrays with specific constraints
4. **Counting Sort**: Linear time sorting for small range values
5. **Index Manipulation**: Using array indices as hash keys

### Key Problem-Solving Insights

- **Binary Search**: Always consider when working with sorted data
- **Two Pointers**: Optimal for merging and partitioning operations
- **Math Operations**: XOR and sum formulas can provide elegant solutions
- **In-place Algorithms**: Constant space solutions are often preferred
- **Pattern Recognition**: Identify when standard algorithms apply

### Time Complexity Patterns

- **Binary Search**: O(log n) for search operations
- **Two Pointers**: O(n) for single-pass array problems
- **Sorting**: O(n log n) for comparison-based sorts
- **Counting Sort**: O(n + k) where k is the range
- **Hash Operations**: O(1) average for lookups

### Space Complexity Considerations

- **In-place Operations**: O(1) extra space preferred
- **Hash Sets**: O(n) space for fast lookups
- **Counting Arrays**: O(k) space where k is value range
- **Temporary Storage**: Minimize when possible

### Interview Success Strategy

1. **Identify Constraints**: Is array sorted? What's the range?
2. **Choose Algorithm**: Binary search for sorted, hash for fast lookup
3. **Optimize Space**: Consider in-place modifications
4. **Handle Edge Cases**: Empty arrays, single elements
5. **Verify Complexity**: Ensure optimal time/space for constraints

### Next Steps

Ready for more challenging problems? Try **[Medium Sorting & Searching Problems](medium-problems.md)** featuring:

- Advanced binary search variants (rotated arrays, 2D matrices)
- Complex sorting algorithms (merge intervals, custom comparators)
- Multi-dimensional search problems
- Optimization problems using search techniques

---

*These easy problems establish the foundation for all advanced sorting and searching algorithms. Master these patterns before moving to more complex variants!*
