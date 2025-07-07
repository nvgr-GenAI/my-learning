# Divide and Conquer - Easy Problems

## 🎯 Learning Objectives

Master the fundamental divide and conquer paradigm by understanding:

- Problem decomposition into smaller subproblems
- Recursive solution design and base cases
- Combining results from subproblems
- Time complexity analysis using recurrence relations
- Classic algorithms: binary search, merge sort, quick select

---

## Problem 1: Binary Search

**Difficulty**: 🟢 Easy  
**Pattern**: Divide Search Space in Half  
**Time**: O(log n), **Space**: O(log n) recursive, O(1) iterative

=== "Problem"

    Given an array of integers `nums` which is sorted in ascending order, and an integer `target`, write a function to search `target` in `nums`. If `target` exists, then return its index. Otherwise, return `-1`.

    You must write an algorithm with `O(log n)` runtime complexity.

    **Example 1:**
    ```
    Input: nums = [-1,0,3,5,9,12], target = 9
    Output: 4
    Explanation: 9 exists in nums and its index is 4
    ```

    **Example 2:**
    ```
    Input: nums = [-1,0,3,5,9,12], target = 2
    Output: -1
    Explanation: 2 does not exist in nums so return -1
    ```

=== "Solution"

    ```python
    def search(nums, target):
        """
        Binary search using divide and conquer.
        
        Time: O(log n) - divide search space by 2 each time
        Space: O(log n) - recursion depth
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
    def search_iterative(nums, target):
        """
        Iterative binary search to avoid recursion overhead.
        
        Time: O(log n), Space: O(1)
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
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Divide**: Split sorted array at middle point
    - **Conquer**: Recursively search in relevant half
    - **Combine**: Return result from recursive call
    - **Base Case**: Element found or search space exhausted
    
    **Critical Details:**
    - Use `left + (right - left) // 2` to avoid integer overflow
    - Ensure loop/recursion terminates with `left > right`
    - Handle edge cases: empty array, single element
    
    **Recurrence Relation**: T(n) = T(n/2) + O(1) = O(log n)

---

## Problem 2: Power Function (Fast Exponentiation)

**Difficulty**: 🟢 Easy  
**Pattern**: Recursive Exponentiation  
**Time**: O(log n), **Space**: O(log n)

=== "Problem"

    Implement `pow(x, n)`, which calculates `x` raised to the power `n` (i.e., x^n).

    **Example 1:**
    ```
    Input: x = 2.00000, n = 10
    Output: 1024.00000
    ```

    **Example 2:**
    ```
    Input: x = 2.10000, n = 3
    Output: 9.26100
    ```

    **Example 3:**
    ```
    Input: x = 2.00000, n = -2
    Output: 0.25000
    Explanation: 2^(-2) = 1/2^2 = 1/4 = 0.25
    ```

=== "Solution"

    ```python
    def myPow(x, n):
        """
        Calculate x^n using fast exponentiation.
        
        Time: O(log n) - divide exponent by 2 each time
        Space: O(log n) - recursion depth
        """
        def power_helper(x, n):
            # Base case
            if n == 0:
                return 1
            
            # Divide: compute x^(n/2)
            half = power_helper(x, n // 2)
            
            # Combine: square the result
            if n % 2 == 0:
                return half * half
            else:
                return half * half * x
        
        # Handle negative exponent
        if n < 0:
            return 1 / power_helper(x, -n)
        else:
            return power_helper(x, n)
    
    # Iterative approach
    def myPow_iterative(x, n):
        """
        Iterative fast exponentiation.
        
        Time: O(log n), Space: O(1)
        """
        if n < 0:
            x = 1 / x
            n = -n
        
        result = 1
        current_power = x
        
        while n > 0:
            if n % 2 == 1:
                result *= current_power
            current_power *= current_power
            n //= 2
        
        return result
    
    # Matrix exponentiation approach for understanding
    def myPow_binary(x, n):
        """Using binary representation of exponent"""
        if n < 0:
            x = 1 / x
            n = -n
        
        result = 1
        base = x
        
        while n > 0:
            if n & 1:  # If current bit is 1
                result *= base
            base *= base
            n >>= 1    # Right shift (divide by 2)
        
        return result
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Binary Exponentiation**: Use binary representation of exponent
    - **Recursive Structure**: x^n = (x^(n/2))^2 for even n
    - **Odd Handling**: x^n = x * (x^(n/2))^2 for odd n
    - **Negative Exponents**: x^(-n) = 1/x^n
    
    **Algorithm Intuition:**
    ```
    x^8 = (x^4)^2 = ((x^2)^2)^2 = (((x)^2)^2)^2
    Instead of 8 multiplications, only 3 needed
    ```
    
    **Time Complexity**: T(n) = T(n/2) + O(1) = O(log n)

---

## Problem 3: Maximum Subarray (Divide and Conquer)

**Difficulty**: 🟢 Easy  
**Pattern**: Divide Array, Combine Results  
**Time**: O(n log n), **Space**: O(log n)

=== "Problem"

    Given an integer array `nums`, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

    **Example 1:**
    ```
    Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
    Output: 6
    Explanation: [4,-1,2,1] has the largest sum = 6
    ```

    **Example 2:**
    ```
    Input: nums = [1]
    Output: 1
    ```

=== "Solution"

    ```python
    def maxSubArray(nums):
        """
        Find maximum subarray sum using divide and conquer.
        
        Time: O(n log n) - T(n) = 2T(n/2) + O(n)
        Space: O(log n) - recursion depth
        """
        def max_subarray_helper(left, right):
            # Base case: single element
            if left == right:
                return nums[left]
            
            # Divide: find middle point
            mid = (left + right) // 2
            
            # Conquer: find max subarray in left and right halves
            left_max = max_subarray_helper(left, mid)
            right_max = max_subarray_helper(mid + 1, right)
            
            # Combine: find max subarray crossing the middle
            # Left side of crossing subarray
            left_sum = float('-inf')
            total = 0
            for i in range(mid, left - 1, -1):
                total += nums[i]
                left_sum = max(left_sum, total)
            
            # Right side of crossing subarray
            right_sum = float('-inf')
            total = 0
            for i in range(mid + 1, right + 1):
                total += nums[i]
                right_sum = max(right_sum, total)
            
            # Maximum crossing subarray
            cross_max = left_sum + right_sum
            
            # Return maximum of all three possibilities
            return max(left_max, right_max, cross_max)
        
        return max_subarray_helper(0, len(nums) - 1)
    
    # Kadane's algorithm for comparison (O(n) solution)
    def maxSubArray_kadane(nums):
        """
        Kadane's algorithm - optimal O(n) solution.
        
        Time: O(n), Space: O(1)
        """
        max_ending_here = max_so_far = nums[0]
        
        for i in range(1, len(nums)):
            max_ending_here = max(nums[i], max_ending_here + nums[i])
            max_so_far = max(max_so_far, max_ending_here)
        
        return max_so_far
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Three Cases**: Max subarray is either in left half, right half, or crosses middle
    - **Crossing Subarray**: Extend from middle to both sides to find maximum
    - **Combine Step**: Take maximum of all three possibilities
    - **Base Case**: Single element array
    
    **Algorithm Steps:**
    1. Divide array into two halves
    2. Recursively find max subarray in each half
    3. Find max crossing subarray
    4. Return maximum of all three
    
    **Note**: While this demonstrates divide and conquer, Kadane's O(n) algorithm is more efficient.

---

## Problem 4: Merge Two Sorted Arrays

**Difficulty**: 🟢 Easy  
**Pattern**: Merge Step of Merge Sort  
**Time**: O(m + n), **Space**: O(1)

=== "Problem"

    You are given two integer arrays `nums1` and `nums2`, sorted in **non-decreasing order**, and two integers `m` and `n`, representing the number of elements in `nums1` and `nums2` respectively.

    **Merge** `nums1` and `nums2` into a single array sorted in **non-decreasing order**.

    **Example 1:**
    ```
    Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
    Output: [1,2,2,3,5,6]
    ```

=== "Solution"

    ```python
    def merge(nums1, m, nums2, n):
        """
        Merge two sorted arrays in-place.
        
        Time: O(m + n) - single pass through both arrays
        Space: O(1) - in-place merge
        """
        # Start from the end to avoid overwriting
        i = m - 1      # Last element in nums1
        j = n - 1      # Last element in nums2
        k = m + n - 1  # Last position in merged array
        
        # Merge from back to front
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
        
        # Note: remaining elements in nums1 are already in place
    
    # Alternative approach using extra space
    def merge_with_space(nums1, m, nums2, n):
        """
        Merge using auxiliary array.
        
        Time: O(m + n), Space: O(m + n)
        """
        # Create copies
        nums1_copy = nums1[:m]
        nums2_copy = nums2[:n]
        
        i = j = k = 0
        
        # Merge while both arrays have elements
        while i < m and j < n:
            if nums1_copy[i] <= nums2_copy[j]:
                nums1[k] = nums1_copy[i]
                i += 1
            else:
                nums1[k] = nums2_copy[j]
                j += 1
            k += 1
        
        # Copy remaining elements
        while i < m:
            nums1[k] = nums1_copy[i]
            i += 1
            k += 1
        
        while j < n:
            nums1[k] = nums2_copy[j]
            j += 1
            k += 1
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Backward Merging**: Start from end to avoid overwriting data
    - **Two Pointers**: Track current position in both arrays
    - **Sorted Property**: Always take smaller element first
    - **Remaining Elements**: Handle leftover elements after one array is exhausted
    
    **Why Start from End?**
    - nums1 has enough space at the end
    - Prevents overwriting unprocessed elements
    - Allows in-place merging

---

## Problem 5: Search Insert Position

**Difficulty**: 🟢 Easy  
**Pattern**: Modified Binary Search  
**Time**: O(log n), **Space**: O(1)

=== "Problem"

    Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

    You must write an algorithm with `O(log n)` runtime complexity.

    **Example 1:**
    ```
    Input: nums = [1,3,5,6], target = 5
    Output: 2
    ```

    **Example 2:**
    ```
    Input: nums = [1,3,5,6], target = 2
    Output: 1
    ```

=== "Solution"

    ```python
    def searchInsert(nums, target):
        """
        Find insert position using binary search.
        
        Time: O(log n) - binary search
        Space: O(1) - iterative approach
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
        
        # When loop ends, left is the insertion point
        return left
    
    # Recursive approach
    def searchInsert_recursive(nums, target):
        """Recursive binary search for insert position"""
        def search_helper(left, right):
            if left > right:
                return left  # Insertion point
            
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                return search_helper(mid + 1, right)
            else:
                return search_helper(left, mid - 1)
        
        return search_helper(0, len(nums) - 1)
    
    # Using built-in bisect for comparison
    def searchInsert_bisect(nums, target):
        """Using Python's bisect module"""
        import bisect
        return bisect.bisect_left(nums, target)
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Modified Binary Search**: Find exact position or insertion point
    - **Loop Invariant**: When loop ends, `left` points to insertion position
    - **Target Not Found**: Return `left` as the correct insertion index
    - **Edge Cases**: Insert at beginning, middle, or end
    
    **Why `left` is the Answer?**
    - When target not found, loop terminates with `left > right`
    - `left` points to the first element greater than target
    - This is exactly where target should be inserted

---

## Problem 6: Find Peak Element

**Difficulty**: 🟢 Easy  
**Pattern**: Binary Search on Unsorted Array  
**Time**: O(log n), **Space**: O(1)

=== "Problem"

    A peak element is an element that is strictly greater than its neighbors.

    Given a **0-indexed** integer array `nums`, find a peak element, and return its index. If the array contains multiple peaks, return the index to **any** of the peaks.

    You may imagine that `nums[-1] = nums[n] = -∞`. In other words, an element is always considered to be strictly greater than a neighbor that is outside the array.

    **Example 1:**
    ```
    Input: nums = [1,2,3,1]
    Output: 2
    Explanation: 3 is a peak element and your function should return the index number 2.
    ```

=== "Solution"

    ```python
    def findPeakElement(nums):
        """
        Find a peak element using binary search.
        
        Time: O(log n) - binary search
        Space: O(1) - iterative approach
        """
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            if nums[mid] > nums[mid + 1]:
                # Peak is in left half (including mid)
                right = mid
            else:
                # Peak is in right half
                left = mid + 1
        
        return left  # left == right, pointing to peak
    
    # Recursive approach
    def findPeakElement_recursive(nums):
        """Recursive binary search for peak"""
        def find_peak_helper(left, right):
            if left == right:
                return left
            
            mid = left + (right - left) // 2
            
            if nums[mid] > nums[mid + 1]:
                return find_peak_helper(left, mid)
            else:
                return find_peak_helper(mid + 1, right)
        
        return find_peak_helper(0, len(nums) - 1)
    
    # Linear scan for comparison
    def findPeakElement_linear(nums):
        """
        Linear approach - O(n) time
        """
        for i in range(len(nums) - 1):
            if nums[i] > nums[i + 1]:
                return i
        return len(nums) - 1
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Peak Property**: Element greater than both neighbors
    - **Binary Search Logic**: Always move toward higher neighbor
    - **Guaranteed Solution**: Array boundaries are -∞, so peak always exists
    - **Multiple Peaks**: Algorithm finds any one peak
    
    **Why Binary Search Works?**
    - If `nums[mid] > nums[mid + 1]`, peak exists in left half
    - If `nums[mid] < nums[mid + 1]`, we can find higher peak in right half
    - Since boundaries are -∞, we're guaranteed to find a peak

---

## Problem 7: Sqrt(x)

**Difficulty**: 🟢 Easy  
**Pattern**: Binary Search on Answer  
**Time**: O(log x), **Space**: O(1)

=== "Problem"

    Given a non-negative integer `x`, return the square root of `x` rounded down to the nearest integer. The returned integer should be **non-negative** as well.

    You **must not use** any built-in exponent function or operator.

    **Example 1:**
    ```
    Input: x = 4
    Output: 2
    Explanation: The square root of 4 is 2, so we return 2.
    ```

    **Example 2:**
    ```
    Input: x = 8
    Output: 2
    Explanation: The square root of 8 is 2.828..., and since we round it down to the nearest integer, 2 is returned.
    ```

=== "Solution"

    ```python
    def mySqrt(x):
        """
        Find integer square root using binary search.
        
        Time: O(log x) - binary search on range [0, x]
        Space: O(1) - iterative approach
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
        
        return right  # right is the largest integer whose square <= x
    
    # Newton's method approach
    def mySqrt_newton(x):
        """
        Using Newton's method for square root.
        
        Time: O(log x), Space: O(1)
        """
        if x < 2:
            return x
        
        # Initial guess
        guess = x
        
        while guess * guess > x:
            guess = (guess + x // guess) // 2
        
        return guess
    
    # Bit manipulation approach
    def mySqrt_bit(x):
        """Using bit manipulation"""
        if x < 2:
            return x
        
        # Find the most significant bit position
        bit = 1
        while bit * bit <= x:
            bit <<= 1
        bit >>= 1
        
        result = 0
        while bit > 0:
            if (result + bit) * (result + bit) <= x:
                result += bit
            bit >>= 1
        
        return result
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Search Space**: Binary search on possible answers [1, x/2]
    - **Overflow Prevention**: Be careful with `mid * mid` calculation
    - **Return Value**: When not exact, return largest integer whose square ≤ x
    - **Optimization**: Search range can be [2, x//2] for x ≥ 2
    
    **Algorithm Choices:**
    1. **Binary Search**: Most straightforward, O(log x)
    2. **Newton's Method**: Faster convergence, O(log x)
    3. **Bit Manipulation**: Good for understanding, O(log x)

---

## Problem 8: Valid Perfect Square

**Difficulty**: 🟢 Easy  
**Pattern**: Binary Search for Exact Match  
**Time**: O(log n), **Space**: O(1)

=== "Problem"

    Given a positive integer `num`, return `true` if `num` is a perfect square or `false` otherwise.

    A **perfect square** is an integer that is the square of an integer. In other words, it is the product of some integer with itself.

    You must not use any built-in library function, such as `sqrt`.

    **Example 1:**
    ```
    Input: num = 16
    Output: true
    Explanation: We return true because 4 * 4 = 16 and 4 is an integer.
    ```

    **Example 2:**
    ```
    Input: num = 14
    Output: false
    Explanation: We return false because 3.742 * 3.742 = 14 and 3.742 is not an integer.
    ```

=== "Solution"

    ```python
    def isPerfectSquare(num):
        """
        Check if number is perfect square using binary search.
        
        Time: O(log num) - binary search
        Space: O(1) - iterative approach
        """
        if num < 2:
            return True
        
        left, right = 2, num // 2
        
        while left <= right:
            mid = left + (right - left) // 2
            square = mid * mid
            
            if square == num:
                return True
            elif square < num:
                left = mid + 1
            else:
                right = mid - 1
        
        return False
    
    # Newton's method approach
    def isPerfectSquare_newton(num):
        """Using Newton's method"""
        if num < 2:
            return True
        
        x = num
        while x * x > num:
            x = (x + num // x) // 2
        
        return x * x == num
    
    # Mathematical approach using odd numbers
    def isPerfectSquare_math(num):
        """
        Using mathematical property: n² = 1 + 3 + 5 + ... + (2n-1)
        
        Time: O(√num), Space: O(1)
        """
        i = 1
        while num > 0:
            num -= i
            i += 2
        
        return num == 0
    
    # Recursive divide and conquer
    def isPerfectSquare_recursive(num):
        """Recursive binary search approach"""
        def check_square(left, right, target):
            if left > right:
                return False
            
            mid = left + (right - left) // 2
            square = mid * mid
            
            if square == target:
                return True
            elif square < target:
                return check_square(mid + 1, right, target)
            else:
                return check_square(left, mid - 1, target)
        
        if num < 2:
            return True
        
        return check_square(2, num // 2, num)
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Exact Match**: Unlike sqrt, we need exact equality
    - **Search Range**: For num ≥ 2, search in [2, num//2]
    - **Perfect Square Property**: Sum of first n odd numbers equals n²
    - **Newton's Method**: Converges quickly to integer square root
    
    **Multiple Approaches:**
    1. **Binary Search**: Most intuitive and efficient
    2. **Newton's Method**: Fast convergence
    3. **Mathematical**: Uses sum of odd numbers property
    4. **Recursive**: Demonstrates divide and conquer structure

---

## Problem 9: First Bad Version

**Difficulty**: 🟢 Easy  
**Pattern**: Binary Search on Predicate  
**Time**: O(log n), **Space**: O(1)

=== "Problem"

    You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.

    Suppose you have `n` versions `[1, 2, ..., n]` and you want to find out the first bad one, which causes all the following ones to be bad.

    You are given an API `bool isBadVersion(version)` which returns whether `version` is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.

    **Example 1:**
    ```
    Input: n = 5, bad = 4
    Output: 4
    Explanation:
    call isBadVersion(3) -> false
    call isBadVersion(5) -> true
    call isBadVersion(4) -> true
    Then 4 is the first bad version.
    ```

=== "Solution"

    ```python
    # The isBadVersion API is already defined for you.
    def isBadVersion(version):
        # This is a mock implementation
        bad_version = 4  # Example: version 4 is the first bad version
        return version >= bad_version
    
    def firstBadVersion(n):
        """
        Find first bad version using binary search.
        
        Time: O(log n) - binary search
        Space: O(1) - iterative approach
        """
        left, right = 1, n
        
        while left < right:
            mid = left + (right - left) // 2
            
            if isBadVersion(mid):
                # First bad version is mid or before mid
                right = mid
            else:
                # First bad version is after mid
                left = mid + 1
        
        return left  # left == right, pointing to first bad version
    
    # Recursive approach
    def firstBadVersion_recursive(n):
        """Recursive binary search approach"""
        def find_first_bad(left, right):
            if left == right:
                return left
            
            mid = left + (right - left) // 2
            
            if isBadVersion(mid):
                return find_first_bad(left, mid)
            else:
                return find_first_bad(mid + 1, right)
        
        return find_first_bad(1, n)
    
    # Alternative implementation with explicit bounds checking
    def firstBadVersion_explicit(n):
        """More explicit boundary handling"""
        left, right = 1, n
        first_bad = n
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if isBadVersion(mid):
                first_bad = mid  # Update first bad found so far
                right = mid - 1  # Look for earlier bad version
            else:
                left = mid + 1   # Look for bad version in right half
        
        return first_bad
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Predicate Search**: Find first position where condition becomes true
    - **Boundary Pattern**: Use `left < right` to find exact boundary
    - **API Minimization**: Each call to isBadVersion costs, so minimize calls
    - **Invariant**: All versions before first bad are good, all after are bad
    
    **Critical Detail:**
    - Use `right = mid` (not `mid - 1`) when bad version found
    - This ensures we don't skip the first bad version
    - Loop terminates when `left == right` at the boundary

---

## Problem 10: Guess Number Higher or Lower

**Difficulty**: 🟢 Easy  
**Pattern**: Interactive Binary Search  
**Time**: O(log n), **Space**: O(1)

=== "Problem"

    We are playing the Guess Game. The game is as follows:

    I pick a number from `1` to `n`. You have to guess which number I picked.

    Every time you guess wrong, I will tell you whether the number I picked is higher or lower than your guess.

    You call a pre-defined API `int guess(int num)`, which returns three possible results:

    - `-1`: Your guess is higher than the number I picked (i.e. `num > pick`).
    - `1`: Your guess is lower than the number I picked (i.e. `num < pick`).
    - `0`: Your guess is correct (i.e. `num == pick`).

    Return the number that I picked.

    **Example 1:**
    ```
    Input: n = 10, pick = 6
    Output: 6
    ```

=== "Solution"

    ```python
    # The guess API is already defined for you.
    def guess(num):
        # This is a mock implementation
        pick = 6  # Example: the picked number is 6
        if num > pick:
            return -1
        elif num < pick:
            return 1
        else:
            return 0
    
    def guessNumber(n):
        """
        Find the picked number using binary search.
        
        Time: O(log n) - binary search
        Space: O(1) - iterative approach
        """
        left, right = 1, n
        
        while left <= right:
            mid = left + (right - left) // 2
            result = guess(mid)
            
            if result == 0:
                return mid  # Found the number
            elif result == -1:
                right = mid - 1  # Guess is too high
            else:  # result == 1
                left = mid + 1   # Guess is too low
        
        return -1  # Should never reach here if input is valid
    
    # Recursive approach
    def guessNumber_recursive(n):
        """Recursive binary search approach"""
        def guess_helper(left, right):
            if left > right:
                return -1  # Should never happen
            
            mid = left + (right - left) // 2
            result = guess(mid)
            
            if result == 0:
                return mid
            elif result == -1:
                return guess_helper(left, mid - 1)
            else:
                return guess_helper(mid + 1, right)
        
        return guess_helper(1, n)
    
    # Ternary search approach (less efficient but educational)
    def guessNumber_ternary(n):
        """
        Using ternary search (divides into 3 parts).
        
        Time: O(log₃ n) - ternary search
        """
        left, right = 1, n
        
        while left <= right:
            mid1 = left + (right - left) // 3
            mid2 = right - (right - left) // 3
            
            result1 = guess(mid1)
            if result1 == 0:
                return mid1
            
            result2 = guess(mid2)
            if result2 == 0:
                return mid2
            
            if result1 == -1:
                right = mid1 - 1
            elif result2 == 1:
                left = mid2 + 1
            else:
                left = mid1 + 1
                right = mid2 - 1
        
        return -1
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Interactive Search**: Algorithm adapts based on API feedback
    - **Three-Way Decision**: API returns -1, 0, or 1
    - **Standard Binary Search**: Despite interactive nature, follows binary search pattern
    - **Guaranteed Solution**: Pick is guaranteed to be in range [1, n]
    
    **API Response Handling:**
    ```
    guess(mid) == -1  →  mid > pick  →  search left half
    guess(mid) == 1   →  mid < pick  →  search right half  
    guess(mid) == 0   →  mid == pick →  found answer
    ```

---

## Problem 11: Arranging Coins

**Difficulty**: 🟢 Easy  
**Pattern**: Binary Search on Mathematical Formula  
**Time**: O(log n), **Space**: O(1)

=== "Problem"

    You have `n` coins and you want to build a staircase with these coins. The staircase consists of `k` rows where the ith row has exactly `i` coins. The last row of the staircase **may be incomplete**.

    Given the integer `n`, return the number of **complete rows** of the staircase you will build.

    **Example 1:**
    ```
    Input: n = 5
    Output: 2
    Explanation: Because the 3rd row is incomplete, we return 2.
    ```

    **Example 2:**
    ```
    Input: n = 8
    Output: 3
    Explanation: Because the 4th row is incomplete, we return 3.
    ```

=== "Solution"

    ```python
    def arrangeCoins(n):
        """
        Find complete rows using binary search.
        
        Time: O(log n) - binary search
        Space: O(1) - iterative approach
        """
        left, right = 0, n
        
        while left <= right:
            mid = left + (right - left) // 2
            # Sum of first mid rows: mid * (mid + 1) / 2
            coins_needed = mid * (mid + 1) // 2
            
            if coins_needed == n:
                return mid
            elif coins_needed < n:
                left = mid + 1
            else:
                right = mid - 1
        
        return right  # right is the largest k where k*(k+1)/2 <= n
    
    # Mathematical approach using quadratic formula
    def arrangeCoins_math(n):
        """
        Using quadratic formula: k*(k+1)/2 = n
        Solving: k² + k - 2n = 0
        k = (-1 + √(1 + 8n)) / 2
        
        Time: O(1), Space: O(1)
        """
        import math
        return int((-1 + math.sqrt(1 + 8 * n)) / 2)
    
    # Iterative approach
    def arrangeCoins_iterative(n):
        """
        Iterative simulation.
        
        Time: O(√n), Space: O(1)
        """
        row = 1
        while n >= row:
            n -= row
            row += 1
        return row - 1
    
    # Newton's method approach
    def arrangeCoins_newton(n):
        """Using Newton's method to solve k*(k+1)/2 = n"""
        if n == 0:
            return 0
        
        # Initial guess
        x = n
        
        # Newton's method: x_new = x - f(x)/f'(x)
        # f(x) = x*(x+1)/2 - n
        # f'(x) = x + 0.5
        while True:
            fx = x * (x + 1) // 2 - n
            if abs(fx) < 1:
                break
            fpx = x + 0.5
            x_new = x - fx / fpx
            x = int(x_new)
        
        # Verify and adjust
        while x * (x + 1) // 2 > n:
            x -= 1
        
        return x
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Mathematical Relationship**: Sum of first k rows = k×(k+1)/2
    - **Search for Maximum k**: Find largest k where k×(k+1)/2 ≤ n
    - **Quadratic Formula**: Can solve directly using math
    - **Binary Search Range**: k can be at most n (when each row has 1 coin)
    
    **Multiple Approaches:**
    1. **Binary Search**: O(log n), most straightforward
    2. **Mathematical**: O(1), direct formula
    3. **Iterative**: O(√n), simulation
    4. **Newton's Method**: O(log log n), numerical method

---

## Problem 12: Sum of Square Numbers

**Difficulty**: 🟢 Easy  
**Pattern**: Two Pointers with Square Root  
**Time**: O(√c), **Space**: O(1)

=== "Problem"

    Given a non-negative integer `c`, decide whether there're two integers `a` and `b` such that `a² + b² = c`.

    **Example 1:**
    ```
    Input: c = 5
    Output: true
    Explanation: 1 * 1 + 2 * 2 = 5
    ```

    **Example 2:**
    ```
    Input: c = 3
    Output: false
    ```

=== "Solution"

    ```python
    def judgeSquareSum(c):
        """
        Check if c can be expressed as sum of two squares.
        
        Time: O(√c) - iterate up to √c
        Space: O(1) - only two pointers
        """
        import math
        
        left = 0
        right = int(math.sqrt(c))
        
        while left <= right:
            current_sum = left * left + right * right
            
            if current_sum == c:
                return True
            elif current_sum < c:
                left += 1
            else:
                right -= 1
        
        return False
    
    # Alternative: iterate through one variable
    def judgeSquareSum_iterate(c):
        """Iterate through possible values of a"""
        import math
        
        for a in range(int(math.sqrt(c)) + 1):
            b_squared = c - a * a
            b = int(math.sqrt(b_squared))
            
            if b * b == b_squared:
                return True
        
        return False
    
    # Using binary search for second number
    def judgeSquareSum_binary(c):
        """Use binary search to find second number"""
        import math
        
        def is_perfect_square(n):
            if n < 0:
                return False
            root = int(math.sqrt(n))
            return root * root == n
        
        for a in range(int(math.sqrt(c)) + 1):
            b_squared = c - a * a
            if is_perfect_square(b_squared):
                return True
        
        return False
    
    # Mathematical approach using Fermat's theorem
    def judgeSquareSum_fermat(c):
        """
        Using Fermat's theorem on sums of two squares:
        A number can be expressed as sum of two squares iff
        every prime factor of form 4k+3 appears to even power.
        """
        # Check for factor of 2
        while c % 2 == 0:
            c //= 2
        
        # Check for odd factors
        i = 3
        while i * i <= c:
            count = 0
            while c % i == 0:
                count += 1
                c //= i
            
            # If prime of form 4k+3 appears odd times
            if i % 4 == 3 and count % 2 == 1:
                return False
            
            i += 2
        
        # If remaining c > 1, it's a prime
        return c % 4 != 3
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Two Pointers**: Start from 0 and √c, move based on sum comparison
    - **Search Space**: Only need to check up to √c for both numbers
    - **Perfect Square Check**: Use integer square root and verify
    - **Mathematical Theory**: Fermat's theorem provides number-theoretic approach
    
    **Algorithm Efficiency:**
    1. **Two Pointers**: O(√c), most intuitive
    2. **Iteration + Perfect Square**: O(√c), simple to implement
    3. **Fermat's Theorem**: O(√c), mathematically elegant

---

## Problem 13: Kth Smallest Element in Sorted Matrix

**Difficulty**: 🟢 Easy  
**Pattern**: Binary Search on Answer Space  
**Time**: O(n log(max-min)), **Space**: O(1)

=== "Problem"

    Given an `n x n` matrix where each of the rows and columns is sorted in ascending order, return the kth smallest element in the matrix.

    Note that it is the kth smallest element **in the sorted order**, not the kth distinct element.

    **Example 1:**
    ```
    Input: matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
    Output: 13
    Explanation: The elements in the matrix are [1,5,9,10,11,12,13,13,15], and the 8th smallest element is 13
    ```

=== "Solution"

    ```python
    def kthSmallest(matrix, k):
        """
        Find kth smallest using binary search on answer.
        
        Time: O(n log(max-min)) where n is matrix dimension
        Space: O(1) - only using variables
        """
        n = len(matrix)
        left, right = matrix[0][0], matrix[n-1][n-1]
        
        def count_less_equal(target):
            """Count elements <= target using sorted property"""
            count = 0
            row, col = n - 1, 0  # Start from bottom-left
            
            while row >= 0 and col < n:
                if matrix[row][col] <= target:
                    count += row + 1  # All elements in this column <= target
                    col += 1
                else:
                    row -= 1
            
            return count
        
        while left < right:
            mid = left + (right - left) // 2
            
            if count_less_equal(mid) < k:
                left = mid + 1
            else:
                right = mid
        
        return left
    
    # Using heap approach (less efficient but intuitive)
    def kthSmallest_heap(matrix, k):
        """
        Using min-heap approach.
        
        Time: O(k log n), Space: O(n)
        """
        import heapq
        
        n = len(matrix)
        heap = []
        
        # Initialize heap with first element of each row
        for i in range(min(n, k)):
            heapq.heappush(heap, (matrix[i][0], i, 0))
        
        # Extract k-1 elements
        for _ in range(k - 1):
            val, row, col = heapq.heappop(heap)
            
            # Add next element from same row if exists
            if col + 1 < n:
                heapq.heappush(heap, (matrix[row][col + 1], row, col + 1))
        
        return heap[0][0]
    
    # Brute force approach for comparison
    def kthSmallest_brute(matrix, k):
        """
        Flatten matrix and sort.
        
        Time: O(n² log n), Space: O(n²)
        """
        elements = []
        for row in matrix:
            elements.extend(row)
        
        elements.sort()
        return elements[k - 1]
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Binary Search on Answer**: Search for the actual value, not index
    - **Counting Function**: Use matrix properties to count elements ≤ target efficiently
    - **Matrix Properties**: Rows and columns are sorted
    - **Search Space**: Between minimum and maximum elements
    
    **Counting Strategy:**
    - Start from bottom-left corner
    - If current ≤ target, all elements above are ≤ target
    - Move right to include more elements
    - If current > target, move up to exclude elements

---

## Problem 14: Find Minimum in Rotated Sorted Array

**Difficulty**: 🟢 Easy  
**Pattern**: Modified Binary Search  
**Time**: O(log n), **Space**: O(1)

=== "Problem"

    Suppose an array of length `n` sorted in ascending order is **rotated** between `1` and `n` times. For example, the array `nums = [0,1,2,4,5,6,7]` might become:

    - `[4,5,6,7,0,1,2]` if it was rotated `4` times.
    - `[0,1,2,4,5,6,7]` if it was rotated `7` times.

    Notice that **rotating** an array `[a[0], a[1], a[2], ..., a[n-1]]` 1 time results in the array `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]`.

    Given the sorted rotated array `nums` of **unique** elements, return the minimum element of this array.

    **Example 1:**
    ```
    Input: nums = [3,4,5,1,2]
    Output: 1
    ```

=== "Solution"

    ```python
    def findMin(nums):
        """
        Find minimum in rotated sorted array.
        
        Time: O(log n) - binary search
        Space: O(1) - iterative approach
        """
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            if nums[mid] > nums[right]:
                # Minimum is in right half
                left = mid + 1
            else:
                # Minimum is in left half (including mid)
                right = mid
        
        return nums[left]
    
    # Alternative approach comparing with left
    def findMin_compare_left(nums):
        """Compare mid with left element"""
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            if nums[mid] < nums[left]:
                # Minimum is in left half (including mid)
                right = mid
            elif nums[mid] > nums[left]:
                # Left half is sorted, minimum is in right half
                left = mid + 1
            else:
                # nums[mid] == nums[left], can't determine, move left
                left += 1
        
        return nums[left]
    
    # Recursive approach
    def findMin_recursive(nums):
        """Recursive binary search"""
        def find_min_helper(left, right):
            if left == right:
                return nums[left]
            
            mid = left + (right - left) // 2
            
            if nums[mid] > nums[right]:
                return find_min_helper(mid + 1, right)
            else:
                return find_min_helper(left, mid)
        
        return find_min_helper(0, len(nums) - 1)
    
    # Linear scan for comparison
    def findMin_linear(nums):
        """
        Linear approach - O(n) time.
        """
        min_val = nums[0]
        for num in nums:
            min_val = min(min_val, num)
        return min_val
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Rotation Detection**: Compare mid with boundaries to determine which half is sorted
    - **Minimum Location**: Minimum is at the rotation point
    - **Sorted Half**: One half is always sorted in rotated array
    - **Binary Search Adaptation**: Modify condition based on rotation
    
    **Algorithm Logic:**
    ```
    If nums[mid] > nums[right]: 
        Minimum is in right half (rotation point is right)
    Else: 
        Minimum is in left half including mid
    ```

---

## Problem 15: Search in Rotated Sorted Array

**Difficulty**: 🟢 Easy  
**Pattern**: Binary Search with Rotation Handling  
**Time**: O(log n), **Space**: O(1)

=== "Problem"

    There is an integer array `nums` sorted in ascending order (with **distinct** values).

    Prior to being passed to your function, `nums` is **possibly rotated** at an unknown pivot index `k` (`1 <= k < nums.length`) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` (**0-indexed**). For example, `[0,1,2,4,5,6,7]` might be rotated at pivot index `3` and become `[4,5,6,7,0,1,2]`.

    Given the array `nums` **after** the possible rotation and an integer `target`, return the index of `target` if it is in `nums`, or `-1` if it is not in `nums`.

    **Example 1:**
    ```
    Input: nums = [4,5,6,7,0,1,2], target = 0
    Output: 4
    ```

=== "Solution"

    ```python
    def search(nums, target):
        """
        Search in rotated sorted array.
        
        Time: O(log n) - binary search
        Space: O(1) - iterative approach
        """
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid
            
            # Determine which half is sorted
            if nums[left] <= nums[mid]:
                # Left half is sorted
                if nums[left] <= target < nums[mid]:
                    right = mid - 1  # Target in left half
                else:
                    left = mid + 1   # Target in right half
            else:
                # Right half is sorted
                if nums[mid] < target <= nums[right]:
                    left = mid + 1   # Target in right half
                else:
                    right = mid - 1  # Target in left half
        
        return -1
    
    # Alternative approach: find pivot first
    def search_find_pivot(nums, target):
        """Find rotation pivot first, then search normally"""
        def find_pivot():
            left, right = 0, len(nums) - 1
            
            while left < right:
                mid = left + (right - left) // 2
                
                if nums[mid] > nums[right]:
                    left = mid + 1
                else:
                    right = mid
            
            return left
        
        def binary_search(left, right, target):
            while left <= right:
                mid = left + (right - left) // 2
                
                if nums[mid] == target:
                    return mid
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return -1
        
        if not nums:
            return -1
        
        pivot = find_pivot()
        
        # Search in appropriate half
        if nums[pivot] <= target <= nums[len(nums) - 1]:
            return binary_search(pivot, len(nums) - 1, target)
        else:
            return binary_search(0, pivot - 1, target)
    
    # Recursive approach
    def search_recursive(nums, target):
        """Recursive search in rotated array"""
        def search_helper(left, right):
            if left > right:
                return -1
            
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid
            
            if nums[left] <= nums[mid]:
                # Left half is sorted
                if nums[left] <= target < nums[mid]:
                    return search_helper(left, mid - 1)
                else:
                    return search_helper(mid + 1, right)
            else:
                # Right half is sorted
                if nums[mid] < target <= nums[right]:
                    return search_helper(mid + 1, right)
                else:
                    return search_helper(left, mid - 1)
        
        return search_helper(0, len(nums) - 1)
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Rotation Property**: At least one half is always sorted
    - **Target Range Check**: Check if target is in the sorted half
    - **Two-Step Process**: Can find pivot first, then search normally
    - **Boundary Handling**: Careful with equality conditions
    
    **Decision Logic:**
    1. Identify which half is sorted
    2. Check if target is in the sorted half's range
    3. Search in appropriate half
    4. Repeat until found or exhausted

---

## 📝 Summary

### Core Divide and Conquer Patterns

| **Pattern** | **Key Insight** | **Example Problems** |
|-------------|-----------------|---------------------|
| **Binary Search** | Eliminate half of search space | Search, Insert Position |
| **Recursive Calculation** | Break into smaller subproblems | Power, Factorial |
| **Array Division** | Divide array and combine results | Maximum Subarray, Merge Sort |
| **Search on Answer** | Binary search on possible answers | Sqrt, Matrix Search |
| **Rotation Handling** | Identify sorted portion | Rotated Array Search |

### Universal Templates

```python
# Basic Binary Search
def binary_search(arr, target):
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

# Recursive Divide and Conquer
def divide_conquer(problem):
    if base_case(problem):
        return solve_directly(problem)
    
    subproblems = divide(problem)
    sub_results = [divide_conquer(sub) for sub in subproblems]
    return combine(sub_results)

# Binary Search on Answer
def search_answer(left, right, check_function):
    while left < right:
        mid = left + (right - left) // 2
        if check_function(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

### Time Complexity Analysis

- **Binary Search**: T(n) = T(n/2) + O(1) = O(log n)
- **Fast Exponentiation**: T(n) = T(n/2) + O(1) = O(log n)  
- **Divide and Conquer on Arrays**: T(n) = 2T(n/2) + O(n) = O(n log n)
- **Search on Answer**: O(log(answer_range) × check_function_time)

### Problem-Solving Strategy

1. **Identify Division**: How can the problem be split?
2. **Define Base Case**: When is the problem small enough to solve directly?
3. **Combine Results**: How are subproblem solutions combined?
4. **Optimize**: Can it be done iteratively? Are there redundant calculations?

### Common Pitfalls

- **Integer Overflow**: Use `left + (right - left) // 2` instead of `(left + right) // 2`
- **Infinite Loops**: Ensure search space decreases each iteration
- **Boundary Conditions**: Handle edge cases like empty arrays, single elements
- **Off-by-One Errors**: Be careful with inclusive vs exclusive bounds

---

## 🎯 Next Steps

- **[Medium Divide and Conquer Problems](medium-problems.md)** - More complex recursive algorithms
- **[Advanced Algorithms](../advanced/index.md)** - Master theorem, complex recurrences
- **[Dynamic Programming](../dp/index.md)** - When divide and conquer meets memoization

These fundamental patterns form the backbone of many advanced algorithms. Master them well!
