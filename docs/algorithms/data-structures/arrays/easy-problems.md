# Arrays: Easy Problems

## 🚀 Foundation Array Challenges

Perfect for building your array manipulation skills and understanding core patterns. These problems are frequently asked in technical interviews and form the foundation for more complex array algorithms.

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Two Sum | Hash Map | Easy | O(n) | O(n) |
    | 2 | Best Time to Buy/Sell Stock | Sliding Window | Easy | O(n) | O(1) |
    | 3 | Remove Duplicates from Sorted Array | Two Pointers | Easy | O(n) | O(1) |
    | 4 | Contains Duplicate | Hash Set | Easy | O(n) | O(n) |
    | 5 | Maximum Subarray (Kadane's) | Dynamic Programming | Easy | O(n) | O(1) |
    | 6 | Merge Sorted Array | Two Pointers | Easy | O(m+n) | O(1) |
    | 7 | Plus One | Array Manipulation | Easy | O(n) | O(1) |
    | 8 | Move Zeroes | Two Pointers | Easy | O(n) | O(1) |
    | 9 | Intersection of Two Arrays II | Hash Map | Easy | O(n+m) | O(min(n,m)) |
    | 10 | Valid Anagram | Hash Map/Sorting | Easy | O(n log n) | O(1) |
    | 11 | Single Number | Bit Manipulation | Easy | O(n) | O(1) |
    | 12 | Find All Numbers Disappeared | Cyclic Sort | Easy | O(n) | O(1) |
    | 13 | Majority Element | Boyer-Moore | Easy | O(n) | O(1) |
    | 14 | Running Sum of 1D Array | Prefix Sum | Easy | O(n) | O(1) |
    | 15 | Shuffle the Array | Array Manipulation | Easy | O(n) | O(n) |

=== "🎯 Interview Tips"

    **📝 Common Patterns to Master:**
    
    - **Hash Map Lookups**: Store seen elements for O(1) access
    - **Two Pointers**: Left/right pointers for in-place operations
    - **Sliding Window**: Fixed or variable window for subarray problems
    - **Prefix Sums**: Running totals for range queries
    - **Cyclic Sort**: Numbers 1 to n pattern
    
    **⚡ Quick Win Strategies:**
    
    - Always ask about constraints (sorted, duplicates, range)
    - Consider in-place modifications to optimize space
    - Think about edge cases: empty array, single element, all same
    - Practice explaining your approach before coding
    
    **🚫 Common Mistakes:**
    
    - Off-by-one errors in loop boundaries
    - Not handling empty arrays
    - Forgetting to return the modified array length
    - Overcomplicating simple two-pointer problems

=== "📚 Study Plan"

    **Week 1-2: Foundation (Problems 1-5)**
    - Master hash map and two-pointer techniques
    - Focus on time/space complexity analysis
    
    **Week 3-4: Intermediate (Problems 6-10)**
    - Practice in-place array modifications
    - Learn prefix sum and sliding window patterns
    
    **Week 5-6: Advanced Easy (Problems 11-15)**
    - Bit manipulation and mathematical approaches
    - Cyclic sort and Boyer-Moore algorithm

---

## Problem 1: Two Sum

**Difficulty:** Easy  
**Pattern:** Hash Map Lookup  
**Time:** O(n) | **Space:** O(n)

=== "Problem Statement"

    Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

    **Example:**
    ```text
    Input: nums = [2,7,11,15], target = 9
    Output: [0,1]
    Explanation: nums[0] + nums[1] = 2 + 7 = 9
    ```

=== "Optimal Solution"

    ```python
    def two_sum(nums, target):
        """
        Hash map approach - O(n) time, O(n) space.
        
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

    # Test
    nums = [2, 7, 11, 15]
    target = 9
    result = two_sum(nums, target)
    print(f"Indices: {result}")  # [0, 1]
    print(f"Values: {nums[result[0]]} + {nums[result[1]]} = {target}")
    ```

=== "Brute Force"

    ```python
    def two_sum_brute_force(nums, target):
        """
        Brute force approach - O(n²) time, O(1) space.
        Check all possible pairs.
        """
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]
        return []

    # Test
    nums = [3, 2, 4]
    target = 6
    print(f"Brute force result: {two_sum_brute_force(nums, target)}")  # [1, 2]
    ```

=== "Two-Pass Hash Map"

    ```python
    def two_sum_two_pass(nums, target):
        """
        Two-pass hash map - clearer logic but less efficient.
        First pass: build hash map
        Second pass: look for complements
        """
        # First pass: create hash map
        num_map = {}
        for i, num in enumerate(nums):
            num_map[num] = i
        
        # Second pass: find complements
        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_map and num_map[complement] != i:
                return [i, num_map[complement]]
        
        return []

    # Test
    nums = [3, 3]
    target = 6
    print(f"Two-pass result: {two_sum_two_pass(nums, target)}")  # [0, 1]
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Hash map trades space for time efficiency
    - One-pass is optimal: build map while searching
    - Handle duplicates by checking `num_map[complement] != i`
    
    **⚡ Interview Tips:**
    
    - Always ask: "Can I assume exactly one solution exists?"
    - Clarify: "Should I return values or indices?"
    - Consider: "What if there are duplicate numbers?"
    
    **🚫 Common Mistakes:**
    
    - Using the same element twice: `nums[i] + nums[i]`
    - Not handling the case where complement equals current number
    - Forgetting to store index before checking complement

---

## Problem 2: Best Time to Buy and Sell Stock

**Difficulty:** Easy  
**Pattern:** Sliding Window / Two Pointers  
**Time:** O(n) | **Space:** O(1)

=== "Problem Statement"

    You are given an array `prices` where `prices[i]` is the price of a given stock on the `i`th day. Find the maximum profit you can achieve by buying and selling the stock once.

    **Example:**
    ```text
    Input: prices = [7,1,5,3,6,4]
    Output: 5
    Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5
    ```

=== "Optimal Solution"

    ```python
    def max_profit(prices):
        """
        Single pass tracking approach - O(n) time, O(1) space.
        
        Keep track of minimum price seen so far and maximum profit.
        For each price, calculate profit if we sell today.
        """
        if not prices:
            return 0
        
        min_price = prices[0]
        max_profit = 0
        
        for price in prices:
            # Update minimum price seen so far
            min_price = min(min_price, price)
            
            # Calculate profit if we sell today
            current_profit = price - min_price
            max_profit = max(max_profit, current_profit)
        
        return max_profit

    # Test
    prices = [7, 1, 5, 3, 6, 4]
    print(f"Max profit: {max_profit(prices)}")  # 5
    ```

=== "Two Pointers"

    ```python
    def max_profit_two_pointers(prices):
        """
        Two pointers approach - more intuitive for some.
        Left pointer: buy day, Right pointer: sell day
        """
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

    # Test
    prices = [7, 6, 4, 3, 1]  # Decreasing prices
    print(f"Two pointers: {max_profit_two_pointers(prices)}")  # 0
    ```

=== "Brute Force"

    ```python
    def max_profit_brute_force(prices):
        """
        Brute force approach - O(n²) time, O(1) space.
        Check all possible buy/sell combinations.
        """
        max_profit = 0
        
        for i in range(len(prices)):
            for j in range(i + 1, len(prices)):
                profit = prices[j] - prices[i]
                max_profit = max(max_profit, profit)
        
        return max_profit

    # Test
    prices = [1, 5, 3, 6, 4]
    print(f"Brute force: {max_profit_brute_force(prices)}")  # 5
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Must buy before selling (can't sell then buy)
    - Track minimum price seen so far, not just current vs previous
    - Profit = current_price - min_price_so_far
    
    **⚡ Interview Tips:**
    
    - Ask: "Can I buy and sell on the same day?" (Usually no)
    - Clarify: "What if prices only decrease?" (Return 0)
    - Consider: "Multiple transactions allowed?" (Different problem)
    
    **🔍 Pattern Recognition:**
    
    - This is a sliding window maximum problem
    - Similar to "Maximum subarray" but with constraints
    - Foundation for stock problems with multiple transactions

---

## Problem 3: Remove Duplicates from Sorted Array

**Difficulty:** Easy  
**Pattern:** Two Pointers  
**Time:** O(n) | **Space:** O(1)

=== "Problem Statement"

    Given an integer array `nums` sorted in non-decreasing order, remove duplicates in-place such that each unique element appears only once. Return the number of unique elements.

    **Example:**
    ```text
    Input: nums = [1,1,2]
    Output: 2, nums = [1,2,_]
    ```

=== "Solution"

    ```python
    def remove_duplicates(nums):
        """
        Two pointers approach for in-place duplicate removal.
        
        Keep one pointer for unique elements position,
        another for scanning the array.
        """
        if not nums:
            return 0
        
        # First element is always unique
        unique_pos = 0
        
        for i in range(1, len(nums)):
            # Found a new unique element
            if nums[i] != nums[unique_pos]:
                unique_pos += 1
                nums[unique_pos] = nums[i]
        
        return unique_pos + 1

    # Test
    nums = [0,0,1,1,1,2,2,3,3,4]
    k = remove_duplicates(nums)
    print(f"Unique count: {k}")  # 5
    print(f"Array: {nums[:k]}")  # [0,1,2,3,4]
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Two pointers technique: one for unique position, one for scanning
    - Sorted array property: duplicates are adjacent
    - In-place modification without extra space
    
    **⚡ Interview Tips:**
    
    - Ask: "Should I preserve the original array?" (Usually no)
    - Clarify: "What should I return?" (Length of unique elements)
    - Remember: Array elements after position k don't matter
    
    **🔍 Follow-up:**
    
    - What if array is not sorted? (Use hash set)
    - Remove duplicates allowing at most k occurrences
    - Remove specific element instead of duplicates

---

## Problem 4: Contains Duplicate

**Difficulty:** Easy  
**Pattern:** Hash Set  
**Time:** O(n) | **Space:** O(n)

=== "Problem Statement"

    Given an integer array `nums`, return `true` if any value appears at least twice in the array, and return `false` if every element is distinct.

    **Example:**
    ```text
    Input: nums = [1,2,3,1]
    Output: true
    ```

=== "Solution"

    ```python
    def contains_duplicate_hash(nums):
        """
        Hash set approach - most efficient.
        """
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False

    def contains_duplicate_sort(nums):
        """
        Sorting approach - O(n log n) time, O(1) space.
        """
        nums.sort()
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                return True
        return False

    def contains_duplicate_pythonic(nums):
        """
        Pythonic one-liner using set length comparison.
        """
        return len(nums) != len(set(nums))

    # Test all approaches
    nums = [1,2,3,1]
    print(f"Hash set: {contains_duplicate_hash(nums)}")  # True
    print(f"Sorting: {contains_duplicate_sort(nums.copy())}")  # True
    print(f"Pythonic: {contains_duplicate_pythonic(nums)}")  # True
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Hash set provides O(1) lookup for duplicate detection
    - Set length comparison is Pythonic but less educational
    - Sorting changes time complexity but saves space
    
    **⚡ Interview Tips:**
    
    - Ask about constraints: array size, value range
    - Space vs time tradeoff: O(n) space vs O(n log n) time
    - Consider early termination in hash set approach
    
    **🔍 Variations:**
    
    - Find the duplicate number (only one duplicate)
    - Contains duplicate within k distance
    - Contains duplicate with specific value difference

---

## Problem 5: Maximum Subarray (Kadane's Algorithm)

**Difficulty:** Easy  
**Pattern:** Dynamic Programming  
**Time:** O(n) | **Space:** O(1)

=== "Problem Statement"

    Given an integer array `nums`, find the contiguous subarray which has the largest sum and return its sum.

    **Example:**
    ```text
    Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
    Output: 6
    Explanation: [4,-1,2,1] has the largest sum = 6
    ```

=== "Solution"

    ```python
    def max_subarray_kadane(nums):
        """
        Kadane's algorithm - optimal DP solution.
        
        At each position, decide whether to:
        1. Start a new subarray from current element
        2. Extend the existing subarray
        """
        max_sum = nums[0]
        current_sum = nums[0]
        
        for i in range(1, len(nums)):
            # Choose max of: start new vs extend existing
            current_sum = max(nums[i], current_sum + nums[i])
            max_sum = max(max_sum, current_sum)
        
        return max_sum

    def max_subarray_with_indices(nums):
        """
        Kadane's with start/end indices tracking.
        """
        max_sum = nums[0]
        current_sum = nums[0]
        start = end = 0
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

    # Test
    nums = [-2,1,-3,4,-1,2,1,-5,4]
    print(f"Max sum: {max_subarray_kadane(nums)}")  # 6
    
    max_sum, start, end = max_subarray_with_indices(nums)
    print(f"Subarray: {nums[start:end+1]} = {max_sum}")  # [4,-1,2,1] = 6
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Kadane's algorithm is a dynamic programming pattern
    - Local optimum leads to global optimum (greedy choice)
    - Negative prefix never helps maximize sum
    
    **⚡ Interview Tips:**
    
    - Ask: "What if all numbers are negative?" (Return the least negative)
    - Clarify: "Empty subarray allowed?" (Usually no, minimum length 1)
    - Practice explaining the "reset strategy" when sum becomes negative
    
    **🔍 Extensions:**
    
    - Maximum product subarray (handle zeros and negatives)
    - Maximum sum of k-length subarray (sliding window)
    - Maximum sum circular array (handle wraparound)

---

## Problem 6: Merge Sorted Array

**Difficulty:** Easy  
**Pattern:** Two Pointers  
**Time:** O(m+n) | **Space:** O(1)

=== "Problem Statement"

    You are given two integer arrays `nums1` and `nums2`, sorted in non-decreasing order, and two integers `m` and `n`. Merge `nums2` into `nums1` as one sorted array.

    **Example:**

    ```text
    Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
    Output: [1,2,2,3,5,6]
    ```

=== "Solution"

    ```python
    def merge_sorted_arrays(nums1, m, nums2, n):
        """
        Merge from the end to avoid overwriting elements.
        
        Start from the largest elements and work backwards.
        """
        # Pointers for nums1, nums2, and merge position
        i = m - 1      # Last element in nums1
        j = n - 1      # Last element in nums2
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
        
        # Copy remaining elements from nums2 (if any)
        while j >= 0:
            nums1[k] = nums2[j]
            j -= 1
            k -= 1
        
        # No need to copy remaining from nums1 (already in place)

    # Alternative: Forward merge (requires extra space)
    def merge_sorted_arrays_forward(nums1, m, nums2, n):
        """
        Forward merge using extra space for clarity.
        """
        result = []
        i = j = 0
        
        while i < m and j < n:
            if nums1[i] <= nums2[j]:
                result.append(nums1[i])
                i += 1
            else:
                result.append(nums2[j])
                j += 1
        
        # Add remaining elements
        result.extend(nums1[i:m])
        result.extend(nums2[j:])
        
        # Copy back to nums1
        nums1[:] = result

    # Test
    nums1 = [1,2,3,0,0,0]
    nums2 = [2,5,6]
    merge_sorted_arrays(nums1, 3, nums2, 3)
    print(f"Merged: {nums1}")  # [1,2,2,3,5,6]
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Backwards merging avoids overwriting unprocessed elements
    - Utilize the fact that nums1 has extra space at the end
    - No additional space needed beyond the provided arrays
    
    **⚡ Interview Tips:**
    
    - Ask: "Can I modify nums1?" (Yes, that's the point)
    - Clarify: "Are both arrays sorted?" (Yes, given in problem)
    - Think: "Why not merge forwards?" (Would overwrite elements)
    
    **🔍 Pattern Recognition:**
    
    - Two pointers on sorted data
    - In-place array manipulation
    - Similar to merge step in merge sort

---

## Problem 7: Plus One

**Difficulty:** Easy  
**Pattern:** Array Manipulation  
**Time:** O(n) | **Space:** O(1) or O(n)

=== "Problem Statement"

    You are given a large integer represented as an integer array `digits`. Increment the integer by one and return the resulting array.

    **Example:**
    ```text
    Input: digits = [1,2,3]
    Output: [1,2,4]
    
    Input: digits = [9,9,9]
    Output: [1,0,0,0]
    ```

=== "Solution"

    ```python
    def plus_one(digits):
        """
        Handle carry propagation from right to left.
        
        Most cases: simple increment
        Edge case: all 9s require new array
        """
        # Start from the least significant digit
        for i in range(len(digits) - 1, -1, -1):
            if digits[i] < 9:
                digits[i] += 1
                return digits
            
            # Current digit is 9, set to 0 and carry over
            digits[i] = 0
        
        # All digits were 9, need extra digit
        return [1] + digits

    def plus_one_alternative(digits):
        """
        Alternative approach using string conversion.
        Less efficient but more intuitive.
        """
        # Convert to number, add 1, convert back
        num = int(''.join(map(str, digits))) + 1
        return [int(d) for d in str(num)]

    # Test cases
    test_cases = [
        [1,2,3],    # Normal case
        [9,9,9],    # All 9s
        [9],        # Single 9
        [1,9],      # Partial carry
    ]
    
    for digits in test_cases:
        result = plus_one(digits.copy())
        print(f"{digits} + 1 = {result}")
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Handle carry propagation from right to left (least significant first)
    - Only all-9s case requires creating a new array
    - Most cases just need simple increment
    
    **⚡ Interview Tips:**
    
    - Ask: "What's the range of digits?" (Usually 0-9)
    - Clarify: "Leading zeros allowed?" (Usually no, except for [0])
    - Edge case: "What about negative numbers?" (Usually not applicable)
    
    **🔍 Similar Problems:**
    
    - Plus one for binary numbers
    - Add two numbers represented as arrays
    - Multiply strings representing numbers

---

## Problem 8: Move Zeroes

**Difficulty:** Easy  
**Pattern:** Two Pointers  
**Time:** O(n) | **Space:** O(1)

=== "Problem Statement"

    Given an integer array `nums`, move all 0's to the end while maintaining the relative order of non-zero elements. Do this in-place.

    **Example:**
    ```text
    Input: nums = [0,1,0,3,12]
    Output: [1,3,12,0,0]
    ```

=== "Solution"

    ```python
    def move_zeroes_optimal(nums):
        """
        Two pointers: one for non-zero position, one for scanning.
        
        Maintain relative order of non-zero elements.
        """
        # Pointer for next non-zero position
        non_zero_pos = 0
        
        # Move all non-zero elements to front
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[non_zero_pos] = nums[i]
                non_zero_pos += 1
        
        # Fill remaining positions with zeros
        while non_zero_pos < len(nums):
            nums[non_zero_pos] = 0
            non_zero_pos += 1

    def move_zeroes_swap(nums):
        """
        Alternative: swap non-zero elements with zeros.
        Fewer writes but same complexity.
        """
        non_zero_pos = 0
        
        for i in range(len(nums)):
            if nums[i] != 0:
                # Swap only if positions are different
                if i != non_zero_pos:
                    nums[i], nums[non_zero_pos] = nums[non_zero_pos], nums[i]
                non_zero_pos += 1

    # Test
    nums1 = [0,1,0,3,12]
    move_zeroes_optimal(nums1)
    print(f"Optimal approach: {nums1}")  # [1,3,12,0,0]
    
    nums2 = [0,1,0,3,12]
    move_zeroes_swap(nums2)
    print(f"Swap approach: {nums2}")     # [1,3,12,0,0]
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Two pointers: one for placement, one for scanning
    - Maintain relative order of non-zero elements
    - Fill zeros at the end rather than shifting during scan
    
    **⚡ Interview Tips:**
    
    - Ask: "Should I maintain relative order?" (Usually yes)
    - Clarify: "In-place modification required?" (Usually yes)
    - Consider: "What about other values to move?" (Similar pattern)
    
    **🔍 Variations:**
    
    - Move all even numbers to end
    - Move all negative numbers to left
    - Remove element instead of moving

---

## Problem 9: Intersection of Two Arrays II

**Difficulty:** Easy  
**Pattern:** Hash Map  
**Time:** O(n+m) | **Space:** O(min(n,m))

=== "Problem Statement"

    Given two integer arrays `nums1` and `nums2`, return an array of their intersection. Each element in the result must appear as many times as it shows in both arrays.

    **Example:**
    ```text
    Input: nums1 = [1,2,2,1], nums2 = [2,2]
    Output: [2,2]
    ```

=== "Solution"

    ```python
    def intersect_hash_map(nums1, nums2):
        """
        Hash map approach - count frequencies.
        
        Use smaller array for hash map to optimize space.
        """
        from collections import Counter
        
        # Use smaller array for counter to save space
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        
        counter = Counter(nums1)
        result = []
        
        for num in nums2:
            if counter[num] > 0:
                result.append(num)
                counter[num] -= 1
        
        return result

    def intersect_sort_two_pointers(nums1, nums2):
        """
        Sort + two pointers approach.
        Good when modifying input is allowed.
        """
        nums1.sort()
        nums2.sort()
        
        i = j = 0
        result = []
        
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                i += 1
            elif nums1[i] > nums2[j]:
                j += 1
            else:
                result.append(nums1[i])
                i += 1
                j += 1
        
        return result

    # Manual counter implementation (if Counter not allowed)
    def intersect_manual_count(nums1, nums2):
        """
        Manual frequency counting without Counter.
        """
        count = {}
        
        # Count frequencies in nums1
        for num in nums1:
            count[num] = count.get(num, 0) + 1
        
        result = []
        for num in nums2:
            if count.get(num, 0) > 0:
                result.append(num)
                count[num] -= 1
        
        return result

    # Test
    nums1 = [4,9,5]
    nums2 = [9,4,9,8,4]
    print(f"Hash map: {intersect_hash_map(nums1, nums2)}")  # [9,4]
    print(f"Two pointers: {intersect_sort_two_pointers(nums1.copy(), nums2.copy())}")  # [4,9]
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Use smaller array for hash map to optimize space
    - Counter is perfect for frequency-based problems
    - Two pointers work well when both arrays are sorted
    
    **⚡ Interview Tips:**
    
    - Ask: "Are duplicates important?" (Yes, count matters)
    - Clarify: "Which order for result?" (Any order usually fine)
    - Follow-up: Handle very large arrays efficiently
    
    **🔍 Follow-up Questions:**
    
    - What if arrays are sorted? (Use two pointers)
    - What if one array is much smaller? (Use it for hash map)
    - What if memory is limited? (External sorting approach)

---

## Problem 10: Valid Anagram

**Difficulty:** Easy  
**Pattern:** Hash Map/Sorting  
**Time:** O(n log n) or O(n) | **Space:** O(1) or O(1)

=== "Problem Statement"

    Given two strings `s` and `t`, return `true` if `t` is an anagram of `s`.

    **Example:**
    ```text
    Input: s = "anagram", t = "nagaram"
    Output: true
    ```

=== "Solution"

    ```python
    def is_anagram_sorting(s, t):
        """
        Sorting approach - simple and clean.
        """
        return sorted(s) == sorted(t)

    def is_anagram_counter(s, t):
        """
        Counter approach - most Pythonic.
        """
        from collections import Counter
        return Counter(s) == Counter(t)

    def is_anagram_array(s, t):
        """
        Array counting - good for lowercase letters only.
        """
        if len(s) != len(t):
            return False
        
        # Assuming only lowercase letters a-z
        count = [0] * 26
        
        for i in range(len(s)):
            count[ord(s[i]) - ord('a')] += 1
            count[ord(t[i]) - ord('a')] -= 1
        
        return all(c == 0 for c in count)

    def is_anagram_hash_map(s, t):
        """
        Hash map approach - works for any characters.
        """
        if len(s) != len(t):
            return False
        
        char_count = {}
        
        # Count characters in s
        for char in s:
            char_count[char] = char_count.get(char, 0) + 1
        
        # Decrement for characters in t
        for char in t:
            if char not in char_count:
                return False
            char_count[char] -= 1
            if char_count[char] == 0:
                del char_count[char]
        
        return len(char_count) == 0

    # Test
    s, t = "anagram", "nagaram"
    print(f"Sorting: {is_anagram_sorting(s, t)}")      # True
    print(f"Counter: {is_anagram_counter(s, t)}")      # True
    print(f"Array: {is_anagram_array(s, t)}")          # True
    print(f"Hash map: {is_anagram_hash_map(s, t)}")    # True
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Multiple valid approaches with different tradeoffs
    - Array counting works only for limited character sets
    - Hash map approach is most general and flexible
    
    **⚡ Interview Tips:**
    
    - Ask: "Only lowercase letters?" (Affects approach choice)
    - Clarify: "Unicode characters?" (Hash map needed)
    - Consider: "Case sensitivity?" (Usually case-sensitive)
    
    **🔍 Pattern Recognition:**
    
    - Frequency counting is common in string problems
    - Sorting can simplify many comparison problems
    - Space-time tradeoffs are important to discuss

---

## Problem 11: Single Number

**Difficulty:** Easy  
**Pattern:** Bit Manipulation  
**Time:** O(n) | **Space:** O(1)

=== "Problem Statement"

    Given a non-empty array of integers `nums`, every element appears twice except for one. Find that single one without using extra space.

    **Example:**
    ```text
    Input: nums = [2,2,1]
    Output: 1
    ```

=== "Solution"

    ```python
    def single_number_xor(nums):
        """
        XOR approach - most elegant solution.
        
        XOR properties:
        - a ^ a = 0
        - a ^ 0 = a
        - XOR is commutative and associative
        """
        result = 0
        for num in nums:
            result ^= num
        return result

    def single_number_hash_set(nums):
        """
        Hash set approach - uses extra space.
        Good for understanding the problem.
        """
        seen = set()
        for num in nums:
            if num in seen:
                seen.remove(num)
            else:
                seen.add(num)
        return seen.pop()

    def single_number_sum(nums):
        """
        Mathematical approach using set sum.
        2 * (sum of unique) - (sum of all) = single number
        """
        return 2 * sum(set(nums)) - sum(nums)

    # Demonstration of XOR properties
    def demonstrate_xor():
        """
        Show why XOR works for this problem.
        """
        nums = [4, 1, 2, 1, 2]
        print(f"Array: {nums}")
        
        result = 0
        for num in nums:
            print(f"{result:b} ^ {num:b} = {result ^ num:b} ({result ^ num})")
            result ^= num
        
        print(f"Single number: {result}")

    # Test
    nums = [4,1,2,1,2]
    print(f"XOR approach: {single_number_xor(nums)}")      # 4
    print(f"Hash set: {single_number_hash_set(nums)}")     # 4
    print(f"Sum approach: {single_number_sum(nums)}")      # 4
    
    print("\nXOR demonstration:")
    demonstrate_xor()
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - XOR is the most elegant solution for this specific constraint
    - Mathematical approach uses the property: 2×(unique sum) - total sum
    - XOR properties: a⊕a=0, a⊕0=a, commutative, associative
    
    **⚡ Interview Tips:**
    
    - Ask: "Exactly one number appears once?" (Critical constraint)
    - Mention: "XOR is the optimal solution" (Shows bit manipulation knowledge)
    - Explain: Why XOR works (Demonstrates deep understanding)
    
    **🔍 Bit Manipulation Pattern:**
    
    - Single Number II: every element appears 3 times except one
    - Single Number III: two elements appear once, rest twice
    - Missing number in range [0,n] using XOR

---

## Problem 12: Find All Numbers Disappeared in Array

**Difficulty:** Easy  
**Pattern:** Cyclic Sort / Array Manipulation  
**Time:** O(n) | **Space:** O(1)

=== "Problem Statement"

    Given an array `nums` of n integers where `nums[i]` is in range `[1, n]`, return all integers in range `[1, n]` that do not appear in `nums`.

    **Example:**
    ```text
    Input: nums = [4,3,2,7,8,2,3,1]
    Output: [5,6]
    ```

=== "Solution"

    ```python
    def find_disappeared_numbers_marking(nums):
        """
        Mark presence by negating values at corresponding indices.
        
        Use array itself as a hash map:
        - Index i represents number i+1
        - Negative value means number i+1 is present
        """
        # Mark numbers as seen by negating values
        for num in nums:
            # Get the index for this number (num-1)
            index = abs(num) - 1
            
            # Mark as seen by making negative (if not already)
            if nums[index] > 0:
                nums[index] = -nums[index]
        
        # Collect indices of positive values
        result = []
        for i in range(len(nums)):
            if nums[i] > 0:
                result.append(i + 1)  # Convert index back to number
        
        return result

    def find_disappeared_numbers_cyclic_sort(nums):
        """
        Cyclic sort approach - place each number at its correct position.
        
        For range [1, n], number i should be at index i-1.
        """
        i = 0
        while i < len(nums):
            # Calculate correct position for current number
            correct_pos = nums[i] - 1
            
            # If number is not at correct position and positions are different
            if nums[i] != nums[correct_pos]:
                nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
            else:
                i += 1
        
        # Find missing numbers
        result = []
        for i in range(len(nums)):
            if nums[i] != i + 1:
                result.append(i + 1)
        
        return result

    def find_disappeared_numbers_set(nums):
        """
        Set approach - uses extra space but clear logic.
        """
        all_numbers = set(range(1, len(nums) + 1))
        present_numbers = set(nums)
        return list(all_numbers - present_numbers)

    # Test
    nums1 = [4,3,2,7,8,2,3,1]
    print(f"Marking: {find_disappeared_numbers_marking(nums1.copy())}")     # [5,6]
    print(f"Cyclic sort: {find_disappeared_numbers_cyclic_sort(nums1.copy())}")  # [5,6]
    print(f"Set approach: {find_disappeared_numbers_set(nums1)}")          # [5,6]
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Array range [1,n] enables using indices as hash map
    - Negative marking technique is space-efficient
    - Cyclic sort pattern: place number i at index i-1
    
    **⚡ Interview Tips:**
    
    - Ask: "Can I modify the input array?" (Usually yes)
    - Recognize: "Numbers in range [1,n]" suggests cyclic sort
    - Alternative: "Use array as hash map" technique
    
    **🔍 Cyclic Sort Pattern:**
    
    - Find all duplicates in array
    - Find missing positive integer
    - First missing positive in unsorted array

---

## Problem 13: Majority Element

**Difficulty:** Easy  
**Pattern:** Boyer-Moore Voting Algorithm  
**Time:** O(n) | **Space:** O(1)

=== "Problem Statement"

    Given an array `nums` of size n, return the majority element. The majority element appears more than ⌊n/2⌋ times.

    **Example:**
    ```text
    Input: nums = [3,2,3]
    Output: 3
    ```

=== "Solution"

    ```python
    def majority_element_boyer_moore(nums):
        """
        Boyer-Moore Voting Algorithm - optimal solution.
        
        Intuition: majority element will survive the "voting battle"
        """
        candidate = None
        count = 0
        
        # Phase 1: Find candidate
        for num in nums:
            if count == 0:
                candidate = num
            
            count += 1 if num == candidate else -1
        
        # Phase 2: Verify (not needed if majority guaranteed)
        # count = sum(1 for num in nums if num == candidate)
        # return candidate if count > len(nums) // 2 else None
        
        return candidate

    def majority_element_hash_map(nums):
        """
        Hash map approach - count frequencies.
        """
        from collections import Counter
        counts = Counter(nums)
        return max(counts.keys(), key=counts.get)

    def majority_element_sorting(nums):
        """
        Sorting approach - majority element will be at middle.
        """
        nums.sort()
        return nums[len(nums) // 2]

    def majority_element_divide_conquer(nums):
        """
        Divide and conquer approach.
        """
        def majority_rec(left, right):
            # Base case
            if left == right:
                return nums[left]
            
            # Divide
            mid = (left + right) // 2
            left_majority = majority_rec(left, mid)
            right_majority = majority_rec(mid + 1, right)
            
            # Conquer
            if left_majority == right_majority:
                return left_majority
            
            # Count occurrences in current range
            left_count = sum(1 for i in range(left, right + 1) if nums[i] == left_majority)
            right_count = sum(1 for i in range(left, right + 1) if nums[i] == right_majority)
            
            return left_majority if left_count > right_count else right_majority
        
        return majority_rec(0, len(nums) - 1)

    # Test
    nums = [2,2,1,1,1,2,2]
    print(f"Boyer-Moore: {majority_element_boyer_moore(nums)}")      # 2
    print(f"Hash map: {majority_element_hash_map(nums)}")            # 2
    print(f"Sorting: {majority_element_sorting(nums.copy())}")       # 2
    print(f"Divide & Conquer: {majority_element_divide_conquer(nums)}")  # 2
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Boyer-Moore voting simulates a "battle" between elements
    - Majority element (>n/2) will always survive the voting process
    - Works because majority element appears more than half the time
    
    **⚡ Interview Tips:**
    
    - Ask: "Is majority element guaranteed?" (Usually yes for this problem)
    - Understand: "Why does Boyer-Moore work?" (Critical insight)
    - Alternative: "What if no majority exists?" (Need verification phase)
    
    **🔍 Voting Algorithm Pattern:**
    
    - Majority Element II: find elements appearing >n/3 times
    - Find celebrity problem (graph variation)
    - Stream processing for majority element

---

## Problem 14: Running Sum of 1D Array

**Difficulty:** Easy  
**Pattern:** Prefix Sum  
**Time:** O(n) | **Space:** O(1)

=== "Problem Statement"

    Given an array `nums`, return the running sum where `runningSum[i] = sum(nums[0]…nums[i])`.

    **Example:**
    ```text
    Input: nums = [1,2,3,4]
    Output: [1,3,6,10]
    Explanation: [1, 1+2, 1+2+3, 1+2+3+4]
    ```

=== "Solution"

    ```python
    def running_sum_in_place(nums):
        """
        In-place modification - most space efficient.
        """
        for i in range(1, len(nums)):
            nums[i] += nums[i-1]
        return nums

    def running_sum_new_array(nums):
        """
        Create new array - preserves original.
        """
        result = [nums[0]]
        for i in range(1, len(nums)):
            result.append(result[-1] + nums[i])
        return result

    def running_sum_itertools(nums):
        """
        Using itertools.accumulate - Pythonic.
        """
        import itertools
        return list(itertools.accumulate(nums))

    def running_sum_manual_accumulate(nums):
        """
        Manual accumulation with running total.
        """
        result = []
        running_total = 0
        
        for num in nums:
            running_total += num
            result.append(running_total)
        
        return result

    # Test
    nums = [1,2,3,4]
    print(f"In-place: {running_sum_in_place(nums.copy())}")          # [1,3,6,10]
    print(f"New array: {running_sum_new_array(nums)}")               # [1,3,6,10]
    print(f"Itertools: {running_sum_itertools(nums)}")              # [1,3,6,10]
    print(f"Manual: {running_sum_manual_accumulate(nums)}")          # [1,3,6,10]
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Prefix sums are fundamental for range query problems
    - Running sum is the simplest form of prefix computation
    - Can be computed in-place or with new array
    
    **⚡ Interview Tips:**
    
    - Ask: "Can I modify the input?" (Affects space complexity)
    - Recognize: "Range sum queries" benefit from prefix sums
    - Extension: "2D prefix sums for matrix range queries"
    
    **🔍 Prefix Sum Applications:**
    
    - Subarray sum equals K
    - Range sum query (immutable/mutable)
    - Product of array except self
    - Maximum size subarray sum equals k

---

## Problem 15: Shuffle the Array

**Difficulty:** Easy  
**Pattern:** Array Manipulation  
**Time:** O(n) | **Space:** O(n)

=== "Problem Statement"

    Given an array `nums` of 2n elements in the form `[x1,x2,...,xn,y1,y2,...,yn]`, return the array in the form `[x1,y1,x2,y2,...,xn,yn]`.

    **Example:**
    ```text
    Input: nums = [2,5,1,3,4,7], n = 3
    Output: [2,3,5,4,1,7]
    Explanation: x1=2, x2=5, x3=1, y1=3, y2=4, y3=7
    ```

=== "Solution"

    ```python
    def shuffle_array_new_array(nums, n):
        """
        Create new array with interleaved elements.
        """
        result = []
        
        for i in range(n):
            result.append(nums[i])      # xi
            result.append(nums[i + n])  # yi
        
        return result

    def shuffle_array_list_comprehension(nums, n):
        """
        Pythonic list comprehension approach.
        """
        return [nums[i // 2 + (i % 2) * n] for i in range(2 * n)]

    def shuffle_array_zip(nums, n):
        """
        Using zip to interleave two halves.
        """
        first_half = nums[:n]
        second_half = nums[n:]
        
        result = []
        for x, y in zip(first_half, second_half):
            result.extend([x, y])
        
        return result

    def shuffle_array_zip_flatten(nums, n):
        """
        Zip with itertools.chain for flattening.
        """
        import itertools
        pairs = zip(nums[:n], nums[n:])
        return list(itertools.chain.from_iterable(pairs))

    def shuffle_array_in_place_encoding(nums, n):
        """
        In-place solution using number encoding (advanced).
        
        Encode two numbers in one using: new = (b * 1001) + a
        Extract: a = new % 1001, b = new // 1001
        """
        # Assuming nums[i] <= 1000 (constraint from problem)
        for i in range(n):
            # Store both nums[i] and nums[i+n] in nums[i]
            nums[i] = nums[i] + (nums[i + n] * 1001)
        
        j = 2 * n - 1
        for i in range(n - 1, -1, -1):
            # Extract the second number (y)
            y = nums[i] // 1001
            # Extract the first number (x)
            x = nums[i] % 1001
            
            nums[j] = y
            nums[j - 1] = x
            j -= 2
        
        return nums

    # Test
    nums = [2,5,1,3,4,7]
    n = 3
    
    print(f"New array: {shuffle_array_new_array(nums, n)}")                    # [2,3,5,4,1,7]
    print(f"List comprehension: {shuffle_array_list_comprehension(nums, n)}")  # [2,3,5,4,1,7]
    print(f"Zip: {shuffle_array_zip(nums, n)}")                              # [2,3,5,4,1,7]
    print(f"Zip flatten: {shuffle_array_zip_flatten(nums, n)}")               # [2,3,5,4,1,7]
    print(f"In-place: {shuffle_array_in_place_encoding(nums.copy(), n)}")     # [2,3,5,4,1,7]
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Array interleaving pattern: alternate between two halves
    - Index mapping: result[2×i] = nums[i], result[2×i+1] = nums[i+n]
    - In-place solution possible with number encoding (advanced)
    
    **⚡ Interview Tips:**
    
    - Ask: "What's the constraint on array values?" (Affects encoding approach)
    - Recognize: "Interleaving pattern" in many array problems
    - Discuss: "Space-time tradeoffs" between approaches
    
    **🔍 Array Manipulation Patterns:**
    
    - Rotate array (cyclic shifting)
    - Rearrange array by signs (positive/negative)
    - Sort array by parity (even/odd positioning)
