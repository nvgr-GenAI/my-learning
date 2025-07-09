# Sorting - Easy Problems

A collection of easy sorting problems to build your algorithmic problem-solving skills.

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | [Sort Array by Parity](#sort-array-by-parity) | Two Pointers | Easy | O(n) | O(1) |
    | 2 | [Sort Colors](#sort-colors) | Counting Sort | Easy | O(n) | O(1) |
    | 3 | [Merge Sorted Array](#merge-sorted-array) | Two Pointers | Easy | O(m+n) | O(1) |
    | 4 | [Contains Duplicate](#contains-duplicate) | Sorting | Easy | O(n log n) | O(1) |
    | 5 | [Valid Anagram](#valid-anagram) | Sorting/Counting | Easy | O(n log n) | O(1) |
    | 6 | [Relative Sort Array](#relative-sort-array) | Counting Sort | Easy | O(n+m) | O(k) |
    | 7 | [Height Checker](#height-checker) | Counting Sort | Easy | O(n) | O(1) |
    | 8 | [Maximum Product of Three Numbers](#maximum-product-of-three-numbers) | Sorting | Easy | O(n log n) | O(1) |
    | 9 | [Rank Transform of an Array](#rank-transform-of-an-array) | Sorting + Mapping | Easy | O(n log n) | O(n) |
    | 10 | [Minimum Absolute Difference](#minimum-absolute-difference) | Sorting | Easy | O(n log n) | O(1) |
    | 11 | [Intersection of Two Arrays](#intersection-of-two-arrays) | Sorting/Set | Easy | O(n log n) | O(n) |
    | 12 | [Majority Element](#majority-element) | Sorting/Counting | Easy | O(n log n) | O(1) |
    | 13 | [Find All Numbers Disappeared in an Array](#find-all-numbers-disappeared-in-an-array) | Cyclic Sort | Easy | O(n) | O(1) |
    | 14 | [Assign Cookies](#assign-cookies) | Greedy + Sorting | Easy | O(n log n) | O(1) |
    | 15 | [Squares of a Sorted Array](#squares-of-a-sorted-array) | Two Pointers | Easy | O(n) | O(n) |

## Sort Array by Parity

=== "🔍 Problem Statement"

    Given an integer array `nums`, move all even integers at the beginning of the array followed by all odd integers. Return any array that satisfies this condition.
    
    **Example:**
    ```
    Input: nums = [3,1,2,4]
    Output: [2,4,3,1]
    Explanation: Outputs [4,2,3,1], [2,4,1,3], and [4,2,1,3] would also be accepted.
    ```
    
    **Constraints:**
    - 1 <= nums.length <= 5000
    - 0 <= nums[i] <= 5000

=== "💡 Solution Approach"

    **Key Insight:** 
    
    This problem is a perfect candidate for a two-pointer approach. We can maintain two pointers, one from the beginning and one from the end of the array, and swap elements when we find an odd element at the beginning and an even element at the end.
    
    **Step-by-step:**
    
    1. Initialize two pointers: `left` at the start and `right` at the end of the array
    2. While `left` < `right`:
       - Move `left` forward if it points to an even number
       - Move `right` backward if it points to an odd number
       - If `left` points to an odd number and `right` points to an even number, swap them
    3. Return the modified array
    
    **Why it works:**
    
    This approach ensures that all even numbers come before odd numbers. We don't need to maintain the relative order within the even or odd groups, so we can swap from opposite ends of the array.

=== "💻 Implementation"

    ```python
    def sortArrayByParity(nums):
        """
        Time: O(n) - we process each element at most once
        Space: O(1) - in-place modification
        """
        i, j = 0, len(nums) - 1
        
        while i < j:
            # Find odd number from left
            while i < j and nums[i] % 2 == 0:
                i += 1
            
            # Find even number from right
            while i < j and nums[j] % 2 == 1:
                j -= 1
            
            # Swap them
            if i < j:
                nums[i], nums[j] = nums[j], nums[i]
        
        return nums
    ```

=== "🔄 Alternative Approaches"

    **Approach 1: Two-Pass Solution**
    
    Create two lists (even and odd) and concatenate them:
    
    ```python
    def sortArrayByParity(nums):
        """
        Time: O(n)
        Space: O(n) for the new array
        """
        return [x for x in nums if x % 2 == 0] + [x for x in nums if x % 2 == 1]
    ```
    
    **Approach 2: Using Built-in Sort**
    
    Use Python's built-in sort with a custom key function:
    
    ```python
    def sortArrayByParity(nums):
        """
        Time: O(n log n)
        Space: O(1) to O(n) depending on sort implementation
        """
        return sorted(nums, key=lambda x: x % 2)
    ```
    
    **Approach 3: Partition (similar to Quicksort partition)**
    
    ```python
    def sortArrayByParity(nums):
        """
        Time: O(n)
        Space: O(1)
        """
        j = 0  # Position to put the next even number
        
        for i in range(len(nums)):
            if nums[i] % 2 == 0:  # If current number is even
                nums[i], nums[j] = nums[j], nums[i]  # Swap with position j
                j += 1  # Increment j to mark the position for next even number
        
        return nums
    ```

=== "💭 Tips & Insights"

    **Interview Tips:**
    
    1. **Clarify First:** Confirm that the relative order within even and odd groups doesn't matter. This allows for more efficient solutions.
    
    2. **Optimization:** The two-pointer approach is more efficient than sorting since it's O(n) instead of O(n log n).
    
    3. **Similar Problems:** This partitioning technique is similar to:
       - Dutch National Flag problem (Sort Colors)
       - Partition in Quicksort
       - Separating positives and negatives
    
    4. **Follow-up Questions:**
       - How would you modify the solution to keep the relative order of elements?
       - What if we want to separate the array into three groups (e.g., divisible by 3, remainder 1, remainder 2)?
    
    **Common Mistakes:**
    
    - Forgetting to check pointer bounds (`i < j`) in the inner while loops
    - Incorrect parity check logic
    - Not handling edge cases (empty array, single element)
    
    **Real-world Applications:**
    
    This partitioning technique is used in:
    - Data preprocessing
    - Organizing records by categorical attributes
    - Image segmentation algorithms

## Sort Colors



=== "🔍 Problem Statement"

    Given an array `nums` with `n` objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue. We will use the integers 0, 1, and 2 to represent the colors red, white, and blue, respectively.
    
    **Example:**
    ```
    Input: nums = [2,0,2,1,1,0]
    Output: [0,0,1,1,2,2]
    ```
    
    **Constraints:**
    - n == nums.length
    - 1 <= n <= 300
    - nums[i] is either 0, 1, or 2

=== "💡 Solution Approach"

    **Key Insight:**
    
    This is the famous "Dutch National Flag" problem introduced by Edsger W. Dijkstra. The goal is to sort an array containing only three distinct values, which can be done in O(n) time with a single pass.
    
    **Step-by-step:**
    
    1. Use three pointers: `low`, `mid`, and `high`
    2. `low` marks the boundary for 0s (all elements before `low` are 0s)
    3. `mid` scans the array and processes unclassified elements
    4. `high` marks the boundary for 2s (all elements after `high` are 2s)
    5. Process elements based on their value:
       - If element is 0: swap with `low`, increment both `low` and `mid`
       - If element is 1: leave it and increment `mid`
       - If element is 2: swap with `high`, decrement `high` (don't increment `mid` yet)
    
    **Why it works:**
    
    The algorithm maintains three regions: [0, low) for 0s, [low, mid) for 1s, and (high, n-1] for 2s. The region [mid, high] contains unclassified elements that we still need to process.

=== "💻 Implementation"

    ```python
    def sortColors(nums):
        """
        Time: O(n) - one-pass solution
        Space: O(1) - in-place sorting
        """
        # Initialize pointers
        low, mid, high = 0, 0, len(nums) - 1
        
        while mid <= high:
            if nums[mid] == 0:
                # Swap with low pointer and increment both
                nums[low], nums[mid] = nums[mid], nums[low]
                low += 1
                mid += 1
            elif nums[mid] == 1:
                # Just increment mid pointer for white (1)
                mid += 1
            else:  # nums[mid] == 2
                # Swap with high pointer and decrement high
                nums[mid], nums[high] = nums[high], nums[mid]
                high -= 1
                # Note: mid is not incremented here as we need to check the swapped value
    ```

=== "🔄 Alternative Approaches"

    **Approach 1: Counting Sort**
    
    This is a two-pass solution that first counts occurrences and then reconstructs the array:
    
    ```python
    def sortColors(nums):
        """
        Time: O(n) - two-pass solution
        Space: O(1) - constant extra space
        """
        # Count occurrences of each color
        counts = [0, 0, 0]
        for num in nums:
            counts[num] += 1
        
        # Overwrite array with sorted colors
        i = 0
        for color in range(3):
            for _ in range(counts[color]):
                nums[i] = color
                i += 1
    ```
    
    **Approach 2: Library Sort**
    
    While not meeting the one-pass requirement, this is a simple solution:
    
    ```python
    def sortColors(nums):
        """
        Time: O(n log n)
        Space: O(1) or O(n) depending on sort implementation
        """
        nums.sort()
    ```

=== "💭 Tips & Insights"

    **Interview Tips:**
    
    1. **Clarify the Requirements:** Ask if a one-pass solution is expected, as this is a common follow-up.
    
    2. **Recognize the Pattern:** This is a well-known problem based on the Dutch National Flag algorithm. Mentioning this shows your knowledge of classic algorithms.
    
    3. **Test Your Solution:** Be sure to trace through the execution with different test cases including edge cases like `[2,0,1]` where all swaps are needed.
    
    4. **Similar Problems:** The partitioning technique is used in:
       - QuickSort algorithm
       - Sort Array By Parity
       - Partition List (in linked lists)
    
    **Common Mistakes:**
    
    - Forgetting to check the swapped value after a high pointer swap
    - Incorrectly incrementing/decrementing pointers
    - Not handling the case when `mid` and `low` or `high` are the same
    
    **Real-world Applications:**
    
    - Image processing where pixels need to be sorted by color
    - Database partitioning by categorical values
    - Organizing data with a small fixed number of categories

=== "� Problem Statement"

    You are given two integer arrays `nums1` and `nums2`, sorted in non-decreasing order, and two integers `m` and `n`, representing the number of elements in `nums1` and `nums2` respectively. Merge `nums1` and `nums2` into a single array sorted in non-decreasing order.
    
    The final sorted array should be stored inside `nums1`. To accommodate this, `nums1` has a length of `m + n`, where the first `m` elements denote the elements that should be merged, and the last `n` elements are set to 0 and should be ignored.
    
    **Example:**
    ```
    Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
    Output: [1,2,2,3,5,6]
    ```
    
    **Constraints:**
    - nums1.length == m + n
    - nums2.length == n
    - 0 <= m, n <= 200
    - 1 <= m + n <= 200
    - -10^9 <= nums1[i], nums2[j] <= 10^9

=== "💡 Solution Approach"

    **Key Insight:**
    
    Since we need to merge in-place without using extra space, the trick is to work from the end of the arrays rather than from the beginning. This way, we can avoid overwriting elements in `nums1` that we still need to consider.
    
    **Step-by-step:**
    
    1. Initialize three pointers:
       - `p1` pointing to the last valid element in `nums1` (at index `m-1`)
       - `p2` pointing to the last element in `nums2` (at index `n-1`)
       - `p` pointing to the last position in the merged array (at index `m+n-1`)
    
    2. Compare elements at `p1` and `p2`, place the larger one at position `p`, and move the corresponding pointer backward.
    
    3. Continue until we've processed all elements from both arrays.
    
    4. If there are remaining elements in `nums2`, copy them to the beginning of `nums1`.
    
    **Why it works:**
    
    By starting from the end, we ensure that we always place elements in positions that have either already been processed or contain zeros, avoiding any data loss.

=== "💻 Implementation"

    ```python
    def merge(nums1, m, nums2, n):
        """
        Time: O(m + n) - we process each element once
        Space: O(1) - in-place merge
        """
        # Initialize pointers to the end of both arrays
        p1 = m - 1  # Last element in nums1
        p2 = n - 1  # Last element in nums2
        p = m + n - 1  # Last position in nums1
        
        # While there are elements in both arrays
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]
                p1 -= 1
            else:
                nums1[p] = nums2[p2]
                p2 -= 1
            p -= 1
        
        # If there are remaining elements in nums2
        # (no need to handle remaining elements in nums1 as they're already in place)
        while p2 >= 0:
            nums1[p] = nums2[p2]
            p2 -= 1
            p -= 1
    ```

=== "🔄 Alternative Approaches"

    **Approach 1: Using Extra Space**
    
    A simpler but less space-efficient approach:
    
    ```python
    def merge(nums1, m, nums2, n):
        """
        Time: O(m + n)
        Space: O(m) - extra space for copying nums1
        """
        # Make a copy of the first m elements of nums1
        nums1_copy = nums1[:m]
        
        # Pointers for nums1_copy and nums2
        p1, p2 = 0, 0
        
        # Pointer for the current position in nums1
        p = 0
        
        # Compare elements from nums1_copy and nums2 and add the smallest to nums1
        while p1 < m and p2 < n:
            if nums1_copy[p1] <= nums2[p2]:
                nums1[p] = nums1_copy[p1]
                p1 += 1
            else:
                nums1[p] = nums2[p2]
                p2 += 1
            p += 1
            
        # If there are remaining elements in nums1_copy or nums2, add them
        if p1 < m:
            nums1[p:p+m-p1] = nums1_copy[p1:m]
        if p2 < n:
            nums1[p:p+n-p2] = nums2[p2:n]
    ```
    
    **Approach 2: Sort After Merging**
    
    A naive but simple solution:
    
    ```python
    def merge(nums1, m, nums2, n):
        """
        Time: O((m+n)log(m+n)) due to sorting
        Space: O(1) or O(m+n) depending on sort implementation
        """
        # Copy nums2 elements to the end of nums1
        for i in range(n):
            nums1[m + i] = nums2[i]
            
        # Sort the entire array
        nums1.sort()
    ```

=== "💭 Tips & Insights"

    **Interview Tips:**
    
    1. **Recognize the Pattern:** This is a variation of the classic "merge" operation in the Merge Sort algorithm, but with the constraint of doing it in-place.
    
    2. **Optimize the Solution:** The backwards approach is more efficient than using extra space or the sort-after-merge approach.
    
    3. **Edge Cases to Consider:**
       - When `m = 0` (nums1 is empty)
       - When `n = 0` (nums2 is empty)
       - When all elements in one array are smaller than all elements in the other
    
    4. **Follow-up Questions:**
       - How would you modify this to merge k sorted arrays?
       - Can you implement this using an iterative merge sort approach?
    
    **Common Mistakes:**
    
    - Starting from the beginning and overwriting elements in `nums1` that are still needed
    - Forgetting to handle remaining elements in `nums2`
    - Incorrectly handling edge cases like empty arrays
    
    **Real-world Applications:**
    
    - Database operations that merge sorted records
    - External sorting algorithms where chunks of data are sorted and merged
    - Implementation of merge sort, which is stable and efficient for certain types of data

=== "🔍 Problem Statement"

    Given an integer array `nums`, return `true` if any value appears at least twice in the array, and return `false` if every element is distinct.
    
    **Example 1:**
    ```
    Input: nums = [1,2,3,1]
    Output: true
    ```
    
    **Example 2:**
    ```
    Input: nums = [1,2,3,4]
    Output: false
    ```
    
    **Constraints:**
    - 1 <= nums.length <= 10^5
    - -10^9 <= nums[i] <= 10^9

=== "💡 Solution Approach"

    **Key Insight:**
    
    There are two main approaches to solve this problem:
    1. Sort the array and check adjacent elements for duplicates
    2. Use a hash set to track seen elements
    
    While the sorting approach is intuitive, the hash set approach is more efficient in terms of time complexity.
    
    **Step-by-step for the Sorting Approach:**
    
    1. Sort the array in ascending order.
    2. Iterate through the array and check if any adjacent elements are equal.
    3. If a duplicate is found, return true; otherwise, return false.
    
    **Step-by-step for the Hash Set Approach:**
    
    1. Create an empty hash set to track seen elements.
    2. Iterate through the array.
    3. For each element, check if it's already in the hash set.
    4. If yes, return true; otherwise, add it to the set and continue.
    5. If the loop completes, return false.

=== "💻 Implementation"

    **Sorting Approach:**
    ```python
    def containsDuplicate(nums):
        """
        Time: O(n log n) for sorting
        Space: O(1) or O(n) depending on sorting implementation
        """
        # Sort the array
        nums.sort()
        
        # Check for adjacent duplicates
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                return True
        
        return False
    ```
    
    **Hash Set Approach:**
    ```python
    def containsDuplicate(nums):
        """
        Time: O(n) - we process each element once
        Space: O(n) - in worst case we store all elements in the set
        """
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False
    ```

=== "🔄 Alternative Approaches"

    **Approach: Brute Force**
    
    Check every pair of elements in the array (not recommended due to O(n²) time complexity):
    
    ```python
    def containsDuplicate(nums):
        """
        Time: O(n²) - comparing each element with all others
        Space: O(1) - no extra space needed
        """
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                if nums[i] == nums[j]:
                    return True
        return False
    ```
    
    **Approach: Using Counter**
    
    Count occurrences of each element and check if any count exceeds 1:
    
    ```python
    from collections import Counter
    
    def containsDuplicate(nums):
        """
        Time: O(n)
        Space: O(n)
        """
        counter = Counter(nums)
        for count in counter.values():
            if count > 1:
                return True
        return False
        
        # One-liner alternative:
        # return any(count > 1 for count in Counter(nums).values())
    ```

=== "💭 Tips & Insights"

    **Interview Tips:**
    
    1. **Time-Space Tradeoff:** This problem illustrates the classic tradeoff between time and space complexity. The hash set approach is O(n) time but requires O(n) space, while the sorting approach is O(n log n) time but can be done with O(1) extra space.
    
    2. **Optimal Approach:** For most practical scenarios, the hash set approach is preferred due to its linear time complexity.
    
    3. **Similar Problems:**
       - Find all duplicates in an array
       - Find the duplicate number (when exactly one duplicate)
       - Check if a string has all unique characters
    
    4. **Follow-up Questions:**
       - How would you modify your solution if you need to find all duplicates?
       - What if the array is already sorted?
       - What if memory is a constraint and we can't use extra space?
    
    **Common Mistakes:**
    
    - Not considering the case where the array is empty
    - Using a complex data structure when a simple set is sufficient
    - Overlooking the possibility of negative numbers in the array
    
    **Real-world Applications:**
    
    - Checking for duplicate entries in databases
    - Validating unique identifiers in a system
    - Error detection in data transmission

=== "� Problem Statement"

    Given two strings `s` and `t`, return `true` if `t` is an anagram of `s`, and `false` otherwise. 
    
    An anagram is a word formed by rearranging the letters of another word, using all the original letters exactly once.
    
    **Example 1:**
    ```
    Input: s = "anagram", t = "nagaram"
    Output: true
    ```
    
    **Example 2:**
    ```
    Input: s = "rat", t = "car"
    Output: false
    ```
    
    **Constraints:**
    - 1 <= s.length, t.length <= 5 * 10^4
    - s and t consist of lowercase English letters

=== "💡 Solution Approach"

    **Key Insight:**
    
    Two strings are anagrams if and only if they have the same characters with the same frequencies. This leads to two main approaches:
    
    1. Sort both strings and check if they're equal
    2. Count character frequencies and compare
    
    **Step-by-step for Sorting Approach:**
    
    1. If the lengths of the two strings are different, they can't be anagrams.
    2. Sort both strings.
    3. Check if the sorted strings are identical.
    
    **Step-by-step for Character Count Approach:**
    
    1. If the lengths of the two strings are different, they can't be anagrams.
    2. Create a counter array/map to track character frequencies.
    3. Increment counts for characters in string `s` and decrement for characters in string `t`.
    4. If all counts are zero at the end, the strings are anagrams.

=== "💻 Implementation"

    **Sorting Approach:**
    ```python
    def isAnagram(s, t):
        """
        Time: O(n log n) for sorting
        Space: O(n) for sorted strings
        """
        if len(s) != len(t):
            return False
        
        return sorted(s) == sorted(t)
    ```
    
    **Character Count Approach:**
    ```python
    def isAnagram(s, t):
        """
        Time: O(n) - we process each character once
        Space: O(1) - constant space as we only have 26 lowercase letters
        """
        if len(s) != len(t):
            return False
        
        # Count characters in both strings
        char_count = [0] * 26
        
        for i in range(len(s)):
            char_count[ord(s[i]) - ord('a')] += 1
            char_count[ord(t[i]) - ord('a')] -= 1
        
        # If all counts are zero, strings are anagrams
        for count in char_count:
            if count != 0:
                return False
        
        return True
    ```

=== "🔄 Alternative Approaches"

    **Approach: Using Hash Map**
    
    More flexible approach that works for any character set (not just lowercase letters):
    
    ```python
    def isAnagram(s, t):
        """
        Time: O(n)
        Space: O(k) where k is the size of the character set
        """
        if len(s) != len(t):
            return False
            
        char_count = {}
        
        # Count characters in s
        for char in s:
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1
                
        # Decrement counts for characters in t
        for char in t:
            if char not in char_count or char_count[char] == 0:
                return False
            char_count[char] -= 1
            
        return True
    ```
    
    **Approach: Using Counter**
    
    A concise approach using Python's Counter class:
    
    ```python
    from collections import Counter
    
    def isAnagram(s, t):
        """
        Time: O(n)
        Space: O(k) where k is the size of the character set
        """
        return Counter(s) == Counter(t)
    ```

=== "💭 Tips & Insights"

    **Interview Tips:**
    
    1. **Early Termination:** Always check if the lengths of the two strings are equal first, as strings of different lengths cannot be anagrams.
    
    2. **Time-Space Tradeoff:** The character count approach is more efficient (O(n) vs O(n log n)) than sorting, especially for long strings.
    
    3. **Character Set Assumptions:** If the input is restricted to lowercase letters, you can use a fixed-size array of 26 elements for constant space. For Unicode strings, a hash map is more appropriate.
    
    4. **Follow-up Questions:**
       - How would you modify your solution to check for case-insensitive anagrams?
       - What if we need to check if a string is an anagram of any permutation of another string?
       - How would you handle Unicode characters or very large strings?
    
    **Common Mistakes:**
    
    - Not checking string lengths at the beginning
    - Incorrect character mapping (e.g., off-by-one errors when mapping to array indices)
    - Using sorted() without considering its space complexity
    
    **Real-world Applications:**
    
    - Word games and puzzles
    - Spell checking and autocorrect features
    - Cryptography and code breaking
    - DNA sequence analysis (checking for permutations of certain genetic sequences)

=== "� Problem Statement"

    Given two arrays `arr1` and `arr2`, the elements of `arr2` are distinct, and all elements in `arr2` are also in `arr1`. Sort the elements of `arr1` such that the relative ordering of items in `arr1` are the same as in `arr2`. Elements that do not appear in `arr2` should be placed at the end of `arr1` in ascending order.
    
    **Example:**
    ```
    Input: arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]
    Output: [2,2,2,1,4,3,3,9,6,7,19]
    ```
    
    **Constraints:**
    - 1 <= arr1.length, arr2.length <= 1000
    - 0 <= arr1[i], arr2[i] <= 1000
    - All elements in arr2 are distinct
    - Each element in arr2 is also in arr1

=== "💡 Solution Approach"

    **Key Insight:**
    
    This problem requires a custom sorting order based on a reference array. We need to:
    1. Follow the order specified in arr2 for elements that appear in it
    2. Sort the remaining elements in ascending order
    
    This can be achieved using either a custom sort function or a counting sort approach.
    
    **Step-by-step for Custom Sort Approach:**
    
    1. Create a map that assigns each element in arr2 its position (rank).
    2. Define a custom sort key function that:
       - Returns the rank from the map if the element is in arr2
       - Returns a value larger than any rank plus the element value otherwise
    3. Sort arr1 using this custom key function.
    
    **Step-by-step for Counting Sort Approach:**
    
    1. Create a count array to track occurrences of each element in arr1.
    2. Iterate through arr2 and add each element to the result the number of times it appears in arr1.
    3. Iterate through the count array and add any remaining elements to the result.

=== "💻 Implementation"

    **Custom Sort Approach:**
    ```python
    def relativeSortArray(arr1, arr2):
        """
        Time: O(n log n) where n is the length of arr1
        Space: O(n) for the rank dictionary
        """
        # Create a mapping of value to position in arr2
        rank = {x: i for i, x in enumerate(arr2)}
        
        # Custom sort key:
        # - If element is in arr2, sort by its position in arr2
        # - If not, sort by the value itself but with an offset
        return sorted(arr1, key=lambda x: rank.get(x, len(arr2) + x))
    ```
    
    **Counting Sort Approach:**
    ```python
    def relativeSortArray(arr1, arr2):
        """
        Time: O(n + k) where n is the length of arr1 and k is the max value in arr1
        Space: O(k) for the counting array
        """
        # Find the maximum value for count array size
        max_val = max(arr1)
        
        # Count occurrences of each number in arr1
        count = [0] * (max_val + 1)
        for num in arr1:
            count[num] += 1
        
        # First add all elements that appear in arr2
        result = []
        for num in arr2:
            result.extend([num] * count[num])
            count[num] = 0  # Reset count to avoid duplicates
        
        # Then add remaining elements in ascending order
        for num in range(max_val + 1):
            if count[num] > 0:
                result.extend([num] * count[num])
        
        return result
    ```

=== "🔄 Alternative Approaches"

    **Approach: Using Dictionary and Two-Part Sort**
    
    This approach separates elements into two groups and then combines them:
    
    ```python
    def relativeSortArray(arr1, arr2):
        """
        Time: O(n log n)
        Space: O(n)
        """
        # Count occurrences of each element in arr1
        count = {}
        for num in arr1:
            count[num] = count.get(num, 0) + 1
        
        # Part 1: Elements that appear in arr2 (in order)
        result = []
        for num in arr2:
            result.extend([num] * count[num])
            del count[num]  # Remove to avoid duplicates
        
        # Part 2: Remaining elements sorted
        remaining = []
        for num in count:
            remaining.extend([num] * count[num])
        remaining.sort()
        
        # Combine both parts
        return result + remaining
    ```
    
    **Approach: Using Buckets**
    
    A variation that uses buckets instead of a counting array:
    
    ```python
    def relativeSortArray(arr1, arr2):
        """
        Time: O(n + m) where n is length of arr1 and m is max value in arr1
        Space: O(n + m)
        """
        # Create buckets for each value in arr1
        buckets = {}
        for num in arr1:
            if num in buckets:
                buckets[num].append(num)
            else:
                buckets[num] = [num]
        
        # Process elements in the order of arr2
        result = []
        for num in arr2:
            result.extend(buckets[num])
            del buckets[num]
        
        # Add remaining elements in sorted order
        for num in sorted(buckets.keys()):
            result.extend(buckets[num])
        
        return result
    ```

=== "💭 Tips & Insights"

    **Interview Tips:**
    
    1. **Algorithm Choice:** If the range of values is small compared to the array size, counting sort is more efficient. For larger ranges, the custom sort approach is better.
    
    2. **Python-specific Optimization:** The `key` parameter in Python's `sorted()` function is a powerful way to implement custom sorting logic without modifying the original comparison function.
    
    3. **Space-Time Tradeoff:** Counting sort uses more space but can be faster for specific input ranges.
    
    4. **Similar Problems:**
       - Custom Sort String
       - Sort Array by Increasing Frequency
       - Sort Characters By Frequency
    
    5. **Follow-up Questions:**
       - How would you handle negative integers?
       - What if the range of values is very large?
       - How would you adapt this for strings or other data types?
    
    **Common Mistakes:**
    
    - Not handling elements that appear in arr1 but not in arr2
    - Incorrectly implementing the counting logic
    - Using inefficient data structures for large input ranges
    
    **Real-world Applications:**
    
    - Custom sorting in databases based on reference tables
    - Prioritizing data based on external rankings
    - Reorganizing inventory based on sales data

=== "� Problem Statement"

    A school is trying to take an annual photo of all the students. The students are asked to stand in a single file line in non-decreasing order by height. Return the minimum number of students that must move in order for all students to be standing in non-decreasing order of height.
    
    **Example 1:**
    ```
    Input: heights = [1,1,4,2,1,3]
    Output: 3
    Explanation: 
    Current array: [1,1,4,2,1,3]
    Expected array: [1,1,1,2,3,4]
    Students at positions 2, 4, and 5 need to move.
    ```
    
    **Example 2:**
    ```
    Input: heights = [5,1,2,3,4]
    Output: 5
    Explanation:
    Current array: [5,1,2,3,4]
    Expected array: [1,2,3,4,5]
    All students need to move.
    ```
    
    **Constraints:**
    - 1 <= heights.length <= 100
    - 1 <= heights[i] <= 100

=== "💡 Solution Approach"

    **Key Insight:**
    
    To find the minimum number of students that must move, we need to compare the current arrangement with the correct arrangement (sorted in non-decreasing order). Each position where the height doesn't match represents a student who needs to move.
    
    **Step-by-step:**
    
    1. Create a copy of the heights array and sort it to get the expected order.
    2. Compare the original array with the sorted array.
    3. Count the positions where the values differ.
    
    **Why it works:**
    
    The sorted array represents the ideal arrangement of students by height. Any student whose current position has a different height than the corresponding position in the sorted array must move to achieve the correct ordering.

=== "💻 Implementation"

    ```python
    def heightChecker(heights):
        """
        Time: O(n log n) for sorting
        Space: O(n) for the sorted array
        """
        # Create expected heights by sorting
        expected = sorted(heights)
        
        # Count positions where heights differ from expected
        count = 0
        for i in range(len(heights)):
            if heights[i] != expected[i]:
                count += 1
                
        return count
    ```

=== "🔄 Alternative Approaches"

    **Approach: Using Counting Sort**
    
    Since the problem states heights are between 1 and 100, we can use counting sort for better time complexity:
    
    ```python
    def heightChecker(heights):
        """
        Time: O(n + k) where k is the range of heights (100 in this case)
        Space: O(k) for the count array
        """
        # Create count array for heights from 1 to 100
        count = [0] * 101
        for height in heights:
            count[height] += 1
        
        # Rebuild the expected array using counts
        result = 0
        curr_height = 0
        
        for i in range(len(heights)):
            # Find the next expected height
            while count[curr_height] == 0:
                curr_height += 1
            
            # Check if current student is at correct height
            if curr_height != heights[i]:
                result += 1
            
            # Mark this height as used
            count[curr_height] -= 1
            
        return result
    ```
    
    **Approach: One-liner with List Comprehension**
    
    A concise but less optimized approach:
    
    ```python
    def heightChecker(heights):
        """
        Time: O(n log n)
        Space: O(n)
        """
        return sum(h1 != h2 for h1, h2 in zip(heights, sorted(heights)))
    ```

=== "💭 Tips & Insights"

    **Interview Tips:**
    
    1. **Problem Simplification:** This problem is essentially asking "how many elements are out of place compared to the sorted array?"
    
    2. **Optimization:** When the range of values is small and known (1-100 here), counting sort can improve time complexity from O(n log n) to O(n).
    
    3. **Edge Cases to Consider:**
       - All students already in correct order (return 0)
       - All students out of place (return the length of the array)
       - Arrays with repeated values
    
    4. **Follow-up Questions:**
       - How would you identify which specific students need to move?
       - Can you solve it with O(1) extra space?
       - What if we want to minimize the total distance moved instead?
    
    **Common Mistakes:**
    
    - Misunderstanding the problem as sorting the array in-place
    - Incorrectly calculating the positions that need to change
    - Using inefficient comparison methods
    
    **Real-world Applications:**
    
    - Measuring the "disorder" in a sequence (similar to Kendall tau distance)
    - Quality control in manufacturing where items should be ordered by size
    - Optimizing rearrangements in sorting networks

=== "� Problem Statement"

    Given an integer array `nums`, find three numbers whose product is maximum and return the maximum product.
    
    **Example 1:**
    ```
    Input: nums = [1,2,3]
    Output: 6
    Explanation: The maximum product is 1 × 2 × 3 = 6.
    ```
    
    **Example 2:**
    ```
    Input: nums = [1,2,3,4]
    Output: 24
    Explanation: The maximum product is 2 × 3 × 4 = 24.
    ```
    
    **Example 3:**
    ```
    Input: nums = [-1,-2,-3]
    Output: -6
    Explanation: The maximum product is -1 × -2 × -3 = -6.
    ```
    
    **Constraints:**
    - 3 <= nums.length <= 10^4
    - -1000 <= nums[i] <= 1000

=== "💡 Solution Approach"

    **Key Insight:**
    
    The maximum product of three numbers can come from either:
    1. The three largest numbers (if they're all positive or if there are less than two negative numbers)
    2. The two smallest numbers (which would be negative) and the largest number (as negatives multiply to become positive)
    
    **Step-by-step for Sorting Approach:**
    
    1. Sort the array in ascending order.
    2. Calculate two potential products:
       - Product of the three largest numbers (nums[-1] * nums[-2] * nums[-3])
       - Product of the two smallest numbers and the largest number (nums[0] * nums[1] * nums[-1])
    3. Return the maximum of these two products.
    
    **Step-by-step for Linear Scan Approach:**
    
    1. Find the two smallest numbers and the three largest numbers in a single pass.
    2. Calculate the two potential products as above.
    3. Return the maximum product.

=== "💻 Implementation"

    **Sorting Approach:**
    ```python
    def maximumProduct(nums):
        """
        Time: O(n log n) for sorting
        Space: O(1) excluding the sort operation
        """
        nums.sort()
        
        # Two cases:
        # 1. Three largest positive numbers
        # 2. Two smallest numbers (negative) and the largest number
        return max(nums[-1] * nums[-2] * nums[-3],
                  nums[0] * nums[1] * nums[-1])
    ```
    
    **Linear Scan Approach:**
    ```python
    def maximumProduct(nums):
        """
        Time: O(n) - single pass through the array
        Space: O(1) - constant extra space
        """
        # Initialize variables to track min and max values
        min1 = min2 = float('inf')  # Two smallest values
        max1 = max2 = max3 = float('-inf')  # Three largest values
        
        for num in nums:
            # Update minimums
            if num <= min1:
                min2 = min1
                min1 = num
            elif num <= min2:
                min2 = num
                
            # Update maximums
            if num >= max1:
                max3 = max2
                max2 = max1
                max1 = num
            elif num >= max2:
                max3 = max2
                max2 = num
            elif num >= max3:
                max3 = num
                
        return max(min1 * min2 * max1, max1 * max2 * max3)
    ```

=== "🔄 Alternative Approaches"

    **Approach: Using a Heap**
    
    This approach uses heaps to find the smallest and largest elements:
    
    ```python
    import heapq
    
    def maximumProduct(nums):
        """
        Time: O(n log k) where k is a small constant (2 and 3 in this case)
        Space: O(1) as we maintain heaps of constant size
        """
        # Initialize min heap for largest elements and max heap for smallest elements
        largest = []  # min heap
        smallest = []  # max heap
        
        for num in nums:
            # Keep track of 3 largest elements
            heapq.heappush(largest, num)
            if len(largest) > 3:
                heapq.heappop(largest)
                
            # Keep track of 2 smallest elements
            heapq.heappush(smallest, -num)  # Use negative for max heap
            if len(smallest) > 2:
                heapq.heappop(smallest)
        
        # Get the actual values
        top3 = [heapq.heappop(largest) for _ in range(3)]
        product1 = top3[0] * top3[1] * top3[2]
        
        bottom2 = [-heapq.heappop(smallest) for _ in range(2)]
        product2 = bottom2[0] * bottom2[1] * top3[2]  # Use largest element
        
        return max(product1, product2)
    ```
    
    **Approach: Using Partial Sort**
    
    If the language supports partial sorting, it can be more efficient:
    
    ```python
    def maximumProduct(nums):
        """
        Time: O(n) for partial sorting of fixed elements
        Space: O(1)
        """
        # Get 3 largest and 2 smallest elements
        largest = sorted(nums, reverse=True)[:3]
        smallest = sorted(nums)[:2]
        
        # Calculate the two potential products
        product1 = largest[0] * largest[1] * largest[2]
        product2 = smallest[0] * smallest[1] * largest[0]
        
        return max(product1, product2)
    ```

=== "💭 Tips & Insights"

    **Interview Tips:**
    
    1. **Edge Cases:** Be careful with arrays containing all negative numbers or mixed positive/negative numbers.
    
    2. **Optimization:** The linear scan approach is optimal with O(n) time complexity and O(1) space complexity.
    
    3. **Intuition Development:** To build intuition, consider:
       - With all positive numbers, always pick the three largest
       - With all negative numbers, pick the three largest (least negative)
       - With mixed numbers, either pick three largest positives or two most negative and one largest positive
    
    4. **Follow-up Questions:**
       - What if we need to find the maximum product of k numbers instead of just 3?
       - How would you handle integer overflow in your solution?
       - Can you optimize the solution further if the array is already sorted?
    
    **Common Mistakes:**
    
    - Only considering the three largest numbers without accounting for negative numbers
    - Incorrectly updating the min/max tracking variables in the linear scan approach
    - Not handling edge cases like arrays with fewer than 3 elements (though the problem constrains this)
    
    **Real-world Applications:**
    
    - Financial portfolio optimization (maximizing returns)
    - Resource allocation problems
    - Optimization problems in operations research

=== "🔍 Problem Statement"

    Given an array of integers `arr`, replace each element with its rank. The rank represents how large the element is - the rank of an element is its position in the array after sorting, starting from 1.
    
    **Example 1:**
    ```
    Input: arr = [40,10,20,30]
    Output: [4,1,2,3]
    Explanation: 40 is the largest element. 10 is the smallest. 20 is the second smallest. 30 is the third smallest.
    ```
    
    **Example 2:**
    ```
    Input: arr = [100,100,100]
    Output: [1,1,1]
    Explanation: Same elements share the same rank.
    ```
    
    **Example 3:**
    ```
    Input: arr = [37,12,28,9,100,56,80,5,12]
    Output: [5,3,4,2,8,6,7,1,3]
    ```
    
    **Constraints:**
    - 0 <= arr.length <= 10^5
    - -10^9 <= arr[i] <= 10^9

=== "💡 Solution Approach"

    **Key Insight:**
    
    To find the rank of each element, we need to:
    1. Create a sorted version of the array
    2. Remove duplicates (as duplicate elements should have the same rank)
    3. Map each unique value to its position in the sorted array (plus 1, as ranks start from 1)
    4. Transform the original array by replacing each element with its rank
    
    **Step-by-step:**
    
    1. Handle the empty array case first.
    2. Create a sorted array of unique elements using a set.
    3. Create a mapping from each unique value to its rank.
    4. Transform the original array using this mapping.
    
    **Why it works:**
    
    The sorted array with duplicates removed gives us the relative ordering of all unique elements. By creating a map from value to position, we can efficiently assign ranks to each element in the original array.

=== "💻 Implementation"

    ```python
    def arrayRankTransform(arr):
        """
        Time: O(n log n) for sorting
        Space: O(n) for the sorted array and mapping
        """
        if not arr:
            return []
            
        # Create a sorted copy without duplicates
        sorted_arr = sorted(set(arr))
        
        # Create rank mapping: value -> rank
        rank_map = {val: i + 1 for i, val in enumerate(sorted_arr)}
        
        # Replace each element with its rank
        return [rank_map[num] for num in arr]
    ```

=== "🔄 Alternative Approaches"

    **Approach: Using Direct Indexing**
    
    This approach uses a list and indexing instead of a dictionary:
    
    ```python
    def arrayRankTransform(arr):
        """
        Time: O(n log n)
        Space: O(n)
        """
        if not arr:
            return []
            
        # Create a list of (value, index) pairs
        indexed = [(val, i) for i, val in enumerate(arr)]
        
        # Sort by value
        indexed.sort()
        
        # Assign ranks, handling duplicates
        result = [0] * len(arr)
        rank = 1
        
        for i in range(len(indexed)):
            value, original_index = indexed[i]
            
            # If this is a new value or the first element, assign a new rank
            if i == 0 or value > indexed[i-1][0]:
                rank = i + 1
                
            result[original_index] = rank
            
        return result
    ```
    
    **Approach: Using NumPy (for large arrays)**
    
    If NumPy is available and the array is large, this can be more efficient:
    
    ```python
    import numpy as np
    
    def arrayRankTransform(arr):
        """
        Time: O(n log n)
        Space: O(n)
        """
        if not arr:
            return []
            
        # Convert to numpy array for efficient operations
        np_arr = np.array(arr)
        
        # Get sorting indices and ranks
        sorted_indices = np.argsort(np_arr)
        ranks = np.empty_like(sorted_indices)
        
        # Assign ranks
        rank = 1
        for i in range(len(sorted_indices)):
            idx = sorted_indices[i]
            
            # First element or different from previous
            if i == 0 or np_arr[sorted_indices[i]] > np_arr[sorted_indices[i-1]]:
                rank = i + 1
                
            ranks[idx] = rank
            
        return ranks.tolist()
    ```

=== "💭 Tips & Insights"

    **Interview Tips:**
    
    1. **Edge Cases:** Always check for empty arrays first.
    
    2. **Handling Duplicates:** Remember that duplicates should have the same rank. This is why we use a set to get unique values before creating the rank mapping.
    
    3. **Optimization:** For very large arrays, consider using more efficient data structures or libraries like NumPy.
    
    4. **Follow-up Questions:**
       - How would you handle very large arrays that don't fit in memory?
       - Can you optimize for the case where the range of values is small?
       - How would you modify your solution if ranks should be consecutive (no gaps)?
    
    **Common Mistakes:**
    
    - Not handling duplicates correctly
    - Starting ranks from 0 instead of 1
    - Confusing indices with ranks
    - Not accounting for empty arrays
    
    **Real-world Applications:**
    
    - Ranking in competitions or leaderboards
    - Data analysis and statistics (computing percentiles)
    - Non-parametric statistical tests that use ranks
    - Feature scaling in machine learning (rank transformation)

=== "� Problem Statement"

    Given an array of distinct integers `arr`, find all pairs of elements with the minimum absolute difference.
    
    **Example 1:**
    ```
    Input: arr = [4,2,1,3]
    Output: [[1,2],[2,3],[3,4]]
    Explanation: The minimum absolute difference is 1, and there are three pairs with this difference: [1,2], [2,3], and [3,4].
    ```
    
    **Example 2:**
    ```
    Input: arr = [1,3,6,10,15]
    Output: [[1,3]]
    Explanation: The minimum absolute difference is 2, and there is only one pair with this difference: [1,3].
    ```
    
    **Example 3:**
    ```
    Input: arr = [3,8,-10,23,19,-4,-14,27]
    Output: [[-14,-10],[19,23],[23,27]]
    Explanation: The minimum absolute difference is 4.
    ```
    
    **Constraints:**
    - 2 <= arr.length <= 10^5
    - -10^6 <= arr[i] <= 10^6
    - All elements in arr are distinct

=== "💡 Solution Approach"

    **Key Insight:**
    
    The absolute difference between two elements is their distance on the number line. If we sort the array, then the minimum absolute difference can only occur between adjacent elements in the sorted array. This allows us to find the minimum difference in a single pass after sorting.
    
    **Step-by-step:**
    
    1. Sort the array in ascending order.
    2. Iterate through the array to find the minimum difference between adjacent elements.
    3. Make a second pass to collect all pairs with this minimum difference.
    
    **Why it works:**
    
    After sorting, any pair with the minimum absolute difference must be adjacent in the sorted array. If two non-adjacent elements had a smaller difference, it would contradict the sorting property.

=== "💻 Implementation"

    ```python
    def minimumAbsDifference(arr):
        """
        Time: O(n log n) for sorting
        Space: O(r) where r is the number of pairs with minimum difference
        """
        # Sort the array
        arr.sort()
        
        # Find the minimum difference
        min_diff = float('inf')
        for i in range(1, len(arr)):
            min_diff = min(min_diff, arr[i] - arr[i-1])
        
        # Collect all pairs with minimum difference
        result = []
        for i in range(1, len(arr)):
            if arr[i] - arr[i-1] == min_diff:
                result.append([arr[i-1], arr[i]])
                
        return result
    ```

=== "🔄 Alternative Approaches"

    **Approach: One-Pass Solution**
    
    We can optimize to find the minimum difference and collect pairs in a single pass:
    
    ```python
    def minimumAbsDifference(arr):
        """
        Time: O(n log n) for sorting
        Space: O(r) where r is the number of pairs with minimum difference
        """
        # Sort the array
        arr.sort()
        
        min_diff = float('inf')
        result = []
        
        # Find minimum difference and collect pairs in one pass
        for i in range(1, len(arr)):
            diff = arr[i] - arr[i-1]
            
            if diff < min_diff:
                min_diff = diff
                result = [[arr[i-1], arr[i]]]
            elif diff == min_diff:
                result.append([arr[i-1], arr[i]])
                
        return result
    ```
    
    **Approach: Using Bucket Sort (for limited range)**
    
    If the range of values is limited, bucket sort can provide a more efficient solution:
    
    ```python
    def minimumAbsDifference(arr):
        """
        Time: O(n + k) where k is the range of values
        Space: O(n + k)
        Note: Only efficient for small ranges
        """
        min_val = min(arr)
        max_val = max(arr)
        range_size = max_val - min_val
        
        # Use bucket sort if range is reasonable
        if range_size > len(arr) * 100:  # arbitrary threshold
            # Fall back to regular sort for large ranges
            return minimumAbsDifference_regular(arr)
            
        # Create a boolean array to represent presence of values
        present = [False] * (range_size + 1)
        for num in arr:
            present[num - min_val] = True
            
        # Find minimum difference and collect pairs
        min_diff = float('inf')
        prev = -1
        pairs = []
        
        for i in range(range_size + 1):
            if present[i]:
                if prev != -1:
                    diff = i - prev
                    if diff < min_diff:
                        min_diff = diff
                        pairs = [[prev + min_val, i + min_val]]
                    elif diff == min_diff:
                        pairs.append([prev + min_val, i + min_val])
                prev = i
                
        return pairs
    ```

=== "💭 Tips & Insights"

    **Interview Tips:**
    
    1. **Optimizing the Algorithm:** The one-pass approach is more efficient than the two-pass approach, as it avoids a second traversal of the array.
    
    2. **Edge Cases to Consider:**
       - Array with only two elements (there's only one pair)
       - Arrays with negative numbers (sorting handles this correctly)
       - Arrays where multiple pairs have the minimum difference
    
    3. **Follow-up Questions:**
       - How would you handle very large arrays that don't fit in memory?
       - Can you optimize for the case where the range of values is small?
       - What if we want pairs with the k-th minimum absolute difference?
    
    4. **Space Complexity Analysis:**
       - The result size can be up to O(n) in the worst case, if every adjacent pair has the same difference.
    
    **Common Mistakes:**
    
    - Using a brute-force approach that checks all pairs (O(n²) time)
    - Not sorting the array first
    - Incorrectly calculating the difference (e.g., forgetting to use absolute value for non-sorted arrays)
    - Not handling edge cases like arrays with only two elements
    
    **Real-world Applications:**
    
    - Finding closest pairs of points in computational geometry
    - Identifying similar items in recommendation systems
    - Data clustering algorithms
    - Time series analysis for detecting similar patterns
