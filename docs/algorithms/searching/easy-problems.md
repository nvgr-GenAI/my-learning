# Searching Algorithms - Easy Problems

## 🎯 Learning Objectives

Master fundamental searching algorithms and binary search applications:

- Binary search implementation and variants
- Search in modified arrays
- Two-pointer search techniques
- Basic optimization with search
- Search space reduction strategies

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Binary Search | Classic Binary Search | Easy | O(log n) | O(1) |
    | 2 | First Bad Version | Binary Search Variant | Easy | O(log n) | O(1) |
    | 3 | Search Insert Position | Binary Search Insertion | Easy | O(log n) | O(1) |
    | 4 | Sqrt(x) | Binary Search on Answer | Easy | O(log x) | O(1) |
    | 5 | Two Sum II (Sorted Array) | Two Pointers | Easy | O(n) | O(1) |
    | 6 | Valid Perfect Square | Binary Search | Easy | O(log n) | O(1) |
    | 7 | Arranging Coins | Binary Search/Math | Easy | O(log n) | O(1) |
    | 8 | Intersection of Two Arrays | Set/Binary Search | Easy | O(n + m) | O(n) |
    | 9 | Intersection of Two Arrays II | Two Pointers | Easy | O(n log n) | O(1) |
    | 10 | Peak Index in Mountain Array | Binary Search | Easy | O(log n) | O(1) |
    | 11 | Find Target in Rotated Array | Modified Binary Search | Easy | O(log n) | O(1) |
    | 12 | Search in BST | Tree Search | Easy | O(log n) | O(1) |
    | 13 | Guess Number Higher or Lower | Binary Search Game | Easy | O(log n) | O(1) |
    | 14 | Count Negative Numbers | Binary Search/Linear | Easy | O(m + n) | O(1) |
    | 15 | Find Smallest Letter Greater | Binary Search Circular | Easy | O(log n) | O(1) |

=== "🎯 Core Patterns"

    **🔍 Binary Search Fundamentals:**
    - Standard binary search implementation
    - Left and right boundary search
    - Search for insertion position
    - Handling duplicates
    
    **🎯 Search Space Design:**
    - Define search boundaries correctly
    - Choose appropriate mid calculation
    - Handle edge cases and termination
    - Verify search invariants
    
    **🔄 Search Variants:**
    - Search in rotated arrays
    - Search in 2D matrices
    - Search for peak elements
    - Search in infinite arrays
    
    **👥 Two-Pointer Techniques:**
    - Search in sorted arrays
    - Finding pairs with target sum
    - Intersection and merging

=== "💡 Solutions"

    === "Binary Search"
        ```python
        def search(nums, target):
            """
            Standard binary search implementation
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
    
    === "First Bad Version"
        ```python
        def firstBadVersion(n):
            """
            Find first bad version using minimum API calls
            Binary search to find first True in boolean array
            """
            left, right = 1, n
            
            while left < right:
                mid = left + (right - left) // 2
                
                if isBadVersion(mid):
                    right = mid  # First bad could be mid
                else:
                    left = mid + 1  # First bad is after mid
            
            return left
        ```
    
    === "Search Insert Position"
        ```python
        def searchInsert(nums, target):
            """
            Find position where target should be inserted
            """
            left, right = 0, len(nums)
            
            while left < right:
                mid = left + (right - left) // 2
                
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid
            
            return left
        ```
    
    === "Sqrt(x)"
        ```python
        def mySqrt(x):
            """
            Binary search for integer square root
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
            
            return right  # Floor of square root
        ```
    
    === "Two Sum II (Sorted Array)"
        ```python
        def twoSum(numbers, target):
            """
            Find two numbers that add up to target
            Use two pointers on sorted array
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
            
            return []  # Should never reach here per problem constraints
        ```
    
    === "Peak Index in Mountain Array"
        ```python
        def peakIndexInMountainArray(arr):
            """
            Find peak element in mountain array
            """
            left, right = 0, len(arr) - 1
            
            while left < right:
                mid = left + (right - left) // 2
                
                if arr[mid] < arr[mid + 1]:
                    left = mid + 1  # Peak is on the right
                else:
                    right = mid  # Peak is on the left or at mid
            
            return left
        ```
    
    === "Search in Rotated Sorted Array"
        ```python
        def search(nums, target):
            """
            Search in rotated sorted array
            """
            left, right = 0, len(nums) - 1
            
            while left <= right:
                mid = left + (right - left) // 2
                
                if nums[mid] == target:
                    return mid
                
                # Determine which half is sorted
                if nums[left] <= nums[mid]:  # Left half is sorted
                    if nums[left] <= target < nums[mid]:
                        right = mid - 1
                    else:
                        left = mid + 1
                else:  # Right half is sorted
                    if nums[mid] < target <= nums[right]:
                        left = mid + 1
                    else:
                        right = mid - 1
            
            return -1
        ```
    
    === "Intersection of Two Arrays"
        ```python
        def intersection(nums1, nums2):
            """
            Find intersection of two arrays
            """
            # Method 1: Using sets
            set1 = set(nums1)
            set2 = set(nums2)
            return list(set1 & set2)
            
            # Method 2: Binary search approach
            # set1 = set(nums1)
            # nums2.sort()
            # result = set()
            # 
            # for num in set1:
            #     if binary_search(nums2, num):
            #         result.add(num)
            # 
            # return list(result)
        
        def binary_search(arr, target):
            left, right = 0, len(arr) - 1
            
            while left <= right:
                mid = left + (right - left) // 2
                if arr[mid] == target:
                    return True
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return False
        ```

=== "📊 Key Insights"

    **🔧 Binary Search Template:**
    ```python
    def binary_search_template(arr, target):
        left, right = 0, len(arr) - 1  # or len(arr) for insertion
        
        while left <= right:  # or left < right for some variants
            mid = left + (right - left) // 2  # Avoid overflow
            
            if condition_met(arr[mid], target):
                return mid  # or process result
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1  # or left for insertion point
    ```
    
    **⚡ Search Strategy Selection:**
    - **Standard Binary Search**: Find exact element
    - **Left Boundary**: First occurrence of target
    - **Right Boundary**: Last occurrence of target
    - **Insertion Point**: Where to insert element
    - **Peak Finding**: Local maximum in array
    
    **🎯 Two-Pointer Applications:**
    - **Sorted Arrays**: Finding pairs, intersections
    - **Opposite Directions**: Start from both ends
    - **Same Direction**: Fast and slow pointers
    - **Window Sliding**: Maintain valid window

=== "🚀 Advanced Tips"

    **💡 Problem-Solving Strategy:**
    1. **Identify Search Space**: What are we searching over?
    2. **Define Boundaries**: Inclusive or exclusive bounds?
    3. **Choose Condition**: What determines search direction?
    4. **Handle Edge Cases**: Empty arrays, single elements
    5. **Verify Termination**: Will loop always terminate?
    
    **🔍 Common Pitfalls:**
    - **Integer Overflow**: Use `mid = left + (right - left) // 2`
    - **Infinite Loops**: Ensure search space reduces each iteration
    - **Boundary Errors**: Off-by-one errors in loop conditions
    - **Duplicate Handling**: Decide how to handle equal elements
    
    **🏆 Best Practices:**
    - Use consistent boundary conventions (inclusive vs exclusive)
    - Test with arrays of size 0, 1, 2 to verify correctness
    - Consider iterative vs recursive implementation
    - Draw diagrams to visualize search space reduction
    - Verify loop invariants throughout the algorithm

## 📝 Summary

These easy searching problems focus on:

- **Binary Search Fundamentals** with various applications
- **Search Space Design** and boundary handling
- **Two-Pointer Techniques** for sorted array problems
- **Modified Arrays** like rotated and mountain arrays
- **Search Optimization** for different problem types

Master these patterns to build a strong foundation for more complex searching algorithms and optimization problems!
