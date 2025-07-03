# Arrays: Medium Problems

## ⚡ Intermediate Array Challenges

These problems require more sophisticated techniques and combining multiple patterns. Master these to excel in technical interviews and build advanced problem-solving skills.

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Three Sum | Two Pointers + Sort | Medium | O(n²) | O(1) |
    | 2 | Container With Most Water | Two Pointers | Medium | O(n) | O(1) |
    | 3 | Product of Array Except Self | Prefix/Suffix Arrays | Medium | O(n) | O(1) |
    | 4 | Rotate Array | Array Rotation | Medium | O(n) | O(1) |
    | 5 | Find First and Last Position | Binary Search | Medium | O(log n) | O(1) |
    | 6 | Spiral Matrix | Matrix Traversal | Medium | O(m×n) | O(1) |
    | 7 | Set Matrix Zeroes | Matrix Manipulation | Medium | O(m×n) | O(1) |
    | 8 | Subarray Sum Equals K | Prefix Sum + HashMap | Medium | O(n) | O(n) |
    | 9 | 3Sum Closest | Two Pointers | Medium | O(n²) | O(1) |
    | 10 | Sort Colors (Dutch Flag) | Three Pointers | Medium | O(n) | O(1) |
    | 11 | Next Permutation | Array Manipulation | Medium | O(n) | O(1) |
    | 12 | Search in Rotated Sorted Array | Modified Binary Search | Medium | O(log n) | O(1) |
    | 13 | Jump Game | Greedy/DP | Medium | O(n) | O(1) |
    | 14 | Merge Intervals | Interval Processing | Medium | O(n log n) | O(n) |
    | 15 | Insert Interval | Interval Merging | Medium | O(n) | O(n) |

=== "🎯 Advanced Patterns"

    **🔄 Array Rotation & Reversal:**
    - In-place rotation techniques
    - Cyclic replacements
    - Reverse-based algorithms
    
    **🎭 Two/Three Pointers:**
    - Sorted array optimizations
    - Avoiding duplicate triplets
    - Window sliding techniques
    
    **📊 Matrix Operations:**
    - Spiral traversal patterns
    - In-place matrix modifications
    - Space-optimized algorithms
    
    **🔍 Modified Binary Search:**
    - Rotated array searches
    - Range finding algorithms
    - Peak element detection

=== "⚡ Interview Strategy"

    **💡 Problem Recognition:**
    
    - **Sum Problems**: Consider two pointers after sorting
    - **Subarray Questions**: Think prefix sums or sliding window
    - **Matrix Problems**: Look for in-place manipulation patterns
    - **Search Problems**: Modified binary search for sorted variations
    
    **🎪 Common Tricks:**
    
    - Sorting to enable two pointers
    - Using array indices as hash map keys
    - In-place algorithms using constant extra space
    - Prefix/suffix array techniques

---

## Problem 1: Three Sum

**Difficulty:** Medium  
**Pattern:** Two Pointers + Sorting  
**Time:** O(n²) | **Space:** O(1)

=== "Problem Statement"

    Given an integer array `nums`, return all unique triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.

    **Example:**
    ```text
    Input: nums = [-1,0,1,2,-1,-4]
    Output: [[-1,-1,2],[-1,0,1]]
    Explanation: nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0
    ```

=== "Optimal Solution"

    ```python
    def three_sum(nums):
        """
        Two pointers approach after sorting - O(n²) time, O(1) space.
        
        Key insight: Fix first element, then use two pointers for remaining two.
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
    result = three_sum(nums)
    print(f"Three sum triplets: {result}")  # [[-1, -1, 2], [-1, 0, 1]]
    ```

=== "Brute Force"

    ```python
    def three_sum_brute_force(nums):
        """
        Brute force approach - O(n³) time, O(1) space.
        Check all possible triplets.
        """
        result = []
        n = len(nums)
        
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if nums[i] + nums[j] + nums[k] == 0:
                        triplet = sorted([nums[i], nums[j], nums[k]])
                        if triplet not in result:
                            result.append(triplet)
        
        return result

    # Test
    nums = [-1, 0, 1, 2, -1, -4]
    result = three_sum_brute_force(nums)
    print(f"Brute force result: {result}")
    ```

=== "Hash Set Approach"

    ```python
    def three_sum_hash_set(nums):
        """
        Hash set approach - O(n²) time, O(n) space.
        For each pair, check if complement exists.
        """
        result = []
        nums.sort()  # Sort to handle duplicates
        n = len(nums)
        
        for i in range(n - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            
            seen = set()
            target = -nums[i]
            
            for j in range(i + 1, n):
                complement = target - nums[j]
                
                if complement in seen:
                    result.append([nums[i], complement, nums[j]])
                    # Skip duplicates
                    while j + 1 < n and nums[j] == nums[j + 1]:
                        j += 1
                
                seen.add(nums[j])
        
        return result

    # Test
    nums = [-1, 0, 1, 2, -1, -4]
    result = three_sum_hash_set(nums)
    print(f"Hash set result: {result}")
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Sorting enables two pointers technique and easy duplicate handling
    - Fix one element, then solve two sum for remaining elements
    - Skip duplicates at all three positions to avoid duplicate triplets
    
    **⚡ Interview Tips:**
    
    - Ask: "Should results contain duplicate triplets?" (Usually no)
    - Mention: "Can extend to k-sum problem" (Shows pattern recognition)
    - Optimize: "Early termination when nums[i] > 0" (All remaining positive)
    
    **🔍 Extensions:**
    
    - 3Sum Closest: find triplet closest to target
    - 4Sum: extend to four numbers
    - 3Sum Smaller: count triplets with sum less than target

---

## Problem 2: Container With Most Water

**Difficulty:** Medium  
**Pattern:** Two Pointers  
**Time:** O(n) | **Space:** O(1)

=== "Problem Statement"

    Given an array of heights, find two lines that together with the x-axis form a container that holds the most water.

    **Example:**
    ```text
    Input: height = [1,8,6,2,5,4,8,3,7]
    Output: 49
    Explanation: The lines at indices 1 and 8 form the container with area 49
    ```

=== "Optimal Solution"

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

    # Test
    height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    result = max_area(height)
    print(f"Max area: {result}")  # 49
    ```

=== "Brute Force"

    ```python
    def max_area_brute_force(height):
        """
        O(n²) brute force solution.
        Check all possible pairs of lines.
        """
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
    result = max_area_brute_force(height)
    print(f"Max area (brute force): {result}")
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Two pointers work because moving the taller line won't increase area
    - Area = min(height[i], height[j]) × (j - i)
    - Always move the pointer with smaller height
    
    **⚡ Interview Tips:**
    
    - Explain why two pointers work: width decreases, so need height increase
    - Draw example to visualize the container concept
    - Mention that we're looking for maximum rectangular area
    
    **🔍 Related Problems:**
    
    - Largest Rectangle in Histogram
    - Trapping Rain Water
    - Maximal Rectangle

---

## Problem 3: Product of Array Except Self

**Difficulty:** Medium  
**Pattern:** Prefix/Suffix Product  
**Time:** O(n) | **Space:** O(1)

=== "Problem Statement"

    Given an array `nums`, return an array `answer` such that `answer[i]` is equal to the product of all elements except `nums[i]`. You cannot use division.

    **Example:**
    ```text
    Input: nums = [1,2,3,4]
    Output: [24,12,8,6]
    Explanation: 24=2×3×4, 12=1×3×4, 8=1×2×4, 6=1×2×3
    ```

=== "Optimal Solution"

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

    # Test
    nums = [1, 2, 3, 4]
    result = product_except_self(nums)
    print(f"Product except self: {result}")  # [24, 12, 8, 6]
    ```

=== "With Extra Arrays"

    ```python
    def product_except_self_verbose(nums):
        """
        Using separate left and right arrays for clarity.
        O(n) space approach.
        """
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
    result = product_except_self_verbose(nums)
    print(f"Product except self (verbose): {result}")
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Break problem into left products and right products
    - Use output array to store left products, then multiply with right products
    - No division needed - build products from both directions
    
    **⚡ Interview Tips:**
    
    - Ask about division constraint upfront
    - Explain the two-pass approach clearly
    - Mention space optimization (reusing output array)
    
    **🔍 Follow-up Questions:**
    
    - What if division was allowed? (Sum all, divide by each)
    - Handle zeros in array? (Count zeros, special cases)
    - Implement in one pass? (Not possible without extra space)

---

## Problem 4: Rotate Array

**Difficulty:** Medium  
**Pattern:** Array Manipulation  
**Time:** O(n) | **Space:** O(1)

=== "Problem Statement"

    Given an array, rotate it to the right by `k` steps, where `k` is non-negative.

    **Example:**
    ```text
    Input: nums = [1,2,3,4,5,6,7], k = 3
    Output: [5,6,7,1,2,3,4]
    Explanation: Rotate right by 3 positions
    ```

=== "Optimal (Reversal)"

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

    # Test
    nums = [1, 2, 3, 4, 5, 6, 7]
    print(f"Original: {nums}")
    rotate(nums, 3)
    print(f"Rotated: {nums}")  # [5, 6, 7, 1, 2, 3, 4]
    ```

=== "Extra Space"

    ```python
    def rotate_extra_space(nums, k):
        """
        Rotate using extra array - O(n) space.
        Simple but uses additional memory.
        """
        n = len(nums)
        k = k % n
        
        rotated = [0] * n
        
        for i in range(n):
            rotated[(i + k) % n] = nums[i]
        
        # Copy back to original array
        for i in range(n):
            nums[i] = rotated[i]

    # Test
    nums = [1, 2, 3, 4, 5, 6, 7]
    rotate_extra_space(nums, 3)
    print(f"Rotated (extra space): {nums}")
    ```

=== "Cyclic Replacement"

    ```python
    def rotate_cyclic(nums, k):
        """
        Rotate using cyclic replacements.
        Move elements to their final position one cycle at a time.
        """
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

    # Test
    nums = [1, 2, 3, 4, 5, 6, 7]
    rotate_cyclic(nums, 3)
    print(f"Rotated (cyclic): {nums}")
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Reversal method is most elegant: reverse whole, then reverse parts
    - Handle k > n case with modulo operation
    - Cyclic replacement avoids extra space but is complex
    
    **⚡ Interview Tips:**
    
    - Start with simple extra space solution, then optimize
    - Draw out the reversal steps to visualize
    - Mention in-place requirement often comes up
    
    **🔍 Variations:**
    
    - Rotate left by k steps
    - Rotate 2D matrix
    - Check if array is rotation of another

---

## Problem 5: Find First and Last Position of Element

**Difficulty:** Medium  
**Pattern:** Binary Search  
**Time:** O(log n) | **Space:** O(1)

=== "Problem Statement"

    Given a sorted array and a target value, return the starting and ending position of the target. If not found, return `[-1, -1]`.

    **Example:**
    ```text
    Input: nums = [5,7,7,8,8,10], target = 8
    Output: [3,4]
    Explanation: Target 8 appears at indices 3 and 4
    ```

=== "Optimal Solution"

    ```python
    def search_range(nums, target):
        """
        Find first and last position using binary search.
        
        Use two separate binary searches: one for leftmost, one for rightmost.
        """
        def find_first(nums, target):
            left, right = 0, len(nums) - 1
            first_pos = -1
            
            while left <= right:
                mid = left + (right - left) // 2
                
                if nums[mid] == target:
                    first_pos = mid
                    right = mid - 1  # Keep searching left
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return first_pos
        
        def find_last(nums, target):
            left, right = 0, len(nums) - 1
            last_pos = -1
            
            while left <= right:
                mid = left + (right - left) // 2
                
                if nums[mid] == target:
                    last_pos = mid
                    left = mid + 1  # Keep searching right
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return last_pos
        
        first = find_first(nums, target)
        if first == -1:
            return [-1, -1]
        
        last = find_last(nums, target)
        return [first, last]

    # Test
    nums = [5, 7, 7, 8, 8, 10]
    result = search_range(nums, 8)
    print(f"Range of 8: {result}")  # [3, 4]
    ```

=== "Linear Search"

    ```python
    def search_range_linear(nums, target):
        """
        Linear approach - O(n) time.
        Simple but not optimal for sorted array.
        """
        first = last = -1
        
        for i, num in enumerate(nums):
            if num == target:
                if first == -1:
                    first = i
                last = i
        
        return [first, last]

    # Test
    nums = [5, 7, 7, 8, 8, 10]
    result = search_range_linear(nums, 8)
    print(f"Range (linear): {result}")
    ```

=== "Single Binary Search"

    ```python
    def search_range_single(nums, target):
        """
        Find any occurrence first, then expand.
        Still O(log n) + O(k) where k is range size.
        """
        # Find any occurrence
        left, right = 0, len(nums) - 1
        found_idx = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                found_idx = mid
                break
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        if found_idx == -1:
            return [-1, -1]
        
        # Expand to find boundaries
        first = last = found_idx
        
        while first > 0 and nums[first - 1] == target:
            first -= 1
        
        while last < len(nums) - 1 and nums[last + 1] == target:
            last += 1
        
        return [first, last]

    # Test
    nums = [5, 7, 7, 8, 8, 10]
    result = search_range_single(nums, 8)
    print(f"Range (single search): {result}")
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Use two modified binary searches: one biased left, one biased right
    - When target found, continue searching in appropriate direction
    - Handle edge cases: empty array, target not found
    
    **⚡ Interview Tips:**
    
    - Explain why two separate searches are needed
    - Draw out the binary search modifications
    - Mention this pattern applies to many "find boundary" problems
    
    **🔍 Related Problems:**
    
    - Search Insert Position
    - Find Peak Element
    - Search in Rotated Sorted Array

---

## Problem 6: Spiral Matrix

**Difficulty:** Medium  
**Pattern:** Matrix Traversal  
**Time:** O(m×n) | **Space:** O(1)

=== "Problem Statement"

    Given an `m x n` matrix, return all elements in spiral order (clockwise from outside to inside).

    **Example:**
    ```text
    Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
    Output: [1,2,3,6,9,8,7,4,5]
    ```

=== "Boundary Tracking"

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

    # Test
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    result = spiral_order(matrix)
    print(f"Spiral order: {result}")  # [1,2,3,6,9,8,7,4,5]
    ```

=== "Direction Vectors"

    ```python
    def spiral_order_direction_vectors(matrix):
        """
        Using direction vectors for cleaner code.
        Track visited cells and change direction when needed.
        """
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
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = spiral_order_direction_vectors(matrix)
    print(f"Spiral (direction vectors): {result}")
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Track four boundaries: top, bottom, left, right
    - Move boundaries inward after completing each direction
    - Handle edge cases: single row/column matrices
    
    **⚡ Interview Tips:**
    
    - Draw the spiral path to visualize
    - Explain boundary conditions carefully
    - Mention that direction vectors approach is more scalable
    
    **🔍 Related Problems:**
    
    - Spiral Matrix II (generate spiral)
    - Rotate Image
    - Set Matrix Zeroes

---

## Problem 7: Set Matrix Zeroes

**Difficulty:** Medium  
**Pattern:** Matrix Manipulation  
**Time:** O(m×n) | **Space:** O(1)

=== "Problem Statement"

    Given an `m x n` matrix, if an element is 0, set its entire row and column to 0. Do it in-place.

    **Example:**
    ```text
    Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
    Output: [[1,0,1],[0,0,0],[1,0,1]]
    ```

=== "Optimal (O(1) Space)"

    ```python
    def set_zeroes(matrix):
        """
        Set matrix zeroes using first row/column as markers.
        
        Use matrix[0][0] to track if first row should be zero.
        Use separate variable for first column.
        """
        if not matrix or not matrix[0]:
            return
        
        rows, cols = len(matrix), len(matrix[0])
        
        # Check if first row/column should be zeroed
        first_row_zero = any(matrix[0][j] == 0 for j in range(cols))
        first_col_zero = any(matrix[i][0] == 0 for i in range(rows))
        
        # Use first row and column as markers
        for i in range(1, rows):
            for j in range(1, cols):
                if matrix[i][j] == 0:
                    matrix[0][j] = 0  # Mark column
                    matrix[i][0] = 0  # Mark row
        
        # Set zeroes based on markers
        for i in range(1, rows):
            for j in range(1, cols):
                if matrix[0][j] == 0 or matrix[i][0] == 0:
                    matrix[i][j] = 0
        
        # Handle first row
        if first_row_zero:
            for j in range(cols):
                matrix[0][j] = 0
        
        # Handle first column
        if first_col_zero:
            for i in range(rows):
                matrix[i][0] = 0

    # Test
    matrix = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    print(f"Original: {matrix}")
    set_zeroes(matrix)
    print(f"After setting zeroes: {matrix}")
    ```

=== "Extra Space (O(m+n))"

    ```python
    def set_zeroes_extra_space(matrix):
        """
        Track zero positions using additional arrays.
        Simpler logic but uses extra space.
        """
        if not matrix or not matrix[0]:
            return
        
        rows, cols = len(matrix), len(matrix[0])
        zero_rows = set()
        zero_cols = set()
        
        # Find all zero positions
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == 0:
                    zero_rows.add(i)
                    zero_cols.add(j)
        
        # Set entire rows to zero
        for row in zero_rows:
            for j in range(cols):
                matrix[row][j] = 0
        
        # Set entire columns to zero
        for col in zero_cols:
            for i in range(rows):
                matrix[i][col] = 0

    # Test
    matrix = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    set_zeroes_extra_space(matrix)
    print(f"Result (extra space): {matrix}")
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Use first row/column as storage to achieve O(1) space
    - Handle first row/column separately to avoid conflicts
    - Process markers after finding all zero positions
    
    **⚡ Interview Tips:**
    
    - Start with extra space solution, then optimize
    - Explain why we can't process while finding zeros
    - Draw example to show first row/column usage
    
    **🔍 Follow-up Questions:**
    
    - What if input is read-only? (Use hash sets)
    - Extend to 3D matrix? (Similar approach)
    - Undo the operation? (Track original zeros)

---

## Problem 8: Subarray Sum Equals K

**Difficulty:** Medium  
**Pattern:** Prefix Sum + HashMap  
**Time:** O(n) | **Space:** O(n)

=== "Problem Statement"

    Given an array of integers and an integer `k`, find the total number of continuous subarrays whose sum equals `k`.

    **Example:**
    ```text
    Input: nums = [1,1,1], k = 2
    Output: 2
    Explanation: [1,1] appears twice
    ```

=== "Optimal (Prefix Sum)"

    ```python
    def subarray_sum(nums, k):
        """
        Find count of subarrays with sum k using prefix sum and hashmap.
        
        If prefix_sum[j] - prefix_sum[i] = k, then subarray[i+1:j+1] sums to k.
        """
        count = 0
        prefix_sum = 0
        sum_count = {0: 1}  # Base case: empty prefix
        
        for num in nums:
            prefix_sum += num
            
            # Check if there's a prefix sum that makes current sum = k
            if prefix_sum - k in sum_count:
                count += sum_count[prefix_sum - k]
            
            # Add current prefix sum to map
            sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1
        
        return count

    # Test
    nums = [1, 1, 1]
    result = subarray_sum(nums, 2)
    print(f"Subarrays with sum 2: {result}")  # 2
    ```

=== "Brute Force"

    ```python
    def subarray_sum_brute_force(nums, k):
        """
        Check all possible subarrays - O(n²) time.
        Simple but inefficient approach.
        """
        count = 0
        n = len(nums)
        
        for i in range(n):
            current_sum = 0
            for j in range(i, n):
                current_sum += nums[j]
                if current_sum == k:
                    count += 1
        
        return count

    # Test
    nums = [1, 1, 1]
    result = subarray_sum_brute_force(nums, 2)
    print(f"Brute force result: {result}")
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Use prefix sum: if prefix[j] - prefix[i] = k, then subarray sums to k
    - HashMap stores count of each prefix sum seen so far
    - Initialize with {0: 1} to handle subarrays starting from index 0
    
    **⚡ Interview Tips:**
    
    - Explain prefix sum concept clearly
    - Walk through example step by step
    - Mention this pattern works for any target sum
    
    **🔍 Related Problems:**
    
    - Two Sum
    - Continuous Subarray Sum
    - Maximum Size Subarray Sum Equals k

---

## Problem 9: 3Sum Closest

**Difficulty:** Medium  
**Pattern:** Two Pointers  
**Time:** O(n²) | **Space:** O(1)

=== "Problem Statement"

    Given an array and a target, find three integers such that their sum is closest to the target.

    **Example:**
    ```text
    Input: nums = [-1,2,1,-4], target = 1
    Output: 2
    Explanation: Closest sum is -1 + 2 + 1 = 2
    ```

=== "Optimal Solution"

    ```python
    def three_sum_closest(nums, target):
        """
        Find three numbers with sum closest to target.
        
        Sort array, then use two pointers for each fixed element.
        """
        nums.sort()
        n = len(nums)
        closest_sum = float('inf')
        
        for i in range(n - 2):
            left, right = i + 1, n - 1
            
            while left < right:
                current_sum = nums[i] + nums[left] + nums[right]
                
                # Update closest if current is better
                if abs(current_sum - target) < abs(closest_sum - target):
                    closest_sum = current_sum
                
                if current_sum == target:
                    return target  # Perfect match
                elif current_sum < target:
                    left += 1
                else:
                    right -= 1
        
        return closest_sum

    # Test
    nums = [-1, 2, 1, -4]
    result = three_sum_closest(nums, 1)
    print(f"Closest sum to 1: {result}")  # 2
    ```

=== "Brute Force"

    ```python
    def three_sum_closest_brute_force(nums, target):
        """
        Check all possible triplets - O(n³) time.
        """
        n = len(nums)
        closest_sum = float('inf')
        
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    current_sum = nums[i] + nums[j] + nums[k]
                    
                    if abs(current_sum - target) < abs(closest_sum - target):
                        closest_sum = current_sum
        
        return closest_sum

    # Test
    nums = [-1, 2, 1, -4]
    result = three_sum_closest_brute_force(nums, 1)
    print(f"Brute force result: {result}")
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Similar to 3Sum but track closest distance instead of exact match
    - Sort array to enable two pointers optimization
    - Early return when exact target is found
    
    **⚡ Interview Tips:**
    
    - Explain difference from regular 3Sum problem
    - Show how sorting enables efficient search
    - Handle edge case: exactly 3 elements
    
    **🔍 Variations:**
    
    - k-Sum Closest
    - 3Sum Smaller (count triplets less than target)
    - 4Sum Closest

---

## Problem 10: Sort Colors (Dutch National Flag)

**Difficulty:** Medium  
**Pattern:** Three Pointers  
**Time:** O(n) | **Space:** O(1)

=== "Problem Statement"

    Given an array with values 0, 1, and 2 (representing colors), sort it in-place. Follow up: do it in one pass.

    **Example:**
    ```text
    Input: nums = [2,0,2,1,1,0]
    Output: [0,0,1,1,2,2]
    ```

=== "One Pass (Dutch Flag)"

    ```python
    def sort_colors(nums):
        """
        Sort colors in one pass using Dutch National Flag algorithm.
        
        Maintain three regions: [0...low-1], [low...high], [high+1...n-1]
        """
        low = mid = 0
        high = len(nums) - 1
        
        while mid <= high:
            if nums[mid] == 0:
                # Swap with low region
                nums[low], nums[mid] = nums[mid], nums[low]
                low += 1
                mid += 1
            elif nums[mid] == 1:
                # Already in correct position
                mid += 1
            else:  # nums[mid] == 2
                # Swap with high region
                nums[mid], nums[high] = nums[high], nums[mid]
                high -= 1
                # Don't increment mid (need to check swapped element)
        
        return nums

    # Test
    nums = [2, 0, 2, 1, 1, 0]
    print(f"Original: {nums}")
    sort_colors(nums)
    print(f"Sorted: {nums}")  # [0, 0, 1, 1, 2, 2]
    ```

=== "Counting Sort"

    ```python
    def sort_colors_counting(nums):
        """
        Count occurrences and reconstruct array.
        Two-pass solution.
        """
        # Count each color
        counts = [0, 0, 0]
        for num in nums:
            counts[num] += 1
        
        # Reconstruct array
        index = 0
        for color in range(3):
            for _ in range(counts[color]):
                nums[index] = color
                index += 1
        
        return nums

    # Test
    nums = [2, 0, 2, 1, 1, 0]
    result = sort_colors_counting(nums.copy())
    print(f"Counting sort: {result}")
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Dutch National Flag algorithm partitions array into three regions
    - Key insight: don't increment mid when swapping with high (unknown element)
    - Three pointers maintain invariants throughout process
    
    **⚡ Interview Tips:**
    
    - Explain why we don't increment mid after swapping with high
    - Draw the three regions to visualize
    - Mention this generalizes to k colors
    
    **🔍 Applications:**
    
    - Quick Sort partitioning
    - 3-way partitioning
    - Sorting with limited values

---

## Problem 11: Next Permutation

**Difficulty:** Medium  
**Pattern:** Array Manipulation  
**Time:** O(n) | **Space:** O(1)

=== "Problem Statement"

    Find the next lexicographically greater permutation. If not possible, return the smallest permutation.

    **Example:**
    ```text
    Input: nums = [1,2,3]
    Output: [1,3,2]
    Input: nums = [3,2,1]
    Output: [1,2,3]
    ```

=== "Optimal Solution"

    ```python
    def next_permutation(nums):
        """
        Find next lexicographically greater permutation.
        
        Algorithm:
        1. Find rightmost ascending pair
        2. Find next larger element to swap with
        3. Reverse suffix to get smallest arrangement
        """
        n = len(nums)
        
        # Step 1: Find rightmost ascending pair
        i = n - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        
        if i >= 0:  # Not the last permutation
            # Step 2: Find next larger element
            j = n - 1
            while nums[j] <= nums[i]:
                j -= 1
            
            # Swap
            nums[i], nums[j] = nums[j], nums[i]
        
        # Step 3: Reverse the suffix
        left, right = i + 1, n - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
        
        return nums

    # Test
    nums1 = [1, 2, 3]
    print(f"Next permutation of {nums1}: {next_permutation(nums1)}")
    
    nums2 = [3, 2, 1]
    print(f"Next permutation of {nums2}: {next_permutation(nums2)}")
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Find rightmost position where nums[i] < nums[i+1]
    - Swap with next larger element in suffix
    - Reverse suffix to get lexicographically smallest arrangement
    
    **⚡ Interview Tips:**
    
    - Work through examples step by step
    - Explain why reversing suffix works
    - Handle edge case: already largest permutation
    
    **🔍 Related Problems:**
    
    - Previous Permutation
    - Permutations
    - Generate all permutations

---

## Problem 12: Search in Rotated Sorted Array

**Difficulty:** Medium  
**Pattern:** Modified Binary Search  
**Time:** O(log n) | **Space:** O(1)

=== "Problem Statement"

    Given a rotated sorted array, search for a target value. Return its index or -1 if not found.

    **Example:**
    ```text
    Input: nums = [4,5,6,7,0,1,2], target = 0
    Output: 4
    ```

=== "Optimal Solution"

    ```python
    def search(nums, target):
        """
        Search in rotated sorted array using modified binary search.
        
        Key insight: One half is always normally sorted.
        """
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid
            
            # Check which half is sorted
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

    # Test
    nums = [4, 5, 6, 7, 0, 1, 2]
    result = search(nums, 0)
    print(f"Index of 0: {result}")  # 4
    ```

=== "Find Pivot First"

    ```python
    def search_with_pivot(nums, target):
        """
        Find pivot point first, then do regular binary search.
        """
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
        
        n = len(nums)
        pivot = find_pivot()
        
        # Search in both halves
        result = binary_search(0, pivot - 1, target)
        if result != -1:
            return result
        
        return binary_search(pivot, n - 1, target)

    # Test
    nums = [4, 5, 6, 7, 0, 1, 2]
    result = search_with_pivot(nums, 0)
    print(f"Index using pivot method: {result}")
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - At least one half of array is always normally sorted
    - Use sorted half to determine which direction to search
    - Handle duplicates by checking left <= mid condition carefully
    
    **⚡ Interview Tips:**
    
    - Draw the rotated array to visualize
    - Explain how to identify the sorted half
    - Consider edge cases: no rotation, single element
    
    **🔍 Follow-ups:**
    
    - Array with duplicates (Search in Rotated Sorted Array II)
    - Find minimum in rotated array
    - Find rotation count

---

## Problem 13: Jump Game

**Difficulty:** Medium  
**Pattern:** Greedy  
**Time:** O(n) | **Space:** O(1)

=== "Problem Statement"

    Given an array where each element represents maximum jump length, determine if you can reach the last index.

    **Example:**
    ```text
    Input: nums = [2,3,1,1,4]
    Output: true
    Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index
    ```

=== "Greedy Solution"

    ```python
    def can_jump(nums):
        """
        Determine if we can reach the end using greedy approach.
        
        Track the furthest position we can reach.
        """
        max_reach = 0
        
        for i in range(len(nums)):
            # If current position is unreachable
            if i > max_reach:
                return False
            
            # Update furthest reachable position
            max_reach = max(max_reach, i + nums[i])
            
            # Early termination if we can reach the end
            if max_reach >= len(nums) - 1:
                return True
        
        return True

    # Test
    nums1 = [2, 3, 1, 1, 4]
    print(f"Can jump {nums1}: {can_jump(nums1)}")  # True
    
    nums2 = [3, 2, 1, 0, 4]
    print(f"Can jump {nums2}: {can_jump(nums2)}")  # False
    ```

=== "Dynamic Programming"

    ```python
    def can_jump_dp(nums):
        """
        DP approach - less efficient but shows the pattern.
        """
        n = len(nums)
        dp = [False] * n
        dp[0] = True
        
        for i in range(1, n):
            for j in range(i):
                if dp[j] and j + nums[j] >= i:
                    dp[i] = True
                    break
        
        return dp[n - 1]

    # Test
    nums = [2, 3, 1, 1, 4]
    result = can_jump_dp(nums)
    print(f"DP result: {result}")
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Greedy approach: track maximum reachable position
    - If current position > max_reach, it's unreachable
    - Early termination when we can reach the end
    
    **⚡ Interview Tips:**
    
    - Start with DP approach, then optimize to greedy
    - Explain why greedy works (optimal substructure)
    - Handle edge cases: single element, empty array
    
    **🔍 Follow-ups:**
    
    - Jump Game II (minimum jumps to reach end)
    - Jump Game III (can reach any zero)
    - Jump Game IV (bidirectional jumps)

---

## Problem 14: Merge Intervals

**Difficulty:** Medium  
**Pattern:** Interval Processing  
**Time:** O(n log n) | **Space:** O(n)

=== "Problem Statement"

    Given a collection of intervals, merge all overlapping intervals.

    **Example:**
    ```text
    Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
    Output: [[1,6],[8,10],[15,18]]
    ```

=== "Optimal Solution"

    ```python
    def merge(intervals):
        """
        Merge overlapping intervals after sorting by start time.
        """
        if not intervals:
            return []
        
        # Sort intervals by start time
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        
        for current in intervals[1:]:
            last = merged[-1]
            
            # Check if intervals overlap
            if current[0] <= last[1]:
                # Merge by extending the end time
                last[1] = max(last[1], current[1])
            else:
                # No overlap, add new interval
                merged.append(current)
        
        return merged

    # Test
    intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
    result = merge(intervals)
    print(f"Merged intervals: {result}")  # [[1, 6], [8, 10], [15, 18]]
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Sort by start time to process intervals in order
    - Two intervals overlap if start ≤ previous end
    - Merge by taking max of end times
    
    **⚡ Interview Tips:**
    
    - Draw timeline to visualize merging
    - Explain why sorting is necessary
    - Consider edge cases: empty list, single interval
    
    **🔍 Related Problems:**
    
    - Insert Interval
    - Non-overlapping Intervals
    - Meeting Rooms

---

## Problem 15: Insert Interval

**Difficulty:** Medium  
**Pattern:** Interval Merging  
**Time:** O(n) | **Space:** O(n)

=== "Problem Statement"

    Given a sorted list of non-overlapping intervals and a new interval, insert and merge if necessary.

    **Example:**
    ```text
    Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
    Output: [[1,5],[6,9]]
    ```

=== "Optimal Solution"

    ```python
    def insert(intervals, new_interval):
        """
        Insert new interval and merge overlaps.
        
        Three phases: before overlap, merge overlaps, after overlap.
        """
        result = []
        i = 0
        n = len(intervals)
        
        # Phase 1: Add intervals that end before new interval starts
        while i < n and intervals[i][1] < new_interval[0]:
            result.append(intervals[i])
            i += 1
        
        # Phase 2: Merge overlapping intervals
        while i < n and intervals[i][0] <= new_interval[1]:
            # Merge with new interval
            new_interval[0] = min(new_interval[0], intervals[i][0])
            new_interval[1] = max(new_interval[1], intervals[i][1])
            i += 1
        
        result.append(new_interval)
        
        # Phase 3: Add remaining intervals
        while i < n:
            result.append(intervals[i])
            i += 1
        
        return result

    # Test
    intervals = [[1, 3], [6, 9]]
    new_interval = [2, 5]
    result = insert(intervals, new_interval)
    print(f"After inserting [2,5]: {result}")  # [[1, 5], [6, 9]]
    ```

=== "💡 Tips & Insights"

    **🎯 Key Insights:**
    
    - Process in three phases: before, during, after overlap
    - Merge by updating new interval boundaries
    - Input already sorted, so no need to sort again
    
    **⚡ Interview Tips:**
    
    - Explain the three-phase approach clearly
    - Show how merging updates new interval
    - Handle edge cases: insert at beginning/end
    
    **🔍 Variations:**
    
    - Remove Interval
    - Interval List Intersections
    - Employee Free Time

---

## 🎯 Problem-Solving Patterns Summary

### 1. Two Pointers with Sorting
- **Use when:** Need to find pairs/triplets with specific sum
- **Pattern:** Sort array, use two pointers to navigate
- **Examples:** Three Sum, 3Sum Closest, Container With Most Water

### 2. Prefix/Suffix Arrays
- **Use when:** Need cumulative information from left/right
- **Pattern:** Build prefix array, then use it for queries
- **Examples:** Product Except Self, Subarray Sum Equals K

### 3. Modified Binary Search
- **Use when:** Array has some sorted property (even if rotated)
- **Pattern:** Identify sorted half, decide search direction
- **Examples:** Search in Rotated Array, Find First/Last Position

### 4. Matrix Traversal
- **Use when:** Need to visit matrix elements in specific order
- **Pattern:** Use boundary tracking or direction vectors
- **Examples:** Spiral Matrix, Set Matrix Zeroes

### 5. Greedy Algorithms
- **Use when:** Local optimal choice leads to global optimum
- **Pattern:** Make best choice at each step
- **Examples:** Jump Game, Sort Colors

### 6. Interval Processing
- **Use when:** Working with ranges or time intervals
- **Pattern:** Sort by start time, then merge/process
- **Examples:** Merge Intervals, Insert Interval

## 💡 Advanced Tips for Medium Problems

!!! tip "Space Optimization"
    Many array problems can be solved in O(1) space using clever index manipulation or the array itself as storage.

!!! note "Edge Cases to Consider"
    - Empty arrays or single elements
    - All elements the same
    - Already sorted/reverse sorted arrays
    - Arrays with maximum/minimum constraints

!!! success "Pattern Recognition"
    - **Sum problems** → Two pointers after sorting
    - **Subarray problems** → Sliding window or prefix sums
    - **Matrix problems** → In-place manipulation or traversal patterns
    - **Search problems** → Modified binary search for sorted variations

## 🚀 Next Steps

Ready for the ultimate challenge? Try:
- [Hard Array Problems](hard-problems.md)
- Practice combining multiple patterns
- Focus on optimizing space complexity

---

*🎉 Excellent work! You've mastered medium array problems. Ready for [Hard Problems](hard-problems.md)?*
